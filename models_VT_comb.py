
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from timm.models.layers import to_2tuple

from transformers import AutoTokenizer, T5EncoderModel
import random
import os
import xformers.ops

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps                                  #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#               Modality Conditioning                                           #
#################################################################################

############################   Text Conditioning    ##############################

class TextEmbedder(nn.Module):
    """
    Embeds text captions into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, hidden_size, dropout_prob):
        super().__init__()

        text_encoder_name = 'google/flan-t5-large'
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_name)
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.dropout_prob = dropout_prob

    def encode_text(self, prompt):
        device = self.text_encoder.device
        batch = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt")
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)

        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        boolean_encoder_mask = (attention_mask == 1).to(device)
        sum_embeddings = (encoder_hidden_states * boolean_encoder_mask.unsqueeze(-1)).sum(dim=-2)
        valid_token_count = boolean_encoder_mask.sum(dim=-1, keepdim=True)
        mean_embeddings = sum_embeddings / valid_token_count
        return encoder_hidden_states, attention_mask, mean_embeddings

    def forward(self, labels, train):
        if train:
            labels = ["" if random.random() < self.dropout_prob else label for label in labels]
        text_embeds, text_mask, text_mean = self.encode_text(labels)
        return text_embeds, text_mask, text_mean 
    
#################################################################################

############################   Visual Conditioning    ##############################
class VisualEmbedder(nn.Module):
    """
    Embeds visual frames into vector representations.
    """
    def __init__(self, hidden_size, dropout_prob, vid_feat_dir, total_num_frames):
        super().__init__()
        BASE_DIR = vid_feat_dir
        self.vid_index_map = torch.load(os.path.join(BASE_DIR, 'vis_idx_map.pt'))
        self.vid_embeds = torch.load(os.path.join(BASE_DIR, 'vis_feat.pt'))
        self.vid_masks = torch.load(os.path.join(BASE_DIR, 'vis_mask.pt'))
        self.vid_flow = torch.load(os.path.join(BASE_DIR, 'vis_flow.pt'))

        self.visual_dim = 512; self.flow_dim = 128; self.idx_dim = 128

        comb_dim = self.visual_dim + self.flow_dim + self.idx_dim
        self.mlp_combine = nn.Sequential(
            nn.Linear(comb_dim, hidden_size, bias=True), # 512 + 128 + 128
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self.total_num_frames = total_num_frames
        self.dropout_prob = dropout_prob
        self.EMPTY = "EMPTY"

        # getting index embeddings before hand
        self.index_embed = self.index_embedding(self.total_num_frames, self.idx_dim, normalize=True)

    @staticmethod
    def index_embedding(total_num_frames, embed_dim, normalize=False):
        position = torch.arange(0, total_num_frames, dtype=torch.float32).unsqueeze(1)
        if normalize:
            position = position / total_num_frames
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(total_num_frames, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def flow_embedding(self, flow_values, mask, embed_dim, normalize=False, scale_factor=10000, epsilon=1e-8):
        if normalize:
            present_mask = mask == 1
            present_flow_values = flow_values[present_mask]
            
            if present_flow_values.numel() != 0:
                flow_min = present_flow_values.min()
                flow_max = present_flow_values.max()
                
                # Prevent division by zero
                denominator = flow_max - flow_min + epsilon
                flow_values_normalized = flow_values.clone()
                flow_values_normalized[present_mask] = (flow_values[present_mask] - flow_min) / denominator
                flow_values = flow_values_normalized
        
        # Ensure flow_values are within a reasonable range
        flow_values = torch.clamp(flow_values, min=-1e4, max=1e4)
        flow_values = flow_values.unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32, device=flow_values.device) *
            (-math.log(scale_factor) / embed_dim)
        )
        pe = torch.zeros(flow_values.size(0), embed_dim, device=flow_values.device)
        
        # Compute positional encoding
        angles = flow_values * div_term
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        
        # Check for NaNs in pe
        if not torch.isfinite(pe).all():
            print("Warning: pe contains NaN or Inf values.")
        
        return pe

    def encode_visual(self, vids, curr_device):
        #cross attention
        batch_vis_embed = torch.zeros(len(vids), self.total_num_frames, self.visual_dim).to(curr_device)
        batch_mask = torch.zeros(len(vids), self.total_num_frames).to(curr_device)
        batch_flow_embed = torch.zeros(len(vids), self.total_num_frames, self.flow_dim).to(curr_device)
        batch_mean = torch.zeros(len(vids), self.visual_dim).to(curr_device)

        for batch_idx, vid in enumerate(vids):
            if vid == self.EMPTY:
                continue

            vid_idx = self.vid_index_map[vid]
            batch_mask[batch_idx] = self.vid_masks[vid_idx]
            batch_vis_embed[batch_idx] = self.vid_embeds[vid_idx]  
            batch_flow_embed[batch_idx] = self.flow_embedding(self.vid_flow[vid_idx], self.vid_masks[vid_idx], self.flow_dim, normalize=True)
            
            curr_embed = self.vid_embeds[vid_idx].to(curr_device)
            valid_mask = batch_mask[batch_idx] == 1
            valid_embeds = curr_embed[valid_mask]

            if valid_embeds.shape[0] > 0:
                batch_mean[batch_idx] = valid_embeds.mean(dim=0)
            else:
                batch_mean[batch_idx] = torch.zeros(self.visual_dim)                  
        
        index_embed_expanded = self.index_embed.unsqueeze(0).expand(len(vids), -1, -1).to(curr_device)
        batch_embed = torch.cat((batch_vis_embed, batch_flow_embed, index_embed_expanded), dim=-1)
        return batch_embed, batch_mask, batch_mean

    def forward(self, vids, train, curr_device):
        if train:
            vids_vis = [self.EMPTY if random.random() < self.dropout_prob else vid for vid in vids]
        else:
            vids_vis = vids

        visual_embeds, visual_masks, visual_embeds_mean = self.encode_visual(vids_vis, curr_device)#.to(curr_device)
        visual_embeds = self.mlp_combine(visual_embeds)
        return visual_embeds, visual_masks, visual_embeds_mean

#################################################################################

class AudPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__() 

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

#################################################################################
#                                 Core SiT Model                                #
#################################################################################

class SiTCrossAttnBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_masking=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.norm_cross_txt = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn_text = MultiHeadCrossAttention(hidden_size, num_heads, use_masking=use_masking, input_cond='text')
        
        self.norm_cross_vis = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn_vis = MultiHeadCrossAttention(hidden_size, num_heads, use_masking=use_masking, input_cond='visual')

        self.weight_text = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, c, c_txt, c_txt_mask, c_vis, c_vis_mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
                
        ca_vis = self.cross_attn_vis(self.norm_cross_vis(x), c_vis, c_vis_mask)
        ca_txt = self.cross_attn_text(self.norm_cross_txt(x), c_txt, c_txt_mask)
        combined = self.weight_text * ca_txt + (1 - self.weight_text) * ca_vis
        x = x + combined

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0., use_masking=False, input_cond='text'):
        super(MultiHeadCrossAttention, self).__init__()
        ## TODO: Only text or only Visual condition is assumed.
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)    
        if input_cond == 'text':
            input_kv_dim = 1024
        elif input_cond == 'visual':            
            input_kv_dim = d_model

        else:
            raise ValueError(f"input_cond should be either 'text' or 'visual' but got {input_cond}")
        self.kv_linear = nn.Linear(input_kv_dim, d_model*2) 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_masking = use_masking

    def forward(self, x, cond, mask=None):
        # x shape: (B, N, C); cond shape: (B, 40, 512), mask shape: (B, 40)
        # places to be masked are 1 in mask

        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).reshape(B, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None

        if self.use_masking and mask is not None:
            attn_bias = torch.zeros([B, self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)            
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_bias = attn_bias.masked_fill(mask == 0.0, float('-inf'))
        
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        x = x.contiguous().reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class MultiHeadCrossAttention_VT(nn.Module):
    def __init__(self, q_dim, kv_dim, d_model, num_heads, attn_drop=0., proj_drop=0., use_masking=False):
        super(MultiHeadCrossAttention_VT, self).__init__()
        ## TODO: Only text or only Visual condition is assumed.
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_linear = nn.Linear(q_dim, d_model)    
        self.kv_linear = nn.Linear(kv_dim, d_model*2) 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, q_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_masking = use_masking

    def forward(self, x, cond, mask=None):
        # x shape: (B, N, C); cond shape: (B, 40, 512), mask shape: (B, 40)
        # places to be masked are 1 in mask

        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).reshape(B, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).reshape(B, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None

        if self.use_masking and mask is not None:
            attn_bias = torch.zeros([B, self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)            
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_bias = attn_bias.masked_fill(mask == 0.0, float('-inf'))
        
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        x = x.contiguous().reshape(B, -1, self.d_model)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x    

class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=[256,16],
        patch_size=2,
        in_channels=8,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        learn_sigma=True,
        use_masking=False,
        vid_feat_dir=None,
        cross_att_vis_text=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size = input_size
        self.use_masking = use_masking
        self.vid_feat_dir = vid_feat_dir
        self.total_num_frames = 40

        self.x_embedder = AudPatchEmbed(img_size=input_size, patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size, bias=True) 
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.y_txt = TextEmbedder(hidden_size, class_dropout_prob)
        self.y_vis = VisualEmbedder(hidden_size, class_dropout_prob, self.vid_feat_dir, self.total_num_frames)

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        mlp_input_dim = 1024 + 512
        self.mlp_comb = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self.cross_att_vis_text = cross_att_vis_text
        if self.cross_att_vis_text:
            TEXT_DIM = 1024; VIS_DIM = hidden_size
            # Q : Text, K,V: Visual
            self.gate_vt_t = torch.nn.Parameter(torch.zeros(1))
            self.norm_cross_vt_text = nn.LayerNorm(TEXT_DIM, elementwise_affine=False, eps=1e-6)
            self.cross_attn_vt_text = MultiHeadCrossAttention_VT(TEXT_DIM, VIS_DIM, hidden_size, num_heads, use_masking=use_masking)

            # Q : Visual, K,V: Text
            self.gate_vt_v = torch.nn.Parameter(torch.zeros(1))
            self.norm_cross_vt_vis = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn_vt_vis = MultiHeadCrossAttention_VT(VIS_DIM, TEXT_DIM, hidden_size, num_heads, use_masking=use_masking)

        self.blocks = nn.ModuleList([
            SiTCrossAttnBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_masking=self.use_masking) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed_audio(self.pos_embed.shape[-1], 
                                                  self.input_size[0] // self.patch_size,
                                                  self.input_size[1] // self.patch_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        
        nn.init.normal_(self.y_vis.mlp_combine[0].weight, std=0.02)
        nn.init.constant_(self.y_vis.mlp_combine[0].bias, 0)
        nn.init.normal_(self.y_vis.mlp_combine[2].weight, std=0.02)
        nn.init.constant_(self.y_vis.mlp_combine[2].bias, 0)

        nn.init.normal_(self.mlp_comb[-1].weight, 0)
        nn.init.constant_(self.mlp_comb[-1].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        h_patches, w_patches = self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size
        c = self.out_channels
        p = self.x_embedder.patch_size[0]

        x = x.reshape(shape=(x.shape[0], h_patches, w_patches, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h_patches * p, w_patches * p))
        return imgs

    def forward(self, x, t, y_txt_in, y_vid_in):
        """
        Forward pass of SiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)

        y_txt_tokens, y_txt_mask, y_txt_mean = self.y_txt(y_txt_in, self.training)
        y_vis_tokens, y_vis_mask, y_vis_mean = self.y_vis(y_vid_in, self.training, x.device)

        if self.cross_att_vis_text:
            ca_vt_text = self.cross_attn_vt_text(self.norm_cross_vt_text(y_txt_tokens), y_vis_tokens, y_vis_mask)
            ca_vt_vis = self.cross_attn_vt_vis(self.norm_cross_vt_vis(y_vis_tokens), y_txt_tokens, y_txt_mask)
            y_txt_tokens = y_txt_tokens + torch.tanh(self.gate_vt_t)*ca_vt_text
            y_vis_tokens = y_vis_tokens + torch.tanh(self.gate_vt_v)*ca_vt_vis

        y_mean = self.mlp_comb(torch.cat([y_txt_mean, y_vis_mean], dim=1))
        c = t + y_mean

        for block in self.blocks:            
            x = block(x, c, y_txt_tokens, y_txt_mask, y_vis_tokens, y_vis_mask)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y_txt_in, y_vid_in, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y_txt_in, y_vid_in)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward_with_cfg_VT(self, x, t, y_txt_in, y_vid_in, s_vis=2.5, s_txt=2.5):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, t, y_txt_in, y_vid_in)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps_v, eps_t, eps_uncond = torch.split(eps, len(eps) // 3, dim=0)
        part_eps = eps_uncond + s_vis * (eps_v - eps_uncond) + s_txt * (eps_t - eps_v)
        eps = torch.cat([part_eps, part_eps, part_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_audio(embed_dim, grid_size_h, grid_size_w, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

#################################################################################
#                                   SiT Configs                                  #
#################################################################################

def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2,  'SiT-XL/4': SiT_XL_4,  'SiT-XL/8': SiT_XL_8,
    'SiT-L/2':  SiT_L_2,   'SiT-L/4':  SiT_L_4,   'SiT-L/8':  SiT_L_8,
    'SiT-B/2':  SiT_B_2,   'SiT-B/4':  SiT_B_4,   'SiT-B/8':  SiT_B_8,
    'SiT-S/2':  SiT_S_2,   'SiT-S/4':  SiT_S_4,   'SiT-S/8':  SiT_S_8,
}