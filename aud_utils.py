import json
import torch
import os
from tqdm import tqdm
from huggingface_hub import snapshot_download
from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL
import tools.torch_tools as torch_tools
import pandas as pd
from transformers import AutoTokenizer, T5EncoderModel

def get_vae_stft(name="declare-lab/tango", device="cuda:0"):
    path = snapshot_download(repo_id=name)

    vae_config = json.load(open("{}/vae_config.json".format(path)))
    stft_config = json.load(open("{}/stft_config.json".format(path)))

    vae = AutoencoderKL(**vae_config).to(device)
    stft = TacotronSTFT(**stft_config).to(device)

    vae_weights = torch.load("{}/pytorch_model_vae.bin".format(path), map_location=device)
    stft_weights = torch.load("{}/pytorch_model_stft.bin".format(path), map_location=device)

    vae.load_state_dict(vae_weights)
    stft.load_state_dict(stft_weights)

    vae.eval()
    stft.eval()
    return vae, stft

def get_latents(aud_paths, texts, vids, vae, stft, target_length=1024, augment=False, augment_num=4):
    mel, _, waveform = torch_tools.wav_to_fbank(list(aud_paths), target_length, stft)
    mel = mel.unsqueeze(1)

    if augment and len(aud_paths) > 1:
        mixed_mel, _, _, mixed_captions, mixed_vids = torch_tools.augment_wav_to_fbank(aud_paths, texts, vids, num_items=augment_num, target_length=target_length, fn_STFT=stft)
        # mixed_mel, _, _, mixed_captions, mixed_vids = torch_tools.augment_wav_to_fbank(aud_paths, texts, vids, num_items=len(aud_paths), target_length, stft, num_items=num_items)
        mixed_mel = mixed_mel.unsqueeze(1)
        mel = torch.cat([mel, mixed_mel], 0)
        texts += mixed_captions
        vids += mixed_vids

    true_latent = vae.get_first_stage_encoding(vae.encode_first_stage(mel))
    return true_latent, texts, vids

class TV2A_dataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, base_aud_dir, split='train', caption_column='caption'):
        self.meta = pd.read_csv(meta_path)
        self.meta = self.meta[self.meta['split'] == split].reset_index(drop=True)
        self.base_aud_dir = base_aud_dir
        self.caption_column = caption_column

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        caption = row[self.caption_column]
        aud_path = os.path.join(self.base_aud_dir, row['label'], row['file'].replace('.mp4', '.wav'))
        vid = f"{row['label']}_{row['file'].split('.')[0]}"
        return aud_path, caption, vid  




class Comp_dataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, base_aud_dir, caption_column='caption'):
        self.meta = pd.read_csv(meta_path)
        self.base_aud_dir = base_aud_dir
        self.caption_column = caption_column

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        caption = row[self.caption_column]
        orig_vid = row['vid_orig']
        curr_vid = row['vid']
        aud_path = os.path.join(self.base_aud_dir, curr_vid+ '.wav')
        return aud_path, caption, curr_vid, orig_vid
    
class VQGAN_Comp_dataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, base_aud_dir, caption_column='caption'):
        self.meta = pd.read_csv(meta_path)
        self.meta = self.meta[self.meta['exists']].reset_index(drop=True)
        self.base_aud_dir = base_aud_dir
        self.caption_column = caption_column

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        caption = row[self.caption_column]
        orig_vid = row['vid_orig']
        curr_vid = row['vid']
        aud_path = os.path.join(self.base_aud_dir, curr_vid+ '.wav')
        return aud_path, caption, curr_vid, orig_vid    

class VGG_dataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, base_aud_dir, split='train', caption_column='caption'):
        self.meta = pd.read_csv(meta_path)
        self.meta = self.meta[self.meta['split'] == split].reset_index(drop=True)
        self.base_aud_dir = base_aud_dir
        self.caption_column = caption_column

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        caption = row[self.caption_column]
        aud_path = os.path.join(self.base_aud_dir, row['vid']+ '.wav')
        vid = row['vid']
        return aud_path, caption, vid 
    
class VQGAN_VGG_dataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, base_aud_dir, caption_column='caption'):
        self.meta = pd.read_csv(meta_path)
        self.meta = self.meta[self.meta['exists']].reset_index(drop=True)
        self.base_aud_dir = base_aud_dir
        self.caption_column = caption_column

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        caption = row[self.caption_column]
        aud_path = os.path.join(self.base_aud_dir, row['vid']+ '.wav')
        vid = row['vid']
        return aud_path, caption, vid     

class AC_dataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, base_aud_dir):
        self.meta = pd.read_csv(meta_path)
        self.meta = self.meta[self.meta['exists']].reset_index(drop=True)
        self.base_aud_dir = base_aud_dir
        self.caption_column = 'caption'

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        row = self.meta.iloc[index]
        caption = row[self.caption_column]
        aud_path = os.path.join(self.base_aud_dir, row['video_id']+ '.wav')
        vid = row['video_id']
        return aud_path, caption, vid     

def get_datasets(data_name, args):
    if data_name == 'vas_train':
        base_aud_dir = '/mnt/data1/saksham/TV2A/retina_share/input_data/VAS_audios'
        meta_path = '/mnt/data1/saksham/TV2A/retina_share/input_data/VAS_audios/aud_captions.csv'
        return TV2A_dataset(meta_path, base_aud_dir, split='train', caption_column=args.caption_column)

    # elif data_name == 'vgg_test':
    #     base_aud_dir = '/mnt/user/wpian/dataset/VGGSound/VGGSound_audios'
    #     meta_path = '/mnt/user/saksham/TV2A/aud_SiT/scripts/vggsound_caption.csv'
    #     return VGG_dataset(meta_path, base_aud_dir, split='test', caption_column=args.caption_column)
    elif data_name == 'vqgan_vgg_test':
        base_aud_dir = '/mnt/user/wpian/dataset/VGGSound/VGGSound_audios'
        meta_path = '/mnt/user/saksham/TV2A/aud_SiT/scripts/vgg_test_vqgan.csv'
        return VQGAN_VGG_dataset(meta_path, base_aud_dir, caption_column=args.caption_column)

    # elif data_name == 'comp_test':
    #     # Only for evaluation
    #     base_aud_dir = '/mnt/user/saksham/TV2A/comp_data/audios'
    #     meta_path = '/mnt/user/saksham/TV2A/comp_data/final_caption.csv'
    #     return Comp_dataset(meta_path, base_aud_dir, caption_column=args.caption_column)
    elif data_name == 'vqgan_comp_test':
        # Only for evaluation
        base_aud_dir = '/mnt/user/saksham/TV2A/comp_data/audios'
        meta_path = '/mnt/user/saksham/TV2A/aud_SiT/scripts/vggmix_vqgan.csv'
        return VQGAN_Comp_dataset(meta_path, base_aud_dir, caption_column=args.caption_column)

    elif data_name == 'audiocaps_test':
        base_aud_dir = '/mnt/user/wpian/dataset/VGGSound/VGGSound_audios' # redundant
        meta_path = '/mnt/user/saksham/TV2A/aud_SiT/scripts/audiocaps/tango_test_audiocaps.csv'
        return AC_dataset(meta_path, base_aud_dir)
        
    else:
        raise ValueError(f"Dataset {data_name} not found")


@torch.no_grad()
def get_text_embedding(prompts):
    text_encoder_name = 'google/flan-t5-large'
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_name)

    for param in text_encoder.parameters():
        param.requires_grad = False
    
    batch = tokenizer(prompts, max_length=tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt")
    input_ids, attention_mask = batch.input_ids, batch.attention_mask
    encoder_hidden_states = text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
    boolean_encoder_mask = (attention_mask == 1)
    sum_embeddings = (encoder_hidden_states * boolean_encoder_mask.unsqueeze(-1)).sum(dim=-2)
    valid_token_count = boolean_encoder_mask.sum(dim=-1, keepdim=True)
    mean_embeddings = sum_embeddings / valid_token_count
    return mean_embeddings


