# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new audios from a pre-trained VT-SiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from download import find_model
from models_VT_comb import SiT_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import create_transport, Sampler
import argparse
import sys
from time import time
import soundfile as sf
from aud_utils import get_vae_stft
import os
from moviepy.editor import VideoFileClip, AudioFileClip
from visual_feats.compute_vid_feats import compute_visual_features

import warnings
warnings.filterwarnings("ignore")

def replace_audio(video_path, new_audio_path, output_path, duration=10):
    video_clip = VideoFileClip(video_path)
    total_duration = min(video_clip.duration, duration)
    new_audio_clip = AudioFileClip(new_audio_path).subclip(0, total_duration)
    video_with_new_audio = video_clip.set_audio(new_audio_clip)
    video_with_new_audio.write_videofile(output_path) # this plays on vscode
    # video_with_new_audio.write_videofile(output_path, codec='libx264', audio_codec='aac') # this plays on mac

def get_vid_name(vid_path):
    basename = os.path.basename(vid_path).rsplit(".", 1)[0]
    vid = 'vid_'+basename.replace(' ', '_') # replace spaces with underscores
    return vid

def main(mode, args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
    else:
        learn_sigma = False       

    #extract visual features
    vid = get_vid_name(args.vid_path)
    compute_visual_features(args.vid_path, vid, args.vid_feat_dir, recalculate=args.recalculate_feat)

    # Load model:
    model = SiT_models[args.model](
        input_size=args.latent_size,
        use_masking=args.use_masking,
        vid_feat_dir = args.vid_feat_dir,
        cross_att_vis_text=args.cross_att_vis_text,
    ).to(device)
    
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)

    if args.txt_wt > 0:
        for key in state_dict.keys():
            if "weight_text" in key:
                state_dict[key] = torch.full_like(state_dict[key], args.txt_wt) 

    model.load_state_dict(state_dict)
    model.eval()  # important!
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
            
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    
    vae, _ = get_vae_stft()
    start_time = time()
    
    os.makedirs(args.save_dir, exist_ok=True)
    TEXT_NULL = ""; VID_NULL = "EMPTY"

    y_txt = [TEXT_NULL, args.text_prompt, TEXT_NULL]
    y_vid = [vid, vid, VID_NULL]

    zs = torch.randn(1, 8, args.latent_size[0], args.latent_size[1], device=device)
    zs = torch.cat([zs,zs,zs], 0)
    model_kwargs = dict(y_txt_in=y_txt, y_vid_in=y_vid, s_vis=args.s_vis, s_txt=args.s_txt)
    
    start_time = time()
    samples = sample_fn(zs, model.forward_with_cfg_VT, **model_kwargs)[-1]
    samples, _, _ = samples.chunk(3, dim=0)  # Remove null class samples
    mel = vae.decode_first_stage(samples)
    wave = vae.decode_to_waveform(mel)[0]

    print(f"Sampling took {time() - start_time:.2f} seconds.")
    
    sav_sr = 16000
    gen_audio_path = "{}/{}.wav".format(args.save_dir, vid+args.save_suffix)
    sf.write(gen_audio_path, wave, samplerate=sav_sr)

    out_vid_path = "{}/{}.mp4".format(args.save_dir, vid+args.save_suffix)
    replace_audio(args.vid_path, gen_audio_path, out_vid_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"
    
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default='ckpts/VTComb_0300000.pt')
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--use-masking", action="store_true")
    parser.add_argument("--cross-att-vis-text", action="store_true", help="Use cross attention between visual and text")
    parser.add_argument("--s-vis", type=float, default=2.5)
    parser.add_argument("--s-txt", type=float, default=2.5)
    parser.add_argument("--save-dir", type=str, default="outputs")
    parser.add_argument("--txt-wt", type=float, default=-1.0)

    parser.add_argument("--vid-path", type=str, default="")
    parser.add_argument("--text-prompt", type=str, default="")
    parser.add_argument("--vid-feat-dir", type=str, default="outputs/feats")
    parser.add_argument("--recalculate-feat", action="store_true")
    parser.add_argument("--save-suffix", type=str, default="")

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
    elif mode == "SDE":
        parse_sde_args(parser)
    
    args = parser.parse_known_args()[0]
    args.aud_size = [1024, 64]
    args.latent_size = [256,16]
    
    main(mode, args)
