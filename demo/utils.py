import re
import os
import argparse
import torch
from moviepy.editor import VideoFileClip, AudioFileClip

def get_parser():
    parser = argparse.ArgumentParser(description="VinTage demo")
    parser = add_demo_args(parser)
    parser = add_models_args(parser)
    return parser

def replace_audio(video_path, new_audio_path, output_path, duration=10):
    video_clip = VideoFileClip(video_path)
    total_duration = min(video_clip.duration, duration)
    new_audio_clip = AudioFileClip(new_audio_path).subclip(0, total_duration)
    video_with_new_audio = video_clip.set_audio(new_audio_clip)
    video_with_new_audio.write_videofile(output_path) # this plays on vscode

def add_demo_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="VinTAGe demo args")
    group.add_argument("--server-name", type=str, default="0.0.0.0",)
    group.add_argument("--port", type=int, default=7860, help="Port for the Gradio server.")
    group.add_argument("--share", action="store_true", help="Share the Gradio app publicly.")
    group.add_argument("--demo", action="store_true", help="sampling is in demo")
    # parser.add_argument("--share", type=bool, default=False)
    return parser

def download_model():
    cmd = 'wget -P ckpts https://huggingface.co/datasets/sakshamsingh1/vintage/resolve/main/VTComb_0300000.pt'
    if not os.path.exists('ckpts/VTComb_0300000.pt'):
        print("Downloading model checkpoint...")
        os.system(cmd)

def vid_feat_dir_exists(vid_feat_dir):
    if not os.path.exists(vid_feat_dir):
        os.makedirs(vid_feat_dir)

    vid_index_map_path = os.path.join(vid_feat_dir, 'vis_idx_map.pt')
    vis_feat_path = os.path.join(vid_feat_dir, 'vis_feat.pt')
    vis_flow_path = os.path.join(vid_feat_dir, 'vis_flow.pt')
    vis_mask_path = os.path.join(vid_feat_dir, 'vis_mask.pt')

    # If the files don't exist, create them
    if not os.path.exists(vid_index_map_path):
        torch.save({'Dummy_vid':0}, vid_index_map_path)
        torch.save(torch.zeros((1, 40, 512)), vis_feat_path)
        torch.save(torch.zeros((1, 40)), vis_flow_path)
        torch.save(torch.zeros((1, 40)), vis_mask_path)

def add_models_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="VinTAGe model args")

    group.add_argument("--model", type=str, default="SiT-XL/2")
    # group.add_argument("--num-sampling-steps", type=int, default=250, help="Number of sampling steps for the model.")
    group.add_argument("--ckpt", type=str, default="ckpts/VTComb_0300000.pt", help="Path to the model checkpoint.")
    group.add_argument("--target-length", type=int, default=1024)
    group.add_argument("--use-masking", action="store_true")
    group.add_argument("--cross-att-vis-text", action="store_true", help="Use cross attention between visual and text")
    # group.add_argument("--s-vis", type=float, default=2.5)
    # group.add_argument("--s-txt", type=float, default=2.5)
    group.add_argument("--txt-wt", type=float, default=-1.0)

    group.add_argument("--save-dir", type=str, default="outputs")
    group.add_argument("--vid-feat-dir", type=str, default="outputs/feats")
    group.add_argument("--recalculate-feat", action="store_true")
    group.add_argument("--save-suffix", type=str, default="")
    group.add_argument("--skip-cache", action="store_true")
    return parser

def get_vid_name(vid_path):
    basename = os.path.basename(vid_path).rsplit(".", 1)[0]
    vid = 'vid_'+basename.replace(' ', '_') # replace spaces with underscores
    return vid

def get_save_text(text, max_words=10):
    save_text = text.replace(" ", "_")
    save_text = re.sub(r"\s+", "_", save_text)
    save_text = "_".join(save_text.split("_")[:max_words])
    return save_text