import os
from visual_feats.av_align import extract_frames, detect_video_peaks
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image

model = SentenceTransformer('clip-ViT-B-32')
model = model.cuda()

def extract_clip_frames(video, dst):
    cmd1 = 'ffmpeg -loglevel error '
    cmd1 += '-i ' + video + " "
    cmd1 += '-y' + " "
    cmd1 += '-r ' + "10 "
    cmd1 += '{0}/frame_%06d.jpg'.format(dst)
    os.system(cmd1)

def get_clip_feats(frames_dir):
    # get clip feats and mask
    curr_feat = torch.zeros((40, 512))
    curr_mask = torch.zeros((40))

    frame_intervals = [1, 4, 7, 10]
    ORIG_FPS = 10
    
    frame_idx = 0
    for sec in range(10):
        for frame_int in frame_intervals:
            curr_frame_num = str(frame_int + sec*ORIG_FPS).zfill(6) + '.jpg'
            curr_frame_path = os.path.join(frames_dir, 'frame_'+curr_frame_num)

            if os.path.exists(curr_frame_path):
                curr_frame = Image.open(curr_frame_path)

                image_embedding = model.encode(curr_frame, convert_to_tensor=True)
                curr_feat[frame_idx] = image_embedding.cpu()
                curr_mask[frame_idx] = 1

            frame_idx += 1    

    return curr_feat, curr_mask

def get_flow_at_intervals(flow_trajectory, fps, interval):
    flow_times = np.arange(len(flow_trajectory)) / fps
    max_time = flow_times[-1]
    desired_times = np.arange(0, max_time + interval, interval)
    flow_values_at_desired_times = np.interp(desired_times, flow_times, flow_trajectory)
    return flow_values_at_desired_times

def get_flow(vid_path):
    try:
        frames, fps = extract_frames(vid_path)
        flow_trajectory = detect_video_peaks(frames, fps)
        flow_values_at_0_25_sec = get_flow_at_intervals(flow_trajectory, fps, interval=0.25)
        flow_025 = np.array(flow_values_at_0_25_sec)
        if len(flow_025) < 40:
            padding = np.zeros(40 - len(flow_025))
            flow_padded = np.concatenate((flow_025, padding))
        else:
            flow_padded = flow_025[:40]
        return flow_padded
    except Exception as e:
        print(f"Error processing video: {e}")
        return np.zeros(40)  # Return zeros if there's an error