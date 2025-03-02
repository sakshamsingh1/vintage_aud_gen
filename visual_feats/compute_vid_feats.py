import os
import torch
from visual_feats.clip_feats import extract_clip_frames, get_clip_feats, get_flow
import shutil

def compute_visual_features(vid_path, vid, vid_feat_dir, recalculate=False):
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

    vid_index_map = torch.load(vid_index_map_path, map_location='cpu')
    vis_feat = torch.load(vis_feat_path, map_location='cpu')
    vis_flow = torch.load(vis_flow_path, map_location='cpu')
    vis_mask = torch.load(vis_mask_path, map_location='cpu')

    if (vid in vid_index_map) and (not recalculate):
        print(f"Skipping {vid} features as it already exists")
        return
    
    # extract frames
    frame_dir = './outputs/tmp_frames'
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    extract_clip_frames(vid_path, frame_dir)
    
    curr_feat, curr_mask = get_clip_feats(frame_dir) # (40, 512) and (40)
    curr_flow = get_flow(vid_path)  # Shape: [40]

    if vid not in vid_index_map:
        # compute clip features
        vid_idx = len(vid_index_map)
        vid_index_map[vid] = vid_idx

        curr_feat = curr_feat.unsqueeze(0)  # Shape: [1, 40, 512]
        curr_mask = curr_mask.unsqueeze(0)  # Shape: [1, 40]
        vis_feat = torch.cat([vis_feat, curr_feat], dim=0)
        vis_mask = torch.cat([vis_mask, curr_mask], dim=0)

        # compute flow features
        curr_flow = torch.tensor(curr_flow).unsqueeze(0)  # Shape: [1, 40]
        vis_flow = torch.cat([vis_flow, curr_flow], dim=0)

    else:
        vid_idx = vid_index_map[vid]
        vis_feat[vid_idx] = curr_feat
        vis_mask[vid_idx] = curr_mask
        vis_flow[vid_idx] = torch.tensor(curr_flow)

    shutil.rmtree(frame_dir) # remove the temporary frames

    # save features
    torch.save(vid_index_map, vid_index_map_path)
    torch.save(vis_feat, vis_feat_path)
    torch.save(vis_flow, vis_flow_path)
    torch.save(vis_mask, vis_mask_path)