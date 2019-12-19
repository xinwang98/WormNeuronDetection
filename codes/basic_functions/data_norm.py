import numpy as np
import os
from codes.utils.normalize import normalize


def data_norm(frames, save_path, norm_patch_area=10):
    if os.path.exists(save_path):
        normalized_frames = np.load(save_path)
    else:
        normalized_frames = frames.copy().astype(np.float32)
        frame_num, frame_size, _, channel_num = frames.shape
        for f in range(frame_num):
            print('Processing frame {}'.format(f))
            normalize(cur_frame=normalized_frames[f], frame_size=frame_size, num_channel=channel_num, patch_size=norm_patch_area)
        np.save(save_path, normalized_frames)
    return normalized_frames
