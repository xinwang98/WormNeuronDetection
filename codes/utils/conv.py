import torch
import torch.nn.functional as F
import numpy as np


def conv3d(filter, frame, gpu, num_channel, frame_size, padding):
    gaussian_filter = torch.from_numpy(np.transpose(filter, axes=(2, 0, 1))).cuda(gpu).unsqueeze(0)
    frame = torch.from_numpy(np.transpose(frame, axes=(2, 0, 1))).cuda(gpu).unsqueeze(0)
    frame_after_conv = torch.zeros_like(frame)
    for ch in range(num_channel):
        if ch == 0:
            frame_slice = torch.zeros(1, 3, frame_size, frame_size).cuda(gpu)
            frame_slice[:, 1:, :, :] = frame[:, ch: ch + 2, :, :]
        elif ch == num_channel - 1:
            frame_slice = torch.zeros(1, 3, frame_size, frame_size).cuda(gpu)
            frame_slice[:, 0: -1, :, :] = frame[:, ch - 1: ch + 1, :, :]
        else:
            frame_slice = frame[:, ch - 1: ch + 2, :, :]
        out = F.conv2d(frame_slice, gaussian_filter, padding=padding)
        frame_after_conv[:, ch, :, :] = out.squeeze()

    frame_after_conv = np.transpose(frame_after_conv.squeeze().cpu().numpy(), axes=(1, 2, 0))
    return frame_after_conv
