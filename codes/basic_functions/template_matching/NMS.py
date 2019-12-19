import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from codes.utils.filters import get_gaussian_filter
from codes.utils.to_image_space import to_image_space
from codes.utils.nms import nms


def nms_match(frames, neuron_positions, gpu):
    kernel_shape = (9, 9, 3)
    sigma = 5
    padding = kernel_shape[0] // 2
    x_overlap = 2
    y_overlap = 2
    z_overlap = 1

    num_frame, frame_size, num_channel = frames.shape[0], frames.shape[1], frames.shape[-1]

    gaussian_filter = get_gaussian_filter(shape=kernel_shape, sigma=sigma).astype(np.float32)
    # gaussian_filter = get_balance_filter().astype(np.float32)
    gaussian_filter = torch.from_numpy(np.transpose(gaussian_filter, axes=(2, 0, 1))).cuda(gpu).unsqueeze(0)
    for frame_idx in range(0, 15):
        cur_frame = frames[frame_idx, :, :, :].astype(np.float32)
        cur_position = neuron_positions[frame_idx].astype(int).transpose()

        frame_cuda = torch.from_numpy(np.transpose(cur_frame, axes=(2, 0, 1))).cuda(gpu).unsqueeze(0)
        frame_after_conv = torch.zeros_like(frame_cuda)
        for ch in range(num_channel):
            if ch == 0:
                frame_slice = torch.zeros(1, 3, frame_size, frame_size).cuda(gpu)
                frame_slice[:, 0, :, :] = frame_cuda[:, 0, :, :]
                frame_slice[:, 1:, :, :] = frame_cuda[:, ch: ch + 2, :, :]
            elif ch == num_channel - 1:
                frame_slice = torch.zeros(1, 3, frame_size, frame_size).cuda(gpu)
                frame_slice[:, 0: -1, :, :] = frame_cuda[:, ch - 1: ch + 1, :, :]
                frame_slice[:, -1, :, :] = frame_cuda[:, -1, :, :]
            else:
                frame_slice = frame_cuda[:, ch - 1: ch + 2, :, :]
            out = F.conv2d(frame_slice, gaussian_filter, padding=padding)
            frame_after_conv[:, ch, :, :] = out.squeeze()

        frame_after_conv = np.transpose(frame_after_conv.squeeze().cpu().numpy(), axes=(1, 2, 0))

        actications = frame_after_conv[cur_position[:, 1], cur_position[:, 0], cur_position[:, 2]]
        threshold = np.min(actications)
        # threshold = np.sort(actications)[20]
        print('threshold is ', threshold)
        gaussian_save_dir = './experiments/show/gaussian_nms/frame_{}'.format(frame_idx)
        if not os.path.exists(gaussian_save_dir):
            os.makedirs(gaussian_save_dir)
        key_points_mask = nms(frame_after_conv, threshold, x_overlap, y_overlap, z_overlap, frame_size, num_channel)
        # key_points = (frame_after_conv >= threshold)

        key_points_mask[cur_position[:, 1], cur_position[:, 0], cur_position[:, 2]] = 0
        key_point_save_dir = './experiments/test_key_points/nms'
        if not os.path.exists(key_point_save_dir):
            os.makedirs(key_point_save_dir)
        key_point_path = os.path.join(key_point_save_dir, 'frame_{}.npy'.format(frame_idx))
        np.save(key_point_path, key_points_mask)

        for ch in range(num_channel):
            print('Num key points in channel {} is {}'.format(ch, np.sum(key_points_mask[:, :, ch])))
            ch_key_point_position = np.argwhere(key_points_mask[:, :, ch] == 1)
            ch_key_points = []
            for i in range(ch_key_point_position.shape[0]):
                ch_key_points.append(cv2.KeyPoint(ch_key_point_position[i, 1], ch_key_point_position[i, 0], 1))
            img = np.uint8(cur_frame[:, :, ch] / 4)
            img_gaussian = cv2.drawKeypoints(img, ch_key_points, np.array([]))
            gaussian_save_path = os.path.join(gaussian_save_dir, 'channel_{}.jpg'.format(ch))
            cv2.imwrite(gaussian_save_path, img_gaussian)
        print('total key points ', np.sum(key_points_mask))




