import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from codes.utils.filters import get_2d_gaussian_filter
from codes.utils.to_image_space import to_image_space

THRESHOLD = 12
KERNEL_SIZE = (7, 7)
SIGMA = 5


def get_threshold(frames, neuron_boxes, train_frames, gaussian_filter):
    gaussian_filter = gaussian_filter.squeeze()
    lightness = []
    for f in train_frames:
        for nn in neuron_boxes[f]:
            neuron_num = len(neuron_boxes[f][nn])
            for box_id in range(neuron_num):
                c, ymin, xmin, dy, dx, _ = neuron_boxes[f][nn][box_id]
                patch = frames[f, xmin: xmin+dx, ymin: ymin+dy, c]
                patch = cv2.resize(patch, KERNEL_SIZE, interpolation=cv2.INTER_LINEAR)
                light = (gaussian_filter * patch).sum()
                lightness.append(light)
    threshold = np.array(lightness).min()
    return threshold


def channel_gaussian_match(frames, neuron_boxes, gpu, train_frames):
    kernel_shape = KERNEL_SIZE
    sigma = SIGMA
    padding = kernel_shape[0] // 2

    num_frame, frame_size, num_channel = frames.shape[0], frames.shape[1], frames.shape[-1]

    gaussian_filter = get_2d_gaussian_filter(shape=kernel_shape, sigma=sigma).astype(np.float32)
    threshold = get_threshold(frames, neuron_boxes, train_frames, gaussian_filter)
    print('threshold is ', threshold)
    gaussian_filter = torch.from_numpy(np.transpose(gaussian_filter, axes=(2, 0, 1))).cuda(gpu).unsqueeze(0)
    for frame_idx in range(frames.shape[0]):
        cur_frame = frames[frame_idx, :, :, :].astype(np.float32)
        frame_cuda = torch.from_numpy(np.transpose(cur_frame, axes=(2, 0, 1))).cuda(gpu).unsqueeze(0)
        frame_after_conv = torch.zeros_like(frame_cuda)
        for ch in range(num_channel):
            frame_slice = frame_cuda[:, ch, :, :].unsqueeze(0)
            out = F.conv2d(frame_slice, gaussian_filter, padding=padding)
            frame_after_conv[:, ch, :, :] = out.squeeze()
        frame_after_conv = np.transpose(frame_after_conv.squeeze().cpu().numpy(), axes=(1, 2, 0))

        gaussian_save_dir = './experiments/show/gaussian/frame_{}'.format(frame_idx)
        if not os.path.exists(gaussian_save_dir):
            os.makedirs(gaussian_save_dir)
        key_points = (frame_after_conv >= threshold)
        key_points[:, :512, :] = 0
        # key_points[:padding, :, :] = 0
        # key_points[-padding:, :, :] = 0
        # key_points[:, :padding, :] = 0
        # key_points[:, -padding:, :] = 0

        key_points_mask = key_points
        if frame_idx in train_frames:
            key_point_save_dir = './experiments/key_points/gaussian/train/'
        else:
            key_point_save_dir = './experiments/key_points/gaussian/test/'
        if not os.path.exists(key_point_save_dir):
            os.makedirs(key_point_save_dir)
        key_point_path = os.path.join(key_point_save_dir, 'frame_{}.npy'.format(frame_idx))
        np.save(key_point_path, key_points_mask)
        for ch in range(num_channel):
            print('Num key points in channel {} is {}'.format(ch, np.sum(key_points[:, :, ch])))
            ch_key_point_position = np.argwhere(key_points[:, :, ch] == 1)
            ch_key_points = []
            for i in range(ch_key_point_position.shape[0]):
                ch_key_points.append(cv2.KeyPoint(ch_key_point_position[i, 1], ch_key_point_position[i, 0], 1))
            img = np.uint8(cur_frame[:, :, ch] / 4)
            img_gaussian = cv2.drawKeypoints(img, ch_key_points, np.array([]))
            gaussian_save_path = os.path.join(gaussian_save_dir, 'channel_{}.jpg'.format(ch))
            cv2.imwrite(gaussian_save_path, img_gaussian)
        print('total key points ', np.sum(key_points))




