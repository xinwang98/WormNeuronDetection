from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import torch
from codes.basic_functions.classification.nms import iou, over_threshold
from codes.utils import split_channel
import pickle
from codes.utils.format_box import format_gt

VALID_CHANNEL = 22
CROP_CHANNEL = 1


def frame_padding(frame):
    frame_size, _, channel_num = frame.shape
    frame_paddings = np.zeros((frame_size, frame_size, channel_num + 2))
    frame_paddings[:, :, 1:-1] = frame
    frame_paddings[:, :, 0] = frame[:, :, 0]
    frame_paddings[:, :, -1] = frame[:, :, -1]
    return frame_paddings


def is_neg_box_valid(format_pos_boxes, format_neg_candidate, iou_threshold):
    if format_pos_boxes.size == 0:
        return True
    for i in range(format_pos_boxes.shape[0]):
        if over_threshold(format_pos_boxes[i], format_neg_candidate, iou_threshold):
            return False
    return True


def extract_box(box):
    pos_box = []
    identifier_idx = -1
    for k in box:
        neuron_boxes = np.array(box[k]).astype(float)
        cross_channel_num = neuron_boxes.shape[0]
        identifier_position = np.argwhere(neuron_boxes[:, identifier_idx] == 1).item()
        for i in range(cross_channel_num):
            distance_to_identifier = abs(i - identifier_position)
            loss_ratio = 1 / (distance_to_identifier + 1)
            neuron_boxes[i, identifier_idx] = loss_ratio
            pos_box.append(neuron_boxes[i])
    return np.array(pos_box)


def extract_pos_neg_box(box):
    pos_box = []
    neg_box = []
    identifier_idx = -1
    for k in box:
        neuron_boxes = np.array(box[k])
        cross_channel_num = neuron_boxes.shape[0]
        center_distance = cross_channel_num // 4
        identifier_position = np.argwhere(neuron_boxes[:, identifier_idx] == 1).item()
        for i in range(cross_channel_num):
            if abs(i - identifier_position) == 0:
                pos_box.append(neuron_boxes[i])
            elif abs(i - identifier_position) <= center_distance and i != 0 and i != cross_channel_num - 1:
                pos_box.append(neuron_boxes[i])
            else:
                neg_box.append(neuron_boxes[i])
    return np.array(pos_box), np.array(neg_box)


class SeqPatchDataset(Dataset):
    def __init__(self, frames, boxes, phase, frame_range,
                 key_mask_root='./experiments/test_key_points/gaussian', patch_size=9):
        patch_root = './experiments/seq_patches/' + phase
        pos_path = os.path.join(patch_root, 'pos_patch.npy')
        neg_path = os.path.join(patch_root, 'neg_patch.npy')
        if os.path.exists(pos_path) and os.path.exists(neg_path):
            pos_patches = np.load(pos_path)
            neg_patches = np.load(neg_path)
        else:
            neg_patches = []
            pos_patches = []
            crop_size = [2, 3, 4]

            neg_box_list = [[] for _ in frame_range]
            pos_box_list = [[] for _ in frame_range]

            # format_gt_boxes = split_channel(boxes)
            for num, i in enumerate(frame_range):
                # format_positive_boxes = format_gt_boxes[i]
                # padding_frame = frame_padding(frames[i])
                frame = frames[i]
                key_masks = np.load(os.path.join(key_mask_root, 'frame_{}.npy'.format(i)))
                frame_box = boxes[i]
                pos_box = extract_box(frame_box)
                format_positive_boxes = format_gt(pos_box)

                plen = pos_box.shape[0]
                for pbox in range(plen):
                    c, ymin, xmin, dy, dx, loss_weight = pos_box[pbox]
                    c = int(c)
                    ymin = int(ymin)
                    xmin = int(xmin)
                    dy = int(dy)
                    dx = int(dx)
                    crop = frame[xmin: xmin + dx, ymin: ymin + dy, c].copy()  # because the frame is padded, the channel needs to add 1
                    crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                    patch = np.zeros((patch_size, patch_size, CROP_CHANNEL + 2))
                    patch[:, :, 0] = (crop - crop.mean()) / crop.std()
                    patch[:, :, 1] = crop.mean()
                    patch[:, :, 2] = crop.std()
                    pos_patches.append((patch, loss_weight))

                for c in range(VALID_CHANNEL):
                    channel_list = np.atleast_1d(np.argwhere(pos_box[:, 0] == c).squeeze())
                    key_mask = key_masks[:, :, c]
                    x, y = np.where(key_mask == 1)
                    patch_choice = np.random.randint(len(crop_size), size=len(x))
                    for j in range(len(x)):
                        s = crop_size[patch_choice[j]]
                        neg_box = np.array([x[j] - s, x[j] + s + 1, y[j] - s, y[j] + s + 1])
                        if not is_neg_box_valid(format_positive_boxes[channel_list, :], neg_box, iou_threshold=0.3):
                            continue
                        neg_box_list[num].append([x[j] - s, x[j] + s + 1, y[j] - s, y[j] + s + 1, c])
                        crop = frame[x[j] - s: x[j] + s + 1, y[j] - s: y[j] + s + 1, c].copy()  # because the frame is padded, the channel needs to add 1
                        crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

                        patch = np.zeros((patch_size, patch_size, CROP_CHANNEL + 2))
                        patch[:, :, 0] = (crop - crop.mean()) / crop.std()
                        patch[:, :, 1] = crop.mean()
                        patch[:, :, 2] = crop.std()
                        neg_patches.append((patch, 1))

                if not os.path.exists(patch_root):
                    os.makedirs(patch_root)

            np.save(pos_path, np.array(pos_patches))
            np.save(neg_path, np.array(neg_patches))

            with open(os.path.join(patch_root, 'pos_list.pkl'), 'wb') as f:
                pickle.dump(pos_box_list, f)
            with open(os.path.join(patch_root, 'neg_list.pkl'), 'wb') as f:
                pickle.dump(neg_box_list, f)
        self.neg_patches = neg_patches
        self.pos_patches = pos_patches

    def __getitem__(self, index):
        neg_len = len(self.neg_patches)
        pos_len = len(self.pos_patches)
        if index < neg_len:
            label = 0
            patch, loss_weight = self.neg_patches[index]
            return torch.tensor(patch.transpose(2, 0, 1).astype(np.float32)), label, loss_weight
        else:
            index = index % pos_len
            label = 1
            patch, loss_weight = self.pos_patches[index]
            return torch.tensor(patch.transpose(2, 0, 1).astype(np.float32)), label, loss_weight

    def __len__(self):
        return 2 * len(self.neg_patches)


class SeqTestSet(Dataset):
    def __init__(self, frames, frame_idx=18, key_mask_root='./experiments/test_key_points/gaussian', patch_size=9):
        patches = []
        # x_one_side_crop_size = [1, 2, 3, 4]
        one_side_crop_size = [2, 3, 4]
        # frame = frames[frame_idx]

        frame = frames[frame_idx]

        key_mask = np.load(os.path.join(key_mask_root, 'frame_{}.npy'.format(frame_idx)))
        x, y, z = np.where(key_mask == 1)
        for j in range(len(x)):
            for sx in one_side_crop_size:
                for sy in one_side_crop_size:
                    crop = frame[x[j] - sx: x[j] + sx + 1, y[j] - sy: y[j] + sy + 1, z[j]].copy()
                    crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                    # patch = np.zeros((patch_size, patch_size, 1))
                    patch = np.zeros((patch_size, patch_size, CROP_CHANNEL + 2))
                    # patch[:, :, 0] = crop
                    patch[:, :, 0] = (crop - crop.mean()) / crop.std()
                    patch[:, :, 1] = crop.mean()
                    patch[:, :, 2] = crop.std()
                    patches.append((patch, np.array([x[j]-sx, x[j] + sx + 1, y[j] - sy, y[j] + sy + 1, z[j]])))
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        patch = torch.tensor(self.patches[index][0].transpose(2, 0, 1).astype(np.float32))
        box = self.patches[index][1]

        return patch, box


