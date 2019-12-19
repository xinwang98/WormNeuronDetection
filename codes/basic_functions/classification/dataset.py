from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import torch
from codes.basic_functions.classification.nms import iou, over_threshold
from codes.utils import split_channel
import pickle

VALID_CHANNEL = 22
CROP_CHANNEL = 3


def frame_padding(frame):
    frame_size, _, channel_num = frame.shape
    frame_paddings = np.zeros((frame_size, frame_size, channel_num + 2))
    frame_paddings[:, :, 1:-1] = frame
    frame_paddings[:, :, 0] = frame[:, :, 0]
    frame_paddings[:, :, -1] = frame[:, :, -1]
    return frame_paddings


def pos_box_augment(boxes):
    # c, xmin, ymin, w, h = boxes[j].astype(int)
    augment_boxes = []
    for f in range(len(boxes)):
        augment_box = boxes[f].copy()
        for i in range(1, boxes[f].shape[1]):
            for modifier in [-1, 1]:
                new_box = boxes[f].copy()
                new_box[:, i] += modifier
                augment_box = np.vstack((augment_box, new_box))
        augment_boxes.append(augment_box)
    return augment_boxes


def is_neg_box_valid(format_pos_boxes, format_neg_candidate, iou_threshold):
    for i in range(format_pos_boxes.shape[0]):
        if over_threshold(format_pos_boxes[i], format_neg_candidate, iou_threshold):
            return False
    return True


class PatchDataset(Dataset):
    def __init__(self, frames, boxes, phase, key_mask_root='./experiments/test_key_points/gaussian', patch_size=9):
        if phase == 'train':
            frame_range = range(0, 15)
        elif phase == 'val':
            frame_range = range(15, 18)
        else:
            raise Exception('Error')
        neg_patches = []
        pos_patches = []
        crop_size = [2, 3, 4]
        # boxes = pos_box_augment(boxes)
        patch_root = './experiments/3_channel_patches/' + phase
        pos_path = os.path.join(patch_root, 'pos_patch.npy')
        neg_path = os.path.join(patch_root, 'neg_patch.npy')
        if os.path.exists(pos_path) and os.path.exists(neg_path):
            pos_patches = np.load(pos_path)
            neg_patches = np.load(neg_path)
        else:
            neg_box_list = [[] for _ in frame_range]
            pos_box_list = [[] for _ in frame_range]

            format_gt_boxes = split_channel(boxes)
            for num, i in enumerate(frame_range):
                format_positive_boxes = format_gt_boxes[i]
                padding_frame = frame_padding(frames[i])
                # frame = frames[i]
                key_masks = np.load(os.path.join(key_mask_root, 'frame_{}.npy'.format(i)))
                for c in range(VALID_CHANNEL):
                    key_mask = key_masks[:, :, c]
                    x, y = np.where(key_mask == 1)
                    patch_choice = np.random.randint(len(crop_size), size=len(x))
                    for j in range(len(x)):
                        s = crop_size[patch_choice[j]]
                        neg_box = np.array([x[j] - s, x[j] + s + 1, y[j] - s, y[j] + s + 1])
                        if not is_neg_box_valid(format_positive_boxes[c], neg_box, iou_threshold=0.3):
                            continue
                        neg_box_list[num].append([x[j] - s, x[j] + s + 1, y[j] - s, y[j] + s + 1, c])
                        crop = padding_frame[x[j] - s: x[j] + s + 1, y[j] - s: y[j] + s + 1, c + 1 - 1: c + 1 + 2].copy()  # because the frame is padded, the channel needs to add 1
                        crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)

                        # patch = np.zeros((patch_size, patch_size, 1))
                        patch = np.zeros((patch_size, patch_size, CROP_CHANNEL + 2))
                        # patch[:, :, 0] = crop
                        patch[:, :, 0:CROP_CHANNEL] = (crop - crop.mean()) / crop.std()
                        patch[:, :, 1] = crop.mean()
                        patch[:, :, 2] = crop.std()
                        neg_patches.append(patch)

                    for j in range(format_positive_boxes[c].shape[0]):
                        xmin, xmax, ymin, ymax = format_positive_boxes[c][j].astype(int)
                        pos_box_list[num].append([xmin, xmax, ymin, ymax, c])
                        crop = padding_frame[xmin: xmax, ymin: ymax, c + 1 - 1: c + 1 + 2].copy()
                        crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                        # patch = np.zeros((patch_size, patch_size, 1))
                        patch = np.zeros((patch_size, patch_size, CROP_CHANNEL + 2))
                        # patch[:, :, 0] = crop
                        patch[:, :, 0:CROP_CHANNEL] = (crop - crop.mean()) / crop.std()
                        patch[:, :, 1] = crop.mean()
                        patch[:, :, 2] = crop.std()
                        pos_patches.append(patch)
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
            return torch.tensor(self.neg_patches[index].transpose(2, 0, 1).astype(np.float32)), 0
        else:
            index = index % pos_len
            return torch.tensor(self.pos_patches[index].transpose(2, 0, 1).astype(np.float32)), 1

    def __len__(self):
        return 2 * len(self.neg_patches)


class TestSet(Dataset):
    def __init__(self, frames, frame_idx=18, key_mask_root='./experiments/test_key_points/gaussian', patch_size=9):
        patches = []
        # x_one_side_crop_size = [1, 2, 3, 4]
        one_side_crop_size = [2, 3, 4]
        # frame = frames[frame_idx]

        padding_frame = frame_padding(frames[frame_idx])

        key_mask = np.load(os.path.join(key_mask_root, 'frame_{}.npy'.format(frame_idx)))
        x, y, z = np.where(key_mask == 1)
        for j in range(len(x)):
            for sx in one_side_crop_size:
                for sy in one_side_crop_size:

                    crop = padding_frame[x[j] - sx: x[j] + sx + 1, y[j] - sy: y[j] + sy + 1, z[j] + 1 - 1: z[j] + 1 + 2].copy()
                    crop = cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                    # patch = np.zeros((patch_size, patch_size, 1))
                    patch = np.zeros((patch_size, patch_size, CROP_CHANNEL + 2))
                    # patch[:, :, 0] = crop
                    patch[:, :, 0:CROP_CHANNEL] = (crop - crop.mean()) / crop.std()
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


