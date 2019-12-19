import h5py
from scipy.io import loadmat
import numpy as np
import os

SLICE = 2
BOX = 4
VALID_CHANNEL = 22


def get_boxes(complete_index, frame_num):
    boxes = [[] for _ in range(frame_num)]
    for f in range(frame_num):
        box_num = complete_index['neuron_boxes'][f][0].shape[1]
        for neuron in range(box_num):
            channel = complete_index['neuron_boxes'][f][0][0][neuron][SLICE] - 1    # align with matlab
            box = complete_index['neuron_boxes'][f][0][0][neuron][BOX] + 0.5 - 1    # align with matlab
            boxes[f].append(np.append(channel, box.astype(int)))
        boxes[f] = np.array(boxes[f])
    return boxes


def get_stack_pos(stack_path, index_pos_path, frame_num=20):
    stack_npy_path = stack_path.replace('mat', 'npy')
    if os.path.exists(stack_npy_path):
        frames = np.load(stack_npy_path)
        # frame_num = frames.shape[0]
    else:
        frames = []
        img_stack = h5py.File(stack_path)
        s = img_stack['img_Stack']
        # _, frame_num = s.shape
        for f in range(frame_num):
            img_stack_frame = img_stack[s[0, f]].value.transpose()
            frames.append(img_stack_frame)
        frames = np.array(frames)
        np.save(stack_npy_path, frames)

    index_and_position = loadmat(index_pos_path)
    # neuron_index = index_and_position['neuron_index_data'].squeeze()
    neuron_boxes = []

    neuron_position = index_and_position['neuron_position_data'][:, 0]
    neuron_pos = []
    neuron_boxes = get_boxes(index_and_position, frame_num)
    for f in range(frame_num):
        pos_list = neuron_position[f]
        neuron_pos.append(pos_list - 1)     # align with matlab

    return frames[..., 0:VALID_CHANNEL], neuron_pos, neuron_boxes
