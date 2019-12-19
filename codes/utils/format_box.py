import numpy as np

CHANNEL_NUM = 22


def format_box(box):
    # box: x, y, z, half_ps_x, half_ps_y
    fb = np.zeros((box.shape[0], 4))
    fb[:, 0] = box[:, 0] - box[:, 3]
    fb[:, 1] = box[:, 0] + box[:, 3] + 1
    fb[:, 2] = box[:, 1] - box[:, 4]
    fb[:, 3] = box[:, 1] + box[:, 4] + 1
    return fb


def format_gt(gt):
    boxes = np.zeros((gt.shape[0], 4))
    boxes[:, 0] = gt[:, 2]
    boxes[:, 1] = gt[:, 2] + gt[:, 4]
    boxes[:, 2] = gt[:, 1]
    boxes[:, 3] = gt[:, 1] + gt[:, 3]
    # boxes[:, 4] = gt[:, 0]
    return boxes.astype(int)


def split_channel(boxes):
    frame_boxes = []
    for f in range(len(boxes)):
        split_boxes = [np.array([]) for _ in range(CHANNEL_NUM)]
        for c in range(CHANNEL_NUM):
            ind = np.atleast_1d(np.argwhere(boxes[f][:, 0] == c).squeeze())
            if len(ind) == 0:
                continue
            split_boxes[c] = format_gt(boxes[f][ind])
        frame_boxes.append(np.array(split_boxes))
    return frame_boxes
