import os
import numpy as np
from PIL import Image, ImageDraw
from .nms import over_threshold, iou
from codes.utils import *
from scipy.io import savemat
from .nms import nms

TP_TRUTH_COLOR = 'red'
TP_PRED_COLOR = 'yellow'
FP_COLOR = 'green'
FN_COLOR = 'pink'

SAVE_ROOT = './experiments/prediction/frame_{}_iou_{}/'
CHANNEL_NUM = 22
IOU = 0.3


def normalize(frame):
    norm_frame = (frame - frame.min()) / (frame.max() - frame.min())
    norm_frame = norm_frame * 255
    return norm_frame.astype(np.uint8)


def merge(frame, pred_box):
    merge_boxes = []
    scores = []
    for c in range(CHANNEL_NUM):
        pred_c = pred_box[c]
        for i in range(pred_c.shape[0]):
            xmin, xmax, ymin, ymax = pred_c[i]
            patch = frame[xmin: xmax, ymin: ymax, c]
            scores.append(patch.mean())
            merge_boxes.append(np.append(pred_c[i], c))
    merge_boxes = np.array(merge_boxes)
    scores = np.array(scores)
    keep = nms(candidate_box=merge_boxes[:, :-1], score=scores, iou_threshold=0.05)
    index, = keep.nonzero()
    print(keep.sum())
    matlab_box = to_matlab(merge_boxes[index])
    py_box = merge_boxes[index]
    return matlab_box, py_box


def to_matlab(boxes):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)   # xmin, xmax, ymin, ymax
    matlab_boxes = np.zeros_like(boxes)

    matlab_boxes[:, 0] = boxes[:, 2] + 1
    matlab_boxes[:, 1] = boxes[:, 0] + 1
    matlab_boxes[:, 2] = boxes[:, 3] - boxes[:, 2] + 1
    matlab_boxes[:, 3] = boxes[:, 1] - boxes[:, 0] + 1
    if boxes.shape[1] == 5:
        matlab_boxes[:, 4] = boxes[:, 4] + 1
    return matlab_boxes


def draw_box(channel_image, tp_truth_box, tp_pred_box, fp_box, fn_box):
    draw = ImageDraw.Draw(channel_image)

    for box in tp_truth_box:
        ymin, ymax, xmin, xmax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=TP_TRUTH_COLOR)
    for box in tp_pred_box:
        ymin, ymax, xmin, xmax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=TP_PRED_COLOR)
    for box in fp_box:
        ymin, ymax, xmin, xmax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=FP_COLOR)
    for box in fn_box:
        ymin, ymax, xmin, xmax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=FN_COLOR)
    return channel_image


def show_prediction(frames, ground_truth, frame_idx=14):
    # ground_truth = ground_truth[frame_idx]

    format_gt_boxes = split_channel(ground_truth)[frame_idx]
    pred_box = np.load('./experiments/pred_box_{}.npy'.format(frame_idx))
    # matlab_box, py_box = merge(frames[frame_idx], pred_box)
    # savemat('./volume_{}_merge.mat'.format(frame_idx + 1), mdict={'merge': matlab_box})
    #
    # merge_root = './experiments/merge'
    # if not os.path.exists(merge_root):
    #     os.makedirs(merge_root)
    #
    # channel_images = [Image.fromarray(normalize(frames[frame_idx, :, :, c])).convert('RGB') for c in range(CHANNEL_NUM)]
    # draws = [ImageDraw.Draw(channel_images[c]) for c in range(CHANNEL_NUM)]
    # for i in range(py_box.shape[0]):
    #     ymin, ymax, xmin, xmax, c = py_box[i]
    #     draws[c].rectangle([xmin, ymin, xmax, ymax], outline='yellow')
    # for c in range(CHANNEL_NUM):
    #     channel_images[c].save(os.path.join(merge_root, 'channel_{}.jpg'.format(c)), quality=95)
    tp = 0
    fn = 0
    fp = 0

    tp_truth_boxes = [[] for _ in range(22)]
    tp_pred_boxes = [[] for _ in range(22)]
    fp_boxes = [[] for _ in range(22)]
    fn_boxes = [[] for _ in range(22)]

    for c in range(CHANNEL_NUM):
        tp_truth_box = []
        tp_pred_box = []
        fp_box = []
        fn_box = []

        gt_c = format_gt_boxes[c]
        pred_c = pred_box[c]

        gt_c_flags = np.zeros(gt_c.shape[0])
        pred_c_flags = np.zeros(pred_c.shape[0])
        if len(gt_c) == 0 and len(pred_c) == 0:
            continue
        elif len(gt_c) == 0 and len(pred_c) != 0:
            fp += pred_c.shape[0]
            for i in range(pred_c.shape[0]):
                fp_box.append(pred_c[i])
        elif len(gt_c) != 0 and len(pred_c) == 0:
            fn += gt_c.shape[0]
            for i in range(gt_c.shape[0]):
                fn_box.append(gt_c[i])
        else:
            for j in range(gt_c.shape[0]):
                flag = True
                for i in range(pred_c.shape[0]):
                    if over_threshold(pred_c[i], gt_c[j], iou_threshold=IOU):
                        # tp += 1
                        # flag = False
                        gt_c_flags[j] = 1
                        pred_c_flags[i] = 1
                        # tp_truth_box.append(gt_c[j])
                        # tp_pred_box.append(pred_c[i])
                        break
                # if flag:
                #     fn += 1
                #     fn_box.append(gt_c[j])

            for i in range(len(gt_c_flags)):
                if gt_c_flags[i] == 1:
                    tp_truth_box.append(gt_c[i])
                    tp += 1
                else:
                    fn_box.append(gt_c[i])
                    fn += 1
            for i in range(len(pred_c_flags)):
                if pred_c_flags[i] == 1:
                    tp_pred_box.append(pred_c[i])
                else:
                    fp_box.append(pred_c[i])
                    fp += 1

        channel_image = Image.fromarray(normalize(frames[frame_idx, :, :, c])).convert('RGB')
        img = draw_box(channel_image, tp_truth_box, tp_pred_box, fp_box, fn_box)

        matlab_tp_pred_box = to_matlab(tp_pred_box)
        matlab_fp_box = to_matlab(fp_box)
        tp_pred_boxes[c] = matlab_tp_pred_box
        fp_boxes[c] = matlab_fp_box

        save_root = SAVE_ROOT.format(frame_idx, IOU)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        img.save(os.path.join(save_root, 'channel_{}.jpg'.format(c)), quality=95)

    savemat('./volume_{}_fp.mat'.format(frame_idx + 1), mdict={'fp': fp_boxes})
    savemat('./volume_{}_tp.mat'.format(frame_idx + 1), mdict={'tp': tp_pred_boxes})
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print((tp + fp))
    print((tp + fn))
    print('precision = ', precision)
    print('recall = ', recall)



