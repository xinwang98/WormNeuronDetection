import numpy as np


def iou(best_box, cur_box):
    xmin1, xmax1, ymin1, ymax1 = best_box[0], best_box[1], best_box[2], best_box[3]
    xmin2, xmax2, ymin2, ymax2 = cur_box[0], cur_box[1], cur_box[2], cur_box[3]
    x_left = max(xmin1, xmin2)
    x_right = min(xmax1, xmax2)
    y_top = max(ymin1, ymin2)
    y_bottom = min(ymax1, ymax2)

    if x_right < x_left or y_bottom < y_top:
        intersect = 0
    else:
        intersect = (x_right - x_left) * (y_bottom - y_top)
    union = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2) - intersect
    return intersect / union


def over_threshold(best_box, cur_box, iou_threshold):
    if iou(best_box, cur_box) > iou_threshold:
        return True
    else:
        return False


def nms(candidate_box, score, iou_threshold=0.5):
    """
    candidate_box: (N, 4)
    score: (N, )
    """
    keep = np.zeros_like(score)
    while np.sum(score) > 0:
        # print('keep {}'.format(np.sum(keep)))
        best_box_id = np.argmax(score)
        score[best_box_id] = 0
        keep[best_box_id] = 1
        best_box = candidate_box[best_box_id]
        for i in range(len(score)):
            if score[i] == 0:
                continue
            cur_box = candidate_box[i]
            if over_threshold(best_box, cur_box, iou_threshold):
                score[i] = 0
    return keep




