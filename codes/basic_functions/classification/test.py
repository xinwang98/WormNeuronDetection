import torch
import numpy as np
from codes.basic_functions.classification.dataset import TestSet
from codes.basic_functions.classification.channel_1_dataset import TestSetC1
from torch.utils.data import DataLoader
from codes.basic_functions.classification.model import PatchNet
from .nms import nms
import torch.nn.functional as F

CHANNEL_NUM = 22


def test(frames, checkpoint_path, gpu, frame_idx):
    model = PatchNet().cuda(gpu)
    check_point = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(check_point['state_dict'])
    model.eval()
    dataset = TestSetC1(frames, frame_idx=frame_idx, key_mask_root='./experiments/key_points/gaussian/test', patch_size=9)
    # dataset = TestSet(frames, frame_idx=frame_idx, key_mask_root='./experiments/key_points/gaussian/test',
    #                     patch_size=9)
    data_loader = DataLoader(dataset, batch_size=1024, shuffle=True, pin_memory=True)
    candidate_box = None
    cls_scores = None
    with torch.no_grad():
        for batch, (images, boxes) in enumerate(data_loader):   # box: x, y, z, half_ps
            images = images.cuda(gpu)
            out = model(images)
            scores = F.softmax(out, dim=1)
            scores, predicted = torch.max(scores.data, 1)
            p = predicted.cpu().numpy()
            positive = np.argwhere(p == 1).squeeze()
            if batch == 0:
                candidate_box = boxes[positive].numpy()
                cls_scores = scores[positive].cpu().numpy()
            else:
                candidate_box = np.vstack((candidate_box, boxes[positive].numpy()))
                cls_scores = np.append(cls_scores, scores[positive].cpu().numpy())

    pred_boxes = [np.array([]) for _ in range(CHANNEL_NUM)]
    for c in range(CHANNEL_NUM):
        ind = np.atleast_1d(np.argwhere(candidate_box[:, -1] == c).squeeze())
        if len(ind) == 0:
            continue
        box_c = candidate_box[ind]
        score_c = cls_scores[ind]
        format_box = box_c[:, :-1]
        keep = nms(candidate_box=format_box, score=score_c, iou_threshold=0.05)
        print(keep.sum())
        index, = keep.nonzero()
        pred_boxes[c] = format_box[index, ...]

    np.save('./experiments/pred_box_{}.npy'.format(frame_idx), np.array(pred_boxes))

