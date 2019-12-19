from codes import get_stack_pos
from codes.basic_functions.classification.train import train
from codes.basic_functions.classification.test import test
from codes.basic_functions.classification.show_results import show_prediction
import torch
import numpy as np
import os
from codes.basic_functions.template_matching.gaussian_matching import gaussian_match


seed = 0
gpu = 3
epoch_num = 15
frame_num = 20
train_frames = [i for i in range(18)]
test_frames = [i for i in range(18, 20)]

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

stack_path = './data/whole_brain_imaging/181208/w10_1_MMStack_Pos0_7.mat'
index_pos_path = './data/whole_brain_imaging/181208/index_and_position_revised_by_wx_Dt191213_20vol.mat'


frames, neuron_pos, neuron_boxes = get_stack_pos(stack_path, index_pos_path, frame_num=frame_num)
model_name = '1_channel_epoch_{}.pth.tar'

# get the interest points
gaussian_match(frames, neuron_pos, gpu, train_frames=train_frames)

# train the classification nn
net, epoch, epoch_acc = train(gpu, frames, neuron_boxes, epoch_num)

model_root = './experiments/models'
if not os.path.exists(model_root):
    os.makedirs(model_root)
check_point = {'epoch': epoch, 'acc': epoch_acc, 'state_dict': net.state_dict()}
check_point_save_path = os.path.join(model_root, model_name.format(epoch))
torch.save(check_point, check_point_save_path)

for frame_idx in [18, 19]:
    test(frames=frames, checkpoint_path='./experiments/models/' + model_name.format(epoch_num - 1), gpu=gpu, frame_idx=frame_idx)
    show_prediction(frames, ground_truth=neuron_boxes, frame_idx=frame_idx)



