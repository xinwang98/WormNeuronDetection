from codes import get_stack_pos
from codes.basic_functions.classification.train import train
from codes.basic_functions.classification.test import test
from codes.basic_functions.classification.show_results import show_prediction
import torch
import numpy as np
import os
from codes.basic_functions.template_matching.gaussian_matching import gaussian_match
from codes.basic_functions.template_matching.channel_gaussian_matching import channel_gaussian_match


seed = 0
gpu = 3
is_weighted_loss = True
learning_rate = 1e-05
epoch_num = 15
frame_num = 30
train_frames = [i for i in range(25)]
test_frames = [i for i in range(25, 30)]

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

stack_path = './data/whole_brain_imaging/181208/w10_1_MMStack_Pos0_7.mat'
# index_pos_path = './data/whole_brain_imaging/181208/index_and_position_revised_by_wx_Dt191213_20vol.mat'
index_pos_path = './data/whole_brain_imaging/181208/revised_wx_20200112v2.mat'

frames, neuron_pos, neuron_boxes = get_stack_pos(stack_path, index_pos_path, frame_num=frame_num)


# get the interest points
# gaussian_match(frames, neuron_pos, gpu, train_frames=train_frames)    # use three channels
# channel_gaussian_match(frames, neuron_boxes, gpu, train_frames)         # use single channel

if is_weighted_loss:
    model_name = 'seq_channel_weighted_loss_epoch_{}.pth.tar'
else:
    model_name = 'seq_channel_epoch_{}.pth.tar'

# train the classification nn
# net, epoch, epoch_acc = train(gpu, frames, train_frames, neuron_boxes, epoch_num, learning_rate, is_weighted_loss)
# model_root = './experiments/models'
# if not os.path.exists(model_root):
#     os.makedirs(model_root)
# check_point = {'epoch': epoch, 'acc': epoch_acc, 'state_dict': net.state_dict()}
# check_point_save_path = os.path.join(model_root, model_name.format(epoch))
# torch.save(check_point, check_point_save_path)

for frame_idx in test_frames[0:2]:
    # test(frames=frames, checkpoint_path='./experiments/models/' + model_name.format(epoch_num - 1), gpu=gpu, frame_idx=frame_idx)
    show_prediction(frames, ground_truth=neuron_boxes, frame_idx=frame_idx)




