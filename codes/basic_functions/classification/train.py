import os
import torch
import torch.nn as nn
from codes.basic_functions.classification.dataset import PatchDataset
from codes.basic_functions.classification.channel_1_dataset import PatchDatasetC1
from codes.basic_functions.classification.seq_dataset import SeqPatchDataset
from torch.utils.data import DataLoader
from codes.basic_functions.classification.model import PatchNet
from torch.utils.tensorboard import SummaryWriter
from codes.utils.weighted_loss import CrossEntropyLoss
from sklearn.model_selection import train_test_split


def train(gpu, frames, train_frames, neuron_boxes, epoch_num, learning_rate=1e-05, is_weighted_loss=True):
    train_list, val_list = train_test_split(train_frames, test_size=0.1, shuffle=False)

    train_set = SeqPatchDataset(frames, neuron_boxes, phase='train', frame_range=train_list,
                                key_mask_root='./experiments/key_points/gaussian/train', patch_size=9)
    val_set = SeqPatchDataset(frames, neuron_boxes, phase='val', frame_range=val_list,
                              key_mask_root='./experiments/key_points/gaussian/train', patch_size=9)

    data_loaders = {'train': DataLoader(train_set, batch_size=2048, shuffle=True, pin_memory=True),
                    'val': DataLoader(val_set, batch_size=2048, shuffle=True, pin_memory=True)}
    loggers = {phase: SummaryWriter('./experiments/loggers' + '/{}'.format(phase)) for phase in ['train', 'val']}

    net = PatchNet().cuda(gpu)
    if is_weighted_loss:
        criterion = CrossEntropyLoss()
        print('Using weighted loss')
    else:
        criterion = nn.CrossEntropyLoss()
        print('NOT using weighted loss')

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print('Val set has {} neg neurons'.format(len(val_set.neg_patches)))
    print('Val set has {} pos neurons'.format(len(val_set.pos_patches)))
    print('The ratio is {}'.format(len(val_set.neg_patches) / len(val_set.pos_patches)))
    for epoch in range(epoch_num):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, epoch_num - 1))

        for phase in ['train', 'val']:
            running_loss = 0
            running_correct = 0
            total = 0

            if phase == 'train':
                net.train()
                fp = 0
            else:
                net.eval()
                fn = 0

            with torch.set_grad_enabled(phase == 'train'):
                for batch, (images, labels, loss_weights) in enumerate(data_loaders[phase]):
                    images = images.cuda(gpu)
                    labels = labels.cuda(gpu)
                    loss_weights = loss_weights.cuda(gpu)
                    out = net(images)
                    if is_weighted_loss:
                        loss = criterion(out, labels, loss_weights)
                    else:
                        loss = criterion(out, labels)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(out.data, 1)
                    running_correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    if phase == 'train':
                        fp += ((predicted == 1) & (labels == 0)).sum().item()
                    if phase == 'val':
                        fn += ((predicted == 0) & (labels == 1)).sum().item()

                epoch_loss = running_loss / (batch + 1)
                epoch_acc = running_correct / total
                print('Loss: {:.2f}'.format(epoch_loss))
                print('Accuracy: {:.2f}'.format(epoch_acc))
                if phase == 'train':
                    print('num of fp is {}'.format(fp))
                    print('fp is {:.4f}'.format(fp / len(data_loaders['train'].dataset) / 2))
                    loggers['train'].add_scalar('fp num', fp, epoch)
                if phase == 'val':
                    print('num of fn is {}'.format(fn))
                    print('Fn is {:.4f}'.format(fn / len(data_loaders['val'].dataset) / 2))
                    loggers['val'].add_scalar('fn num', fn, epoch)

                loggers[phase].add_scalar('Loss', epoch_loss, epoch)
                loggers[phase].add_scalar('Accuracy', epoch_acc, epoch)
    return net, epoch, epoch_acc
