import os
import torch
import torch.nn as nn
from codes.basic_functions.classification.dataset import PatchDataset
from codes.basic_functions.classification.channel_1_dataset import PatchDatasetC1
from torch.utils.data import DataLoader
from codes.basic_functions.classification.model import PatchNet
from torch.utils.tensorboard import SummaryWriter


def train(gpu, frames, neuron_boxes, epoch_num, learning_rate=1e-05):
    train_set = PatchDatasetC1(frames, neuron_boxes, phase='train',
                             key_mask_root='./experiments/key_points/gaussian/train', patch_size=9)
    val_set = PatchDatasetC1(frames, neuron_boxes, phase='val',
                           key_mask_root='./experiments/key_points/gaussian/train', patch_size=9)

    # train_set = PatchDataset(frames, neuron_boxes, phase='train',
    #                          key_mask_root='./experiments/key_points/gaussian/train', patch_size=9)
    # val_set = PatchDataset(frames, neuron_boxes, phase='val',
    #                        key_mask_root='./experiments/key_points/gaussian/train', patch_size=9)

    data_loaders = {'train': DataLoader(train_set, batch_size=2048, shuffle=True, pin_memory=True),
                    'val': DataLoader(val_set, batch_size=2048, shuffle=True, pin_memory=True)}
    loggers = {phase: SummaryWriter('./experiments/loggers' + '/{}'.format(phase)) for phase in ['train', 'val']}

    net = PatchNet().cuda(gpu)
    criterion = nn.CrossEntropyLoss()
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
                for batch, (images, labels) in enumerate(data_loaders[phase]):
                    images = images.cuda(gpu)
                    labels = labels.cuda(gpu)
                    out = net(images)
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
