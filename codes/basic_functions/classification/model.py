import torch.nn as nn
import torch


# class PatchNet(nn.Module):
#     # without dropout
#     # precision =  0.59
#     # recall =  0.85
#     # epoch: 20
#
#     def __init__(self):
#         super(PatchNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16,  kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
#             nn.ReLU(True),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm2d(64),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
#             # nn.Dropout2d(),
#             nn.ReLU(True),
#             nn.BatchNorm2d(64),
#
#             nn.MaxPool2d(kernel_size=2, stride=2)
#          )
#         self.linear = nn.Sequential(
#             nn.Linear(3*3*64, 1024),
#             nn.ReLU(True),
#             nn.BatchNorm1d(1024),
#             nn.Dropout(),
#
#             nn.Linear(1024, 512),
#             nn.ReLU(True),
#             nn.BatchNorm1d(512),
#             nn.Dropout(),
#
#             nn.Linear(512, 2)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         y = self.linear(x)
#         return y


class PatchNet(nn.Module):
    # without dropout
    # precision =  0.59
    # recall =  0.85
    # epoch: 20

    def __init__(self):
        super(PatchNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16,  kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.Dropout2d(),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2),
            # nn.Dropout2d(),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(kernel_size=2, stride=2)
         )
        self.linear = nn.Sequential(
            nn.Linear(3*3*64, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Dropout(),

            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(),

            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        y = self.linear(x)
        return y

# class PatchNet(nn.Module):
#     def __init__(self):
#         super(PatchNet, self).__init__()
#         self.pre_conv = nn.Sequential(
#             nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm2d(16),
#         )
#         self.block1 = nn.Sequential(
#             nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm2d(16),
#             nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm2d(16),
#         )
#         self.middle_conv = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm2d(32),
#         )
#         self.block2 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm2d(32),
#         )
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.linear = nn.Sequential(
#             nn.Linear(2*2*32, 1024),
#             nn.ReLU(True),
#             nn.BatchNorm1d(1024),
#             nn.Dropout(),
#
#             nn.Linear(1024, 512),
#             nn.ReLU(True),
#             nn.BatchNorm1d(512),
#             nn.Dropout(),
#
#             nn.Linear(512, 2)
#         )
#
#     def forward(self, x):
#         x = self.pre_conv(x)
#         out = self.block1(x)
#         out += x
#
#         x = self.middle_conv(out)
#
#         out = self.block2(x)
#         out += x
#         out = self.pool(out)
#         out = torch.flatten(out, 1)
#         y = self.linear(out)
#         return y
