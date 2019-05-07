import os
import warnings

import torch

from sa_energy_model import FixHWConv2d, conv2d_out_dim, SparseConv2d
import torch.nn as nn
from pt_models import myalexnet
from pt_models import mysqueezenet1_0
from torchvision.models import alexnet, squeezenet1_0


class MyLeNet5(nn.Module):
    def __init__(self, conv_class=FixHWConv2d):
        super(MyLeNet5, self).__init__()
        h = 32
        w = 32
        feature_layers = []
        # conv
        feature_layers.append(conv_class(h, w, 1, 6, kernel_size=5))
        h = conv2d_out_dim(h, kernel_size=5)
        w = conv2d_out_dim(w, kernel_size=5)
        feature_layers.append(nn.ReLU(inplace=True))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        h = conv2d_out_dim(h, kernel_size=2, stride=2)
        w = conv2d_out_dim(w, kernel_size=2, stride=2)
        # conv
        feature_layers.append(conv_class(h, w, 6, 16, kernel_size=5))
        h = conv2d_out_dim(h, kernel_size=5)
        w = conv2d_out_dim(w, kernel_size=5)
        feature_layers.append(nn.ReLU(inplace=True))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        h = conv2d_out_dim(h, kernel_size=2, stride=2)
        w = conv2d_out_dim(w, kernel_size=2, stride=2)

        self.features = nn.Sequential(*feature_layers)

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 16 * 5 * 5)
        x = self.classifier(x)
        return x


mnist_pretrained_lenet5_path = os.path.dirname(os.path.realpath(__file__)) + '/mnist_pretrained_lenet5.pkl'


def get_net_model(net='alexnet', pretrained_dataset='imagenet', dropout=False, pretrained=True, input_mask=False):
    if input_mask:
        conv_class = SparseConv2d
    else:
        conv_class = FixHWConv2d
    if net == 'alexnet':
        model = myalexnet(pretrained=(pretrained_dataset == 'imagenet') and pretrained, dropout=dropout, conv_class=conv_class)
        teacher_model = alexnet(pretrained=(pretrained_dataset == 'imagenet'))
    elif net == 'squeezenet':
        model = mysqueezenet1_0(pretrained=(pretrained_dataset == 'imagenet') and pretrained, dropout=dropout, conv_class=conv_class)
        teacher_model = squeezenet1_0(pretrained=(pretrained_dataset == 'imagenet'))
    elif net == 'lenet-5':
        model = MyLeNet5(conv_class=conv_class)
        if pretrained and pretrained_dataset == 'mnist-32':
            model.load_state_dict(torch.load(mnist_pretrained_lenet5_path), strict=False)
        teacher_model = MyLeNet5()
        if os.path.isfile(mnist_pretrained_lenet5_path):
            teacher_model.load_state_dict(torch.load(mnist_pretrained_lenet5_path), strict=False)
        else:
            warnings.warn('failed to import teacher model!')
    else:
        raise NotImplementedError

    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    return model, teacher_model
