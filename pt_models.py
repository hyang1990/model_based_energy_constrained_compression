import torch
import torch.utils.model_zoo as model_zoo
from torch import nn as nn
import torch.nn.init as init

from sa_energy_model import FixHWConv2d, conv2d_out_dim

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
}

################################################################
#########################  Alex NET   ##########################
################################################################

class MyAlexNet(nn.Module):
    def __init__(self, h=224, w=224, conv_class=FixHWConv2d, num_classes=1000, dropout=True):
        super(MyAlexNet, self).__init__()
        feature_layers = []

        # conv
        feature_layers.append(conv_class(h, w, 3, 64, kernel_size=11, stride=4, padding=2))
        h = conv2d_out_dim(h, kernel_size=11, stride=4, padding=2)
        w = conv2d_out_dim(w, kernel_size=11, stride=4, padding=2)
        feature_layers.append(nn.ReLU(inplace=True))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        h = conv2d_out_dim(h, kernel_size=3, stride=2)
        w = conv2d_out_dim(w, kernel_size=3, stride=2)

        # conv
        feature_layers.append(conv_class(h, w, 64, 192, kernel_size=5, padding=2))
        h = conv2d_out_dim(h, kernel_size=5, padding=2)
        w = conv2d_out_dim(w, kernel_size=5, padding=2)
        feature_layers.append(nn.ReLU(inplace=True))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        h = conv2d_out_dim(h, kernel_size=3, stride=2)
        w = conv2d_out_dim(w, kernel_size=3, stride=2)

        # conv
        feature_layers.append(conv_class(h, w, 192, 384, kernel_size=3, padding=1))
        h = conv2d_out_dim(h, kernel_size=3, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, padding=1)
        feature_layers.append(nn.ReLU(inplace=True))

        # conv
        feature_layers.append(conv_class(h, w, 384, 256, kernel_size=3, padding=1))
        h = conv2d_out_dim(h, kernel_size=3, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, padding=1)
        feature_layers.append(nn.ReLU(inplace=True))

        # conv
        feature_layers.append(conv_class(h, w, 256, 256, kernel_size=3, padding=1))
        h = conv2d_out_dim(h, kernel_size=3, padding=1)
        w = conv2d_out_dim(w, kernel_size=3, padding=1)
        feature_layers.append(nn.ReLU(inplace=True))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        h = conv2d_out_dim(h, kernel_size=3, stride=2)
        w = conv2d_out_dim(w, kernel_size=3, stride=2)

        self.features = nn.Sequential(*feature_layers)

        fc_layers = [nn.Dropout(p=0.5 if dropout else 0.0),
                     nn.Linear(256 * 6 * 6, 4096),
                     nn.ReLU(inplace=True),
                     nn.Dropout(p=0.5 if dropout else 0.0),
                     nn.Linear(4096, 4096),
                     nn.ReLU(inplace=True),
                     nn.Linear(4096, num_classes)]

        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def get_inhw(self, x):
        res = []
        for module in self.features._modules.values():
            if isinstance(module, nn.Conv2d):
                res.append((x.size(2), x.size(3)))
            x = module(x)
        for module in self.classifier._modules.values():
            if isinstance(module, nn.Linear):
                res.append((1, 1))
        return res


def myalexnet(pretrained=False, model_root=None, **kwargs):
    model = MyAlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], model_root), strict=False)
    return model


################################################################
########################  Squeeze NET   ########################
################################################################

class MySqueezeNet(nn.Module):
    class MyFire(nn.Module):

        def __init__(self, inplanes, squeeze_planes,
                     expand1x1_planes, expand3x3_planes, h_in, w_in, conv_class=FixHWConv2d):
            super(MySqueezeNet.MyFire, self).__init__()
            h = h_in
            w = w_in

            self.inplanes = inplanes
            self.squeeze = conv_class(h, w, inplanes, squeeze_planes, kernel_size=1)
            self.squeeze_activation = nn.ReLU(inplace=True)
            h = conv2d_out_dim(h, kernel_size=1)
            w = conv2d_out_dim(w, kernel_size=1)

            self.expand1x1 = conv_class(h, w, squeeze_planes, expand1x1_planes, kernel_size=1)
            self.expand1x1_activation = nn.ReLU(inplace=True)
            self.expand3x3 = conv_class(h, w, squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
            self.expand3x3_activation = nn.ReLU(inplace=True)
            h = conv2d_out_dim(h, kernel_size=3, padding=1)
            w = conv2d_out_dim(w, kernel_size=3, padding=1)

        def forward(self, x):
            x = self.squeeze_activation(self.squeeze(x))
            return torch.cat([
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x))
            ], 1)

    def __init__(self, version=1.0, h=224, w=224, conv_class=FixHWConv2d, num_classes=1000, dropout=True):
        MyFire = self.MyFire
        super(MySqueezeNet, self).__init__()
        if version not in [1.0]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0".format(version=version))
        self.num_classes = num_classes

        feature_layers = []
        # conv
        feature_layers.append(conv_class(h, w, 3, 96, kernel_size=7, stride=2))
        h = conv2d_out_dim(h, kernel_size=7, stride=2)
        w = conv2d_out_dim(w, kernel_size=7, stride=2)
        feature_layers.append(nn.ReLU(inplace=True))
        # pooling
        feature_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        h = conv2d_out_dim(h, kernel_size=3, stride=2, ceil_mode=True)
        w = conv2d_out_dim(w, kernel_size=3, stride=2, ceil_mode=True)

        # fire block
        feature_layers.append(MyFire(96, 16, 64, 64, h_in=h, w_in=w, conv_class=conv_class))
        feature_layers.append(MyFire(128, 16, 64, 64, h_in=h, w_in=w, conv_class=conv_class))
        feature_layers.append(MyFire(128, 32, 128, 128, h_in=h, w_in=w, conv_class=conv_class))
        feature_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        h = conv2d_out_dim(h, kernel_size=3, stride=2, ceil_mode=True)
        w = conv2d_out_dim(w, kernel_size=3, stride=2, ceil_mode=True)

        feature_layers.append(MyFire(256, 32, 128, 128, h_in=h, w_in=w, conv_class=conv_class))
        feature_layers.append(MyFire(256, 48, 192, 192, h_in=h, w_in=w, conv_class=conv_class))
        feature_layers.append(MyFire(384, 48, 192, 192, h_in=h, w_in=w, conv_class=conv_class))
        feature_layers.append(MyFire(384, 64, 256, 256, h_in=h, w_in=w, conv_class=conv_class))
        feature_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
        h = conv2d_out_dim(h, kernel_size=3, stride=2, ceil_mode=True)
        w = conv2d_out_dim(w, kernel_size=3, stride=2, ceil_mode=True)

        feature_layers.append(MyFire(512, 64, 256, 256, h_in=h, w_in=w, conv_class=conv_class))

        self.features = nn.Sequential(*feature_layers)
        # Final convolution is initialized differently form the rest
        final_conv = conv_class(h, w, 512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5 if dropout else 0.0),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def mysqueezenet1_0(pretrained=False, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MySqueezeNet(version=1.0, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_0']), strict=False)
    return model
