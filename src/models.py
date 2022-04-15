import torch.nn as nn
import torch
import math
from efficientnet_pytorch import EfficientNet

def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

class MeanPooling(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(MeanPooling, self).__init__()
        self.cla_activation = cla_activation
        self.cla = nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.cla)

    def activate(self, x, activation):
        return torch.sigmoid(x)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """
        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)
        cla = cla[:, :, :, 0]
        x = torch.mean(cla, dim=2)
        return x

class EffNetMean(nn.Module):
    def __init__(self, label_dim=527, level=0, pretrain=True):
        super(EffNetMean, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        if pretrain == False:
            print('not using imagenet pretrained network')
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(level), in_channels=1)
        else:
            print('using imagenet pretrained network')
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(level), in_channels=1)
        self.attention = MeanPooling(
            self.middim[level],
            label_dim,
            att_activation='sigmoid',
            cla_activation='sigmoid')
        self.avgpool = nn.AvgPool2d((4, 1))
        self.effnet._fc = nn.Identity()

    def forward(self, x):
        if x.shape[2] < x.shape[1]:
            x = x.transpose(1, 2)

        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.effnet.extract_features(x)
        x = self.avgpool(x)
        x = x.transpose(2,3)
        out = self.attention(x)
        return out