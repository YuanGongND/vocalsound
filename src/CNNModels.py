import torch.nn as nn
import torch
from .HigherModels import *
from efficientnet_pytorch import EfficientNet
import torchvision

class MBNet(nn.Module):
    def __init__(self, label_dim=527, level=0, pretrain=True):
        super(MBNet, self).__init__()

        self.model = torchvision.models.mobilenet_v2(pretrained=pretrain)

        self.model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = torch.nn.Linear(in_features=1280, out_features=527, bias=True)

    def forward(self, x, nframes):
        # this only for baseline that has reverse order of time and feat dim

        if x.shape[2] < x.shape[1]:
            x = x.transpose(1, 2)

        if x.dim() == 3:
            x = x.unsqueeze(1)

        #x = x.expand([-1, 3, -1, -1])
        out = torch.sigmoid(self.model(x))
        return out, out[:,0]

class EffNetFullAttention(nn.Module):
    def __init__(self, label_dim=527, level=0, pretrain=True, head_num=4):
        super(EffNetFullAttention, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        if pretrain == False:
            print('not using imagenet pretrained network')
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(level), in_channels=1)
        else:
            print('using imagenet pretrained network')
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(level), in_channels=1)
        self.attention = MHeadAttention(
            self.middim[level],
            label_dim,
            att_activation='sigmoid',
            cla_activation='sigmoid')
        self.avgpool = nn.AvgPool2d((4, 1))
        self.effnet._fc = nn.Identity()
        #self.avgpool = nn.MaxPool2d((4, 1))

    def forward(self, x, nframes=1056):
        # # this only for baseline that has reverse order of time and feat dim

        if x.shape[2] < x.shape[1]:
            x = x.transpose(1, 2)

        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.effnet.extract_features(x)
        x = self.avgpool(x)
        x = x.transpose(2,3)
        out, norm_att = self.attention(x)
        return out, norm_att

# with different attention layer
class EffNetRevAttention(nn.Module):
    def __init__(self, label_dim=527, level=0, pretrain=True, head_num=4):
        super(EffNetRevAttention, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        if pretrain == False:
            print('not using imagenet pretrained network')
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(level), in_channels=1)
        else:
            print('using imagenet pretrained network')
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(level), in_channels=1)
        self.attention = Attention(
            self.middim[level],
            label_dim,
            att_activation='sigmoid',
            cla_activation='sigmoid')
        self.avgpool = nn.AvgPool2d((4, 1))
        self.effnet._fc = nn.Identity()
        #self.avgpool = nn.MaxPool2d((4, 1))

    def forward(self, x, nframes=1056):
        # # this only for baseline that has reverse order of time and feat dim

        if x.shape[2] < x.shape[1]:
            x = x.transpose(1, 2)

        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.effnet.extract_features(x)
        x = self.avgpool(x)
        x = x.transpose(2,3)
        out, norm_att = self.attention(x)
        return out, norm_att

class EffNetMean(nn.Module):
    def __init__(self, label_dim=527, level=0, pretrain=True, head_num=4):
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

    def forward(self, x, nframes=1056):
        # # this only for baseline that has reverse order of time and feat dim

        if x.shape[2] < x.shape[1]:
            x = x.transpose(1, 2)

        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.effnet.extract_features(x)
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = x.transpose(2,3)
        out = self.attention(x)
        #print(out.shape)
        return out, out[:,0]

class EffNetFullAttentionMin(nn.Module):
    def __init__(self, label_dim=527, level=0, pretrain=True, head_num=4):
        super(EffNetFullAttentionMin, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        if pretrain == False:
            print('not using imagenet pretrained network')
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(level), in_channels=1)
        else:
            print('using imagenet pretrained network')
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(level), in_channels=1)
        self.attention = Attention(self.middim[level], label_dim, att_activation='sigmoid', cla_activation='sigmoid')
        self.avgpool = nn.AvgPool2d((2, 1))

    def forward(self, x, nframes=1056):
        # # this only for baseline that has reverse order of time and feat dim
        if x.shape[2] < x.shape[1]:
            x = x.transpose(1, 2)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.effnet.extract_features(x)
        x = self.avgpool(x)
        x = x.transpose(2,3)
        out, norm_att = self.attention(x)
        return out, norm_att


audio_mdl = EffNetMean(label_dim=6, level=0, pretrain=False)
test_input = torch.rand(5, 1024, 128)
out = audio_mdl(test_input)