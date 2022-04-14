import math
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.cuda.amp import autocast

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


def init_bn(bn):
    bn.weight.data.fill_(1.)

class Attention(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(Attention, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()


    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return torch.sigmoid(x)

        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        att = att[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        epsilon = 1e-7
        att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        return x, norm_att

class MeanPooling(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(MeanPooling, self).__init__()

        self.cla_activation = cla_activation

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

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
        #print(cla.shape)
        cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        #print(cla.shape)
        x = torch.mean(cla, dim=2)
        #print(x.shape)

        return x

class Attention2(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(Attention, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()


    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return torch.sigmoid(x)

        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        att = self.att(x)
        att = self.activate(att, self.att_activation)

        cla = self.cla(x)
        cla = self.activate(cla, self.cla_activation)

        att = att[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        cla = cla[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        # epsilon = 1e-7
        # att = torch.clamp(att, epsilon, 1. - epsilon)

        norm_att = att / torch.sum(att, dim=2)[:, :, None]
        x = torch.sum(norm_att * cla, dim=2)

        return x, norm_att

class MultiHeadAttention(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation):
        super(MultiHeadAttention, self).__init__()

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att1 = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.att2 = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla1 = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla2 = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()
        self.head_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

    def init_weights(self):
        init_layer(self.att1)
        init_layer(self.cla1)
        init_layer(self.att2)
        init_layer(self.cla2)

    def activate(self, x, activation):

        if activation == 'linear':
            return x

        elif activation == 'relu':
            return F.relu(x)

        elif activation == 'sigmoid':
            return torch.sigmoid(x)

        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        att1 = self.att1(x)
        att1 = self.activate(att1, self.att_activation)

        cla1 = self.cla1(x)
        cla1 = self.activate(cla1, self.cla_activation)

        att2 = self.att2(x)
        att2 = self.activate(att2, self.att_activation)

        cla2 = self.cla2(x)
        cla2 = self.activate(cla2, self.cla_activation)

        att1 = att1[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        cla1 = cla1[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        att2 = att2[:, :, :, 0]   # (samples_num, classes_num, time_steps)
        cla2 = cla2[:, :, :, 0]   # (samples_num, classes_num, time_steps)

        epsilon = 1e-7
        att1 = torch.clamp(att1, epsilon, 1. - epsilon)
        att2 = torch.clamp(att2, epsilon, 1. - epsilon)

        norm_att1 = att1 / torch.sum(att1, dim=2)[:, :, None]
        norm_att2 = att2 / torch.sum(att2, dim=2)[:, :, None]

        x1 = torch.sum(norm_att1 * cla1, dim=2)
        x2 = torch.sum(norm_att2 * cla2, dim=2)

        x = x1 * self.head_weight[0] + x2 * self.head_weight[1]

        return x, []

class MHeadAttention(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation, head_num=4):
        super(MHeadAttention, self).__init__()

        self.head_num = head_num

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.ModuleList([])
        self.cla = nn.ModuleList([])
        for i in range(self.head_num):
            self.att.append(nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))
            self.cla.append(nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))

        self.head_weight = nn.Parameter(torch.tensor([1.0/self.head_num] * self.head_num))

    def activate(self, x, activation):
        if activation == 'linear':
            return x
        elif activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        x_out = []
        for i in range(self.head_num):
            att = self.att[i](x)
            att = self.activate(att, self.att_activation)

            cla = self.cla[i](x)
            cla = self.activate(cla, self.cla_activation)

            att = att[:, :, :, 0]  # (samples_num, classes_num, time_steps)
            cla = cla[:, :, :, 0]  # (samples_num, classes_num, time_steps)

            epsilon = 1e-7
            att = torch.clamp(att, epsilon, 1. - epsilon)

            norm_att = att / torch.sum(att, dim=2)[:, :, None]
            x_out.append(torch.sum(norm_att * cla, dim=2) * self.head_weight[i])

        x = (torch.stack(x_out, dim=0)).sum(dim=0)

        return x, []


class MHeadAttention3(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation, head_num=4):
        super(MHeadAttention3, self).__init__()

        self.head_num = head_num

        self.att_activation = att_activation
        self.cla_activation = cla_activation

        self.att = nn.ModuleList([])
        self.cla = nn.ModuleList([])
        for i in range(self.head_num):
            self.att.append(nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))
            self.cla.append(nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))

        self.linear_proj = nn.Linear(n_out * self.head_num, n_out)

    def activate(self, x, activation):
        if activation == 'linear':
            return x
        elif activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        x_out = []
        for i in range(self.head_num):
            att = self.att[i](x)
            att = self.activate(att, self.att_activation)

            cla = self.cla[i](x)
            cla = self.activate(cla, self.cla_activation)

            att = att[:, :, :, 0]  # (samples_num, classes_num, time_steps)
            cla = cla[:, :, :, 0]  # (samples_num, classes_num, time_steps)

            epsilon = 1e-7
            att = torch.clamp(att, epsilon, 1. - epsilon)

            norm_att = att / torch.sum(att, dim=2)[:, :, None]
            x_out.append(torch.sum(norm_att * cla, dim=2))
        x = torch.cat(x_out, 1)
        #print(x.shape)
        x = self.linear_proj(x)
        x = torch.sigmoid(x)
        return x, []


class MHeadAttention2(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation, head_num=4):
        super(MHeadAttention2, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(n_out, num_heads=1)

        self.q_linear = nn.Linear(n_in, n_out)
        self.k_linear = nn.Linear(n_in, n_out)
        self.v_linear = nn.Linear(n_in, n_out)

    def activate(self, x, activation):
        if activation == 'linear':
            return x
        elif activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """
        x = x.squeeze()
        x = x.permute([2, 0, 1])
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = self.q_linear(x)
        attn_output, _ = self.multihead_attn(q, k, v)
        attn_output = torch.mean(attn_output, dim=0)
        attn_output = attn_output.squeeze()
        attn_output = torch.sigmoid(attn_output)
        return attn_output, []



class MHeadAttentionT(nn.Module):
    def __init__(self, n_in, n_out, att_activation, cla_activation, head_num=4):
        super(MHeadAttentionT, self).__init__()

        self.transformer = torch.nn.Transformer(n_in, nhead=1, num_encoder_layers=1, num_decoder_layers=0)
        self.linear = nn.Linear(n_in, n_out)

    def activate(self, x, activation):
        if activation == 'linear':
            return x
        elif activation == 'relu':
            return F.relu(x)
        elif activation == 'sigmoid':
            return torch.sigmoid(x)
        elif activation == 'softmax':
            return F.softmax(x, dim=1)

    def forward(self, x):
        """input: (samples_num, freq_bins, time_steps, 1)
        """
        x = x.squeeze()
        x = x.permute([2, 0, 1])
        print(x.shape)
        x = self.transformer(x, x)
        # output torch.Size([66, 12, 1408])
        x = self.linear(x)
        # output torch.Size([66, 12, 527])
        x = x.permute([1, 2, 0])
        x = torch.mean(x, dim=2)
        print(x.shape)
        print(torch.mean(torch.abs(x)))
        print(torch.max(x))
        #x = torch.sigmoid(x)
        return x, []

class EmbeddingLayers(nn.Module):

    def __init__(self, freq_bins, hidden_units, drop_rate):
        super(EmbeddingLayers, self).__init__()

        self.freq_bins = freq_bins
        self.hidden_units = hidden_units
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv2d(
            in_channels=freq_bins, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=hidden_units, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.conv3 = nn.Conv2d(
            in_channels=hidden_units, out_channels=hidden_units,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

        self.bn0 = nn.BatchNorm2d(freq_bins)
        self.bn1 = nn.BatchNorm2d(hidden_units)
        self.bn2 = nn.BatchNorm2d(hidden_units)
        self.bn3 = nn.BatchNorm2d(hidden_units)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)

        init_bn(self.bn0)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)

    def forward(self, input, return_layers=False):
        """input: (samples_num, time_steps, freq_bins)
        """

        drop_rate = self.drop_rate

        # (samples_num, freq_bins, time_steps)
        x = input.transpose(1, 2)

        # Add an extra dimension for using Conv2d
        # (samples_num, freq_bins, time_steps, 1)
        x = x[:, :, :, None].contiguous()

        a0 = self.bn0(x)
        a1 = F.dropout(F.relu(self.bn1(self.conv1(a0))),
                       p=drop_rate,
                       training=self.training)

        a2 = F.dropout(F.relu(self.bn2(self.conv2(a1))),
                       p=drop_rate,
                       training=self.training)

        emb = F.dropout(F.relu(self.bn3(self.conv3(a2))),
                        p=drop_rate,
                        training=self.training)

        if return_layers is False:
            # (samples_num, hidden_units, time_steps, 1)
            return emb

        else:
            return [a0, a1, a2, emb]


class DecisionLevelMaxPooling(nn.Module):

    def __init__(self, freq_bins, classes_num, hidden_units, drop_rate):

        super(DecisionLevelMaxPooling, self).__init__()

        self.emb = EmbeddingLayers(freq_bins, hidden_units, drop_rate)
        self.fc_final = nn.Linear(hidden_units, classes_num)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, time_steps, freq_bins)
        """

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, time_steps, hidden_units)
        b1 = b1[:, :, :, 0].transpose(1, 2)

        b2 = torch.sigmoid(self.fc_final(b1))

        # (samples_num, classes_num)
        (output, _) = torch.max(b2, dim=1)

        return output


class DecisionLevelAveragePooling(nn.Module):

    def __init__(self, freq_bins, classes_num, hidden_units, drop_rate):

        super(DecisionLevelAveragePooling, self).__init__()

        self.emb = EmbeddingLayers(freq_bins, hidden_units, drop_rate)
        self.fc_final = nn.Linear(hidden_units, classes_num)

    def init_weights(self):

        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, time_steps, hidden_units)
        b1 = b1[:, :, :, 0].transpose(1, 2)

        b2 = torch.sigmoid(self.fc_final(b1))

        # (samples_num, classes_num)
        output = torch.mean(b2, dim=1)

        return output


class DecisionLevelSingleAttention(nn.Module):

    def __init__(self, freq_bins, classes_num, hidden_units, drop_rate):

        super(DecisionLevelSingleAttention, self).__init__()

        self.emb = EmbeddingLayers(freq_bins, hidden_units, drop_rate)
        self.attention = Attention(
            hidden_units,
            classes_num,
            att_activation='sigmoid',
            cla_activation='sigmoid')

    def init_weights(self):
        pass

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1)
        """

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, classes_num, time_steps, 1)
        output, norm_att = self.attention(b1)

        return output


class DecisionLevelMultiAttention(nn.Module):

    def __init__(self, freq_bins, classes_num, hidden_units, drop_rate):

        super(DecisionLevelMultiAttention, self).__init__()

        self.emb = EmbeddingLayers(freq_bins, hidden_units, drop_rate)
        self.attention = Attention(
            hidden_units,
            classes_num,
            att_activation='sigmoid',
            cla_activation='sigmoid')
        self.fc_final = nn.Linear(classes_num * 2, classes_num)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc_final)

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1)
        """


        # (samples_num, hidden_units, time_steps, 1)
        emb_layers = self.emb(input, return_layers=True)

        # (samples_num, classes_num)
        output1, norm_att1 = self.attention(emb_layers[-1])
        output2, norm_att2 = self.attention(emb_layers[-2])

        # (samples_num, classes_num * 2)
        cat_output = torch.cat((output1, output2), dim=1)

        # (samples_num, class_num)
        output = torch.sigmoid(self.fc_final(cat_output))

        return output


class FeatureLevelSingleAttention(nn.Module):

    def __init__(self, freq_bins, classes_num, hidden_units, drop_rate):

        super(FeatureLevelSingleAttention, self).__init__()

        self.emb = EmbeddingLayers(freq_bins, hidden_units, drop_rate)

        self.attention = Attention(
            hidden_units,
            hidden_units,
            att_activation='sigmoid',
            cla_activation='linear')

        self.fc_final = nn.Linear(hidden_units, classes_num)
        self.bn_attention = nn.BatchNorm1d(hidden_units)

        self.drop_rate = drop_rate

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc_final)
        init_bn(self.bn_attention)

    def forward(self, input):
        """input: (samples_num, freq_bins, time_steps, 1)
        """
        drop_rate = self.drop_rate

        # (samples_num, hidden_units, time_steps, 1)
        b1 = self.emb(input)

        # (samples_num, hidden_units)
        b2, norm_att = self.attention(b1)
        b2 = F.dropout(
            F.relu(
                self.bn_attention(b2)),
            p=drop_rate,
            training=self.training)

        # (samples_num, classes_num)
        output = torch.sigmoid(self.fc_final(b2))

        return output

