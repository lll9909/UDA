import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import torch

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)



class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)
        #
        # out += residual
        # out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)
        #
        # out += residual
        # out = self.relu(out)

        return out



class BasicBlock7x7(nn.Module):
    expansion = 1


    def __init__(self, inplanes7, planes, stride=1):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     residual = self.downsample(x)
        #
        # out += residual
        # out = self.relu(out)
        return out
class Attention(nn.Module):
    def __init__(self, features, M=3, r=16, L=32):
        # features:input channel
        # M 通路数
        # r 压缩比率
        # L 压缩后的最小特征
        super(Attention, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x, y, z):
        feas = torch.stack([x, y, z], dim=3)
        fea_U = feas.sum(dim=3)
        fea_s = fea_U.mean(-1)
        fea_z = self.fc(fea_s)
        for i, fcc in enumerate(self.fcs):
            vector = fcc(fea_z).unsqueeze_(dim=-1).unsqueeze_(dim=-1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=3)
        attention_vectors = self.softmax(attention_vectors)
        # att_vis = attention_vectors.mean(dim=0)
        # att_vis = att_vis[10, 0, :]
        # print('att:', att_vis)
        out = feas*attention_vectors
        out = out.sum(dim=-1)

        return out




class MSResNet(nn.Module):
    def __init__(self, input_channel, layers=[1, 1, 1, 1], num_classes=10):
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(MSResNet, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        self.layer3x3_4 = self._make_layer3(BasicBlock3x3, 512, layers[3], stride=2)

        # maxplooing kernel size: 16, 11, 6
        # self.maxpool3 = nn.AvgPool1d(kernel_size=32, stride=1, padding=0)
        # self.maxpool3 = nn.AdaptiveAvgPool1d(1)

        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        self.layer5x5_4 = self._make_layer5(BasicBlock5x5, 512, layers[3], stride=2)
        # self.maxpool5 = nn.AvgPool1d(kernel_size=26, stride=1, padding=0)
        # self.maxpool5 = nn.AdaptiveAvgPool1d(1)

        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=2)
        self.layer7x7_4 = self._make_layer7(BasicBlock7x7, 512, layers[3], stride=2)
        # self.maxpool7 = nn.AvgPool1d(kernel_size=21, stride=1, padding=0)
        # self.maxpool7 = nn.AdaptiveAvgPool1d(1)
        # self.drop = nn.Dropout(p=0.2)
        # self.fc = nn.Linear(512*3, num_classes)
        self.in_features = 512*3

        self.downsample0 = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1,bias=False),
                nn.BatchNorm1d(64),
            )
        self.downsample1 = nn.Sequential(
                nn.Conv1d(64, 128,kernel_size=3, stride=2, padding=1,bias=False),
                nn.BatchNorm1d(128),
            )
        self.downsample2 = nn.Sequential(
                nn.Conv1d(128, 256,kernel_size=3, stride=2, padding=1,bias=False),
                nn.BatchNorm1d(256),
            )
        self.downsample3 = nn.Sequential(
                nn.Conv1d(256, 512,kernel_size=3, stride=2, padding=1,bias=False),
                nn.BatchNorm1d(512),
            )
        self.attention0 = Attention(features=64)
        self.attention1 = Attention(features=128)
        self.attention2 = Attention(features=256)
        self.attention3 = Attention(features=512)
        self.avgpool = nn.AdaptiveAvgPool1d(3)

        # todo: modify the initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer3(self, block, planes, blocks, stride=2):
        layers = []
        layers.append(block(self.inplanes3, planes, stride))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))
        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        layers = []
        layers.append(block(self.inplanes5, planes, stride))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))
        return nn.Sequential(*layers)


    def _make_layer7(self, block, planes, blocks, stride=2):
        layers = []
        layers.append(block(self.inplanes7, planes, stride))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))
        return nn.Sequential(*layers)

    def forward(self, x0):
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x = self.layer3x3_1(x0)
        y = self.layer5x5_1(x0)
        z = self.layer7x7_1(x0)
        x1 = self.attention0(x, y, z) + self.downsample0(x0)
        x1 = self.relu(x1)

        x = self.layer3x3_2(x1)
        y = self.layer5x5_2(x1)
        z = self.layer7x7_2(x1)
        x2 = self.attention1(x, y, z) + self.downsample1(x1)
        x2 = self.relu(x2)

        x = self.layer3x3_3(x2)
        y = self.layer5x5_3(x2)
        z = self.layer7x7_3(x2)
        x3 = self.attention2(x, y, z) + self.downsample2(x2)
        x3 = self.relu(x3)

        x = self.layer3x3_4(x3)
        y = self.layer5x5_4(x3)
        z = self.layer7x7_4(x3)
        # x4 = x + y + z + self.downsample3(x3)
        x4 = self.attention3(x, y, z) + self.downsample3(x3)
        x4 = self.relu(x4)
        out = self.avgpool(x4)
        out = out.view(out.size(0), -1)
        return out
# sig = torch.randn(6,1,2120).cuda()
# msresnet = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=4)
# model = msresnet.cuda()
# pre = model(sig)
# print(pre.shape)
# sig = torch.randn(6,1,2100).cuda()
# msresnet = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=4)
# model = msresnet.cuda()
# pre = model(sig)
# print(model)
# att = Attention(features=64).cuda()
# fea = torch.randn(5, 64, 124).cuda()
# a = att(fea, fea, fea)








