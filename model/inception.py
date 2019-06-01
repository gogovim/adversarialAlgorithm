import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import os

class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(in_channels, 64, kernel_size=1))

        self.branch1 = nn.Sequential(BasicConv2d(in_channels, 48, kernel_size=1),
             BasicConv2d(48, 64, kernel_size=5, padding=2))

        self.branch2 = nn.Sequential(BasicConv2d(in_channels, 64, kernel_size=1),
                     BasicConv2d(64, 96, kernel_size=3, padding=1),
                     BasicConv2d(96, 96, kernel_size=3, padding=1))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_channels, pool_features, kernel_size=1))

    def forward(self, x):
        branch1x1 = self.branch0(x)

        branch5x5 = self.branch1(x)

        branch3x3dbl = self.branch2(x)

        branch_pool = self.branch3(x)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(in_channels, 384, kernel_size=3, stride=2))

        self.branch1 = nn.Sequential(BasicConv2d(in_channels, 64, kernel_size=1),
         BasicConv2d(64, 96, kernel_size=3, padding=1),
         BasicConv2d(96, 96, kernel_size=3, stride=2))

    def forward(self, x):
        branch3x3 = self.branch0(x)

        branch3x3dbl = self.branch1(x)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(in_channels, 192, kernel_size=1))

        c7 = channels_7x7
        self.branch1 = nn.Sequential(BasicConv2d(in_channels, c7, kernel_size=1),
              BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
              BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0)))

        self.branch2 = nn.Sequential(BasicConv2d(in_channels, c7, kernel_size=1),
              BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
              BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
             BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
              BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3)))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_channels, 192, kernel_size=1))

    def forward(self, x):
        branch1x1 = self.branch0(x)

        branch7x7 = self.branch1(x)

        branch7x7dbl = self.branch2(x)

        branch_pool = self.branch3(x)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(in_channels, 192, kernel_size=1),
          BasicConv2d(192, 320, kernel_size=3, stride=2))

        self.branch1 = nn.Sequential(BasicConv2d(in_channels, 192, kernel_size=1),
          BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
          BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
          BasicConv2d(192, 192, kernel_size=3, stride=2))

    def forward(self, x):
        branch3x3 = self.branch0(x)

        branch7x7x3 = self.branch1(x)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)

class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch0 = nn.Sequential(BasicConv2d(in_channels, 320, kernel_size=1))
             
        self.branch1_0 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch1_1 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch1_2 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch2_0 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch2_1 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch2_2 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch2_3 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_channels, 192, kernel_size=1))

    def forward(self, x):
        branch1x1 = self.branch0(x)

        branch3x3 = self.branch1_0(x)
        branch3x3 = [
            self.branch1_1(branch3x3),
            self.branch1_2(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch2_0(x)
        branch3x3dbl = self.branch2_1(branch3x3dbl)
        branch3x3dbl = [
            self.branch2_2(branch3x3dbl),
            self.branch2_3(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = self.branch3(x)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

 



class Bottleneck(nn.Module):
    def __init__(self, n_in, n_out, stride = 1, expansion = 4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(n_out)
        self.conv3 = nn.Conv2d(n_out, n_out * expansion, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(n_out * expansion)

        self.downsample = None
        if stride != 1 or n_in != n_out * expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(n_in, n_out * expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(n_out * expansion))

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Null(nn.Module):
    def __init__(self):
        super(Null, self).__init__()

    def forward(self, x):
        return x




class Inception3(nn.Module):

    def __init__(self, num_classes=1001):
        super(Inception3, self).__init__()
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classif = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, need_feature=False):

        # 299 x 299 x 3
        x = self.conv2d_1a(x)
        # 149 x 149 x 32
        x = self.conv2d_2a(x)
        # 147 x 147 x 32
        x = self.conv2d_2b(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.conv2d_3b(x)
        # 73 x 73 x 80
        x = self.conv2d_4a(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        feature=x

        # 8 x 8 x 2048
        x = self.avgpool(x)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.classif(x)
        # 1000 (num_classes)
        if need_feature:
            return x,feature
        return x