import torch
from torch import nn
import os
import time
import sys
import argparse
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile, Image
import random
import math
import numpy as np


class Conv(nn.Module):
    def __init__(self, n_in, n_out, stride=1, bias=True, activate=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1, bias=bias)
        if activate:
            self.bn = nn.BatchNorm2d(n_out)
            self.elu = nn.ELU(inplace=True)
        self.activate = activate

    def forward(self, x):
        out = self.conv(x)

        if self.activate:
            out = self.bn(out)
            out = self.elu(out)
        return out


class Compress(nn.Module):
    def __init__(self, n_in, n_out):
        super(Compress, self).__init__()
        com_in = [n_in, 16, 32, 64, 128, 256, 128, 64, 32]
        com_out = [16, 32, 64, 128, 256, 128, 64, 32, n_out]

        com = []
        for i in range(len(com_in)):
            com.append(Conv(com_in[i], com_out[i], activate=(False if i == len(com_in) - 1 else True)))
        self.com = nn.Sequential(*com)
        # self.com_shortlink=nn.Sequential(nn.Conv2d(com_in[0],com_out[-1],kernel_size=1,stride=1,bias=True))

    def forward(self, x):
        com_x = self.com(x)  # +self.com_shortlink(x)
        return com_x


class Recover(nn.Module):
    def __init__(self, n_in, n_out):
        super(Recover, self).__init__()
        rec_in = [n_in, 32, 64, 128, 256, 128, 64, 32, 16]
        rec_out = [32, 64, 128, 256, 128, 64, 32, 16, n_out]

        rec = []
        for i in range(len(rec_in)):
            rec.append(Conv(rec_in[i], rec_out[i], activate=(False if i == len(rec_in) - 1 else True)))
        self.rec = nn.Sequential(*rec)
        # self.rec_shortlink=nn.Sequential(nn.Conv2d(rec_in[0],rec_out[-1],kernel_size=1,stride=1,bias=True))

    def forward(self, x):
        rec_x = self.rec(x)  # +self.rec_shortlink(x)
        return rec_x


class Rectifi(nn.Module):
    def __init__(self, n_in=3, n_out=3, bit=12):
        super(Rectifi, self).__init__()
        self.com = Compress(n_in, bit)
        self.rec = Recover(bit, n_out)

        self.ac = nn.Hardtanh(0, 1)

    def forward(self, x):
        com_x = self.com(x)

        # com_x=com_x+torch.randn(com_x.shape).cuda()*20.
        # x=self.ac(com_x)
        x = (com_x > 0).float()
        # print(x)
        rec_x = self.rec(x)
        return rec_x

    def get_loss(self, images, ad_images, cost, flag, classify_model=None):

        com_x = self.com(images)

        rec_x = self.rec(self.ac(com_x + 0.5))

        com_xbar = self.com(ad_images)
        rec_xbar = self.rec(self.ac(com_xbar + 0.5))

        loss1 = cost(images, rec_x).mean()
        if classify_model is None:
            loss2 = cost(images, rec_xbar).mean()
        else:
            loss2 = cost(images, rec_xbar, classify_model).mean()

        loss = loss1 + loss2  # +loss3
        if flag:
            print(torch.abs(com_x).mean().item(), torch.abs(com_xbar).mean().item(),
                  (torch.abs(com_x) < 0.5).sum().item() * 1.0 / (torch.abs(com_x) > -1).sum().item(),
                  (torch.abs(com_xbar) < 0.5).sum().item() * 1.0 / (torch.abs(com_xbar) > -1).sum().item())
        # print("loss1 = ",loss1.item(),"loss2 = ",loss2.item())
        return loss, rec_x, rec_xbar, loss1.item(), loss2.item()
