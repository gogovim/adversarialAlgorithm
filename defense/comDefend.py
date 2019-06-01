import torch
from  torch import nn
import os
import time
import sys
import argparse
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile,Image
import random
import math
import numpy as np


class Conv(nn.Module):
    def __init__(self, n_in, n_out, stride = 1,bias=True,activate=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1, bias = bias)
        if activate:
            self.bn = nn.BatchNorm2d(n_out)
            self.elu = nn.ELU(inplace = True)
        self.activate=activate

    def forward(self, x):
        out = self.conv(x)
        
        if self.activate:
            out = self.bn(out)
            out = self.elu(out)
        return out

class ComDefend(nn.Module):
    def __init__(self,pho_size=299,short_link=True):
        super(ComDefend, self).__init__()
        self.pho_size=pho_size
        self.short_link=short_link
        com_in=[3,16,32,64,128,256,128,64,32]
        com_out=[16,32,64,128,256,128,64,32,12]

        rec_in=[12,32,64,128,256,128,64,32,16]
        rec_out=[32,64,128,256,128,64,32,16,3]

        com=[]
        for i in range(len(com_in)):
            com.append(Conv(com_in[i],com_out[i],activate=(False if i==len(com_in)-1 else True)))
        self.com=nn.Sequential(*com)
        if short_link:
            self.com_shortlink=nn.Sequential(nn.Conv2d(com_in[0],com_out[-1],kernel_size=1,stride=1,bias=False),nn.BatchNorm2d(com_out[-1]))

        rec=[]
        for i in range(len(rec_in)):
            rec.append(Conv(rec_in[i],rec_out[i],activate=(False if i==len(rec_in)-1 else True)))
        self.rec=nn.Sequential(*rec)
        if short_link:
            self.rec_shortlink=nn.Sequential(nn.Conv2d(rec_in[0],rec_out[-1],kernel_size=1,stride=1,bias=False),nn.BatchNorm2d(rec_out[-1]))
        

    def forward(self, x,mode='test',noise=None):
        com_x=self.com(x)
        if self.short_link:
            com_x+=self.com_shortlink(x)
        if noise is None:
            noise=torch.randn(com_x.shape).cuda()

        if noise is not None:
            #print(noise.mean())
            x=com_x-noise
        else:
            x=com_x
        x=torch.sigmoid(x)
        x=(x>0.5).float()

        #print(x.mean(),com_x.mean(),noise.mean())
        rec_x=self.rec(x)
        if self.short_link:
            rec_x+=self.rec_shortlink(x)
        if mode=='test':
            return rec_x
        return com_x,rec_x
