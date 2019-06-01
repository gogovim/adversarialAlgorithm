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
    def __init__(self, n_in, n_out, stride = 1,bias=True,activate=True,kernel_size=3):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = kernel_size, stride = stride, padding = (0 if kernel_size==1 else 1), bias = bias)
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

class BasicBlock(nn.Module):

    def __init__(self, n_in, n_out, stride=1, downsample=None, activate=True):
        super(BasicBlock, self).__init__()
        self.activate=activate
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv(n_in, n_out, stride=stride,bias=True,activate=True)
        self.conv2 = Conv(n_out, n_out,bias=True,activate=False)
        self.downsample = downsample

        if self.activate:
            self.bn = nn.BatchNorm2d(n_out)
            self.elu = nn.ELU(inplace = True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        #print(out.shape,identity.shape)
        
        if self.activate:
            out=self.bn(out)
            out+=identity
            out=self.elu(out)
        else:
            out += identity
        return out
class Mymodel(nn.Module):
    def __init__(self,pho_size=299):
        super(Mymodel, self).__init__()
        self.pho_size=pho_size
        com_in=[3,64,128,256]
        com_out=[64,128,256,512]
        nums_in=[2,4,2,2]
        com=[]
        for i in range(len(com_in)):
            for j in range(nums_in[i]):
                n_in=com_in[i] if j==0 else com_out[i]
                n_out=com_out[i]
                stride=(2 if j==0 else 1)
                activate=(False if i==len(com_in)-1 and j==nums_in[i]-1 else True)
                downsample=None
                if n_in!=n_out or stride!=1:
                    downsample=nn.Sequential(Conv(n_in, n_out, stride = stride,bias=True,activate=False,kernel_size=1),
                        nn.BatchNorm2d(n_out)) if activate else nn.Sequential(Conv(n_in, n_out, stride = stride,bias=True,activate=False,kernel_size=1))
                com.append(BasicBlock(n_in,n_out,stride=stride,downsample=downsample,activate=activate))
        self.com=nn.Sequential(*com)

        rec_in=[512,256,128,64]
        rec_out=[256,128,64,3]
        nums_in=[2,2,4,2]
        rec=[]
        pho_sizes=[]
        for i in range(len(rec_in)):
            pho_sizes.append(pho_size)
            pho_size=(pho_size+1)//2
        pho_sizes=pho_sizes[::-1]
        for i in range(len(rec_in)):
            rec.append(nn.Upsample(size = (pho_sizes[i], pho_sizes[i]), mode = 'bilinear'))
            #pho_size=(pho_size+1)//2
            for j in range(nums_in[i]):
                n_in=rec_in[i] if j==0 else rec_out[i]
                n_out=rec_out[i]
                #stride=(1 if n_in==n_out else 2)
                activate=(False if i==len(rec_in)-1 and j==nums_in[i]-1 else True)

                downsample=None
                if n_in!=n_out:
                    downsample=nn.Sequential(Conv(n_in, n_out,activate=False,kernel_size=1),
                        nn.BatchNorm2d(n_out)) if activate else nn.Sequential(Conv(n_in, n_out,activate=False,kernel_size=1))
                rec.append(BasicBlock(n_in,n_out,activate=activate,downsample=downsample))
        self.rec=nn.Sequential(*rec)
        

    def forward(self, x,noise=None):
        com_x=self.com(x)
        if noise is None:
            noise=torch.randn(com_x.shape).cuda()
        if noise is not None:
            #print(noise)
            x=com_x-noise
        else:
            x=com_x
        x=torch.sigmoid(x)
        x=(x>0.5).float()
        rec_x=self.rec(x)
        return rec_x
