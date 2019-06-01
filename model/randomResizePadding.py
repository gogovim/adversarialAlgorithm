import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os

class RandomResizePadding(nn.Module):
    def __init__(self,input_size=299,output_size=331):
        super(RandomResizePadding, self).__init__()
        self.input_size=input_size
        self.output_size=output_size

    def forward(self, x,y=None):
        rnd=random.randint(self.input_size,self.output_size-1)
        x=F.interpolate(x,size=(rnd,rnd))
        #print(x.shape)
        l,t=random.randint(0,self.output_size-rnd),random.randint(0,
            self.output_size-rnd)
        r,b=self.output_size-rnd-l,self.output_size-rnd-t
        x=nn.ZeroPad2d((l,r,t,b))(x)

        if y is not None:
            y=F.interpolate(y,size=(rnd,rnd))
            y=nn.ZeroPad2d((l,r,t,b))(y)
            return x,y
        return x