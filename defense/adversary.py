"""adversary.py"""
from pathlib import Path

import torch
from torch.autograd import Variable
from utils import where


def i_fgsm(x, y, net, criterion,targeted=False, eps=0.03, alpha=1, iteration=1, x_val_min=-1, x_val_max=1):
    x_adv = Variable(x.data, requires_grad=True)
    for i in range(iteration):
        h_adv = net(x_adv)
        if targeted:
            cost = criterion(h_adv, y)
        else:
            cost = -criterion(h_adv, y)

        net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - alpha * x_adv.grad
        x_adv = where(x_adv > x + eps, x + eps, x_adv)
        x_adv = where(x_adv < x - eps, x - eps, x_adv)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)

    h_adv = net(x_adv)

    return x_adv, h_adv


def fgsm(x, y, net, criterion,targeted=False, eps=0.03, x_val_min=-1, x_val_max=1):
    x_adv = Variable(x.data, requires_grad=True)
    h_adv = net(x_adv)
    #print(x.size(),y.size())
    if targeted:
        cost = criterion(h_adv, y)
    else:
        cost = -criterion(h_adv, y)

    net.zero_grad()
    if x_adv.grad is not None:
        x_adv.grad.data.fill_(0)
    cost.backward()

    x_adv.grad.sign_()
    #print("change eps={},real change1={},change2={}".format(eps,eps*0.3*255,-eps*0.3*255))
    x_adv = x_adv - eps * x_adv.grad

    return x_adv

