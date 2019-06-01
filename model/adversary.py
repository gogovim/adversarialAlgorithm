"""adversary.py"""
from pathlib import Path

import torch
from torch.autograd import Variable

def step_ll(x,y,net, criterion, eps=0.03):
    net.eval()
    x_adv = Variable(x.data, requires_grad=True)
    h_adv = net(x_adv)
    #print(x.size(),y.size())
    cost = -criterion(h_adv, y)

    net.zero_grad()
    if x_adv.grad is not None:
        x_adv.grad.data.fill_(0)
    cost.backward()

    x_adv.grad.sign_()
    #print("change eps={},real change1={},change2={}".format(eps,eps*0.3*255,-eps*0.3*255))
    x_adv = x_adv - eps * x_adv.grad

    return x_adv

def fgsm(x, y, net, criterion,targeted=False, eps=0.03, x_val_min=-1, x_val_max=1):
    net.eval()

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

    
    #x_adv.grad=x_adv.grad.sign_()
    '''
    tmp=torch.abs(x_adv.grad.view(x_adv.shape[0],-1))
    u=tmp.sort()[0][:,int(tmp.shape[1]*0.95)]
    u=u.view(u.shape[0],-1)
    #print(tmp.shape,u.shape)
    tmp=tmp>u
    tmp=tmp.view(x_adv.shape).float()
    #print(tmp*x_adv)
    x_adv.grad=x_adv.grad*tmp
    #print(x_adv.shape)
    #print((torch.abs(x_adv.grad)>0).sum())
    '''
    std_c=x_adv.grad.view(x_adv.shape[0],3,-1).std(2).mean(1)
    std_c=std_c.view(std_c.shape[0],1,1,1)
    print(x_adv.shape,std_c.shape)
    x_adv.grad=x_adv.grad/std_c
    #x_adv.grad.sign_()
    #print("change eps={},real change1={},change2={}".format(eps,eps*0.3*255,-eps*0.3*255))

    x_adv = x_adv - eps * x_adv.grad

    return x_adv

