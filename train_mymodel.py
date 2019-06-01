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
import cv2

from model.vgg import VGG
from model.resnet import resnet101
from model.densenet import DenseNet
from model.inceptionresnetv2 import InceptionResNetV2
from model.inceptionv4 import InceptionV4
from model.inception import Inception3
from model import adversary
from model.HGD import get_denoise,denoise_loss,HGD
from model.randomResizePadding import RandomResizePadding
from model.comDefend import ComDefend
from model.mymodel import Mymodel

import train_HGD

from utils import AverageMeter,Dataset_filename,AdversarialDataset,Dataset_noise,mean,stdv,get_model,where,cal_distance,train_adversarial_dirs,valid_adversarial_dirs

os.environ["CUDA_VISIBLE_DEVICES"]="0"
def cal_loss1(x):
    v=x*x
    v=v.view(v.shape[0],-1)

    return v.mean(1)
def cal_loss2(x,y):
    return cal_loss1(x-y)
def convert(adv_inputs,mean,stdv,inputs=None):

    #print("bufore D:",torch.pow(torch.abs(adv_inputs - inputs), 1).mean())

    adv_inputs=adv_inputs.transpose(1,2).transpose(2,3)
    adv_inputs=(adv_inputs*stdv+mean)
    adv_inputs=torch.clamp(adv_inputs,0,1)
    if inputs is not None:
        inputs=inputs.transpose(1,2).transpose(2,3)
        inputs=(inputs*stdv+mean)
        inputs=torch.clamp(inputs,0,1)
        #print(adv_inputs-inputs)
        return adv_inputs.cpu().numpy(),inputs.cpu().numpy()
    
    return adv_inputs.cpu().numpy()
def valid_epoch(model,valid_dataloader, lam,print_freq=40,classify_model=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error1= AverageMeter()
    error2= AverageMeter()
    end = time.time()
    model.eval()
    cost=nn.CrossEntropyLoss()
    classify_model.eval()
    custdv,cumean=torch.tensor(stdv).cuda(),torch.tensor(mean).cuda()
    for i,(images,labels,noise) in enumerate(valid_dataloader):
        #print(type(images),type(labels))
        #print(images.shape,labels.shape)
        #if i>30:
        #    return batch_time.avg, losses.avg,error2.avg
        images,labels,noise=images.cuda(),labels.cuda(),noise.cuda().float()*args.phi
        #print(images.shape)
        #print(images.shape,labels.shape)
        #optimizer.zero_grad()
        #outputs=model(images,noise=noise)

        outputs=model(images,noise=noise)
        with torch.no_grad():
            y1=classify_model(images).detach()
            y2=classify_model(outputs)
        #loss=torch.abs(y1-y2).mean()
        l2,perception_loss=cal_loss2(images,outputs).mean(),cost(y2,labels).mean()
        if i%print_freq==0:
            print('l2={},perception_loss={}'.format(l2,perception_loss))
        loss=l2+perception_loss

        batch_size=labels.size(0)
        losses.update(loss.item(), batch_size)

        y1=y1.max(1)[1]
        y2=y2.max(1)[1]


        with torch.no_grad():
            y1=classify_model(images).detach()
            y2=classify_model(outputs)
        #loss=torch.abs(y1-y2).mean()
        loss=cal_loss2(images,outputs).mean()
        #loss.backward()
        #optimizer.step()

        batch_size=labels.size(0)
        losses.update(loss.item(), batch_size)

        y1=y1.max(1)[1]
        y2=y2.max(1)[1]
        #if i%print_freq==0:
        #    d_clean_y=d_clean_y.max(1)[1]
        #    adv_y=adv_y.max(1)[1]
        error1.update(torch.ne(y1.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        error2.update(torch.ne(y2.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)

        batch_time.update(time.time() - end)
        end = time.time()


        if i % print_freq == 0:

            ims,ims1=convert(outputs,cumean,custdv,images)
            for im in ims1:
                im=im*255.
                im=np.rint(im).astype(np.uint8)
                #print(im.shape)
                cv2.imwrite(str(i//print_freq%print_freq)+"valid.jpg",im)
                #cv2.waitKey(1)
            for im in ims:
                im=im*255.
                im=np.rint(im).astype(np.uint8)
                #print(im.shape)
                cv2.imwrite(str(i//print_freq%print_freq)+"validrec.jpg",im)
            res = '\t'.join([
                'valid:',
                'Iter: [%d/%d]' % (i + 1, len(valid_dataloader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error2.val, error2.avg),
                'raw Error %.4f (%.4f)' % (error1.val, error1.avg)
            ])
            print(res)
    return batch_time.avg, losses.avg,error2.avg
def train_epoch(model,train_dataloader,optimizer, lam,epoch, n_epochs,print_freq=40,classify_model=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error1= AverageMeter()
    error2= AverageMeter()
    end = time.time()
    model.train()
    classify_model.eval()
    cost=nn.CrossEntropyLoss()
    custdv,cumean=torch.tensor(stdv).cuda(),torch.tensor(mean).cuda()
    for i,(images,labels,noise) in enumerate(train_dataloader):
        #print(type(images),type(labels))
        #print(images.shape,labels.shape)
        #if i>30:
        #    return batch_time.avg, losses.avg,error2.avg
        images,labels,noise=images.cuda(),labels.cuda(),noise.cuda().float()*args.phi
        #print(images.shape)
        #print(images.shape,labels.shape)
        optimizer.zero_grad()
        outputs=model(images,noise=noise)
        with torch.no_grad():
            y1=classify_model(images).detach()
        y2=classify_model(outputs)
        #loss=torch.abs(y1-y2).mean()
        l2,perception_loss=cal_loss2(images,outputs).mean(),cost(y2,labels).mean()
        if i%print_freq==0:
            print('l2={},perception_loss={}'.format(l2,perception_loss))
        loss=l2+perception_loss
        loss.backward()
        optimizer.step()

        batch_size=labels.size(0)
        losses.update(loss.item(), batch_size)

        y1=y1.max(1)[1]
        y2=y2.max(1)[1]
        #if i%print_freq==0:
        #    d_clean_y=d_clean_y.max(1)[1]
        #    adv_y=adv_y.max(1)[1]
        error1.update(torch.ne(y1.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        error2.update(torch.ne(y2.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)

        batch_time.update(time.time() - end)
        end = time.time()


        if i % print_freq == 0:

            ims,ims1=convert(outputs.detach(),cumean,custdv,images)
            for im in ims1:
                im=im*255.
                im=np.rint(im).astype(np.uint8)
                #print(im.shape)
                cv2.imwrite(str(i//print_freq%print_freq)+".jpg",im)
                #cv2.waitKey(1)
            for im in ims:
                im=im*255.
                im=np.rint(im).astype(np.uint8)
                #print(im.shape)
                cv2.imwrite(str(i//print_freq%print_freq)+"rec.jpg",im)
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (i + 1, len(train_dataloader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error2.val, error2.avg),
                'raw Error %.4f (%.4f)' % (error1.val, error1.avg)
            ])
            print(res)
    return batch_time.avg, losses.avg,error2.avg
def train(model,train_dataloader,lr,save_dir,num_epoches,model_name,valid_dataloader=None,classify_model=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    end = time.time()
    best_error=1.
    nobetter_num=1
    for epoch in range(num_epoches):
        if nobetter_num >=5:
            print("train done .lr={},best_loss={}".format(lr,best_error))
            break
        if nobetter_num >=3:
            lr=lr*0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        _, train_loss,train_error = train_epoch(
            model,
            train_dataloader,
            optimizer,
            0.0001,
            epoch,
            num_epoches,
            classify_model=classify_model
        )
        if valid_dataloader:
            with torch.no_grad():
                _, valid_loss,valid_error = valid_epoch(
                    model,
                    valid_dataloader,
                    0.0001,
                    classify_model=classify_model
                )
        if valid_dataloader and valid_error< best_error:
            best_error = valid_error
            if valid_error+0.0005 < best_error:
                nobetter_num+=1
            else:
                nobetter_num=1
            print('New best error: %.4f' % best_error)
            torch.save(model.state_dict(), os.path.join(save_dir, model_name+'_model.dat'))
        else:
            #torch.save(model.state_dict(), os.path.join(save_dir, 'vgg16_model.dat'))
            nobetter_num+=1

        with open(os.path.join(save_dir, model_name+'_results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.6f,%.06f\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error
            ))
def main(args):
    model=Mymodel(pho_size=args.pho_size)
    print(model)
    #model=torch.nn.DataParallel(model).cuda()
    print(args.pre_train)
    #model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(args.pre_train)).items()})
    if args.pre_train:
        try:
            model.load_state_dict(torch.load(os.path.join(args.pre_train)))
            print("ok")
        except:
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(args.pre_train)).items()})
            print("ok")

    model=torch.nn.DataParallel(model)
    model=model.cuda()
    print(model)
    model_name=args.model_name+"_"+str(args.pho_size)

    classify_model=get_model("resnet101",pho_size=299,num_classes=110)
    try:
        classify_model.load_state_dict(torch.load(os.path.join("./weights/resnet101_clean_299_model.dat")))
    except:
        classify_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join("./weights/resnet101_clean_299_model.dat")).items()})
    classify_model=torch.nn.DataParallel(classify_model).cuda()
    print(classify_model)
    if args.mode == "train":
        train_transforms = transforms.Compose([
            transforms.Resize((args.pho_size ,args.pho_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        valid_transforms = transforms.Compose([
            transforms.Resize((args.pho_size,args.pho_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        train_data = Dataset_noise(train_dir,transform=train_transforms,shape=[512,19,19])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,num_workers=4)

        #交叉验证
        valid_data = Dataset_noise(valid_dir,transform=valid_transforms,shape=[512,19,19])
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True,num_workers=4)
        train(model,train_loader,args.lr,args.save_dir,args.epoch,model_name,valid_loader,classify_model)


if __name__=="__main__":

    train_dir='/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_train'
    valid_dir='/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_test'
    batch_size=8
    #./weights/Mymodel_299_model.dat
    parser = argparse.ArgumentParser(description='Mymodel')

    parser.add_argument('--mode', type=str, default='train', help='train | valid | test')

    #photo_size
    parser.add_argument('--pho_size',type=int,default=299,help='photo size')
    parser.add_argument('--phi', type=float, default=1.0, help='phi')

    #load model
    parser.add_argument('--model_name', type=str, default='Mymodel', help='model name')
    parser.add_argument('--pre_train', type=str, default='./weights/Mymodel_299_model.dat', help='weights')
    
    #train lr
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')


    #train eopch
    parser.add_argument('--epoch', type=int, default=300, help='epoch size')
    #train clean image
    parser.add_argument('--train_dir', type=str, default=train_dir, help='train_dir directory path')
    parser.add_argument('--valid_dir', type=str, default=valid_dir, help='valid_dir directory path')
    parser.add_argument('--test_dir', type=str, default='', help='test_dir directory path')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='mini-batch size')

    #the dir of weights/generate images
    parser.add_argument('--save_dir', type=str, default='./weights', help='output directory path')

    args=parser.parse_args()
    main(args)
