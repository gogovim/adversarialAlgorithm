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

from model.vgg import VGG
from model.resnet import resnet101
from model.densenet import DenseNet
from model.inceptionresnetv2 import InceptionResNetV2
from model.inceptionv4 import InceptionV4
from model.inception import Inception3
from model import adversary
from model.HGD import get_denoise,denoise_loss,HGD
from model.randomResizePadding import RandomResizePadding
from model.resnet_noise import resnet101_noise

import train_HGD

from utils import AverageMeter,Dataset_filename,AdversarialDataset,mean,stdv,get_model,where,cal_distance,train_adversarial_dirs,valid_adversarial_dirs,Dataset_noise

#from scipy.misc import imsave

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#device=torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True
#torch.backends.cudnn.benchmark = False
def valid_epoch(model,valid_dataloader,cost,print_freq=40,batch_num=None,random_layer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    end = time.time()
    model.eval()
    for i,(images,labels,noises) in enumerate(valid_dataloader):
        images,labels,noises=images.cuda(),labels.cuda(),noises.cuda()
        noises=noises*args.phi
        if random_layer:
            images=random_layer(images)
        outputs=model(images,noises)
        loss=cost(outputs,labels)

        batch_size=labels.size(0)
        outputs=outputs.max(1)[1]
        error.update(torch.ne(outputs.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            res = '\t'.join([
                'Valid',
                'Iter: [%d/%d]' % (i + 1, batch_num),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)
    return batch_time.avg, losses.avg, error.avg

def train_epoch(model,train_dataloader,optimizer, cost,epoch, n_epochs, print_freq=40,batch_num=None,random_layer=None):
    print(random_layer)
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    end = time.time()
    model.train()
    for i,(images,labels,noises) in enumerate(train_dataloader):
        #if i>30:
        #    return batch_time.avg, losses.avg, error.avg
        #print(type(images),type(labels))
        #print(images.shape,labels.shape)
        images,labels,noises=images.cuda(),labels.cuda(),noises.cuda()
        noises=noises*args.phi
        if random_layer:
            images=random_layer(images)
        #print(images.shape)
        #print(images.shape,labels.shape)
        optimizer.zero_grad()
        outputs=model(images,noises)
        #print("outpits.size={},labels.size={}",outputs.size(),labels.size())
        loss=cost(outputs,labels)
        loss.backward()
        #optimizer.step()

        batch_size=labels.size(0)
        outputs=outputs.max(1)[1]
        error.update(torch.ne(outputs.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()


        if i % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (i + 1, batch_num),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)
    return batch_time.avg, losses.avg, error.avg

def train(model,train_dataloader,lr=0.01,save_dir='./weights',num_epoches=200,model_name="",valid_dataloader=None,batch_num=None,train_type="clean",random_layer=None):
    if random_layer:
        model_name=model_name+"_randomLayer"
    print(train_dataloader,valid_dataloader)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    cost=nn.CrossEntropyLoss()

    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    end = time.time()
    best_error=1.0
    nobetter_num=1
    for epoch in range(num_epoches):
        if nobetter_num >=5:
            print("train done .lr={},best_error={}".format(lr,best_error))
            break
        if nobetter_num >=3:
            lr=lr*0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        _, train_loss, train_error = train_epoch(
            model,
            train_dataloader if train_type=="clean" else combind_loader(*train_dataloader),
            optimizer,
            cost,
            epoch,
            num_epoches,
            batch_num=batch_num,
            random_layer=random_layer
        )
        if valid_dataloader:
            with torch.no_grad():
                _, valid_loss, valid_error = valid_epoch(
                    model,
                    valid_dataloader if train_type=="clean" else combind_loader(*valid_dataloader),
                    cost,
                    batch_num=len(valid_dataloader)if train_type=="clean" else len(valid_dataloader[0]),
                    random_layer=random_layer
                )
        if valid_dataloader and valid_error< best_error:
            best_error = valid_error
            if valid_error+0.005 < best_error:
                nobetter_num+=1
            else:
                nobetter_num=1
            print('New best error: %.4f' % best_error)
            torch.save(model.state_dict(), os.path.join(save_dir, model_name+'_model.dat'))
        else:
            #torch.save(model.state_dict(), os.path.join(save_dir, 'vgg16_model.dat'))
            nobetter_num+=1

        with open(os.path.join(save_dir, model_name+'_results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
            ))

def main(args):
    model=get_model(args.model_name,num_classes=110,pho_size=args.pho_size)
    #model=torch.nn.DataParallel(model).cuda()

    if args.pre_train:
        try:
            model.load_state_dict(torch.load(os.path.join(args.pre_train)))
        except:
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(args.pre_train)).items()})

    model=torch.nn.DataParallel(model)
    model=model.cuda()
    model_name=args.model_name+"_"+args.train_type+"_"+str(args.pho_size)


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
        train_data = Dataset_noise(args.train_dir,transform=train_transforms,shape=[2048])
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,num_workers=4)

        #
        valid_data = Dataset_noise(args.valid_dir,transform=valid_transforms,shape=[2048])
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True,num_workers=4)

        batch_num=len(train_loader)
        model_name+=str(args.phi)
        train(model,train_loader,args.lr,args.save_dir,args.epoch,model_name,valid_loader,batch_num,args.train_type)



if __name__=="__main__":

    #train_dir='/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_train_densenet_clean_299_model_dat_fgsm_0_03'
    train_dir='/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_train'
    valid_dir='/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_test'
    batch_size=2
    dd='./weights/resnet101_noise_clean_299_model.dat'
    #./weights/Mymodel_299_model.dat
    parser = argparse.ArgumentParser(description='Mymodel')

    parser.add_argument('--mode', type=str, default='train', help='train | valid | test')
    parser.add_argument('--train_type', type=str, default='clean', help='clean | adversarial')

    #photo_size
    parser.add_argument('--pho_size',type=int,default=299,help='photo size')

    #load model
    parser.add_argument('--model_name', type=str, default='resnet101_noise', help='model name')
    parser.add_argument('--pre_train', type=str, default='', help='weights')
    
    #train lr
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--phi', type=float, default=1.0, help='phi')


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


