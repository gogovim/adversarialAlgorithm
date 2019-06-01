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

import train_HGD

import cv2

from utils import AverageMeter,Dataset_filename,AdversarialDataset,mean,stdv,get_model,where,cal_distance,train_adversarial_dirs,valid_adversarial_dirs

#from scipy.misc import imsave

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#device=torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True
#torch.backends.cudnn.benchmark = False
def FGSM(x, y_true, net,criterion,y_target=None, eps=0.03, x_min=[0,0,0],x_max=[1,1,1]):
    #print(eps)
    if y_target is not None:
        x_adv = adversary.fgsm(x, y_target, net,criterion,True, eps)
    else:
        x_adv = adversary.fgsm(x, y_true, net, criterion, False, eps)
    #print('----------------------')
    #print(x_min,x_max,x_adv.shape,x_adv[:][0][:][:].shape,x_adv[:][1][:][:].shape)
    x_adv[:,0,:,:]=torch.clamp(x_adv[:,0,:,:],x_min[0],x_max[0])
    x_adv[:,1,:,:]=torch.clamp(x_adv[:,1,:,:],x_min[1],x_max[1])
    x_adv[:,2,:,:]=torch.clamp(x_adv[:,2,:,:],x_min[2],x_max[2])
    #print(x_adv[0][2])
    #print(x_adv[0][0].shape)


    return x_adv.detach()
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

def cal_distance(x,y):
    diff = np.abs(x.reshape((-1, 3)) - y.reshape((-1, 3)))
    #print(diff.shape)
    #print((diff>0).sum(),299*299*3)
    #print(np.sum((diff ** 2), axis=1))

    return np.mean(np.sqrt(np.sum((diff ** 2), axis=1)))
def test(model,test_loader,cost,print_freq=40,batch_num=None,denoise=None,random_layer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    raw_error=AverageMeter()
    cat_error={}
    raw_cat_error={}
    meanD=AverageMeter()

    end = time.time()
    model.eval()
    if denoise:
        denoise.eval()
    if random_layer:
        random_layer.eval()
    custdv,cumean=torch.tensor(stdv).cuda(),torch.tensor(mean).cuda()
    """
    print(mean,stdv)
    tran = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
        ])
    
    data = datasets.ImageFolder("/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_test",transform=tran)

    test_loader1=DataLoader(data, batch_size=args.batch_size, shuffle=True,num_workers=4)
    """
    nx=np.array([[0.,0.,0.],[1.,1.,1.]])
    x_min,x_max=(nx-mean)/stdv
    epss=[40/255]
    for i,(clean_images,labels) in enumerate(test_loader):
        clean_images,labels=clean_images.cuda(),labels.cuda()
        adversarial_images=FGSM(clean_images,labels,model,torch.nn.functional.cross_entropy,eps=epss[random.randint(0,len(epss)-1)],x_min=x_min,x_max=x_max)
        #tmp=adversarial_images  
        #print(adversarial_images)
        
        if i % print_freq == 0:

            ims1,clean_imgs=convert(adversarial_images.detach(),cumean,custdv,clean_images.cuda())
            for im,imc in zip(ims1,clean_imgs):
                im=np.rint(im*255.)
                imc=np.rint(imc*255.)

                #print(imc.shape)
                #print(imc[:5][:5][:5])
                #print(imc[0][0],im[0][0])
                #print(im.shape,imc.shape)
                dd=cal_distance(im,imc)
                #print("distance between clean adn  adversarial:",diffs.item(),dd)
                im=np.rint(im).astype(np.uint8)
                imc=np.rint(imc).astype(np.uint8)
                #print(im.shape)
                cv2.imwrite("./output"+"/"+str(i//print_freq%print_freq)+"ad.jpg",im)
                cv2.imwrite("./output"+"/"+str(i//print_freq%print_freq)+"test.jpg",imc)
        
        if denoise:
            d_adversarial_images=denoise(adversarial_images)
            d_clean_images=denoise(clean_images)

        #print(d_clean_images[:][:5][:5])
        
        if i % print_freq == 0:

            ims1,clean_imgs=convert(d_adversarial_images.detach(),cumean,custdv,clean_images.cuda())
            for im,imc in zip(ims1,clean_imgs):
                im=im*255.
                imc=imc*255.
                #print(imc.shape)
                #print(imc[:5][:5][:5])
                dd=cal_distance(im,imc)
                #print("distance after denoise:",diffs.item(),dd)
                im=np.rint(im).astype(np.uint8)
                imc=np.rint(imc).astype(np.uint8)
                #print(im.shape)
                cv2.imwrite("./output"+"/"+str(i//print_freq%print_freq)+"adrec.jpg",im)
                cv2.imwrite("./output"+"/"+str(i//print_freq%print_freq)+"testrec.jpg",imc)

            #ims1,clean_imgs=convert(d_adversarial_images.detach(),cumean,custdv,d_clean_images.cuda())
        
        if random_layer:
            d_adversarial_images=random_layer(d_adversarial_images)
        #print(diffs)
        #diffs=diffs.cu


        outputs=model(d_adversarial_images)
        outputs_clean=model(d_clean_images)
        loss=cost(outputs,labels)

        batch_size=labels.size(0)
        outputs=outputs.max(1)[1]
        outputs_clean=outputs_clean.max(1)[1]



        #print(batch_size,outputs.shape)
        #print(outputs,labels)
        for _ in range(batch_size):
            real_cat=labels[_].item()
            raw_cat=outputs_clean[_].item()
            ad_cat=outputs[_].item()
            if real_cat not in cat_error.keys():
                cat_error[real_cat]=AverageMeter()
            cat_error[real_cat].update(1.0 if real_cat==ad_cat else 0.0)
            if real_cat not in raw_cat_error.keys():
                raw_cat_error[real_cat]=AverageMeter()
            raw_cat_error[real_cat].update(1.0 if real_cat==raw_cat else 0.0) 
        error.update(torch.ne(outputs.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        raw_error.update(torch.ne(outputs_clean.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)
        #meanD.update(where((outputs==labels).cpu(),diffs.float(),0.).mean().item(),batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            res = '\t'.join([
                'test',
                'Iter: [%d/%d]' % (i + 1, batch_num),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
                'raw_error %.4f (%.4f)'%(raw_error.val,raw_error.avg),
                'meanD %.4f (%.4f)' %(meanD.val,meanD.avg)
            ])
            print(res)
    return batch_time.avg, losses.avg, error.avg,meanD.avg,raw_cat_error,cat_error



def main(args):
    denoise=None
    if args.denoise:
        denoise=get_model(args.denoise)
        try:
            denoise.load_state_dict(torch.load(os.path.join(args.denoise_weight)))
        except:
            denoise.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(args.denoise_weight)).items()})
        denoise=torch.nn.DataParallel(denoise).cuda()
        denoise.eval()
    random_layer=None
    if args.random_layer:
        random_layer=RandomResizePadding()
        random_layer=torch.nn.DataParallel(random_layer).cuda()
    model=get_model(args.model_name,num_classes=110,pho_size=args.pho_size)
    #model=torch.nn.DataParallel(model).cuda()

    if args.pre_train:
        try:
            model.load_state_dict(torch.load(os.path.join(args.pre_train)))
        except:
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(args.pre_train)).items()})

    model=torch.nn.DataParallel(model)
    print("over")
    print(model)
    print(denoise)
    print(random_layer)
    model=model.cuda()
    model_name=args.model_name+"_"+args.train_type+"_"+str(args.pho_size)

    if args.mode=="test":
        #self, clean_dirs, adversarial_dirs, pho_size,mean,stdv):
        trans=transforms.Compose([
            transforms.Resize((args.pho_size,args.pho_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        data=datasets.ImageFolder(args.input_dir,transform=trans)
        test_loader=DataLoader(data, batch_size=args.batch_size, shuffle=True,num_workers=4)
        batch_time, losses, error,meanD,raw_cat_error,cat_error=test(model,test_loader,nn.CrossEntropyLoss(),print_freq=40,batch_num=len(test_loader),
            denoise=denoise,random_layer=random_layer)
        for i in range(110):
            print(i,"raw error:",raw_cat_error[i].avg,"ad error:",cat_error[i].avg)
        with open(os.path.join(args.output), 'a') as f:
            f.write('test = %s\n'% (args.model_name+" "+args.pre_train+" "+args.denoise+" "+str(args.random_layer)) )
            f.write('%s,%0.6f,%0.6f,%0.6f,%.6f\n' % (
                args.input_dir,
                batch_time*len(test_loader),
                losses,
                error,
                meanD
            ))

if __name__=="__main__":
    """
    resnet101_clean_299_model.dat
    InceptionResNetV2_clean_299_model.dat
    Inception3_clean_299_model.dat
    resnet101_adversarial_model.dat
    resnet101_adversarial_224_model.dat
    Inception3_clean_299_randomLayer_model.dat
    InceptionResNetV2_clean_299_model.dat
    

    Mymodel_299_model.dat Mymodel
    Inception3_clean_299_HGD_model.dat denoise
    comdefend_299_model.dat  Comdefend
    """
    model_name="Inception3"
    pre_train="weights/Inception3_clean_299_model.dat"
    denoise="Rectifi"
    denoise_weight="weights/Rectifi_299_model.dat"
    input_dir="/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_test"
    output="test_model.csv"


    parser = argparse.ArgumentParser(description='train template')

    parser.add_argument('--mode', type=str, default='test', help='train | valid | generate | test')
    parser.add_argument('--train_type', type=str, default='clean', help='clean | adversarial')
    parser.add_argument('--random_layer', type=int, default=0, help='has random_layer')

    
    #photo_size
    parser.add_argument('--pho_size',type=int,default=299,help='photo size')

    #load model
    parser.add_argument('--model_name', type=str, default=model_name, help='model name')
    parser.add_argument('--pre_train', type=str, default=pre_train, help='weights')
    parser.add_argument('--denoise',type=str,default=denoise,help='denoise model')
    parser.add_argument('--denoise_weight',type=str,default=denoise_weight,help='denoise model')
    

    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')



    #valid,generate input_dir
    parser.add_argument('--input_dir', type=str, default=input_dir, help='input_dir directory path')

    #the dir of weights/generate images
    parser.add_argument('--output', type=str, default=output, help='output directory path')


    args=parser.parse_args()
    main(args)


