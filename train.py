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

from utils import AverageMeter,Dataset_filename,AdversarialDataset,mean,stdv,get_model,where,cal_distance,train_adversarial_dirs,valid_adversarial_dirs

#from scipy.misc import imsave

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#device=torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True
#torch.backends.cudnn.benchmark = False

def FGSM(x, y_true, net,criterion,y_target=None, eps=0.03, alpha=2/255, iteration=1):
    if y_target is not None:
        x_adv = adversary.fgsm(x, y_target, net,criterion,True, eps)
    else:
        x_adv = adversary.fgsm(x, y_true, net, criterion, False, eps)
    return x_adv
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
        return (adv_inputs*255.).cpu().numpy(),(inputs*255.).cpu().numpy()
    
    return (adv_inputs*255.).cpu().numpy()
def generate_adversial_examles(model,data_dir,adversarial_method,save_dir,batch_size=16,true_label=True,pho_size=299,eps=0.03,iteration=None):
    batch_time = AverageMeter()
    meanD = AverageMeter()
    end = time.time()
    data = Dataset_filename(data_dir,w=pho_size,h=pho_size,mean=mean,stdv=stdv,need_filename=True)
    dataLoader = DataLoader(data, batch_size=batch_size, shuffle=True,num_workers=4)
    custdv,cumean=torch.tensor(stdv).cuda(),torch.tensor(mean).cuda()
    for i,(inputs,labels,filenames) in enumerate(dataLoader):
        #print("gendrate adversarial samples [{}/{}]".format(i,len(dataLoader)))
        #print(filenames,inputs.shape,inputs.dtype,type(labels),type(filenames))
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels=labels.cuda()
        model.eval()
        output = labels if true_label else model(inputs).max(1)[1] if (adversarial_method=="fgsm" or adversarial_method=="i-fgsm") else model(inputs).min(1)[1]
        output=output.detach()
        if adversarial_method=="fgsm":
            adv_input=FGSM(inputs,output,model,torch.nn.functional.cross_entropy,eps=eps)
        elif adversarial_method=="step-ll":
            adv_input=adversary.step_ll(inputs,output,model,torch.nn.functional.cross_entropy,eps=eps)
        elif adversarial_method=="i-fgsm":
            nx=np.array([[0.,0.,0.],[1.,1.,1.]])
            x_min,x_max=(nx-mean)/stdv
            adv_input=adversary.i_fgsm(inputs, output, model, torch.nn.functional.cross_entropy,targeted=False, 
                eps=eps, alpha=1.0/255/0.3, iteration=iteration, x_val_min=x_min, x_val_max=x_max)

        adv_input=adv_input.data.cuda()

        if i%40==0:
            adv_input,inputs=convert(adv_input,cumean,custdv,inputs)
            #print(adv_input)
            adv_input=np.rint(adv_input).astype(np.uint8)
            inputs=np.rint(inputs).astype(np.uint8)
        else:
            adv_input=convert(adv_input,cumean,custdv)
            #print(adv_input)
            adv_input=np.rint(adv_input).astype(np.uint8)

        for idx,filename in enumerate(filenames):
            #print("write",args.output_dir+'/'+filename)
            save_path=os.path.join(save_dir,str(labels[idx].item()))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            im=Image.fromarray(adv_input[idx])
            im.save(os.path.join(save_path,filename))
            #print(cal_distance(adv_input[idx],inputs[idx]))
            if i%40==0:
                meanD.update(cal_distance(adv_input[idx].astype(np.float),inputs[idx].astype(np.float)))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % 40==0:
            res = '\t'.join([
                'generate',
                'Iter: [%d/%d]' % (i + 1, len(dataLoader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'meanD %.3f (%.3f)' % (meanD.val, meanD.avg),
            ])
            print(res)
def test(model,test_dataloader,cost,print_freq=40,batch_num=None,denoise=None,random_layer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    meanD=AverageMeter()

    end = time.time()
    model.eval()
    for i,(images,adversarial_images,labels,diffs) in enumerate(test_dataloader):
        adversarial_images,labels=adversarial_images.cuda(),labels.cuda()
        if denoise:
            adversarial_images=denoise(adversarial_images)
        if random_layer:
            adversarial_images=random_layer(adversarial_images)
        
        #diffs=diffs.cu
        outputs=model(adversarial_images)
        loss=cost(outputs,labels)

        batch_size=labels.size(0)
        outputs=outputs.max(1)[1]
        error.update(torch.ne(outputs.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)
        meanD.update(where((outputs==labels).cpu(),diffs.float(),0.).mean().item(),batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            res = '\t'.join([
                'test',
                'Iter: [%d/%d]' % (i + 1, batch_num),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
                'meanD %.4f (%.4f)' %(meanD.val,meanD.avg)
            ])
            print(res)
    return batch_time.avg, losses.avg, error.avg,meanD.avg
def valid_epoch(model,valid_dataloader,cost,print_freq=40,batch_num=None,random_layer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    end = time.time()
    model.eval()
    for i,(images,labels) in enumerate(valid_dataloader):
        images,labels=images.cuda(),labels.cuda()
        if random_layer:
            images=random_layer(images)
        outputs=model(images)
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
    for i,(images,labels) in enumerate(train_dataloader):
        #print(type(images),type(labels))
        #print(images.shape,labels.shape)
        images,labels=images.cuda(),labels.cuda()
        if random_layer:
            images=random_layer(images)
        #print(images.shape)
        #print(images.shape,labels.shape)
        optimizer.zero_grad()
        outputs=model(images)
        #print("outpits.size={},labels.size={}",outputs.size(),labels.size())
        loss=cost(outputs,labels)
        loss.backward()
        optimizer.step()

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
def combind_loader(clean_images_dataloader,adversial_samples_dataloader):
    if len(clean_images_dataloader)!=len(adversial_samples_dataloader):
        print("len(clean_images_dataloader)!=len(adversial_samples_dataloader)")
        sys.exit(1)
    for X1,X2 in zip(clean_images_dataloader,adversial_samples_dataloader):
        imgs1,label1=X1
        imgs2,label2=X2

        yield torch.cat([imgs1,imgs2],0),torch.cat([label1,label2],0)

def train(model,train_dataloader,lr=0.01,save_dir='./weights',num_epoches=200,model_name="",valid_dataloader=None,batch_num=None,train_type="clean",random_layer=None):
    print(lr)
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
    denoise=None
    if args.denoise:
        denoise=get_model("denoise")
        try:
            denoise.load_state_dict(torch.load(os.path.join(args.denoise)))
        except:
            denoise.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(args.denoise)).items()})
        denoise=torch.nn.DataParallel(denoise).cuda()
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
        train_data = datasets.ImageFolder(args.train_dir,transform=train_transforms)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,num_workers=8)

        #交叉验证
        valid_data = datasets.ImageFolder(args.valid_dir,transform=valid_transforms)
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True,num_workers=8)


        if args.train_type=="adversial":
            adversial_train_loader=DataLoader(datasets.ImageFolder(args.adversial_train_dir,transform=train_transforms), 
                batch_size=args.adversial_batch_size, shuffle=True,num_workers=4)
            adversial_valid_loader = DataLoader(datasets.ImageFolder(args.adversial_valid_dir,transform=valid_transforms),
                batch_size=args.adversial_batch_size, shuffle=True,num_workers=4)

        batch_num=len(train_loader)
        train(model,train_loader if args.train_type=="clean" else (train_loader,adversial_train_loader),args.lr,args.save_dir,
            args.epoch,model_name,valid_loader if args.train_type=="clean" else (valid_loader,adversial_valid_loader),batch_num,args.train_type,
            random_layer=random_layer)
    elif args.mode=='train_HGD':
        #train_adversarial_dirs=[args.train_dir]
        #valid_adversarial_dirs=[args.valid_dir]

        train_data = AdversarialDataset(args.train_dir,train_adversarial_dirs,args.pho_size,mean,stdv,need_diff=True,phi=0.5)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,num_workers=4)

        #交叉验证
        valid_data = AdversarialDataset(args.valid_dir,valid_adversarial_dirs,args.pho_size,mean,stdv,need_diff=True,phi=0.5)
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True,num_workers=4)

        if denoise is None:
            denoise=get_model("denoise")
            denoise=torch.nn.DataParallel(denoise).cuda()
        model_name=model_name+"_HGD"
        train_HGD.train(denoise,model,train_loader,model_name=model_name,valid_dataloader=valid_loader,batch_num=len(train_loader),random_layer=random_layer)
    elif args.mode=="valid":

        
        valid_transforms = transforms.Compose([
            transforms.Resize((args.pho_size,args.pho_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        #valid_data=MyDataset(args.input_dir, args.pho_size,mean,stdv)
        valid_data = datasets.ImageFolder(args.input_dir,transform=valid_transforms)
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True,num_workers=4)
        batch_time, losses, error=valid_epoch(model,valid_loader,nn.CrossEntropyLoss(),print_freq=40,batch_num=len(valid_loader),
            random_layer=random_layer)

        with open(os.path.join(args.save_dir, model_name+'_valid.csv'), 'a') as f:
            f.write('%s,%0.6f,%0.6f,%0.6f\n' % (
                args.input_dir,
                batch_time*len(valid_loader),
                losses,
                error
            ))
    elif args.mode=="test":
        #self, clean_dirs, adversarial_dirs, pho_size,mean,stdv):
        data=AdversarialDataset(args.test_dir,valid_adversarial_dirs,args.pho_size,mean,stdv,need_diff=True,phi=0,mode='train')
        test_loader=DataLoader(data, batch_size=args.batch_size, shuffle=True,num_workers=4)
        with torch.no_grad():
            batch_time, losses, error,meanD=test(model,test_loader,nn.CrossEntropyLoss(),print_freq=40,batch_num=len(test_loader),denoise=denoise,
                random_layer=random_layer)
        with open(os.path.join(args.save_dir, model_name+'_test.csv'), 'a') as f:
            f.write('denoise = %s\n'% (args.denoise if args.denoise else 'None') )
            f.write('%s,%0.6f,%0.6f,%0.6f,%.6f\n' % (
                args.adversarial_test_dir,
                batch_time*len(test_loader),
                losses,
                error,
                meanD
            ))
    elif args.mode=="generate":
        if args.save_dir:
            save_dir=args.save_dir
        else:
            save_dir=args.input_dir+"_"+model_name+"_"+args.adversarial_method+"_"+str(args.eps).replace('.','_')
        #generate_adversial_examles(model,data_dir,adversarial_method,save_dir,batch_size=16,true_label=True,pho_size=299,eps=0.03):
        generate_adversial_examles(model,args.input_dir,args.adversarial_method,save_dir=save_dir,batch_size=args.batch_size,
            true_label=False,pho_size=args.pho_size,eps=args.eps,iteration=args.iteration)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='train template')

    parser.add_argument('--mode', type=str, default='train', help='train | valid | generate | test')
    parser.add_argument('--train_type', type=str, default='clean', help='clean | adversarial')
    parser.add_argument('--random_layer', type=int, default=0, help='has random_layer')

    
    #photo_size
    parser.add_argument('--pho_size',type=int,default=299,help='photo size')

    #load model
    parser.add_argument('--model_name', type=str, default='', help='model name')
    parser.add_argument('--pre_train', type=str, default='', help='weights')
    parser.add_argument('--denoise',type=str,default='',help='denoise model')
    
    #train lr
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')


    #train eopch
    parser.add_argument('--epoch', type=int, default=300, help='epoch size')
    #train clean image
    parser.add_argument('--train_dir', type=str, default='', help='train_dir directory path')
    parser.add_argument('--valid_dir', type=str, default='', help='valid_dir directory path')
    parser.add_argument('--test_dir', type=str, default='', help='test_dir directory path')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')

    #train adversarial image
    parser.add_argument('--adversarial_train_dir', type=str, default='', help='adversial_train_dir directory path')
    parser.add_argument('--adversarial_valid_dir', type=str, default='', help='adversial_valid_dir directory path')
    parser.add_argument('--adversarial_test_dir', type=str, default='', help='adversial_test_dir directory path')
    parser.add_argument('--adversarial_batch_size', type=int, default=0, help='adversial_batch_size directory path')


    #valid,generate input_dir
    parser.add_argument('--input_dir', type=str, default='', help='input_dir directory path')
    parser.add_argument('--adversarial_method', type=str, default='fgsm', help='fgsm | i-fgsm | step-ll')
    parser.add_argument('--eps', type=float, default=0.03, help='the attack step size')

    #the dir of weights/generate images
    parser.add_argument('--save_dir', type=str, default='output', help='output directory path')

    #the iteration num of i-fgsm
    parser.add_argument('--iteration', type=int, default=1, help='the iteration num of i-fgsm')
    


    args=parser.parse_args()
    print(args.mode,args.pho_size,args.model_name,args.train_dir,args.random_layer)
    main(args)


