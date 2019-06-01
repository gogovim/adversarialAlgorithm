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
from PIL import ImageFile


from model.vgg import VGG
from model.resnet import resnet101
from model.densenet import DenseNet
from model.inceptionresnetv2 import InceptionResNetV2
from model.inceptionv4 import InceptionV4
from model.inception import Inception3
from model import adversary
from model.HGD import get_denoise,denoise_loss,HGD
from model.randomResizePadding import RandomResizePadding

from utils import AverageMeter,Dataset_filename,AdversarialDataset,mean,stdv,get_model,where,cal_distance

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#device=torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True
#torch.backends.cudnn.benchmark = False
def valid_epoch(model_denoise,model_classify,valid_dataloader,cost,print_freq=40,batch_num=None,random_layer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error1 = AverageMeter()
    error2 = AverageMeter()
    error3 = AverageMeter()
    error4 = AverageMeter()
    meanD=AverageMeter()

    end = time.time()
    model_denoise.eval()
    model_classify.eval()
    for i,(clean_images,adversarial_images,labels,diffs) in enumerate(valid_dataloader):
        #print(type(images),type(labels))
        #print(images.shape,labels.shape)
        #if i>10:
        #    return batch_time.avg, losses.avg, error1.avg,error2.avg,meanD.avg
        #print(clean_images.shape,adversarial_images.shape)
        clean_images,adversarial_images,labels=clean_images.cuda(),adversarial_images.cuda(),labels.cuda()
        #print(images.shape,labels.shape)

        #d_clean_images= model_denoise(clean_images)
        d_adversarial_images= model_denoise(adversarial_images)

        if random_layer:
            clean_images=random_layer(clean_images)
            d_adversarial_images=random_layer(d_adversarial_images)

        y1,f1=model_classify(clean_images,need_feature=True)
        y2,f2=model_classify(d_adversarial_images,need_feature=True)

        #clean_y=model_classify(clean_images)
        #adv_y=model_classify(adversarial_images)


        loss=cost(y1,y2).mean()

        batch_size=labels.size(0)
        y1=y1.max(1)[1]
        y2=y2.max(1)[1]
        #clean_y=clean_y.max(1)[1]
        #adv_y=adv_y.max(1)[1]
        error1.update(torch.ne(y1.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        error2.update(torch.ne(y2.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        meanD.update(where((y2==labels).cpu(),diffs.float(),0.).mean().item(),batch_size)
        #error3.update(torch.ne(clean_y.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        #error4.update(torch.ne(adv_y.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)

        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()


        if i % print_freq == 0:
            res = '\t'.join([
                'Valid:',
                'Iter: [%d/%d]' % (i + 1, batch_num),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error_clean %.4f (%.4f)' % (error1.val, error1.avg),
                'Error_adversaril %.4f (%.4f)' % (error2.val, error2.avg),
                'meanD %.4f (%.4f)' %(meanD.val,meanD.avg)
                #'Error_row %.4f/%.4f' % (error3.avg, error4.avg),
            ])
            print(res)
    return batch_time.avg, losses.avg, error1.avg,error2.avg,meanD.avg

def train_epoch(model_denoise,model_classify,train_dataloader,optimizer, cost,epoch, n_epochs, print_freq=40,batch_num=None,random_layer=None):

    batch_time = AverageMeter()
    losses = AverageMeter()
    error1 = AverageMeter()
    error2 = AverageMeter()
    error3 = AverageMeter()
    error4 = AverageMeter()
    meanD=AverageMeter()

    end = time.time()
    model_denoise.train()
    model_classify.eval()
    for i,(clean_images,adversarial_images,labels,diffs) in enumerate(train_dataloader):
        #print(i)
        #print(type(images),type(labels))
        #print(images.shape,labels.shape)
        #print(clean_images.shape,adversarial_images.shape)
        #if i>200:
        #    return batch_time.avg, losses.avg, error1.avg,error2.avg,meanD.avg
        #print(labels)
        clean_images,adversarial_images,labels=clean_images.cuda(),adversarial_images.cuda(),labels.cuda()
        #print(clean_images.shape,adversarial_images.shape)
        #print(images.shape,labels.shape)
        optimizer.zero_grad()
        #if i%print_freq==0:
        #    d_clean_images= model_denoise(clean_images)
        d_adversarial_images= model_denoise(adversarial_images)

        if random_layer:
            clean_images,d_adversarial_images=random_layer(clean_images,d_adversarial_images)

        #print(clean_images.shape,d_adversarial_images.shape)
        y1,f1=model_classify(clean_images,need_feature=True)
        y2,f2=model_classify(d_adversarial_images,need_feature=True)

        #if i%print_freq==0:
        #    #d_clean_images= model_denoise(clean_images)
        #    if random_layer:
        #        d_clean_y,adv_y=random_layer(d_clean_images,adversarial_images)
        #d_clean_y=model_classify(clean_images)
        #    adv_y=model_classify(adversarial_images)

        y1=y1.detach()
        f1=f1.detach()

        loss=cost(y1,y2).mean()
        loss.backward()
        optimizer.step()

        batch_size=labels.size(0)
        y1=y1.max(1)[1]
        y2=y2.max(1)[1]
        #if i%print_freq==0:
        #    d_clean_y=d_clean_y.max(1)[1]
        #    adv_y=adv_y.max(1)[1]
        error1.update(torch.ne(y1.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        error2.update(torch.ne(y2.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        meanD.update(where((y2==labels).cpu(),diffs.float(),0.).mean().item(),batch_size)
        #if i%print_freq==0:
        #    error3.update(torch.ne(d_clean_y.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        #    error4.update(torch.ne(adv_y.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)

        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()


        if i % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (i + 1, batch_num),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error_clean %.4f (%.4f)' % (error1.val, error1.avg),
                'Error_adversaril %.4f (%.4f)' % (error2.val, error2.avg),
                #'Error_row %.4f(clean with denoise)/%.4f(adversarial without denoise)' % (error3.avg, error4.avg),
                'meanD %.4f (%.4f)' %(meanD.val,meanD.avg)
            ])
            print(res)
    return batch_time.avg, losses.avg, error1.avg,error2.avg,meanD.avg

def train(model_denoise,model_classify,train_dataloader,model_name="",lr=0.01,save_dir='./weights',num_epoches=200,valid_dataloader=None,batch_num=None,random_layer=None):

    #optimizer = torch.optim.SGD(model_denoise.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    optimizer = torch.optim.Adam(model_denoise.parameters(),lr = lr, weight_decay = 0.0005)
    cost=denoise_loss

    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    end = time.time()
    best_error=1.0
    nobetter_num=1
    for epoch in range(num_epoches):
        if nobetter_num >=7:
            print("train done .lr={},best_error={}".format(lr,best_error))
            break
        if nobetter_num >=5:
            lr=lr*0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        _, train_loss, train_error1,train_error2,train_meanD = train_epoch(
            model_denoise,
            model_classify,
            train_dataloader,
            optimizer,
            cost,
            epoch,
            num_epoches,
            batch_num=batch_num,
            random_layer=random_layer
        )

        if valid_dataloader:
            with torch.no_grad():
                _, valid_loss, valid_error1,valid_error2,valid_meanD = valid_epoch(
                    model_denoise,
                    model_classify,
                    valid_dataloader,
                    cost,
                    batch_num=len(valid_dataloader),
                    random_layer=random_layer
                )
        if valid_dataloader and valid_error2< best_error:
            best_error = valid_error2
            if valid_error2+0.001 < best_error:
                nobetter_num+=1
            else:
                nobetter_num=1
            print('New best error: %.4f' % valid_error2)
            torch.save(model_denoise.state_dict(), os.path.join(save_dir, model_name+'_model.dat'))
        else:
            #torch.save(model.state_dict(), os.path.join(save_dir, 'vgg16_model.dat'))
            nobetter_num+=1

        with open(os.path.join(save_dir, model_name+'_results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,%0.5f,%.5f\n' % (
                (epoch + 1),
                train_loss,
                train_error2,
                train_meanD,
                valid_loss,
                valid_error2,
                valid_meanD,
            ))

def main(model_classify_name,pre_train=None,batch_size=16,pho_size=224):
    train_clean_dir="/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_train"
    train_adversarial_dir="/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_train_InceptionResNetV2_fgsm_299"
    valid_clean_dir="/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_test"
    valid_adversarial_dir="/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_test_InceptionResNetV2_fgsm_299"
    mean = [0.65, 0.60, 0.59]
    stdv = [0.32, 0.33, 0.33]

    save_dir="./weights"

    train_data = AdversarialDataset(train_clean_dir,train_adversarial_dir,pho_size,mean,stdv)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=4)

    #交叉验证
    valid_data = AdversarialDataset(valid_clean_dir,valid_adversarial_dir,pho_size,mean,stdv)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True,num_workers=4)


    model_denoise=get_denoise()
    model_classify=get_model(model_classify_name)
    model_denoise=torch.nn.DataParallel(model_denoise)
    model_classify=torch.nn.DataParallel(model_classify)
    """
    pretrain_dict = torch.load(os.path.join(dir_path,'inceptionv3_state.pth'))
    state_dict = model.net.state_dict()
    for key in pretrain_dict.keys():
        assert state_dict.has_key(key)
        value = pretrain_dict[key]
        if not isinstance(value, torch.FloatTensor):
            value = value.data
        state_dict[key] = value
    """
    model_denoise=model_denoise.cuda()
    model_classify=model_classify.cuda()

    model_classify.load_state_dict(torch.load(os.path.join(save_dir, model_classify_name+"_clean_299_model.dat")))
    print(model_denoise,model_classify)
    model_name="HGD_"+str(pho_size)
    train(model_denoise,model_classify,train_loader,model_name=model_name,valid_dataloader=valid_loader,batch_num=len(train_loader))

if __name__=="__main__":
	main(model_classify_name="Inception3",pre_train=False,batch_size=4,pho_size=299)


