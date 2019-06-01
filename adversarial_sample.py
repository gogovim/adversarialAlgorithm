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
from model.HGD import get_denoise,denoise_loss

from utils import AverageMeter,Dataset_filename,AdversarialDataset,mean,stdv,get_model

from scipy.misc import imsave


os.environ["CUDA_VISIBLE_DEVICES"]="0"
#device=torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True
#torch.backends.cudnn.benchmark = False




def convert(adv_inputs,mean,stdv,inputs=None):

    #print("bufore D:",torch.pow(torch.abs(adv_inputs - inputs), 1).mean())

    adv_inputs=adv_inputs.transpose(1,2).transpose(2,3)
    adv_inputs=(adv_inputs*stdv+mean)
    if inputs is not None:
        inputs=inputs.transpose(1,2).transpose(2,3)
        inputs=(inputs*stdv+mean)
        #print("difference:",torch.pow(torch.abs(adv_inputs - inputs), 1).mean())
    adv_inputs=torch.clamp(adv_inputs,0,1)
    return adv_inputs.cpu().numpy()
def generate_adversial_examles(model,data_dir,attack_mode,save_dir,true_label=True,pho_size=299,batch_size=16):
    batch_time = AverageMeter()
    end = time.time()

    batch_size=16
    data = Dataset_filename(data_dir,w=pho_size,h=pho_size,mean=mean,stdv=stdv,need_filename=True)
    dataLoader = DataLoader(data, batch_size=batch_size, shuffle=True,num_workers=4)
    custdv,cumean=torch.tensor(stdv).cuda(),torch.tensor(mean).cuda()
    for i,(inputs,labels,filenames) in enumerate(dataLoader):
        #print("gendrate adversarial samples [{}/{}]".format(i,len(dataLoader)))
        #print(filenames,inputs.shape,inputs.dtype,type(labels),type(filenames))
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels=labels.cuda()
        output = labels if true_label else model(inputs).max(1)[1]
        adv_input=FGSM(inputs,output,model,torch.nn.functional.cross_entropy,eps=0.03)
        adv_input=adv_input.data.cuda()

        adv_input=convert(adv_input,cumean,custdv)
        for idx,filename in enumerate(filenames):
            #print("write",args.output_dir+'/'+filename)
            save_path=os.path.join(save_dir,str(labels[idx].item()))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            imsave(os.path.join(save_path,filename),adv_input[idx])
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 40==0:
            res = '\t'.join([
                'generate',
                'Iter: [%d/%d]' % (i + 1, len(dataLoader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
            ])
            print(res)
def cal_distance(raw_images,adversarial_images):
    #res=0.
    pass
def valid_epoch(model,valid_dataloader,cost,print_freq=40):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    end = time.time()
    model.eval()
    for i,(images,labels) in enumerate(valid_dataloader):
        images,labels=images.cuda(),labels.cuda()
        outputs=model(images)

        loss=cost(outputs,labels)

        batch_size=labels.size(0)
        outputs=outputs.max(1)[1]
        #print(outputs,labels)
        error.update(torch.ne(outputs.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            res = '\t'.join([
                'Valid',
                'Iter: [%d/%d]' % (i + 1, len(valid_dataloader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)
    return batch_time.avg, losses.avg, error.avg
def evaluate_adversial_examles(model,data_dir):
    pho_size = 224
    batch_size=8
    valid_transforms = transforms.Compose([
        transforms.Resize((pho_size,pho_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    valid_data = datasets.ImageFolder(data_dir,transform=valid_transforms)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True,num_workers=4)
    valid_epoch(model,valid_loader,torch.nn.functional.cross_entropy)
def main(model_name,attack_mode,defense_model=[],pho_size=299):
    data_dir="/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_test"
    save_dir=data_dir+"_"+model_name+"_"+attack_mode+"_"+str(pho_size)
    mean = [0.65, 0.60, 0.59]
    stdv = [0.32, 0.33, 0.33]
    
    model=get_model(model_name).cuda()
    model=torch.nn.DataParallel(model)
    #model=model.cuda()
    print(model)
    model_name=model_name+"_"+"clean"+"_"+str(pho_size)
    model.load_state_dict(torch.load(os.path.join("./weights", model_name+'_model.dat')))
    generate_adversial_examles(model,data_dir,attack_mode,save_dir=save_dir,pho_size=pho_size)
    del model
    """
    for model_name in defense_model:
        model=get_model(model_name).cuda()
        model=torch.nn.DataParallel(model)
        #model=model.cuda()
        print(model)
        model.load_state_dict(torch.load(os.path.join("./weights", model_name+'_model.dat')))
    
        evaluate_adversial_examles(model,save_dir)
        evaluate_adversial_examles(model,data_dir)
    """

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='toynet template')
    # parser.add_argument('--epoch', type=int, default=20, help='epoch size')
    # parser.add_argument('--batch_size', type=int, default=100, help='mini-batch size')
    # parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    # parser.add_argument('--y_dim', type=int, default=10, help='the number of classes')
    # parser.add_argument('--target', type=int, default=-1, help='target class for targeted generation')
    # parser.add_argument('--eps', type=float, default=1e-9, help='epsilon')
    parser.add_argument('--input_dir', type=str, default='input', help='input directory path')
    # parser.add_argument('--output_dir', type=str, default='datasets', help='dataset directory path')
    # parser.add_argument('--summary_dir', type=str, default='summary', help='summary directory path')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory path')
    #parser.add_argument('--output_file', type=str, default='res.csv', help='output directory path')
    # parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='checkpoint directory path')
    # parser.add_argument('--load_ckpt', type=str, default='', help='')
    # parser.add_argument('--mode', type=str, default='train', help='generate / defense /train')
    # parser.add_argument('--seed', type=int, default=1, help='random seed')
    # parser.add_argument('--iteration', type=int, default=1, help='the number of iteration for FGSM')
    parser.add_argument('--epsilon', type=float, default=0.03, help='epsilon for FGSM and i-FGSM')
    # parser.add_argument('--alpha', type=float, default=2/255, help='alpha for i-FGSM')
    args = parser.parse_args()
    #print(args.input_dir,args.output_dir)
    main(model_name="InceptionResNetV2",attack_mode="fgsm",defense_model=["vgg16"],pho_size=299)
    #main(model_name="vgg16",attack_mode="fgsm",defense_model=["resnet101"])
