import argparse, torch
import numpy as np
from torch.utils.data import Dataset
from torch import nn
from torch.autograd import Variable
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
from PIL import ImageFile,Image


from model.vgg import VGG
from model.resnet import resnet101,resnet152
from model.resnet_noise import resnet101_noise
from model.densenet import DenseNet
from model.inceptionresnetv2 import InceptionResNetV2
from model.inceptionv4 import InceptionV4
from model.inception import Inception3
from model.HGD import get_denoise,denoise_loss,HGD
from model.comDefend import ComDefend
from model.mymodel import Mymodel
from model.rectifi import Rectifi

ImageFile.LOAD_TRUNCATED_IMAGES = True

mean = [0.6460, 0.5981, 0.5895]
stdv = [0.3220, 0.3280, 0.3295]
model_weight_size=[
["vgg16","vgg16_clean_299_model.dat",299],

["densenet","densenet_clean_166_model.dat",166],
["densenet","densenet_clean_299_model.dat",299],

["resnet101","resnet101_clean_192_model.dat",192],
["resnet101","resnet101_clean_299_model.dat",299],

["InceptionResNetV2","InceptionResNetV2_clean_299_model.dat",299],

["Inception3","Inception3_clean_299_model.dat",299],

["InceptionV4","Inceptionv4_clean_299_model.dat",299]

]
adversarial_methods=["fgsm","step-ll","i-fgsm"]
train_adversarial_dirs=[]
valid_adversarial_dirs=[]
inputs=[
"/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_test",
"/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_train"
]
for i in [6]:
    for j in inputs[:1]:
        for k in adversarial_methods[0:1]:
            for e in [0.03]:
                valid_adversarial_dirs.append(j+"_"+model_weight_size[i][1].replace('.','_')+"_"+k+"_"+str(e).replace('.','_'))
for i in [2,4,5,6,7]:
    for j in inputs[1:]:
        for k in adversarial_methods[:1]:
            for e in [0.03,0.36,0.48]:
                train_adversarial_dirs.append(j+"_"+model_weight_size[i][1].replace('.','_')+"_"+k+"_"+str(e).replace('.','_'))
def get_model(model_name,pho_size=299,num_classes=110):
    if model_name=="vgg16":
        model=VGG(num_classes=num_classes,pho_size=299)
    elif model_name=="resnet101":
        model=resnet101(num_classes=num_classes)
    elif model_name=="resnet152":
        model=resnet152(num_classes=num_classes)
    elif model_name=="densenet":
        model=DenseNet(
        growth_rate=12,
        block_config=[(100 - 4) // 6 for _ in range(3)],
        num_classes=num_classes,
        small_inputs=False,
        efficient=True,
        pho_size=pho_size
    )
    elif model_name=="InceptionResNetV2":
        model=InceptionResNetV2(num_classes=num_classes)
    elif model_name=="InceptionV4":
        model=InceptionV4(num_classes=num_classes)
    elif model_name=="Inception3":
        model=Inception3(num_classes=num_classes)
    elif model_name=="denoise":
        model=get_denoise()
    elif model_name=="Mymodel":
        model=Mymodel()
    elif model_name=='Comdefend':
        model=ComDefend()
    elif model_name=='Rectifi':
        model=Rectifi()
    return model
class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class Dataset_noise(Dataset):
    def __init__(self, root_dir,transform,shape):
        self.root_dir = root_dir
        self.data=datasets.ImageFolder(root_dir,transform=transform)
        self.shape=shape
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        noise=torch.randn(self.shape)
        #print('--------')
        x,y=self.data[idx]
        #x=x//16*16
        return x,y,noise
class Dataset_filename(Dataset):
    """
    return image,filename
    """
    def __init__(self, root_dir,w=None,h=None,mean=0.5,stdv=1,need_filename=False):
        self.root_dir = root_dir
        self.filenames=[]
        self.labels=[]
        self.imagespath=[]
        self.mean=mean
        self.stdv=stdv
        self.w,self.h=w,h
        self.need_filename=need_filename

        for filepath in os.listdir(root_dir):
            for image in os.listdir(os.path.join(root_dir,filepath)):
                self.filenames.append(image)
                self.labels.append(int(filepath))
                self.imagespath.append(os.path.join(root_dir,filepath,image))
 
    def __len__(self):
        return len(self.filenames)
 
    def __getitem__(self, idx):
        X = np.zeros((3,self.w,self.h),dtype='float32')
        try:
            #print(input_dir+'/'+filepath)
            image=Image.open(self.imagespath[idx])
            #print('read over')
            image=np.array(image.resize((self.w,self.h)),dtype='float32')
            image=(image / 255.0) 
            image=(image-self.mean)/self.stdv
            X[0, :, :] = image[:,:,0]
            X[1, :, :] = image[:,:,1]
            X[2, :, :] = image[:,:,2]
        except:
            pass
        if self.need_filename:
            return X,self.labels[idx],self.filenames[idx]
        return X,self.labels[idx]
class AdversarialDataset(Dataset):
    def __init__(self, clean_dirs, adversarial_dirs, pho_size,mean,stdv,need_diff=False,phi=0,mode="train",shape=None):
        self.need_diff=need_diff
        self.phi=phi
        self.files=[]
        self.clean_dirs=clean_dirs
        self.adversarial_dirs=adversarial_dirs
        self.pho_size=pho_size
        self.mean=mean
        self.stdv=stdv
        self.mode=mode

        self.shape=shape
        #标准化和随机水平翻转
        self.transform=transforms.Compose([transforms.Resize([pho_size,pho_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)])
        for filepaths in os.listdir(clean_dirs):
            #print(clean_dirs,filepaths)
            for filename in os.listdir(os.path.join(clean_dirs,filepaths)):
                #print(filepaths)
                self.files.append((os.path.join(filepaths,filename),filename,filepaths))
        #random.shuffle(self.filenames)

    def __getitem__(self, idx):
        #print(self.adversarial_dirs)
        if self.mode=='train':
            filepath,filename,label=self.files[idx]
        else:
            filepath,filename,label=self.files[idx//len(self.adversarial_dirs)]

        clean_image=Image.open(os.path.join(self.clean_dirs,filepath))
        try:
            if self.mode=='train':
                adversarial_path=os.path.join(self.adversarial_dirs[random.randint(0,len(self.adversarial_dirs)-1)],label,filename)
            else:
                adversarial_path=os.path.join(self.adversarial_dirs[idx%len(self.adversarial_dirs)],label,filename)
            adversarial_image=Image.open(adversarial_path)
        except:
            if self.mode=='train':
                adversarial_path=os.path.join(self.adversarial_dirs[random.randint(0,len(self.adversarial_dirs)-1)],str(int(label)),filename)
            else:
                adversarial_path=os.path.join(self.adversarial_dirs[idx%len(self.adversarial_dirs)],str(int(label)),filename)
            adversarial_image=Image.open(adversarial_path)
        
        

        if clean_image.mode!="RGB":
        	clean_image=clean_image.convert("RGB")
        if adversarial_image.mode!="RGB":
        	adversarial_image=adversarial_image.convert("RGB")

        if self.need_diff:
            t1=np.array(clean_image.resize((299, 299),Image.BILINEAR))
            #print("t1:",t1[0][0])
            t2=np.array(adversarial_image.resize((299, 299),Image.BILINEAR))
            diff=cal_distance(t1,t2)
            #print("t1:",t1[0][0],"t2:",t2[0][0],diff)
            #print(diff,np.abs(t1,t2).mean())
        #print("mode:",clean_image.mode,adversarial_image.mode)
        #print("shaoe:",clean_image.size,adversarial_image.size,clean_image.getbands(),adversarial_image.getbands())
        p=random.random()
        if self.phi>p:
            #print('RandomHorizontalFlip')
            trh=transforms.RandomHorizontalFlip(p=1.0)
            clean_image=trh(clean_image)
            adversarial_image=trh(adversarial_image)

        #print("-----\n",np.array(adversarial_image))
        clean_image = self.transform(clean_image)
        adversarial_image = self.transform(adversarial_image)

        #print(adversarial_image)
        if self.need_diff:
            #print(diff)
            if self.shape is not None:
                return clean_image,adversarial_image,int(label),diff,torch.randn(self.shape)
            return clean_image,adversarial_image,int(label),diff
        if self.shape is not None:
            return clean_image,adversarial_image,int(label),torch.randn(self.shape)
        return clean_image,adversarial_image,int(label)

    def __len__(self):
        if self.mode=='train':
            return len(self.files)
        if self.mode=='test':
            return len(self.files)*len(self.adversarial_dirs)
def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    #x=x.float()
    #y=y.float()
    #print("----------------------------",type(x),type(y))
    #print(x,y)
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)

def cal_distance(x,y):

    x=x.astype(np.float64)
    y=y.astype(np.float64)
    #print(x.reshape((-1, 3))[:5] ,y.reshape((-1, 3))[])
    #print(x.dtype,y.dtype)
    diff = np.abs(x.reshape((-1, 3)) - y.reshape((-1, 3)))
    #print(x[:2][:2][:2][:2],y[:2][:2][:2][:2])
    #print(diff.shape)
    #print((diff>0).sum(),299*299*3)
    #print(np.sum((diff ** 2), axis=1))
    return np.mean(np.sqrt(np.sum((diff ** 2), axis=1)))
if __name__=="__main__":
    clean_dirs="/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_train"
    adversarial_dirs="/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_train_vgg16_fgsm"
    mean = [0.65, 0.60, 0.59]
    stdv = [0.32, 0.33, 0.33]
    data=AdversarialDataset(clean_dirs,adversarial_dirs,224,mean,stdv)
    dataLoader = DataLoader(data, batch_size=32, shuffle=True,num_workers=4)
    for idx,(x1,x2,y) in enumerate(dataLoader):
    	if idx%4==0:
    	    print("[{}/{}]".format(idx,len(dataLoader)))
