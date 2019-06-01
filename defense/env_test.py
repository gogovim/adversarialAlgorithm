#coding:utf-8
from vgg import VGG
from resnet import resnet101
from densenet import DenseNet
from inceptionresnetv2 import InceptionResNetV2
from inceptionv4 import InceptionV4
from inception import Inception3
from randomResizePadding import RandomResizePadding
from mymodel import Mymodel
from HGD import get_denoise,denoise_loss
from comDefend import ComDefend
from rectifi import Rectifi

from torchvision import datasets, transforms
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
from PIL import ImageFile
import sys
import random
os.environ["CUDA_VISIBLE_DEVICES"]="0"
ImageFile.LOAD_TRUNCATED_IMAGES = True
mean = [0.6460, 0.5981, 0.5895]
stdv = [0.3220, 0.3280, 0.3295]

def load_images(input_dir, batch_shape):
    images = np.random.random(batch_shape)
    #print(batch_shape,images.dtype)
    pad=random.randint(0,6)
    pad=0
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in os.listdir(input_dir):
        try:
            image=Image.open(input_dir+'/'+filepath)
            #image=image.resize((224,224))
            #image=image.resize()
            #print(filepath)
            image=np.array(image.resize((batch_shape[2]-2*pad,batch_shape[3]-2*pad)),dtype='float32')
            image=(image / 255.0) 
            image=(image-mean)/stdv

            #cv2.imshow("0",image)
            #cv2.waitKey(0)
            w1,h1=random.randint(0,2*pad),random.randint(0,2*pad)
            #print("here",pad,w1,h1)
            #print(w1,h1,images)
            #print(w1,batch_shape[2]-pad+w1, h1,batch_shape[3]-pad+h1)
            images[idx, 0, w1:batch_shape[2]-2*pad+w1, h1:batch_shape[3]-2*pad+h1] = image[:,:,0]
            images[idx, 1, w1:batch_shape[2]-2*pad+w1, h1:batch_shape[3]-2*pad+h1] = image[:,:,1]
            images[idx, 2, w1:batch_shape[2]-2*pad+w1, h1:batch_shape[3]-2*pad+h1] = image[:,:,2]

            #print(images[idx,0,10:20,10:20])
            #cv2.imshow("1",images[idx])


        except :
            print('error')
            continue
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames,images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0

    if idx > 0:
        #print("output image",filenames,images.shape,images1.shape)
        yield filenames,images
def get_model(model_name,pho_size=299,num_classes=110):
    if model_name=="vgg16":
        model=VGG(num_classes=num_classes,pho_size=299)
    elif model_name=="resnet101":
        model=resnet101(num_classes=num_classes)
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
def test(input_dir,model_name,weight_name,pho_size,num_classes,random_layer=None,denoise='Comdefend',denoise_weight='comdefend_299_model.dat'):
    print(denoise,denoise_weight)
    denoise=get_model(denoise)
    try:
        denoise.load_state_dict(torch.load(os.path.join(denoise_weight)))
    except:
        #model=torch.nn.DataParallel(model)
        denoise.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(denoise_weight)).items()})
    
    model=get_model(model_name,pho_size=pho_size,num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(os.path.join(weight_name)))
    except:
        #model=torch.nn.DataParallel(model)
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(weight_name)).items()})
    #model=torch.nn.DataParallel(model)
    #model.load_state_dict(torch.load(os.path.join(weight_name)))
    model=model.cuda()
    denoise.cuda()
    model.eval()
    denoise.eval()
    res=[]
    for filenames,inputs in load_images(input_dir,(1,3,pho_size,pho_size)):
        inputs=inputs.astype(np.float32)
        inputs=torch.from_numpy(inputs)
        inputs = inputs.cuda()
        #inputs=denoise(inputs)
        #print(inputs)
        if random_layer:
            inputs=random_layer(inputs)
        #print(inputs.shape)
        output = model(inputs)
        _, preds= output.data.cpu().topk(1, dim=1)

        #print(preds)
        for filename,  label in zip(filenames,  preds):
        	res.append((filename,label.item()))
    return res
def merge_pred(preds):
    for i in range(len(preds)):
        preds[i]=pd.DataFrame(preds[i]).set_index(0)
    preds=pd.concat(preds,axis=1)
    print(preds)
    
    preds=pd.DataFrame(preds.mode(axis=1)[0])
    #print(preds)
    preds.columns=['value']
    

    preds=preds.reset_index()
    print(preds.values)

    return preds.values


def defense(input_dir,output_file):
    model_names=["Inception3","InceptionResNetV2","InceptionV4","resnet101","densenet","Inception3","resnet101"]
    mode_weights=["Inception3_clean_299_model.dat","InceptionResNetV2_clean_299_model.dat","Inceptionv4_clean_299_model.dat",
    "resnet101_clean_299_model.dat","densenet_clean_166_model.dat","Inception3_clean_299_randomLayer_model.dat","resnet101_adversarial_model.dat"]
    random_layer=RandomResizePadding().cuda()
    preds=[]
    #with torch.no_grad():
    #    preds.append(test(input_dir,model_names[3],mode_weights[3],299,110))
    #with torch.no_grad():
    #    preds.append(test(input_dir,model_names[0],mode_weights[0],299,110))
    #with torch.no_grad():
    #    preds.append(test(input_dir,model_names[6],mode_weights[6],299,110,random_layer=random_layer))
    with torch.no_grad():
        preds.append(test(input_dir,model_names[0],mode_weights[0],299,110,denoise='Rectifi',denoise_weight='Rectifi_299_model.dat'))

    #print(preds1,preds2,preds3)
    preds=merge_pred(preds)
    test_path = input_dir
    with open(output_file,'w') as out_file:
        for filename,pred in preds:
            out_file.write('{0},{1}\n'.format(filename, int(pred)))
"""
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda()

# model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(os.path.join(path))['state_dict'].items()})

model.load_state_dict(torch.load(os.path.join(path)))
model.eval()
"""
#pho_size = 224
"""
test_transforms = transforms.Compose([
    transforms.Resize((pho_size ,pho_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=stdv),
])
"""
test_path = './test'
if len(sys.argv)>1:
    test_path=sys.argv[1]
if len(sys.argv)>2:
    outputfile=sys.argv[2]

#test_data = datasets.ImageFolder(test_path,transform=test_transforms)
#test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
#print('begin test')
defense(test_path,outputfile)
"""
with open(outputfile,'w') as out_file:
    for filenames,inputs in load_images(test_path,(1,3,pho_size,pho_size)):
        inputs=inputs.astype(np.float32)
        #print(inputs.dtype)
        inputs=torch.from_numpy(inputs)
        #print(inputs.dtype)
        #print(filenames,inputs)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        output = model(inputs)
        _, preds = output.data.cpu().topk(1, dim=1)
        #output=np.ones([1,1],dtype=np.int)
        #preds=torch.from_numpy(output);
        #print(preds)
        for filename, label in zip(filenames, preds):
            out_file.write('{0},{1}\n'.format(filename, label.item()))
"""






