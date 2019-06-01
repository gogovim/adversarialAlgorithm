#NonTargetedAttack.py
from torchvision import datasets, transforms
import os
import torch
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

def cal_distance(x,y):
    diff = np.abs(x.reshape((-1, 3)) - y.reshape((-1, 3)))
    #print((diff>0).sum(),299*299*3)
    #print(diff)
    return np.mean(np.sqrt(np.sum((diff ** 2), axis=1)))

def load_images(input_dir, OutputDirectory):
    for filepath in os.listdir(input_dir):
        print(os.path.join(input_dir,filepath))
        try:
            image=Image.open(input_dir+'/'+filepath)
            #image=image.resize(224,224)
            #image=image.resize()
            print(filepath)
            #print(image.shape())
            image=np.array(image.resize((299,299)),dtype='float32')
            print('over')
            left = 100
            right = 199
            top = 100
            bottom = 199
            new_image = image.copy()
            n = 1

            for i in range(2):
                new_image[left:right, top:bottom] += image[(left - i):(right - i), top:bottom]
                new_image[left:right, top:bottom] += image[(left + i):(right + i), top:bottom]
                new_image[left:right, top:bottom] += image[left:right, (top - i):(bottom - i)]
                new_image[left:right, top:bottom] += image[left:right, (top + i):(bottom + i)]
            print('hehehh')
            n += 4
            new_image[left:right, top:bottom] /= n
            new_image = new_image.astype(np.uint8)
            print("meanD=",cal_distance(image,new_image))
            Image.fromarray(np.asarray(new_image, np.int8), "RGB").save(OutputDirectory + "/" + filepath)
            print('over')
        except:
        	print('error')

InputDirectory = sys.argv[1]
OutputDirectory = sys.argv[2]
load_images(InputDirectory,OutputDirectory)
