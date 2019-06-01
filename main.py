import argparse
from model.vgg import VGG
from model.resnet import resnet101
from model.densenet import DenseNet
    
import os
import numpy as np
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mean = [0.65, 0.60, 0.59]
stdv = [0.32, 0.33, 0.33]
def load_images(input_dir, batch_shape):
    #print("load images",input_dir)
    images = np.zeros(batch_shape,dtype='float32')
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in os.listdir(input_dir):
        #print(filepath)
        try:
            #print(input_dir+'/'+filepath)
            image=Image.open(input_dir+'/'+filepath)
            #print('read over')
            image=np.array(image.resize((batch_shape[2],batch_shape[3])),dtype='float32')
            image=(image / 255.0) 
            image=(image-mean)/stdv
            images[idx, 0, :, :] = image[:,:,0]
            images[idx, 1, :, :] = image[:,:,1]
            images[idx, 2, :, :] = image[:,:,2]
            #print('ok')
        except :
            continue
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

def FGSM(x, y_true, net,criterion,y_target=None, eps=0.03, alpha=2/255, iteration=1):

    h = net(x)
    prediction = h.max(1)[1]
    accuracy = torch.eq(prediction, y_true).float().mean()
    cost = criterion(h, y_true)

    if y_target is not None:
        x_adv, h_adv = adversary.fgsm(x, y_target, net,criterion,True, eps)
    else:
        x_adv, h_adv = adversary.fgsm(x, y_true, net, criterion, False, eps)
    pre_adv=h_adv.max(1)[1]
    accuracy_adv = torch.eq(pre_adv, y_true).float().mean()
    #print('y_true.size=',h_adv.size(),y_true.size())
    cost_adv = criterion(h_adv, y_true)

    print("x_adv.size={0}".format(x_adv.size()))
    print("before FGSM ,accuracy={0},cost={1}.".format(accuracy,cost))
    print("after  FGSM ,accuracy={0},cost={1}.".format(accuracy_adv,cost_adv))

    return x_adv,pre_adv

def generate_adversial_examles(mode,input_dir,output_dir):
    pho_size=220
    mean = [0.65, 0.60, 0.59]
    stdv = [0.32, 0.33, 0.33]
    batch_size=16
    depth = 100
    block_config = [(depth - 4) // 6 for _ in range(3)]

    """
    valid_path="/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_train"
    valid_transforms = transforms.Compose([
        transforms.Resize((pho_size, pho_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    valid_data = datasets.ImageFolder(valid_path,
                                      transform=valid_transforms)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    """
    

    

    custdv,cumean=torch.tensor(stdv).cuda(),torch.tensor(mean).cuda()
    for filenames,inputs in load_images(args.input_dir,(16,3,pho_size,pho_size)):
        print(filenames)
        inputs=inputs.astype(np.float32)
        #print(inputs.dtype)
        inputs=torch.from_numpy(inputs)
        #print(inputs.dtype)
        #print(filenames,inputs)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        output = model(inputs).max(1)[1]
        adv_input,pre_adv=FGSM(inputs,output,model,torch.nn.functional.cross_entropy)
        adv_input=adv_input.data.cuda()
        #print(input,type(custdv))
        adv_input=gpucvrtimgs(adv_input,cumean,custdv)
        #input=gpucvrtimgs(input,cumean,custdv)
        for idx,filename in enumerate(filenames):
            #print("write",args.output_dir+'/'+filename)
            im=Image.fromarray(adv_input[idx])
            im.save(args.output_dir+'/'+filename)
def get_model(model_name):
    if model_name=="vgg16":
        model=VGG(num_classes=110)
    elif model_name=="resnet101":
        model=resnet101(num_classes=1000)
    elif model_name=="densenet":
        model=DenseNet(
        growth_rate=12,
        block_config=[(100 - 4) // 6 for _ in range(3)],
        num_classes=110,
        small_inputs=False,
        efficient=True,
    )
    else:
        model=None
    return model
def gpucvrtimgs(X,mean,stdv):
    #print(X.shape)
    #X=X.view(X.shape[0],X.shape[2],X.shape[3],3)
    X=X.transpose(1,2).transpose(2,3)
    #print(X.shape)
    X=(X*stdv+mean)*255
    X=torch.clamp(X,0,255).cpu().numpy().astype(np.uint8)
    return X
def attack():
    pass
def defense(input_dir,output_file):
    model1,model2,model3=get_model("vgg16"),get_model("resnet101"),get_model("densenet")
    model1,model2,model3=torch.nn.DataParallel(model1).cuda(),torch.nn.DataParallel(model2).cuda(),torch.nn.DataParallel(model3).cuda()
    path1,path2,path3 ="./weights/vgg16_model.dat","./weights/resnet101_model.dat","./weights/model.dat"
    model1.load_state_dict(torch.load(os.path.join(path1)))
    model2.load_state_dict(torch.load(os.path.join(path2)))
    model3.load_state_dict(torch.load(os.path.join(path3)))

    pho_size = 224
    test_path = input_dir

    with open(output_file,'w') as out_file:
        for filenames,inputs in load_images(test_path,(1,3,pho_size,pho_size)):
            inputs=inputs.astype(np.float32)
            #print(inputs.dtype)
            inputs=torch.from_numpy(inputs)
            #print(inputs.dtype)
            #print(filenames,inputs)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            output1 = torch.nn.functional.softmax(model1(inputs))
            output2 = torch.nn.functional.softmax(model2(inputs))
            output3 = torch.nn.functional.softmax(model3(inputs))
            #print(output2[:,:110])
            #print(output3)
            _, preds1 = output1.data.cpu().topk(1, dim=1)
            _, preds2 = output2.data.cpu().topk(1, dim=1)
            _, preds3 = output3.data.cpu().topk(1, dim=1)
            print(filenames,preds1,preds2,preds3)
            #output=np.ones([1,1],dtype=np.int)
            #preds=torch.from_numpy(output);
            #print(preds)
            for filename, label1,label2,label3 in zip(filenames, preds1,preds2,preds3):
                labels=np.array([label1.item(),label2.item(),label3.item()])
                bin_count=np.bincount(labels)
                label=np.argmax(bin_count)
                if bin_count[label]==1:
                    label=label2.item()
                out_file.write('{0},{1}\n'.format(filename, label))

def main(args):
    defense(args.input_dir,args.output_file)

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
    parser.add_argument('--output_file', type=str, default='res.csv', help='output directory path')
    # parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='checkpoint directory path')
    # parser.add_argument('--load_ckpt', type=str, default='', help='')
    # parser.add_argument('--mode', type=str, default='train', help='generate / defense /train')
    # parser.add_argument('--seed', type=int, default=1, help='random seed')
    # parser.add_argument('--iteration', type=int, default=1, help='the number of iteration for FGSM')
    # parser.add_argument('--epsilon', type=float, default=0.03, help='epsilon for FGSM and i-FGSM')
    # parser.add_argument('--alpha', type=float, default=2/255, help='alpha for i-FGSM')
    args = parser.parse_args()
    #print(args.input_dir,args.output_dir)
    main(args)
