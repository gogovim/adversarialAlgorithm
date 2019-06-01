import torch
from torch import nn
import os
import time
import sys
import argparse
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile, Image
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
from model.HGD import get_denoise, denoise_loss, HGD
from model.randomResizePadding import RandomResizePadding
from model.comDefend import ComDefend

import train_HGD

from utils import AverageMeter, Dataset_filename, AdversarialDataset, Dataset_noise, mean, stdv, get_model, where, \
    cal_distance, train_adversarial_dirs, valid_adversarial_dirs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def cal_loss1(x):
    v = x * x
    v = v.view(v.shape[0], -1)

    return v.mean(1)


def cal_loss2(x, y):
    return cal_loss1(x - y)


def cal_loss(x, y):
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)
    #print("x:",x.device,"y:",y.device)
    res=torch.abs(x - y).mean(1)
    #print("res:",res.device)
    return res


def cal_hloss(x, y, classify_model=None):
    if classify_model is None:
        return cal_loss2(x, y)
    o1 = classify_model(x)
    o2 = classify_model(y)

    return cal_loss(o1, o2)


def convert(adv_inputs, mean, stdv, inputs=None):
    # print("bufore D:",torch.pow(torch.abs(adv_inputs - inputs), 1).mean())

    adv_inputs = adv_inputs.transpose(1, 2).transpose(2, 3)
    adv_inputs = (adv_inputs * stdv + mean)
    adv_inputs = torch.clamp(adv_inputs, 0, 1)
    if inputs is not None:
        inputs = inputs.transpose(1, 2).transpose(2, 3)
        inputs = (inputs * stdv + mean)
        inputs = torch.clamp(inputs, 0, 1)
        # print(adv_inputs-inputs)
        return adv_inputs.cpu().numpy(), inputs.cpu().numpy()

    return adv_inputs.cpu().numpy()


def FGSM(x, y_true, net, criterion, y_target=None, eps=0.03, x_min=[0, 0, 0], x_max=[1, 1, 1]):
    # print(eps)
    if y_target is not None:
        x_adv = adversary.fgsm(x, y_target, net, criterion, True, eps)
    else:
        x_adv = adversary.fgsm(x, y_true, net, criterion, False, eps)
    # print('----------------------')
    # print(x_min,x_max,x_adv.shape,x_adv[:][0][:][:].shape,x_adv[:][1][:][:].shape)
    x_adv[:, 0, :, :] = torch.clamp(x_adv[:, 0, :, :], x_min[0], x_max[0])
    x_adv[:, 1, :, :] = torch.clamp(x_adv[:, 1, :, :], x_min[1], x_max[1])
    x_adv[:, 2, :, :] = torch.clamp(x_adv[:, 2, :, :], x_min[2], x_max[2])
    # print(x_adv[0][2])
    # print(x_adv[0][0].shape)

    return x_adv.detach()


def valid_epoch(model, valid_dataloader, lam, print_freq=40, classify_models=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error1 = AverageMeter()
    error2 = AverageMeter()
    end = time.time()
    model.eval()
    # classify_model.eval()
    cost = cal_hloss
    custdv, cumean = torch.tensor(stdv).cuda(), torch.tensor(mean).cuda()
    nx = np.array([[0., 0., 0.], [1., 1., 1.]])
    x_min, x_max = (nx - mean) / stdv
    classify_model = classify_models[0]
    classify_model.eval()
    epss = [20/255/0.3]
    for i, (images, labels) in enumerate(valid_dataloader):

        gen_classify = classify_models[random.randint(0, len(classify_models) - 1)]

        # print(type(images),type(labels))
        # print(images.shape,labels.shape)
        #if i>30:
        #    return batch_time.avg, losses.avg,error2.avg
        images, labels = images.cuda(), labels.cuda()
        ad_images = FGSM(images, labels, gen_classify, torch.nn.functional.cross_entropy,
                         eps=epss[random.randint(0, len(epss) - 1)], x_min=x_min, x_max=x_max)
        # noise=noise.float().cuda()
        # print(images.shape)
        # print(images.shape,labels.shape)

        # for t in classify_model.parameters():
        #    print(t.grad)
        #    break
        # optimizer.zero_grad()
        # classify_model.zero_grad()
        with torch.no_grad():
            rec_x = model(images)
            rec_xbar = model(ad_images)

            y = classify_model(images)
            y1 = classify_model(rec_x)
            y2 = classify_model(rec_xbar)

        loss1=cal_loss2(images,rec_x).mean()
        loss2 = cal_loss(y, y2).mean()

        loss = loss1 + loss2

        batch_size = labels.size(0)
        losses.update(loss.item(), batch_size)

        y1 = y1.max(1)[1]
        y2 = y2.max(1)[1]
        # if i%print_freq==0:
        #    d_clean_y=d_clean_y.max(1)[1]
        #    adv_y=adv_y.max(1)[1]
        error1.update(torch.ne(y1.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        error2.update(torch.ne(y2.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:

            print("loss1 = ", loss1, "loss2 = ", loss2)
            ims, ims1 = convert(rec_x.detach(), cumean, custdv, rec_xbar.detach())
            imsc = convert(images, cumean, custdv)
            for im in ims1:
                im = im * 255.
                im = np.rint(im).astype(np.uint8)
                # print(im.shape)
                cv2.imwrite(str(i // print_freq % print_freq) + "validrecnoise.jpg", im)
                # cv2.waitKey(1)
            for im in ims:
                im = im * 255.
                im = np.rint(im).astype(np.uint8)
                # print(im.shape)
                cv2.imwrite(str(i // print_freq % print_freq) + "validrec.jpg", im)
            for im in imsc:
                im = im * 255.
                im = np.rint(im).astype(np.uint8)
                # print(im.shape)
                cv2.imwrite(str(i // print_freq % print_freq) + "validraw.jpg", im)
            res = '\t'.join([
                'valid',
                'Iter: [%d/%d]' % (i + 1, len(valid_dataloader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'clean image Error %.4f (%.4f)' % (error1.val, error1.avg),
                'noise image Error %.4f (%.4f)' % (error2.val, error2.avg)
            ])
            print(res)
    return batch_time.avg, losses.avg, error2.avg


def train_epoch(model, train_dataloader, optimizer, lam, epoch, n_epochs, print_freq=40, classify_models=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error1 = AverageMeter()
    error2 = AverageMeter()
    end = time.time()
    model.train()
    # classify_model.eval()
    cost = cal_hloss
    custdv, cumean = torch.tensor(stdv).cuda(), torch.tensor(mean).cuda()
    nx = np.array([[0., 0., 0.], [1., 1., 1.]])
    x_min, x_max = (nx - mean) / stdv
    classify_model = classify_models[0]
    classify_model.eval()
    epss = [0.12,0.24,0.36,0.48,0.60,0.72,0.84,0.96,1.08]
    for i, (images, labels) in enumerate(train_dataloader):

        gen_classify = classify_models[random.randint(0, len(classify_models) - 1)]

        # print(type(images),type(labels))
        # print(images.shape,labels.shape)
        #if i>30:
        #    return batch_time.avg, losses.avg,error2.avg
        images, labels = images.cuda(), labels.cuda()
        ad_images = FGSM(images, labels, gen_classify, torch.nn.functional.cross_entropy,
                         eps=epss[random.randint(0, len(epss) - 1)], x_min=x_min, x_max=x_max)
        # noise=noise.float().cuda()
        # print(images.shape)
        # print(images.shape,labels.shape)

        # for t in classify_model.parameters():
        #    print(t.grad)
        #    break
        optimizer.zero_grad()
        classify_model.zero_grad()
        #print(images.device,ad_images.device)
        rec_x=model(images)
        rec_xbar=model(ad_images)

        y= classify_model(images)
        y1 = classify_model(rec_x)
        y2 = classify_model(rec_xbar)

        loss1=cal_loss2(images,rec_x).mean()
        loss2=cal_loss(y,y2).mean()

        loss=loss1+loss2
        loss.backward()
        optimizer.step()


        batch_size = labels.size(0)
        losses.update(loss.item(), batch_size)

        y1 = y1.max(1)[1]
        y2 = y2.max(1)[1]
        # if i%print_freq==0:
        #    d_clean_y=d_clean_y.max(1)[1]
        #    adv_y=adv_y.max(1)[1]
        error1.update(torch.ne(y1.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        error2.update(torch.ne(y2.cpu(), labels.cpu()).float().sum().item() / batch_size, batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:

            print("loss1 = ", loss1, "loss2 = ", loss2)
            ims, ims1 = convert(rec_x.detach(), cumean, custdv, rec_xbar.detach())
            imsc, imsa = convert(images, cumean, custdv, ad_images)
            for im in ims1:
                im = im * 255.
                im = np.rint(im).astype(np.uint8)
                # print(im.shape)
                cv2.imwrite(str(i // print_freq % print_freq) + "recnoise.jpg", im)
                # cv2.waitKey(1)
            for im in ims:
                im = im * 255.
                im = np.rint(im).astype(np.uint8)
                # print(im.shape)
                cv2.imwrite(str(i // print_freq % print_freq) + "rec.jpg", im)
            for im in imsc:
                im = im * 255.
                im = np.rint(im).astype(np.uint8)
                # print(im.shape)
                cv2.imwrite(str(i // print_freq % print_freq) + "raw.jpg", im)
            for im in imsa:
                im = im * 255.
                im = np.rint(im).astype(np.uint8)
                # print(im.shape)
                cv2.imwrite(str(i // print_freq % print_freq) + "noise.jpg", im)
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (i + 1, len(train_dataloader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'clean image Error %.4f (%.4f)' % (error1.val, error1.avg),
                'noise image Error %.4f (%.4f)' % (error2.val, error2.avg)
            ])
            print(res)
    return batch_time.avg, losses.avg, error2.avg


def train(model, train_dataloader, lr, save_dir, num_epoches, model_name, valid_dataloader=None, classify_models=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    end = time.time()
    best_error = 1.
    nobetter_num = 1
    for epoch in range(num_epoches):
        if nobetter_num >= 5:
            print("train done .lr={},best_loss={}".format(lr, best_error))
            break
        if nobetter_num >= 3:
            lr = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        _, train_loss, train_error = train_epoch(
            model,
            train_dataloader,
            optimizer,
            0.0001,
            epoch,
            num_epoches,
            classify_models=classify_models
        )
        if valid_dataloader:
            _, valid_loss, valid_error = valid_epoch(
                model,
                valid_dataloader,
                0.0001,
                classify_models=classify_models
            )
        if valid_dataloader and valid_error < best_error:
            best_error = valid_error
            if valid_error + 0.005 < best_error:
                nobetter_num += 1
            else:
                nobetter_num = 1
            print('New best error: %.4f' % best_error)
            torch.save(model.state_dict(), os.path.join(save_dir, model_name + '_model.dat'))
        else:
            nobetter_num += 1

        with open(os.path.join(save_dir, model_name + '_results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.6f,%.6f\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error
            ))


def main(args):
    model = get_model(args.model_name)
    model=torch.nn.DataParallel(model).cuda()
    print(args.pre_train)
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(args.pre_train)).items()})
    if args.pre_train:
        try:
            model.load_state_dict(torch.load(os.path.join(args.pre_train)))
            print("ok")
        except:
            model.load_state_dict(
                {k.replace('module.', ''): v for k, v in torch.load(os.path.join(args.pre_train)).items()})
            print("ok")

    # model=torch.nn.DataParallel(model)
    model = model.cuda()
    print(model)
    model_name = args.model_name + "_" + str(args.pho_size)

    if args.mode == "train":

        train_transforms = transforms.Compose([
            transforms.Resize((args.pho_size, args.pho_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        valid_transforms = transforms.Compose([
            transforms.Resize((args.pho_size, args.pho_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ])
        train_data = datasets.ImageFolder(args.train_dir, transform=train_transforms)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # 交叉验证
        valid_data = datasets.ImageFolder(args.valid_dir, transform=valid_transforms)
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

        classify_models = []

        classify_model = get_model("Inception3", pho_size=args.pho_size, num_classes=110)
        try:
            classify_model.load_state_dict(torch.load(os.path.join("./weights/Inception3_clean_299_model.dat")))
        except:
            classify_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(
                os.path.join("./weights/Inception3_clean_299_model.dat")).items()})
        classify_model = torch.nn.DataParallel(classify_model).cuda()

        classify_models.append(classify_model)

        classify_model = get_model("resnet101", pho_size=args.pho_size, num_classes=110)
        try:
            classify_model.load_state_dict(torch.load(os.path.join("./weights/resnet101_clean_299_model.dat")))
        except:
            classify_model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(
                os.path.join("./weights/resnet101_clean_299_model.dat")).items()})
        classify_model = torch.nn.DataParallel(classify_model).cuda()
        classify_models.append(classify_model)

        classify_model = get_model("densenet", pho_size=args.pho_size, num_classes=110)
        try:
            classify_model.load_state_dict(torch.load(os.path.join("./weights/densenet_clean_299_model.dat")))
        except:
            classify_model.load_state_dict({k.replace('module.', ''): v for k, v in
                                            torch.load(os.path.join("./weights/densenet_clean_299_model.dat")).items()})
        classify_model = torch.nn.DataParallel(classify_model).cuda()
        classify_models.append(classify_model)

        train(model, train_loader, args.lr, args.save_dir, args.epoch, model_name, valid_loader,
              classify_models=classify_models)

if __name__=="__main__":

    train_dir='/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_train'
    valid_dir='/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_test'
    batch_size=2
    #./weights/comdefend_299_model.dat./weights/Rectifi_299_model.dat
    parser = argparse.ArgumentParser(description='Rectifi')

    parser.add_argument('--mode', type=str, default='train', help='train | valid | test')
    parser.add_argument('--phi', type=float, default=1.0, help='phi')
    parser.add_argument('--bits', type=int, default=12, help='bits')

    #photo_size
    parser.add_argument('--pho_size',type=int,default=299,help='photo size')

    #load model
    parser.add_argument('--model_name', type=str, default='Rectifi', help='model name')
    parser.add_argument('--pre_train', type=str, default='', help='weights')
    
    #train lr
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')


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
