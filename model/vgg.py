import torch
from  torch import nn
import os
import time
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#device=torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True
#torch.backends.cudnn.benchmark = True
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


class VGG(nn.Module):
    def __init__(self,modeltype="VGG16",num_classes=110,pho_size=224):
        super(VGG, self).__init__()
        self.pho_size=pho_size
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2))
        w=int(pho_size/32)
        self.classify=nn.Sequential(
            nn.Linear(w*w*512, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        out=self.feature(x)
        #print(out.size())
        out=out.view(out.size(0),-1)
        out=self.classify(out)
        return out
def vgg_valid(model,valid_dataloader,cost,print_freq=40):
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

def vgg_train_epoch(model,train_dataloader,optimizer, cost,epoch, n_epochs, print_freq=40):

    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    end = time.time()
    model.train()
    for i,(images,labels) in enumerate(train_dataloader):
        #if i>10:
        #    return batch_time.avg, losses.avg, error.avg
        images,labels=images.cuda(),labels.cuda()
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
                'Iter: [%d/%d]' % (i + 1, len(train_dataloader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)
    return batch_time.avg, losses.avg, error.avg
def vgg_train(model,train_dataloader,lr=0.01,save_dir='./weights',num_epoches=200,valid_dataloader=None):
    
    
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
        _, train_loss, train_error = vgg_train_epoch(
            model,
            train_dataloader,
            optimizer,
            cost,
            epoch,
            num_epoches
        )
        if valid_dataloader:
        	_, valid_loss, valid_error = vgg_valid(
                model,
                valid_dataloader,
                cost
            )
        if valid_dataloader and valid_error< best_error:
            best_error = valid_error
            if valid_error+0.005 < best_error:
                nobetter_num+=1
            else:
                nobetter_num=1
            print('New best error: %.4f' % best_error)
            torch.save(model.state_dict(), os.path.join(save_dir, 'vgg16_model.dat'))
        else:
            #torch.save(model.state_dict(), os.path.join(save_dir, 'vgg16_model.dat'))
            nobetter_num+=1

        with open(os.path.join(save_dir, 'vgg16_results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
            ))
        
def main():
    train_dir="/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_train"
    valid_dir="/home/data/wanghao/tianchi/data/IJCAI_2019_AAAC_test"
    save_dir="./weights"
    mean = [0.65, 0.60, 0.59]
    stdv = [0.32, 0.33, 0.33]
    pho_size = 224
    batch_size=16
    train_transforms = transforms.Compose([
        transforms.Resize((pho_size ,pho_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize((pho_size,pho_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    train_data = datasets.ImageFolder(train_dir,
                                      transform=train_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=4)

    #交叉验证
    valid_data = datasets.ImageFolder(valid_dir,
                                                transform=valid_transforms)
    valid_loader = DataLoader(valid_data, batch_size=8, shuffle=False,num_workers=4)

    model=VGG(num_classes=110)
    #print(device)
    print(torch.cuda.current_device(),torch.cuda.device_count())
    #torch.cuda.set_device(1)
    print(torch.cuda.current_device(),torch.cuda.get_device_name(0))
    model.cuda()
    model=torch.nn.DataParallel(model)
    #model=model.cuda()
    print(model)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'vgg16_model.dat')))
    vgg_train(model,train_loader,0.01,save_dir,300,valid_loader)

if __name__=="__main__":
	main()


