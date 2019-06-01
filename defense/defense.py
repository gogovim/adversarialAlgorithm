from torchvision import datasets, transforms
from densenet import DenseNet
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
from PIL import ImageFile
import sys
ImageFile.LOAD_TRUNCATED_IMAGES = True
mean = [0.5071, 0.4867, 0.4408]
stdv = [0.2675, 0.2565, 0.2761]
def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape,dtype='float32')
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in os.listdir(input_dir):
        try:
            image=Image.open(input_dir+'/'+filepath)
            image=np.array(image.resize((batch_shape[2],batch_shape[3])),dtype='float32')
            image=(image / 255.0) 
            image=(image-mean)/stdv
            images[idx, 0, :, :] = image[:,:,0]
            images[idx, 1, :, :] = image[:,:,1]
            images[idx, 2, :, :] = image[:,:,2]
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

depth=100
growth_rate=12
efficient=True
block_config = [(depth - 4) // 6 for _ in range(3)]
model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=110,
        small_inputs=False,
        efficient=efficient,
    )

path = 'model.dat'
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model).cuda()
# model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(os.path.join(path))['state_dict'].items()})
model.load_state_dict(torch.load(os.path.join(path)))
model.eval()
pho_size = 220
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
	    #print(preds)
	    for filename, label in zip(filenames, preds):
	        out_file.write('{0},{1}\n'.format(filename, label.item()))







