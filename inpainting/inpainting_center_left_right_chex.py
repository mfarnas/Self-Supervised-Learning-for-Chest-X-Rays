import os
import glob
import cv2

import numpy as np
from skimage.exposure import equalize_hist as equalize
from PIL import Image
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from torchvision.models.densenet import DenseNet, densenet121

# set device
if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")
print("Using "+str(device))

inp_augs = [None]

params = {
    "equalize_hist": False,
    "num_classes": 4,
    "batch_size": 32,
    "num_workers": 8,

    # set mask configurations
    "image_shape": [256, 256, 3],
    "mask_shape": [100, 100],
    "margin": [0, 0],
    "mask_batch_same": True,
    "left_right_crop": True,
    "max_delta_shape": [32, 32],
    "mask_type": "hole",
    "mosaic_unit_size": 12
}

OUTPUT_PATH = "./"
     
class PNGReader:
    def __init__(self, equalize_hist=False):
        self.equalize_hist = equalize_hist
        
    def png2array(self, img):

        data = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        data = np.reshape(data, (1, data.shape[0], data.shape[1]))

        data = data - np.min(data)
        data = data / np.max(data)

        if self.equalize_hist:
            data = equalize(data)

        return data

    def read(self, img_path) -> torch.Tensor:
        img_data = self.png2array(img_path)
        img_data = torch.Tensor(img_data)

        return img_data   
        
def random_bbox(params, batch_size):
    """Generate a random tlhw with configuration.

    Args:
        params: Config should have configuration including img

    Returns:
        tuple: (top, left, height, width)

    """
    img_height, img_width, _ = params['image_shape']
    h, w = params['mask_shape']
    margin_height, margin_width = params['margin']
    maxt = img_height - margin_height - h
    maxl = img_width - margin_width - w
    bbox_list = []
    if params['mask_batch_same']:

        if params['left_right_crop']:
            if np.random.choice(2, 1)[0] % 2 == 0:
                ## left targeted mask
                # bbx ends at 20% from the bottom
                maxt = img_height - 0.2*img_height - h
                # bbx ends at the center from the left
                maxl = img_width - 0.5*img_width - w
                # bbx starts from 15% from the top
                t = np.random.randint(0.15*img_height, maxt)
                # bbx starts from 10% from the left
                l = np.random.randint(0.1*img_width, maxl)
                bbox_list.append((t, l, h, w))
                bbox_list = bbox_list * batch_size

            else:
                # right targeted mask
                # bbx ends at 20% from the bottom
                maxt = img_height - 0.2*img_height - h
                # bbx ends at 10% from the left
                maxl = img_width - 0.1*img_width - w
                # bbx starts from 15% from the top
                t = np.random.randint(0.15*img_height, maxt)
                # bbx starts from 50% from the left
                l = np.random.randint(0.5*img_width, maxl)
                bbox_list.append((t, l, h, w))
                bbox_list = bbox_list * batch_size
        
        else:
            t = np.random.randint(margin_height, maxt)
            l = np.random.randint(margin_width, maxl)
            bbox_list.append((t, l, h, w))
            bbox_list = bbox_list * batch_size

    else:
        for i in range(batch_size):
            t = np.random.randint(margin_height, maxt)
            l = np.random.randint(margin_width, maxl)
            bbox_list.append((t, l, h, w))

    return torch.tensor(bbox_list, dtype=torch.int64)

def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w):
    batch_size = bboxes.size(0)
    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
    for i in range(batch_size):
        bbox = bboxes[i]
        delta_h = np.random.randint(max_delta_h // 2 + 1)
        delta_w = np.random.randint(max_delta_w // 2 + 1)
        mask[i, :, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.
    return mask

def mask_image(x, bboxes, params):
    height, width, _ = params['image_shape']
    max_delta_h, max_delta_w = params['max_delta_shape']
    mask = bbox2mask(bboxes, height, width, max_delta_h, max_delta_w)

    if x.is_cuda:
        mask = mask.cuda()
        # cropped_result = cropped_result.cuda()

    if params['mask_type'] == 'hole':
        result = x * (1. - mask)
        cropped_result = x * mask
    else:
        raise NotImplementedError('Not implemented mask type.')

    ImageWithCenterRemoved = result
    ImageWithLRCrop = cropped_result
    return ImageWithCenterRemoved, ImageWithLRCrop #result, mask

# Prepare inputs    
class CHEXPERT_CENTER_CROP(Dataset):
    def __init__(self, path, size=(256,256), trans=None):

        self.transforms = trans
        self.img_reader = PNGReader()
        self.resize = transforms.Resize(size, interpolation=Image.BILINEAR)

        self.dataset = {}
        self.dataset["inputs"] = []
        self.dataset["studies"] = []
        
        filepath = glob.glob(os.path.join(path, "*", "*", "*.jpg"))
          
        for img_path in filepath:
            self.dataset["inputs"].append(img_path)


    def __len__(self):
        return len(self.dataset["inputs"])

    def __getitem__(self, index):
        
        img_path = self.dataset["inputs"][index]

        img = self.img_reader.read(img_path)
        
        img = torch.Tensor(img)

        img = self.resize(img)

        img = img.expand((3, img.shape[1], img.shape[2]))
        
        IMG_SIZE = img.shape[1]
        CROP_SIZE = 100

        start_height = IMG_SIZE // 2 - (CROP_SIZE // 2)
        start_width = IMG_SIZE // 2 - (CROP_SIZE // 2)
        
        target = torch.zeros(img.size())
        target[
            ...,
            start_height:start_height + CROP_SIZE,
            start_width:start_width + CROP_SIZE
        ] = 1
        
        img2 = img.clone()
        img2[
            ...,
            start_height:start_height + CROP_SIZE,
            start_width:start_width + CROP_SIZE
        ] = 0
        
        if self.transforms is not None:
            img2 = self.transforms(img2)

        return img2, target

class CHEXPERT_LR_CROP(Dataset):
    def __init__(self, path, size=(256,256), trans=None):

        self.transforms = trans
        self.img_reader = PNGReader()
        self.resize = transforms.Resize(size, interpolation=Image.BILINEAR)

        self.dataset = {}
        self.dataset["inputs"] = []
        self.dataset["studies"] = []
        
        filepath = glob.glob(os.path.join(path, "*", "*", "*.jpg"))

          
        for img_path in filepath:
            self.dataset["inputs"].append(img_path)


    def __len__(self):
        return len(self.dataset["inputs"])

    def __getitem__(self, index):
        
        img_path = self.dataset["inputs"][index]

        img = self.img_reader.read(img_path)
        
        img = torch.Tensor(img)

        img = self.resize(img)

        img = img.expand((3, img.shape[1], img.shape[2]))

        bboxes = random_bbox(params, batch_size=img.size(0))
        img, target = mask_image(img, bboxes, params)

        img = img[0,...]
        target = target[0,...]
        
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
    
def collate_fn(batch):
    """
    Function from:
    https://github.com/pytorch/vision/blob/master/references/detection/utils.py

    """
    return tuple(zip(*batch))

train_transform = None
TRAIN_PATH = "/share/data_drive1/X-Ray/CheXpert/train/"
train_set = CHEXPERT_LR_CROP(path=TRAIN_PATH, trans=train_transform, split="train")
train_loader = DataLoader(train_set, batch_size=params["batch_size"], num_workers=params["num_workers"], 
                         shuffle=False, collate_fn=collate_fn)

val_transform = None
VAL_PATH = "/share/data_drive1/X-Ray/CheXpert/valid/"
val_set = CHEXPERT_LR_CROP(path=VAL_PATH, trans=val_transform, split="val")
val_loader = DataLoader(val_set, batch_size=params["batch_size"], num_workers=params["num_workers"], 
                         shuffle=False, collate_fn=collate_fn)

# Prepare model
class DenseNetEncoder(DenseNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # the initial layer to convolve into 3 channels
        # idea from https://www.kaggle.com/aleksandradeis/bengali-ai-efficientnet-pytorch-starter
        self.input_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

    def forward(self, inputs):
        x = self.input_conv(inputs)
        return self.extract_features(x)
    
    @classmethod
    def load_pretrained(cls):
        model_name = 'densenet121'
        model = densenet121(pretrained=True)
        model_dict = model.state_dict()

        model = torch.nn.Sequential(*(list(model.children())[:-1]))

        return model

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Decoder(nn.Module):

    def __init__(self, encoder, n_channels, out_channels=1):
        super().__init__()

        self.encoder = encoder

        self.up_conv0 = up_conv(n_channels, 512)    # added
        self.up_conv1 = up_conv(512, 256)    
        self.up_conv2 = up_conv(256, 128)    
        self.up_conv3 = up_conv(128, 64)    
        self.up_conv4 = up_conv(64, 32)    # added
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.encoder(x)     # input: 1x128x128, output: 1024x4x4
        x = self.up_conv0(x)
        x = self.up_conv1(x)    
        x = self.up_conv2(x)    
        x = self.up_conv3(x)    
        x = self.up_conv4(x)    
        x = self.final_conv(x)  # input: 64x32x32, output: 1x32x32
       
        return x

encoder = DenseNetEncoder.load_pretrained()
inpainter = Decoder(encoder, n_channels=1024).to(device) #1024: densenet121 output. To do: don't hardcode this.
 
model = inpainter
softmax_layer = torch.nn.Softmax(-1)


# TRAINING
lr = 1e-3
optimizer = optim.Adam(model.parameters(),lr=lr)
criterion = nn.MSELoss()

train_loss = []
val_loss = []
best_loss = 10000
epochs = 50000
    
for epoch_num in range(0, epochs):
    for batch_num, data in tqdm(enumerate(train_loader), "Training..", total=len(train_loader)):

        inp = torch.stack(data[0])
        target = torch.stack(data[1])
        model.train()
        epoch_total_loss = 0

        class_outputs = []
        for inp_aug in inp_augs:
            if inp_aug is None:
                optimizer.zero_grad()

                new_inp = inp.to(device)
                output = model(new_inp)

                batch_loss = criterion(output,target.to(device))

                epoch_total_loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()
            avrg_loss = epoch_total_loss / train_set.__len__()

        labels = []
        predictions = []
        epoch_total_loss = 0

        for (inp, target) in val_loader:
            model.eval()
            with torch.no_grad():
                
                target = torch.stack(list(target), dim=0)[:,0:1,...].to(device)

                inp = torch.stack(list(inp), dim=0)
                batch_prediction = model(inp.to(device))

                batch_loss = criterion(batch_prediction,target)
                epoch_total_loss += batch_loss.item()

            avrg_loss_val = epoch_total_loss / val_set.__len__()
            val_loss.append(avrg_loss_val)

        if epoch_num % 5 == 0:

            torch.save(model.state_dict(), OUTPUT_PATH + "/chex_ctr_acc_epoch_{}.pt".format(epoch_num))
            fig = plt.figure(figsize=(64,32))
            IDX = 0
            batch_prediction2 = batch_prediction.detach().clone()
            batch_prediction2 = batch_prediction2.cpu().numpy()
            inp2 = inp.detach().clone()
            inp2 = inp2.cpu().numpy()
            for idx in range(len(batch_prediction)):

                try:
                    plt.subplot(4, 8, IDX+1)
                    plt.axis('off')
                    plt.imshow(inp2[idx,0,...], cmap="gray")

                    plt.subplot(4, 8, IDX+2)
                    plt.axis('off')
                    batch_prediction3 = batch_prediction2[idx,0,...] + inp2[idx,0,...]
                    plt.imshow(batch_prediction3, cmap="gray")

                    IDX += 2
                except: pass

            fig.savefig(OUTPUT_PATH + "/chex_ctr_img_{}.png".format(epoch_num))
            fig.clf()
            plt.close()

        if avrg_loss_val < best_loss:
            best_loss = avrg_loss_val
            torch.save(model.state_dict(), OUTPUT_PATH + "/best_acc_chex_ctr_epoch_{}.pt".format(epoch_num))

        print("Epoch %d - train-loss=%0.4f - val-loss=%0.4f" % (epoch_num, avrg_loss, avrg_loss_val))

