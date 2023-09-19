import torch
from model import ResidualBlock, ResNet_gabor_cat2
from torchvision import utils, transforms
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import math
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import time
import pandas as pd

use_gpu = torch.cuda.is_available()


def cal_orientation(Qxx, Qxy):
    S = 2 * torch.sqrt(Qxx ** 2 + Qxy ** 2)
    Qxx = Qxx / S
    Qxy = Qxy / S
    # Evaluate nx and ny from normalized Qxx and Qxy
    nx = torch.sqrt(Qxx + 0.5)
    ny = torch.abs(Qxy / nx)  # This ensures ny>0, such that theta lies between 0 to pi
    nx = nx * torch.sign(Qxy)
    theta = (torch.atan2(ny, nx))
    orientation = theta / math.pi  # do not multiply 255 here, save_image function will multiply
    return orientation


def load_model(path, channels, layers, lam):
    lam = lam.split(',')
    lam = [int(i) for i in lam]
    model = ResNet_gabor_cat2(ResidualBlock, layers, channels, 8, 8, 11, pc=True, lam=lam)

    if use_gpu:
        model = torch.nn.DataParallel(model)
        print('multiple GPUs!')
        model.cuda()
    checkpoint = torch.load(path)  # , map_location=torch.device('cpu')
    print('loading checkpoint', path)
    model.load_state_dict(checkpoint['model'], strict=False)

    return model


class pipline_data(Dataset):
    def __init__(self, csv_path, size=None):
        data_df = pd.read_csv(csv_path)
        self.rootdir = data_df['root']
        self.files = data_df['file']
        self.tag = data_df['tag']
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        root = self.rootdir[item]
        file = self.files[item]
        tag = self.tag[item]
        path = os.path.join(root, file)
        img = Image.open(path).convert('L')
        high1 = np.max(img)
        high2 = np.percentile(img, 99)
        # hi = 0.01 * high1 + 0.99 * high2
        hi = 0.2 * high1 + 0.8 * high2
        # hi = high1
        lo = np.min(img)
        input = (img - lo) / (hi - lo)
        input = transforms.ToTensor()(input)
        if self.size is not None:
            input = transforms.CenterCrop(self.size)(input)
        # print(input.shape)
        input = input.float()

        return input, tag


def process_img(file_path, size=None):
    img = Image.open(file_path).convert('L')
    high1 = np.max(img)
    high2 = np.percentile(img, 99)
    # hi = 0.01 * high1 + 0.99 * high2
    hi = 0.2 * high1 + 0.8 * high2
    # hi = high1
    lo = np.min(img)
    input = (img - lo) / (hi - lo)
    input = transforms.ToTensor()(input)
    if size is not None:
        input = transforms.CenterCrop(size)(input)
    print(input.shape)
    input = input.float()

    return input.unsqueeze(0)


channels = [64, 128, 256, 512]
layers = [3, 4, 6, 3]
lam = '1,15'
cat = 2
att = 0

outdir = 'evaluation_output/'

if not os.path.exists(outdir):
    os.mkdir(outdir)

data = pipline_data('data/test_samples_various_data.csv')



dataloader = DataLoader(data, batch_size=1, shuffle=False)

print('Loading original model..........')
path = '/work/yunruili/unconfined_orientation/save_weights/320_gr34_aug_smooth.epoch.187'
model = load_model(path, channels, layers, lam=lam)
model.eval()

for i, (data, tag) in enumerate(dataloader):
    t = time.time()

    with torch.no_grad():
        if use_gpu:
            data = data.cuda()
        else:
            data = Variable(data)
        with autocast():
            pred = model(data)

    print('total time: ', time.time() - t)

    utils.save_image(
        pred,
        os.path.join(outdir, '%s_orientation.png' % tag), nrow=1)
    utils.save_image(
        data,
        os.path.join(outdir, '%s_raw.png' % tag), nrow=1)



# for idx in range(0,len(exp)):
#     print('Loading pretrained model.....', idx)
#     path = os.path.join(rootpath, exp[idx])
#     model = load_model(path, channels[idx], layers[idx], lam=lam, cat=cat, att=att)
#     model.eval()
#
#     for data, tag in dataloader:
#
#         t = time.time()
#
#         with torch.no_grad():
#             if use_gpu:
#                 data = data.cuda()
#             else:
#                 data = Variable(data)
#             with autocast():
#                 pred = model(data)
#
#         print('total time: ', time.time() - t)
#
#         utils.save_image(
#             pred,
#             os.path.join(outdir, '%s_%s_orientation_250.png' % (exp[idx], tag[0])), nrow=1)
#         # utils.save_image(
#         #     data,
#         #     os.path.join(outdir, '%s_raw.png' % (tag[0])), nrow=1)

