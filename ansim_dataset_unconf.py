import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageOps
import PIL
import torch, torchvision
from torchvision import transforms, datasets, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math
import random
import os
import numpy as np

import torchvision.transforms.functional as TF


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    mask = mask.astype(int)
    return mask

mask = create_circular_mask(128,128)


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        # print(self.mask_count)
        # print(mask_idx)
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class ansimDataset_mix_smoothmask_s(Dataset):
    """input- retardance, output- orientation, mask"""
    """rotate angel from 1 - 359 """
    """if rotate, output resized to 320 """
    ''' return s for each target image '''

    def __init__(self,img_list_csv, seq_csv, root_dir_input, root_dir_target, step, gap, rotate_angel = 1, rotate=True, image_size=128, rand_range=10, patch_size = 10, model_patch_size=5, mask_ratio=0.5):
        """
        Args:
            image_csv (string): Path to the csv file with image path.
            seq_csv (string): Path to the csv file with indices of heads of sequence.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_list_csv = pd.read_csv(img_list_csv, header=None)
        # self.img_list_target = pd.read_csv(img_list_csv_target, header=None)
        self.seq_list = pd.read_csv(seq_csv, header=None)
        # self.seq_list_target = pd.read_csv(seq_csv_target, header=None)

        self.root_dir_input = root_dir_input
        self.root_dir_target = root_dir_target
        self.rotate = rotate
        self.step = step
        self.gap = gap
        self.rotate_angel = rotate_angel
        self.mask = mask
        if rotate:
            self.image_size = int(image_size * np.sqrt(2)/2)
            self.image_size = int(self.image_size // patch_size) * patch_size
            # print(self.image_size)

            self.image_size_to_mask = 320  # to resize image so that the image size fits the restrictions of the patch size, encoder downsampling, etc.

        else:
            self.image_size = image_size
            self.image_size_to_mask = image_size

        self.mask_generator = MaskGenerator(input_size=self.image_size_to_mask,
                                            mask_patch_size=patch_size,
                                            model_patch_size=model_patch_size,
                                            mask_ratio=mask_ratio)

        self.rand_range = rand_range

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        # print(idx)
        seq_head = self.seq_list.iloc[idx,0]
        # seq_head_target = self.seq_list_target.iloc[idx, 0]
        randint = random.randint(0,self.rand_range)
        seq_head = seq_head + randint
        # seq_head_target = seq_head_target + randint
        seq_input = torch.empty(self.step//2, 1, self.image_size_to_mask, self.image_size_to_mask, dtype=torch.float) #self.image_size, self.image_size,
        seq_target = torch.empty(self.step//2, 1, self.image_size_to_mask, self.image_size_to_mask, dtype=torch.float)
        seq_s = torch.empty(self.step//2, 1, self.image_size_to_mask, self.image_size_to_mask, dtype=torch.float)

        for i in np.arange(self.step//2):
            img_idx = seq_head + self.gap * i
            img_idx_target = img_idx #+ self.step//2    # for sequence prediction
            # print('input', img_idx)
            # print('target', img_idx_target)
            img_name_input = os.path.join(self.root_dir_input, self.img_list_csv.iloc[img_idx,0])
            img_name_target = os.path.join(self.root_dir_target, self.img_list_csv.iloc[img_idx_target, 0])
            # print(img_name_input)
            # print(img_name_target)
            image_input = Image.open(img_name_input)
            image_input = image_input.convert('L')

            image_target = Image.open(img_name_target)
            image_target = image_target.convert('L')

            # image_resized = image_resized * self.mask
            # image_tensor = torch.from_numpy(image_resized)
            # seq[i][0] = image_tensor

            if self.rotate:
                ## rotate and crop
                image_target = TF.rotate(image_target, self.rotate_angel)
                image_input = TF.rotate(image_input, self.rotate_angel)
                image_target = transforms.CenterCrop(self.image_size)(image_target)
                image_input = transforms.CenterCrop(self.image_size)(image_input)  #self.image_size

                ### resize to 320*320 to use the smooth model simmim
                image_input = TF.resize(image_input, (self.image_size_to_mask, self.image_size_to_mask),
                                        interpolation=0)
                image_target = TF.resize(image_target, (self.image_size_to_mask, self.image_size_to_mask),
                                        interpolation=0)
                ### interpolation, NEAREST, BILINEAR, BICUBIC

            else:
                image_target = torchvision.transforms.functional.resize(image_target,
                                                                           (self.image_size, self.image_size),
                                                                           interpolation=2)
                image_input = torchvision.transforms.functional.resize(image_input,
                                                                           (self.image_size, self.image_size),
                                                                           interpolation=2)
            #####   The ToTensor transformation will normalize the uint8 input image by dividing by 255   #####
            seq_input[i][0] = transforms.ToTensor()(image_input)
            seq_target[i][0] = transforms.ToTensor()(image_target)

            #generate s
            image_target = transforms.ToTensor()(image_target)
            image_target = image_target * math.pi

            target_qxx = torch.pow(torch.cos(image_target), 2) - 1 / 2
            target_qxy = torch.cos(image_target) * torch.sin(image_target)
            qxx = transforms.GaussianBlur(kernel_size=(5, 5), sigma=1.0)(target_qxx)
            qxy = transforms.GaussianBlur(kernel_size=(5, 5), sigma=1.0)(target_qxy)
            s = 2 * torch.sqrt(torch.pow(qxx, 2) + torch.pow(qxy, 2))
            seq_s[i][0] = s

            mask = self.mask_generator()

        return seq_input, seq_target, seq_s, mask, img_idx


class ansimDataset_mix_smoothmask_s_randnoise(Dataset):
    """input- retardance, output- orientation, mask"""
    """rotate angel from 1 - 359 """
    """if rotate, output resized to 320 """
    ''' return s for each target image '''
    ''' add random line noise to simulate extra microtubule '''

    def __init__(self,img_list_csv, seq_csv, root_dir_input, root_dir_target, step, gap, rotate_angel = 1, rotate=True,
                    noise_pct = 0.03, noise_len = 30,
                 image_size=128, rand_range=10, patch_size = 10, model_patch_size=5, mask_ratio=0.5):
        """
        Args:
            image_csv (string): Path to the csv file with image path.
            seq_csv (string): Path to the csv file with indices of heads of sequence.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_list_csv = pd.read_csv(img_list_csv, header=None)
        # self.img_list_target = pd.read_csv(img_list_csv_target, header=None)
        self.seq_list = pd.read_csv(seq_csv, header=None)
        # self.seq_list_target = pd.read_csv(seq_csv_target, header=None)

        self.root_dir_input = root_dir_input
        self.root_dir_target = root_dir_target
        self.rotate = rotate
        self.step = step
        self.gap = gap
        self.rotate_angel = rotate_angel
        self.mask = mask
        if rotate:
            self.image_size = int(image_size * np.sqrt(2)/2)
            self.image_size = int(self.image_size // patch_size) * patch_size
            # print(self.image_size)

            self.image_size_to_mask = 320  # to resize image so that the image size fits the restrictions of the patch size, encoder downsampling, etc.

        else:
            self.image_size = image_size
            self.image_size_to_mask = image_size

        self.mask_generator = MaskGenerator(input_size=self.image_size_to_mask,
                                            mask_patch_size=patch_size,
                                            model_patch_size=model_patch_size,
                                            mask_ratio=mask_ratio)

        self.rand_range = rand_range

        self.noise_pct = noise_pct

        self.noise_len = noise_len


    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        # print(idx)
        seq_head = self.seq_list.iloc[idx,0]
        # seq_head_target = self.seq_list_target.iloc[idx, 0]
        randint = random.randint(0,self.rand_range)
        seq_head = seq_head + randint
        # seq_head_target = seq_head_target + randint
        seq_input = torch.empty(self.step//2, 1, self.image_size_to_mask, self.image_size_to_mask, dtype=torch.float) #self.image_size, self.image_size,
        seq_target = torch.empty(self.step//2, 1, self.image_size_to_mask, self.image_size_to_mask, dtype=torch.float)
        seq_s = torch.empty(self.step//2, 1, self.image_size_to_mask, self.image_size_to_mask, dtype=torch.float)

        for i in np.arange(self.step//2):
            img_idx = seq_head + self.gap * i
            img_idx_target = img_idx #+ self.step//2    # for sequence prediction
            # print('input', img_idx)
            # print('target', img_idx_target)
            img_name_input = os.path.join(self.root_dir_input, self.img_list_csv.iloc[img_idx,0])
            img_name_target = os.path.join(self.root_dir_target, self.img_list_csv.iloc[img_idx_target, 0])
            # print(img_name_input)
            # print(img_name_target)
            image_input = Image.open(img_name_input)
            image_input = image_input.convert('L')

            image_target = Image.open(img_name_target)
            image_target = image_target.convert('L')

            # image_resized = image_resized * self.mask
            # image_tensor = torch.from_numpy(image_resized)
            # seq[i][0] = image_tensor

            if self.rotate:
                ## rotate and crop
                image_target = TF.rotate(image_target, self.rotate_angel)
                image_input = TF.rotate(image_input, self.rotate_angel)
                image_target = transforms.CenterCrop(self.image_size)(image_target)
                image_input = transforms.CenterCrop(self.image_size)(image_input)  #self.image_size

                ### resize to 320*320 to use the smooth model simmim
                image_input = TF.resize(image_input, (self.image_size_to_mask, self.image_size_to_mask),
                                        interpolation=0)
                image_target = TF.resize(image_target, (self.image_size_to_mask, self.image_size_to_mask),
                                        interpolation=0)
                ### interpolation, NEAREST, BILINEAR, BICUBIC

            else:
                image_target = torchvision.transforms.functional.resize(image_target,
                                                                           (self.image_size, self.image_size),
                                                                           interpolation=2)
                image_input = torchvision.transforms.functional.resize(image_input,
                                                                           (self.image_size, self.image_size),
                                                                           interpolation=2)

            # add noise
            #####   The ToTensor transformation will normalize the uint8 input image by dividing by 255   #####
            image_target = transforms.ToTensor()(image_target)
            image_input = transforms.ToTensor()(image_input)

            n_noise_pt = int(self.noise_pct * image_input.shape[1])

            for pt in range(n_noise_pt):
                x = np.random.randint(0,image_input.shape[1])
                y = np.random.randint(0,image_input.shape[2])

                noise_len = self.noise_len + np.random.randint(-10, 10)

                x_end = np.random.randint(max(x - noise_len, 0), min(x + noise_len, image_input.shape[1]-1))
                y_end = np.random.randint(max(y - noise_len, 0), min(y + noise_len, image_input.shape[2]-1))

                if x_end == x:  # vertical line
                    if y<=y_end:
                        image_input[:, x, y:y_end] = 1
                    else:
                        image_input[:, x, y_end:y] = 1

                else:
                    tan = (y_end - y) / (x_end - x)
                    if x>x_end:
                        tmp = x
                        x = x_end
                        x_end = tmp
                    for x_i in range(x, x_end):
                        y_i = min(image_input.shape[2]-1, int(y_end + (x_i - x_end) * tan))
                        image_input[:, x_i, y_i] = 1
                    image_input[:, x_end, y_end] = 1

            seq_input[i][0] = image_input
            seq_target[i][0] = image_target

            #generate s
            image_target = image_target * math.pi

            target_qxx = torch.pow(torch.cos(image_target), 2) - 1 / 2
            target_qxy = torch.cos(image_target) * torch.sin(image_target)
            qxx = transforms.GaussianBlur(kernel_size=(5, 5), sigma=1.0)(target_qxx)
            qxy = transforms.GaussianBlur(kernel_size=(5, 5), sigma=1.0)(target_qxy)
            s = 2 * torch.sqrt(torch.pow(qxx, 2) + torch.pow(qxy, 2))
            seq_s[i][0] = s

            mask = self.mask_generator()

        return seq_input, seq_target, seq_s, mask, img_idx

