from torch.utils.data import Dataset
import numpy as np
import random
from scipy.io import loadmat
import h5py
import os
from utils import mask_input

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif", ".mat", ".h5"])

def randcrop(a, b, crop_size):
    [wid, hei, nband]=a.shape
    crop_size1 = crop_size
    Width = random.randint(0, wid - crop_size1 - 1)
    Height = random.randint(0, hei - crop_size1 - 1)
    return a[Width:(Width + crop_size1),  Height:(Height + crop_size1), :], b[Width:(Width + crop_size1),  Height:(Height + crop_size1), :]

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

# Train with croped image patch
class TrainARADDataset(Dataset):
    def __init__(self, data_root, msfa_size=4, patch_size=160, augment=True, add_noise_std=None):
        self.image_filenames = [os.path.join(data_root, x) for x in os.listdir(data_root) if is_image_file(x)]
        self.msfa_size = msfa_size
        self.mosaic_bands = self.msfa_size ** 2
        self.crop_size = calculate_valid_crop_size(patch_size, msfa_size)
        self.augment = augment
        self.add_noise_std = add_noise_std  # None or 10 or 30 or 50

    def data_arguement(self, img, rotTimes, vFlip, hFlip):
        # [c,h,w]
        # Random rotation
        for j in range(rotTimes):
            img_ = np.rot90(img.copy(), axes=(1, 2))
            img = img_.copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, :, ::-1].copy()
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, index):
        with h5py.File(self.image_filenames[index], 'r') as mat:
            img = np.float32(np.array(mat['cube'])) # [c,w,h]
            img = np.transpose(img, (2, 1, 0)) # [c,w,h] -> [h,w,c]
        mat.close()
        clean_img = img.copy()  # [h,w,c]

        # adding gaussian noise
        if self.add_noise_std != None:
            noise = np.float32(np.random.randn(*img.shape) * self.add_noise_std / 255.0)
            img = np.clip(img + noise, a_min=0, a_max=1.0)

        # 切块：拿一张图出来就随机切出一块crop_size大小的patch，并不是一张图切出好多块patches
        img, clean_img = randcrop(img, clean_img, self.crop_size)
        # generate sparse_raw image from GT MSI
        sparse_raw = mask_input(img, msfa_size=self.msfa_size) # [h,w,c]
        # generate raw image
        raw = sparse_raw.sum(axis=2, keepdims=True) # [h,w,c]
        raw = np.transpose(raw, (2, 0, 1))  # [c,h,w]
        sparse_raw = np.transpose(sparse_raw, (2, 0, 1)) # [c,h,w]
        target = np.transpose(clean_img.copy(), (2, 0, 1))  # [c,h,w]
        # argument
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.augment:
            raw = self.data_arguement(raw, rotTimes, vFlip, hFlip)
            sparse_raw = self.data_arguement(sparse_raw, rotTimes, vFlip, hFlip)
            target = self.data_arguement(target, rotTimes, vFlip, hFlip)

        return np.ascontiguousarray(raw), np.ascontiguousarray(sparse_raw), np.ascontiguousarray(target)

    def __len__(self):
        return len(self.image_filenames)

# Test with image
class TestARADDataset(Dataset):
    def __init__(self, data_root, msfa_size=4, add_noise_std=None):
        self.image_filenames = [os.path.join(data_root, x) for x in os.listdir(data_root) if is_image_file(x)]
        self.image_filenames.sort()
        self.msfa_size = msfa_size
        self.mosaic_bands = self.msfa_size ** 2
        self.add_noise_std = add_noise_std  # None or 10 or 30 or 50

    def __getitem__(self, index):
        with h5py.File(self.image_filenames[index], 'r') as mat:
            img = np.float32(np.array(mat['cube'])) # [c,w,h]
            img = np.transpose(img, (2, 1, 0)) # [c,w,h] -> [h,w,c]
        mat.close()
        clean_img = img.copy()  # [h,w,c]

        # adding gaussian noise
        if self.add_noise_std != None:
            noise = np.float32(np.random.randn(*img.shape) * self.add_noise_std / 255.0)
            img = np.clip(img + noise, a_min=0, a_max=1.0)

        # generate sparse_raw image from GT MSI
        sparse_raw = mask_input(img, msfa_size=self.msfa_size) # [h,w,c]
        # generate raw image
        raw = sparse_raw.sum(axis=2, keepdims=True) # [h,w,c]
        raw = np.transpose(raw, (2, 0, 1)) # [c,h,w]
        sparse_raw = np.transpose(sparse_raw, (2, 0, 1)) # [c,h,w]
        target = np.transpose(clean_img.copy(), (2, 0, 1))  # [c,h,w]

        return np.ascontiguousarray(raw), np.ascontiguousarray(sparse_raw), np.ascontiguousarray(target)

    def __len__(self):
        return len(self.image_filenames)





