import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from masks import center_crop_masks, perlin_masks


class MNISTDataset(Dataset):
    def __init__(self, split="train", digits=range(10), mask_type="perlin", mask_area=0.2, mask_scale=1.0):
        """
        split: in ['train', 'test']
               which data split to use
        digits: subset of [0,1,2,3,4,5,6,7,8,9]
                which digits to include in the dataset
        mask_type: in ['center', 'perlin']
                   which mask type to use
        mask_area: between 0 and 1
                   how much of the image to be masked
        mask_scale: at least 1
                    frequency of noise for perlin mask
                    1-4 -> big blobs, 8-64 -> fine texture
        """

        assert split in ["train", "test"]
        assert len(digits) > 0
        for d in digits:
            assert d in range(10)
        assert mask_type in ["center", "perlin"]
        assert 0 <= mask_area <= 1
        assert mask_scale >= 1

        self.mask_type = mask_type
        self.mask_area = mask_area
        self.mask_scale = mask_scale

        self.imgs = []
        for d in digits:
            img_paths = sorted(glob(f"mnist/{split}/{d}/*.jpg"))
            for img_path in img_paths:
                img = Image.open(img_path).convert('L')
                self.imgs.append(img)

        self.w = self.imgs[0].size[0]
        self.h = self.imgs[0].size[1]

        if mask_type == "center":
            self.masks = center_crop_masks(self.h, self.w, self.mask_area, batch_size=1).unsqueeze(1)
        else:
            self.masks = perlin_masks(self.h, self.w, self.mask_scale, self.mask_area, batch_size=10000, seed=2840).unsqueeze(1)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1), # [0,1] -> [-1,1]
        ])

    
    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        img = self.transform(self.imgs[idx])

        mask_idx = torch.randint(0, self.masks.shape[0], (1,)).item()
        mask = self.masks[mask_idx]

        return img, mask

