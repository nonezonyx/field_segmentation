import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import cv2
import numpy as np
import os
from pathlib import Path

class SatelliteDataset(Dataset):
    def __init__(self, images_dir, masks_dir, paired_transform=None, image_transform=None):
        self.paired_transform = paired_transform
        self.image_transform = image_transform
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.eval = False
        
        self.image_files = sorted(self.images_dir.glob("*.png"))
        self.mask_files = sorted(self.masks_dir.glob("*.png"))
        
        assert len(self.image_files) == len(self.mask_files), "Mismatched number of images/masks"
        for img_path, mask_path in zip(self.image_files, self.mask_files):
            assert img_path.name == mask_path.name, "Mismatched filename pairs"
        
    def __len__(self):
        return len(self.image_files)

    def eval():
        self.eval = True

    def train():
        self.eval = False

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(str(mask_path)), cv2.COLOR_BGR2RGB)
        
        image = TF.to_pil_image(image)
        mask = TF.to_pil_image(mask)
        if not self.eval:
            if self.paired_transform is not None:
                image, mask = self.paired_transform(image, mask)
            if self.image_transform is not None:
                image = self.image_transform(image)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask