import torch
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T

class PairedRandomCrop(torch.nn.Module):
    def __init__(self, crop_size=512):
        super().__init__() 
        self.crop_size = crop_size

    def forward(self, image, mask):
        crop_params = T.RandomCrop.get_params(image, (self.crop_size, self.crop_size))
        image = TF.crop(image, *crop_params)
        mask = TF.crop(mask, *crop_params)
        return image, mask