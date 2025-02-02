import torch
import random
import torchvision.transforms.functional as TF

class PairedRandomHorizontalFlip(torch.nn.Module):
    def __init__(self, prob=0.5):
        super().__init__() 
        self.prob = prob

    def forward(self, image, mask):
        if random.random() > self.prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class PairedRandomVerticalFlip(torch.nn.Module):
    def __init__(self, prob=0.5):
        super().__init__() 
        self.prob = prob

    def forward(self, image, mask):
        if random.random() > self.prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask