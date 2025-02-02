import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class RandomHue(nn.Module):
    def __init__(self, hue_factor=(-0.2, 0.2)):
        super(RandomHue, self).__init__()
        self.hue_factor = hue_factor

    def forward(self, img):
        factor = torch.empty(1).uniform_(*self.hue_factor).item()
        return TF.adjust_hue(img, factor)


class RandomSaturation(nn.Module):
    def __init__(self, saturation_range=(0.8, 1.2)):
        super(RandomSaturation, self).__init__()
        self.saturation_range = saturation_range

    def forward(self, img):
        factor = torch.empty(1).uniform_(*self.saturation_range).item()
        return TF.adjust_saturation(img, factor)


class RandomBrightness(nn.Module):
    def __init__(self, brightness_range=(0.8, 1.2)):
        super(RandomBrightness, self).__init__()
        self.brightness_range = brightness_range

    def forward(self, img):
        factor = torch.empty(1).uniform_(*self.brightness_range).item()
        return TF.adjust_brightness(img, factor)


class RandomContrast(nn.Module):
    def __init__(self, contrast_range=(0.8, 1.2)):
        super(RandomContrast, self).__init__()
        self.contrast_range = contrast_range

    def forward(self, img):
        factor = torch.empty(1).uniform_(*self.contrast_range).item()
        return TF.adjust_contrast(img, factor)


class AddNoise(nn.Module):
    def __init__(self, mean=0.0, std=0.05):
        super(AddNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, img):
        noise = torch.randn_like(img) * self.std + self.mean
        return img + noise