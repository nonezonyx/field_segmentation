import torch
import random
import torchvision.transforms.functional as TF

class PairedRandomRotation(torch.nn.Module):
    def forward(self, image, mask):
        angle = 90 * random.randint(0, 3)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        return image, mask