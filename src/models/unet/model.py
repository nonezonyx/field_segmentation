import torch
import torch.nn as nn
from src.models.unet.blocks import EncoderBlock, DecoderBlock

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features = 64):
        super(UNet, self).__init__()
        self.features = features

        self.enc0 = EncoderBlock(in_channels, features)
        self.enc1 = EncoderBlock(features, features * 2)
        self.enc2 = EncoderBlock(features * 2, features * 4, depth=3)
        self.enc3 = EncoderBlock(features * 4, features * 8, depth=3)

        self.bottleneck_enc = EncoderBlock(features * 8, features * 8, depth=3)
        self.bottleneck_dec = DecoderBlock(features * 8, features * 8, depth=3)

        self.dec0 = DecoderBlock(features * 8 + features * 8, features * 4, depth=3)
        self.dec1 = DecoderBlock(features * 4 + features * 4, features * 2, depth=3)
        self.dec2 = DecoderBlock(features * 2 + features * 2, features)
        self.dec3 = DecoderBlock(features + features, out_channels, classification=True)

    def forward(self, x):
        e0, ind0 = self.enc0(x)
        e1, ind1 = self.enc1(e0)
        e2, ind2 = self.enc2(e1)
        e3, ind3 = self.enc3(e2)

        b0, indb = self.bottleneck_enc(e3)
        b1 = self.bottleneck_dec(b0, indb)

        d0 = self.dec0(torch.cat((e3, b1), dim=1), torch.cat((ind3, ind3), 1))
        d1 = self.dec1(torch.cat((e2, d0), dim=1), torch.cat((ind2, ind2), 1))
        d2 = self.dec2(torch.cat((e1, d1), dim=1), torch.cat((ind1, ind1), 1))
        output = self.dec3(torch.cat((e0, d2), dim=1), torch.cat((ind0, ind0), 1))

        return output