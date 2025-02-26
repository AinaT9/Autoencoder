import torch
import torchvision
import torch.nn as nn
from torchvision.models import vgg19_bn


def load_vgg19():
    return vgg19_bn(weights=None, progress=True)



class AutoEncoder(nn.Module):
    def __init__(self, block, activ):
        super().__init__()
        self.encoders = nn.ModuleDict({
            "1": self.block1(),
            "2": self.block2(),
            "3": self.block3(),
            "4": self.block4(),
            "5": self.block5(),
        })

        self.decoders = nn.ModuleDict({
            "1": self.decoder_block1(),
            "2": self.decoder_block2(),
            "3": self.decoder_block3(),
            "4": self.decoder_block4(),
            "5": self.decoder_block5(),
        })
        self.block = block
        self.activ = activ 
    
    def forward(self,x):
        encoder = self.encoders[self.block]
        x = encoder(x)
        for i in range(int(self.block), 0, -1):
            x = self.decoders[str(i)](x) 
        if self.activ: 
            output = self.activ(x)
        else: 
            output = x
        return output
    
    def block1(self): 
        model=load_vgg19()
        return model.features[:7]

    def block2(self): 
        model=load_vgg19()
        return model.features[:14]

    def block3(self):
        model=load_vgg19()
        return model.features[:27]

    def block4(self):
        model=load_vgg19()
        return model.features[:40]

    def block5(self):
        model=load_vgg19()
        return model.features

    def decoder_block1(self): 
        return nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(inplace=True),

                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True),

                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True),

                nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            )
    def decoder_block2(self): 
        return nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(inplace=True),
            )
    def decoder_block3(self): 
        return nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(inplace=True),
            )
    def decoder_block4(self): 
        return nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(inplace=True),
            )
    def decoder_block5(self): 
        return nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(inplace=True),
            )

        



     
     