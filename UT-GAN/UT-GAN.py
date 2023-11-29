import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1) 
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128)
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), 
            nn.BatchNorm2d(256)
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512)
        )
        self.enc5 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4, 2, 1), 
            nn.BatchNorm2d(512)
        )
        self.enc6 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4, 2, 1), 
            nn.BatchNorm2d(512)
        )
        self.enc7 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4, 2, 1), 
            nn.BatchNorm2d(512)
        )
        self.enc8 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, 4, 2, 1), 
            nn.BatchNorm2d(512)
        )
        self.dec1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512 + z_dim, 512, 4, 1, 0), 
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )
        self.dec2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1), 
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )
        self.dec3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5)
        )
        self.dec4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512 * 2, 512, 4, 2, 1), 
            nn.BatchNorm2d(512)
        )
        self.dec5 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512 * 2, 256, 4, 2, 1), 
            nn.BatchNorm2d(256)
        )
        self.dec6 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256 * 2, 128, 4, 2, 1), 
            nn.BatchNorm2d(128)
        )
        self.dec7 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128 * 2, 64, 4, 2, 1), 
            nn.BatchNorm2d(64)
        )
        self.dec8 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64 * 2, 1, 4, 2, 1), 
            nn.Sigmoid()
        )
        self.double_tanh = DoubleTanh()
        self.stc = STC()

    def forward(self, x, z):
        e1 = self.enc1(x) 
        e2 = self.enc2(e1) 
        e3 = self.enc3(e2) 
        e4 = self.enc4(e3) 
        e5 = self.enc5(e4)
        e6 = self.enc6(e5) 
        e7 = self.enc7(e6) 
        e8 = self.enc8(e7)
        
        z = z.view(-1, z.size(1), 1, 1) 
        c = torch.cat([e8, z], dim=1)
        
        d1 = self.dec1(c) 
        d1 = torch.cat([d1, e7], dim=1) 
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], dim=1) 
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], dim=1) 
        d4 = self.dec4(d3) 
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.dec5(d4) 
        d5 = torch.cat([d5, e3], dim=1) 
        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], dim=1) 
        d7 = self.dec7(d6) 
        d7 = torch.cat([d7, e1], dim=1)
        d8 = self.dec8(d7) 

        return d8

# Define the discriminator based on XuNet
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1) 
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            ResBlock(64, 128, 4, 2, 1), 
            SelfAttention(128) 
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            ResBlock(128, 256, 4, 2, 1), 
            SelfAttention(256) 
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            ResBlock(256, 512, 4, 2, 1), 
            SelfAttention(512) 
        )
        self.enc5 = nn.Sequential(
            nn.LeakyReLU(0.2),
            ResBlock(512, 512, 4, 2, 1), 
            SelfAttention(512) 
        )
        self.enc6 = nn.Sequential(
            nn.LeakyReLU(0.2),
            ResBlock(512, 512, 4, 2, 1), 
            SelfAttention(512) 
        )
        self.enc7 = nn.Sequential(
            nn.LeakyReLU(0.2),
            ResBlock(512, 512, 4, 2, 1), 
            SelfAttention(512) 
        )
        self.enc8 = nn.Sequential(
            nn.LeakyReLU(0.2),
            ResBlock(512, 512, 4, 2, 1), 2)
            SelfAttention(512) 
        )
        self.dec1 = nn.Sequential(
            nn.ReLU(),
            ResBlock(512, 512, 4, 2, 1), 
            SelfAttention(512) 
        )
        self.dec2 = nn.Sequential(
            nn.ReLU(),
            ResBlock(512 * 2, 512, 4, 2, 1), 
            SelfAttention(512) 
        )
        self.dec3 = nn.Sequential(
            nn.ReLU(),
            ResBlock(512 * 2, 512, 4, 2, 1), 
            SelfAttention(512) 
        )
        self.dec4 = nn.Sequential(
            nn.ReLU(),
            ResBlock(512 * 2, 512, 4, 2, 1), 
            SelfAttention(512) 
        )
        self.dec5 = nn.Sequential(
            nn.ReLU(),
            ResBlock(512 * 2, 256, 4, 2, 1), 
            SelfAttention(256) 
        )
        self.dec6 = nn.Sequential(
            nn.ReLU(),
            ResBlock(256 * 2, 128, 4, 2, 1), 
            SelfAttention(128) 
        )
        self.dec7 = nn.Sequential(
            nn.ReLU(),
            ResBlock(128 * 2, 64, 4, 2, 1), 
            SelfAttention(64) 
        )
        self.dec8 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64 * 2, 1, 4, 2, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x) 
        e2 = self.enc2(e1) 
        e3 = self.enc3(e2) 
        e4 = self.enc4(e3) 
        e5 = self.enc5(e4) 
        e6 = self.enc6(e5) 
        e7 = self.enc7(e6) 
        e8 = self.enc8(e7) 
        d1 = self.dec1(e8) 
        d1 = torch.cat([d1, e7], dim=1)
        d2 = self.dec2(d1) 
        d2 = torch.cat([d2, e6], dim=1)
        d2 = self.dec2(d2)
        d3 = torch.cat([d3, e5], dim=1) 
        d4 = self.dec4(d3) 
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.dec5(d4) 
        d5 = torch.cat([d5, e3], dim=1) 
        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], dim=1) 
        d7 = self.dec7(d6) 
        d7 = torch.cat([d7, e1], dim=1)
        d8 = self.dec8(d7)

    return d8
