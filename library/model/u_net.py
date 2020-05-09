import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(DepthPrediction, self).__init__()
        self.conv64 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Relu(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Relu(),
        )
        self.mp1 = nn.MaxPool2d(2)

        self.conv128 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Relu(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Relu(),
        )
        self.mp2 = nn.MaxPool2d(2)

        self.conv256 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Relu(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Relu(),
        )
        self.mp3 = nn.MaxPool2d(2)

        self.conv512 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Relu(),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Relu(),
        )
        self.mp4 = nn.MaxPool2d(2)

        self.conv1024 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.Relu(),

            nn.Conv2d(1024, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(1024),
            nn.Relu(),
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.Relu(),

            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.Relu(),
        )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.Relu(),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Relu(),
        )

        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.Relu(),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Relu(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Relu(),

            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.Relu(),
        )

    def forward(self, x):
        out = self.conv64(x)
        out_conv64 = out 
        out = self.mp1(out)
        out = self.conv128(out)
        out_conv128 = out
        out = self.mp2(out)
        out = self.conv256(out)
        out_conv256 = out 
        out = self.mp3(out)
        out = self.conv512(out)
        out_conv512 = out 
        out = self.mp4(out)
        out = self.conv1024(out)
        out = torch.cat([self.upsample1(out), out_conv512], dim=1)
        out = torch.cat([self.upsample2(out), out_conv256, dim=1)
        out = torch.cat([self.upsample3(out), out_conv128, dim=1)
        out = torch.cat([self.upsample4(out), out_conv64, dim=1)

        return out 
