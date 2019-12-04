import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self,img_ch,output_ch, nb_dates):
        super(U_Net,self).__init__()
        self.nb_dates = nb_dates
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.Conv1 = conv_block(ch_in=img_ch,ch_out=16)
        self.Conv2 = conv_block(ch_in=16,ch_out=32)
        self.Conv3 = conv_block(ch_in=32,ch_out=64)
        self.Conv4 = conv_block(ch_in=64,ch_out=128)
        self.Conv5 = conv_block(ch_in=128,ch_out=256)

        self.Up5 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv5 = conv_block(ch_in=256+(len(nb_dates)-1)*128, ch_out=128)

        self.Up4 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv4 = conv_block(ch_in=128+(len(nb_dates)-1)*64, ch_out=64)
        
        self.Up3 = up_conv(ch_in=64,ch_out=32)
        self.Up_conv3 = conv_block(ch_in=64+(len(nb_dates)-1)*32, ch_out=32)
        
        self.Up2 = up_conv(ch_in=32,ch_out=16)
        self.Up_conv2 = conv_block(ch_in=32+(len(nb_dates)-1)*16, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16,output_ch,kernel_size=1,stride=1,padding=0)

    def encoder(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        return x1,x2,x3,x4,x5

    def forward(self,input):
	#5D input in the form of (Dates, Batchsize, Channels, Height, Width)			
        encodings = []
        for nd in self.nb_dates:
            out = self.encoder(input[self.nb_dates.index(nd)])
            encodings.append(out)

        cats = []
        for enc_level in range(3, -1, -1):
            cat_i = []
            for nd in range(len(self.nb_dates)-1, -1, -1):
                cat_i.append(encodings[nd][enc_level])
            cats.append(cat_i)

        d5 = self.Up5(encodings[-1][-1])
        d5 = torch.cat((d5,   torch.cat(cats[0], dim=1)), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((d4,  torch.cat(cats[1], dim=1)), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((d3, torch.cat(cats[2], dim=1)), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((d2, torch.cat(cats[3], dim=1)), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
