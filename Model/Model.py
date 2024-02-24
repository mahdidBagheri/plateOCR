import torch
from torch import nn
from Config.DatasetConfig import plate_format, image_width, image_height, alphabet_length
from Config.DatasetConfig import prediction_head_num
from torch.nn import functional as F
import numpy as np

class OCRNet(nn.Module):
    def __init__(self, input_dim=(image_height, image_width), nOut=alphabet_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_block2 = ConvBlock(in_channel=3, out_channel=128, kernel_size=(5,5), padding=(2,2), stride=(2,2), act="LReLU", pooling_size=(2,1)) # 128 -> 32
        self.c_block3 = ConvBlock(in_channel=128, out_channel=256, kernel_size=(5,5), padding=(2,2), stride=(2,2), act="LReLU", pooling_size=(2,1)) # 32 -> 8
        self.c_block4 = ConvBlock(in_channel=256, out_channel=256, kernel_size=(3,3), padding=(1,1), stride=(2,2), act="LReLU", pooling_size=(2,1)) # 8 -> 4
        self.c_block5 = ConvBlock(in_channel=256, out_channel=512, kernel_size=(2,2), padding=(0,0), stride=(1,1), act="ReLU", pooling_size=(1,1)) # 4 -> 1

        self.lstm = nn.LSTM(input_size=63, hidden_size=int(prediction_head_num/2), bidirectional=True)
        self.embedding = nn.Linear(512, nOut)

        self.softmax = nn.Softmax()

    def conv_block(self, in_channel, out_channel, kernel_size=(5,5), stride=(2,2), padding=(2,2), act="LReLU", pooling_size=(2,2)):
        block = nn.Sequential()
        conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        bn = nn.BatchNorm2d(out_channel)
        pooling = nn.MaxPool2d(kernel_size=pooling_size)

        if (act == "LReLU"):
            act = nn.LeakyReLU()
        else:
            act = nn.ReLU()

        block.add_module("conv", conv )
        block.add_module("bn", bn )
        block.add_module("pooling", pooling)
        block.add_module("act", act )

        return block


    def forward(self, X):
        X = self.c_block2(X)
        X = self.c_block3(X)
        X = self.c_block4(X)
        X = self.c_block5(X)
        X = torch.squeeze(X)
        X, _ = self.lstm(X)
        X = torch.transpose(X, 1, 2)
        B, H, V = X.size()
        X = X.reshape(B*H,V)
        X = self.embedding(X)
        X = X.view(B, H, -1)
        out = torch.nn.functional.log_softmax(X)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=(5,5), stride=(2,2), padding=(2,2), act="LReLU", pooling_size=(2,2), *args, **kwargs,):
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.pooling = nn.MaxPool2d(kernel_size=pooling_size)

        if (act == "LReLU"):
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.ReLU()

    def forward(self, X):
        X = self.conv(X)
        X = self.bn(X)
        X = self.pooling(X)
        X = self.act(X)
        return X