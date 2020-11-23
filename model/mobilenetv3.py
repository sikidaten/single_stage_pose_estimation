import torch
import torch.nn as nn
from model.net.convbnhswish import  ConvBnHswish
from model.net.convbnhswish import hardswish
HS=hardswish
RE=nn.ReLU()

small_params = [
    [3, 16, 16, True, RE, 2],
    [3, 72, 24, False, RE, 2],
    [3, 88, 24, False, RE, 1],
    [5, 96, 40, True, HS, 2],
    [5, 240, 40, True, HS, 1],
    [5, 240, 40, True, HS, 1],
    [5, 120, 48, True, HS, 1],
    [5, 144, 48, True, HS, 1],
    [5, 288, 96, True, HS, 2],
    [5, 576, 96, True, HS, 1],
    [5, 576, 96, True, HS, 1],
]
small_params_layers = [0, 2, 7, 10]
class MobileNetV3(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(MobileNetV3, self).__init__()
        self.preconv=ConvBnHswish(in_ch,16,3,2)
        self.MBConvBlocks=nn.ModuleList([MBConvBlock() for param in small_params ])
        self.lastconv=nn.Conv2d(512,out_ch,1)

    def forward(self,x):
        x=self.preconv(x)
        features=[]
        for idx,layer in enumerate(self.MBConvBlocks):
            x=layer(x)
            if idx in small_params_layers:
                features+=[x]
        