import torch
from thop import profile

from model.unet import UNet
from model.softgatedskipconnection import SoftGatedSkipConnection
from model.hourglass import HourglassNet
model=HourglassNet(num_stacks=8,num_classes=25)
# model=SoftGatedSkipConnection(8,128,3,25)
# model = UNet()
input = torch.randn(1, 3, 256,256)
macs, params = profile(model, inputs=(input, ))
print(f'MACS:{macs//(1000**2)}M, FLOPs:{params//(1000**2)}M')