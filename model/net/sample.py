import torch
import torch.nn as nn
import torch.nn.functional as F

class Sample(nn.Module):
    def __init__(self,sec,feature):
        super(Sample, self).__init__()
        self.conv=nn.Conv2d(feature,feature,1)
        self.bn=nn.BatchNorm2d(feature)
        assert sec in ['up','down']
        self.sec=sec
    def forward(self,x):
        if self.sec=='up':
            x=F.upsample(x,scale_factor=2)
        else:
            x=F.max_pool2d(x)
        x=self.conv(x)
        x=self.bn(x)
        x=F.relu(x)
        return x