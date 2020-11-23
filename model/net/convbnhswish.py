import torch
import torch.nn as nn
import torch.nn.functional as F

def hardswish(x):
    if x<-3:
        return 0
    elif x>=3:
        return x
    else:
        return x*(x+3)/6
class ConvBnHswish(nn.Module):
    def __init__(self,in_feature,out_feature,kernel,stride=1,padding=0):
        super(ConvBnHswish,self).__init__()
        self.conv=nn.Conv2d(in_feature,out_feature,kernel,stride,padding)
        self.bn=nn.BatchNorm2d(out_feature)
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=hardswish(x)
        return x