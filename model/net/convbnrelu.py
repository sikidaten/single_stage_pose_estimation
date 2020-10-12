import torch.nn as nn
import torch.nn.functional as F

class ConvBNRelu(nn.Module):
    def __init__(self,in_feature,out_feature,kernel,stride=1,padding=0):
        super(ConvBNRelu, self).__init__()
        self.conv=nn.Conv2d(in_feature,out_feature,kernel,stride,padding,bias=False)
        self.bn=nn.BatchNorm2d(out_feature)

    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=F.relu(x)
        return x