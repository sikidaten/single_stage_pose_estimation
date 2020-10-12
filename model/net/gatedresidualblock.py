import torch
import torch.nn as nn
import torch.nn.functional as F

from model.net.convbnrelu import ConvBNRelu
class GatedResidualBlock(nn.Module):
    def __init__(self,feature):
        super(GatedResidualBlock, self).__init__()
        features=[feature,feature//2,feature//4]
        self.cbr1=ConvBNRelu(features[0],features[1],3,1,1)
        self.cbr2=ConvBNRelu(features[1],features[2],3,1,1)
        self.cbr3=ConvBNRelu(features[2],features[2],3,1,1)
        self.alpha=nn.Parameter(torch.ones(1,features[0],1,1)/2)

    def forward(self,x):
        x0=x
        x1=self.cbr1(x)
        x2=self.cbr2(x1)
        x3=self.cbr3(x2)

        x=torch.cat([x1,x2,x3],1)
        x=F.relu(x0*self.alpha+x)

        return x
