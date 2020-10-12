import torch.nn as nn
import torch.nn.functional as F
from model.net.gatedresidualblock import GatedResidualBlock

class Base(nn.Module):
    def __init__(self,depth,feature,num_ch,num_key):
        super(Base, self).__init__()
        self.depth=depth
        self.conv0=nn.Conv2d(num_ch,feature,1)
        self.enc_res=nn.ModuleList([GatedResidualBlock(feature) for _ in range(depth)])
        self.skip=nn.ModuleList([nn.Conv2d(feature,feature,1) for _ in range(depth)])
        self.dec_res=nn.ModuleList([GatedResidualBlock(feature) for _ in range(depth)])
        self.lastconv=nn.Conv2d(feature,num_key,1)

    def forward(self,x):
        x=self.conv0(x)
        skip=[]
        for d in range(self.depth):
            x=self.enc_res[d](x)
            skip+=[self.skip[d](x)]
            x=F.max_pool2d(x,kernel_size=2)

        for d in range(self.depth):
            x=self.dec_res[d](x)
            x=F.upsample(x,scale_factor=2)
            x=x+skip[self.depth-d-1]

        x=self.lastconv(x)
        return x
