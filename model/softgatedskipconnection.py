import torch
import torch.nn as nn
from model.net.base import Base
class SoftGatedSkipConnection(nn.Module):
    def __init__(self,stack,feature,num_ch,num_key):
        super(SoftGatedSkipConnection, self).__init__()
        self.out_ch= 2 * num_key + 1
        self.stack=stack
        self.bases=nn.ModuleList([Base(4, feature, num_ch, self.out_ch) for _ in range(stack)])
        self.convs=nn.ModuleList([nn.Conv2d(self.out_ch, num_ch, 1) for _ in range(stack)])

    def forward(self,x):
        ret=[]
        for s in range(self.stack):
            x=self.bases[s](x)
            _x=torch.sigmoid(x)
            ret+=[torch.split(_x, [1, self.out_ch - 1], 1)]
            x=self.convs[s](x)
        return ret

if __name__=='__main__':
    model=SoftGatedSkipConnection(4,256,3,18)
    optim=torch.optim.Adam(model.parameters())
    input=torch.randn(1,3,128,128)
    output=model(input)
    print(f'{input.shape} -> {output[0].shape}')
    loss=torch.stack(output).mean()
    optim.step()
    optim.zero_grad()