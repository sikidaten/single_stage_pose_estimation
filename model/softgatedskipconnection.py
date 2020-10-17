import torch
import torch.nn as nn
from model.net.base import Base
class SoftGatedSkipConnection(nn.Module):
    def __init__(self,stack,feature,num_ch,num_key):
        super(SoftGatedSkipConnection, self).__init__()
        self.out_ch= 2 * num_key + 1
        self.stack=stack
        self.bases=nn.ModuleList([Base(depth=4, feature=feature, num_ch=num_ch if i==0 else self.out_ch, num_key=self.out_ch) for i in range(stack)])

    def forward(self,x):
        ret=[]
        for s in range(self.stack):
            x=self.bases[s](x)
            ret+=[torch.split(x, [1, self.out_ch - 1], 1)]
        return ret

if __name__=='__main__':
    model=SoftGatedSkipConnection(stack=4,feature=256,num_ch=3,num_key=12)
    optim=torch.optim.Adam(model.parameters())
    input=torch.randn(1,3,128,128)
    output=model(input)
    [print(f'{input.shape} -> {output[i][0].shape},{output[i][1].shape}') for i in range(len(output))]
    loss=torch.stack([o[0] for o in output]).mean()
    optim.step()
    optim.zero_grad()