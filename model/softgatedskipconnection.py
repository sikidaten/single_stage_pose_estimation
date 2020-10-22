import torch
import torch.nn as nn
from model.net.base import Base
from model.net.convbnrelu import ConvBNRelu
from model.net.gatedresidualblock import GatedResidualBlock
class SoftGatedSkipConnection(nn.Module):
    def __init__(self,stack,feature,num_ch,num_key):
        super(SoftGatedSkipConnection, self).__init__()
        self.out_ch= 2 * num_key + 1
        self.stack=stack
        self.prelayer=nn.Sequential(ConvBNRelu(num_ch,feature,7,2,3),nn.MaxPool2d(2),GatedResidualBlock(feature))
        self.bases=nn.ModuleList([Base(depth=4, feature=feature) for i in range(stack)])
        self.out=nn.ModuleList([nn.Conv2d(feature,num_key*2+1,1) for _ in range(stack)])
    def forward(self,x):
        x=self.prelayer(x)
        ret=[]
        for s in range(self.stack):
            x=self.bases[s](x)
            out=self.out[s](x)
            center,kps=torch.split(out, [1, self.out_ch - 1], 1)
            ret+=[torch.sigmoid(center),torch.tanh(kps)]
        return ret

if __name__=='__main__':
    from torchviz import make_dot
    model=SoftGatedSkipConnection(stack=4,feature=256,num_ch=3,num_key=12)
    optim=torch.optim.Adam(model.parameters())
    input=torch.randn(1,3,128,128)
    output=model(input)
    dot=make_dot(output[0][0],params=dict(model.named_parameters()))
    dot.format='png'
    dot.render('graph_image')
    [print(f'{input.shape} -> {output[i][0].shape},{output[i][1].shape}') for i in range(len(output))]
    loss=torch.stack([o[0] for o in output]).mean()
    optim.step()
    optim.zero_grad()