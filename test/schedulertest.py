import torch
import torch.nn as nn
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.fc=nn.Linear(3,3)
    def forward(self,x):
        return self.fc(x)
model=Model()
optimizer=torch.optim.Adam(model.parameters())
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.5)
for e in range(20):
    for i in range(10):
        optimizer.step()
        print(e,i,scheduler.get_lr())
    scheduler.step()