import torch
import torch.nn as nn
import torch.nn.functional as F

class decoder(nn.Module):
    def __init__(self,outlevel,sec,features):
        super(decoder, self).__init__()
