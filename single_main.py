from model.softgatedskipconnection import SoftGatedSkipConnection
from utils.mpii import MPIIDataset
if __name__=='__main__':
    model=SoftGatedSkipConnection(stack=4,feature=128,num_ch=3,num_key=16)
