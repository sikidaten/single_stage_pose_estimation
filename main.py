import torch
import argparse
import torch.nn.functional as F
from model.softgatedskipconnection import SoftGatedSkipConnection as Model
from utils.dataset import COCODataset
from utils.core import *
def operate(phase):
    if phase=='train':
        model.train()
        loader=trainloader
    else:
        model.val()
        loader=valloader
    with torch.set_grad_enabled(phase=='train'):
        for idx,(img,center_map,center_mask,kps_offset,kps_weight) in enumerate(loader):
            img=img.to(device)
            center_map=center_map.to(device)
            center_mask=center_mask.to(device)
            kps_offset=kps_offset.to(device)
            kps_weight=kps_weight.to(device)

            output=model(img)
            loss=0
            for outcenter,outkps in output:
                center_loss = F.mse_loss(center_map * center_mask, outcenter * center_mask)
                center_offset_loss = F.smooth_l1_loss(kps_offset * kps_weight, outkps * kps_weight)
                loss=loss+center_loss+center_offset_loss
            loss.backward()
            optmizer.step()
            optmizer.zero_grad()
            print(f'{e:3d}, {idx:3d}/{len(loader)}, {loss*100:.6f}, {phase}')
            addvalue(writer,f'loss:{phase}',loss.item(),e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batchsize',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--size',type=int,default=256)
    parser.add_argument('--stack',default=4,type=int)
    args=parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=Model(args.stack,args.size,3,12).to(device)
    batchsize=args.batchsize
    optmizer=torch.optim.Adam(model.parameters())
    size=(args.size,args.size)
    trainloader=torch.utils.data.DataLoader(COCODataset('../data/coco/train2017','../data/coco/person_keypoints_train2017.json',size),batch_size=batchsize,shuffle=True)
    valloader=torch.utils.data.DataLoader(COCODataset('../data/coco/val2017','../data/coco/person_keypoints_val2017.json',size),batch_size=batchsize,shuffle=True)
    epochs=args.epochs
    writer={}


    for e in range(epochs):
        operate('train')
        operate('val')
        save(writer,model,'data')