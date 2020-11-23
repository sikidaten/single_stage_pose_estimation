import argparse
import os

import torch.nn.functional as F
import torchvision.transforms as T
from multiprocessing import cpu_count
from model.softgatedskipconnection import SoftGatedSkipConnection
from model.hourglass import HourglassNet
from utils.core import *
from utils.dataset import COCODataset
from utils.decoder import decoder
from utils.eval import cocoevaluate
from utils.utils import makeresultimg
from utils.kfac import KFAC


def operate(phase):
    resultslist = []
    if phase == 'train':
        model.train()
        loader = trainloader
    else:
        model.eval()
        loader = valloader
    with torch.set_grad_enabled(phase == 'train'):
        for idx, (img, center_map, center_mask, kps_offset, kps_weight, img_id) in enumerate(loader):
            B, C, H, W = img.shape
            img = img.to(device)
            center_map = center_map.to(device)
            center_mask = center_mask.to(device)
            kps_offset = kps_offset.to(device)
            kps_weight = kps_weight.to(device)

            output = model(img)
            center_losses=0
            center_offset_losses=0
            for outcenter, outkps in output:
                # sigmoid here
                center_loss = F.mse_loss(center_map * center_mask, outcenter * center_mask,reduction='sum')
                center_offset_loss = F.smooth_l1_loss(kps_offset * kps_weight, outkps * kps_weight,reduction='sum')
                center_losses=center_losses+center_loss/(B*args.stack)
                center_offset_losses=center_offset_losses+center_offset_loss/(B*args.stack)
            loss = (args.beta*center_offset_losses+center_losses)
            print(f'{e:3d}, {idx:3d}/{len(loader)}, {loss.item()/args.stack:3.6f}, {phase}')
            addvalue(writer, f'loss:{phase}', loss.item(), e)
            addvalue(writer, f'centerloss:{phase}', center_losses.item(), e)
            addvalue(writer, f'kpsloss:{phase}', center_offset_losses.item(), e)

            if phase == 'train':
                loss.backward()
                optimizer.step()
                if do_preoptim:preoptimizer.step()
                optimizer.zero_grad()
                if idx==0 and do_schedule:scheduler.step()

            for b in range(B):
                results = decoder(4, 4, size[0]//4, size[1]//4, output[-1][0][b].detach().cpu(),
                                  output[-1][1][b].detach().cpu(), 12)
                resultslist += [(int(img_id[b]), results)]
                if idx == 0 and b==0:
                    gt_result=decoder(4,4,size[0]//4,size[1]//4,center_map.cpu()[b],kps_offset.cpu()[b],12)
                    resultimg = makeresultimg(T.ToPILImage()((img[b] + 0.5).detach().cpu()).convert("RGB"), gt_result)
                    resultimg.save(f'{savefolder}/{e}_{b}_{phase}_gt.png')
                    resultimg = makeresultimg(T.ToPILImage()((img[b]+0.5).detach().cpu()).convert("RGB"), results)
                    resultimg.save(f'{savefolder}/{e}_{b}_{phase}.png')
                    # T.ToPILImage()((img[b] + 0.5).detach().cpu()).convert("RGB").save(f'{savefolder}/{e}{b}_{phase}_img.png')

    if phase != 'train':
        acc = cocoevaluate(resultslist, size, phase)
        addvalue(writer, f'mAP:{phase}', acc, e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--size', type=int, default=384)
    parser.add_argument('--features',type=int,default=256)
    parser.add_argument('--stack', default=4, type=int)
    parser.add_argument('--savefolder', default='tmp')
    parser.add_argument('--num_workers',default=cpu_count(),type=int)
    parser.add_argument('--grey',default=False,action='store_true')
    parser.add_argument('--aug',default=False,action='store_true')
    parser.add_argument('--beta',default=0.01,type=float)
    parser.add_argument('--tau',default=7,type=int)
    parser.add_argument('--optim',default='adam')
    parser.add_argument('--preoptim',default=None)
    parser.add_argument('--model',default='sgsc')
    args = parser.parse_args()
    savefolder = f'data/{args.savefolder}'
    os.makedirs(f'{savefolder}', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device='cpu'
    in_ch=3 if not args.grey else 1
    if args.model=='sgsc':
        model = SoftGatedSkipConnection(args.stack, args.features, in_ch, 12).to(device)
    elif args.model=='hourglass':
        model=HourglassNet(num_stacks=8,num_classes=12*2+1).to(device)
    # model.load_state_dict(torch.load('data/tmp/model.pth'))
    # torch.save(model.state_dict(),'model.pth')
    batchsize = args.batchsize
    do_schedule=False
    do_preoptim=False
    if args.optim=='adam':
        optimizer=torch.optim.Adam(model.parameters())
    elif args.optim=='rmsprop':
        do_schedule=True
        optimizer=torch.optim.RMSprop(model.parameters(),lr=0.003)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.5)

    elif args.optim=='sgd':
        optimizer=torch.optim.SGD(model.parameters(),1e-3)
    else:
        assert False, f'{args.optim} is not allowed. Set correctly.'

    if args.preoptim=='kfac':
        preoptimizer=KFAC(model,1e-3)
        do_preoptim=True

    # optmizer = torch.optim.Adam(model.parameters())
    size = (args.size, args.size)
    trainloader = torch.utils.data.DataLoader(
        COCODataset('../data/coco/train2017', '../data/coco/person_keypoints_train2017.json', size,grey=args.grey,do_aug=args.aug,tau=args.tau),
        batch_size=batchsize, shuffle=True,num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(
        COCODataset('../data/coco/val2017', '../data/coco/person_keypoints_val2017.json', size,grey=args.grey,do_aug=args.aug,tau=args.tau), batch_size=batchsize,
        shuffle=True,num_workers=args.num_workers)
    epochs = args.epochs
    writer = {}

    for e in range(epochs):
        operate('train')
        operate('val')
        save(writer, model, savefolder)
