import argparse
import os

import torch.nn.functional as F
import torchvision.transforms as T
from multiprocessing import cpu_count
from model.softgatedskipconnection import SoftGatedSkipConnection as Model
from utils.core import *
from utils.dataset import COCODataset
from utils.decoder import decoder
from utils.eval import cocoevaluate
from utils.utils import makeresultimg


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
            loss = 0
            for outcenter, outkps in output:
                # sigmoid here
                center_loss = F.mse_loss(center_map * center_mask, torch.sigmoid(outcenter) * center_mask)
                center_offset_loss = F.smooth_l1_loss(kps_offset * kps_weight, outkps * kps_weight)
                loss = loss + center_loss + center_offset_loss
            loss = loss / args.stack
            print(f'{e:3d}, {idx:3d}/{len(loader)}, {loss:.6f}, {phase}')
            addvalue(writer, f'loss:{phase}', loss.item(), e)

            if phase == 'train':
                loss.backward()
                optmizer.step()
                optmizer.zero_grad()
            else:
                for b in range(B):
                    results = decoder(1, 1, size[0], size[1], output[-1][0][b].detach().cpu(),
                                      output[-1][1][b].detach().cpu(), 12)
                    resultslist += [(int(img_id[b]), results)]
                    if idx == 0:
                        resultimg = makeresultimg(T.ToPILImage()(img[b].detach().cpu()), results)
                        resultimg.save(f'{savefolder}/{epochs}_{b}.png')

    if phase != 'train':
        acc = cocoevaluate(resultslist, size, phase)
        addvalue(writer, f'mAP:{phase}', acc, e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--stack', default=4, type=int)
    parser.add_argument('--savefolder', default='tmp')
    parser.add_argument('--num_workers',default=cpu_count(),type=int)
    args = parser.parse_args()
    savefolder = f'data/{args.savefolder}'
    os.makedirs(f'{savefolder}', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(args.stack, args.size, 3, 12).to(device)
    # model.load_state_dict(torch.load('data/tmp/model.pth'))
    batchsize = args.batchsize
    optmizer = torch.optim.Adam(model.parameters())
    size = (args.size, args.size)
    trainloader = torch.utils.data.DataLoader(
        COCODataset('../data/coco/train2017', '../data/coco/person_keypoints_train2017.json', size),
        batch_size=batchsize, shuffle=True,num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(
        COCODataset('../data/coco/val2017', '../data/coco/person_keypoints_val2017.json', size), batch_size=batchsize,
        shuffle=True,num_workers=args.num_workers)
    epochs = args.epochs
    writer = {}

    for e in range(epochs):
        # operate('train')
        operate('val')
        save(writer, model, savefolder)
