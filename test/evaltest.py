from utils.dataset import COCODataset
from utils.decoder import decoder
import torch
from utils.eval import cocoevaluate
import random
size = (384,384)

dataset=COCODataset('../data/coco/val2017', '../data/coco/person_keypoints_val2017.json', size,tau=7)
resultslist = []
for i in range(len(dataset)):
    print(f'\r{i}/{len(dataset)}',end='')
    img, centermap, center_mask, kps_offset, kps_weight, img_id =dataset[i]
    results = decoder(4, 4, size[0]//4, size[1]//4, centermap, kps_offset, 12)
    resultslist += [(img_id,results)]
print()
cocoevaluate(resultslist,size,'val')