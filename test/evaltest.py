from utils.dataset import COCODataset
from utils.decoder import decoder
import torch
from utils.eval import cocoevaluate
import random
size = (128, 128)

dataset=COCODataset('../data/coco/val2017', '../data/coco/person_keypoints_val2017.json', size)
resultslist = []
for i in range(len(dataset)):
    print(f'\r{i}/{len(dataset)}',end='')
    img, centermap, center_mask, kps_offset, kps_weight, img_id =dataset[i]
    results = decoder(1, 1, size[0], size[1], centermap, kps_offset, 12)
    resultslist += [(img_id,results)]
print()
cocoevaluate(resultslist,size)