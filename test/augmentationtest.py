from utils.dataset import COCODataset
from utils.decoder import decoder
import cv2
import matplotlib.pyplot as plt
from PIL import ImageDraw
from torchvision.transforms import ToPILImage
import numpy as np
import random
size=(512,512)
dataset=COCODataset('../data/coco/train2017','../data/coco/person_keypoints_train2017.json',size,True)
img,centermap,center_mask,kps_offset,kps_weight,img_id=dataset[113]
img=ToPILImage()(img)
draw=ImageDraw.Draw(img)
results=decoder(1,1,size[0],size[1],centermap,kps_offset,12)
print(results)
r=5
fontcolor=(255,255,255)
colors = [(0,0,255),(255,0,0),(0,255,0),(0,255,255),(255,0,255),(255,255,0),(255,255,255)]
for j, result in enumerate(results):
    # print (result)
    center = result['center']
    single_person_joints = result['joints']
    draw.ellipse(((int(center[0])-r, int(center[1])-r), (int(center[0])+r, int(center[1])+r)), fill=colors[j % 3])
    draw.text((center[0],center[1]),'c',fill=fontcolor)
    for i in range(12):
        x = int(single_person_joints[2 * i])
        y = int(single_person_joints[2 * i + 1])
        if not (x==0 and y==0):
            draw.ellipse(((x-r, y-r), (x+r, y+r)), colors[j % 3])
            draw.text((x, y),str(i), fontcolor)
            draw.line(((center[0],center[1]),(x,y)))
img.show()
