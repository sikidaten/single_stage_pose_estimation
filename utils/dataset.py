import torch
from pycocotools.coco import COCO
from utils.singlestagelabel import singlestagelabel
import torchvision.transforms as T

class COCODataset(torch.utils.data.Dataset):
    def __init__(self,root,json,size):
        self.coco=COCO(json)
        self.root=root
        self.size=size
        self.cat_ids=self.coco.getCatIds(['person'])
        self.img_ids=self.coco.getImgIds(catIds=self.cat_ids)
        self.scale=1
        self.num_joints=12
        self.transformer=T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_info=self.coco.loadImgs(self.img_ids[idx])[0]
        ann_ids=self.coco.getAnnIds(self.img_ids[idx],self.cat_ids)
        annos=self.coco.loadAnns(ann_ids)
        img,centermap,center_mask,kps_offset,kps_weight=singlestagelabel(img_info,self.root,annos,self.size,self.scale,self.num_joints)
        if img.mode!='RGB':img=img.convert('RGB')
        return self.transformer(img), centermap, center_mask, kps_offset, kps_weight
if __name__=='__main__':
    dataset=COCODataset('../data/coco/train2017','../data/coco/person_keypoints_train2017.json',(128,128))
    _=dataset[0]