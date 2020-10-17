from utils.cocoeval import COCOeval
from utils.coco import COCO
import json
import numpy as np
def cocoevaluate(resultslist,size):
    cocoGt=COCO('../data/coco/person_keypoints_val2017.json')
    # cocoGt=COCO('../data/coco/person_keypoints_train2017.json')
    det=[]
    kpslevel = np.array([[6, 8, 10], [5, 7, 9], [12, 14, 16], [11, 13, 15]])
    for imgid,results in resultslist:
        img_info=cocoGt.loadImgs(imgid)[0]
        # anno=cocoGt.loadAnns(cocoGt.getAnnIds(imgid,[1]))
        scale=np.array([img_info['width']/size[0],img_info['height']/size[1]])
        for result in results:
            keypoints=np.zeros((17,3))
            joints=np.array(result['joints'])
            for kpsid ,xy in zip(kpslevel.reshape(-1),joints.reshape(-1,2)):
                keypoints[kpsid,:2]=xy*scale
            score=result['center'][2]
            if keypoints.sum()!=0:
                det+=[{
                    "image_id":imgid,
                    "category_id": 1,
                    "keypoints": keypoints.reshape(-1).tolist(),
                    "score":float(score),
                }]
    with open('det.json','w') as f:
        json.dump(det,f)
    cocoDt=cocoGt.loadRes('det.json')
    cocoEval=COCOeval(cocoGt,cocoDt,'keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats[0]