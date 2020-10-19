from PIL import Image
import numpy as np
from utils.augmentation import data_aug
def encoder(img_info, root, annos, size, scale, num_joints,do_data_aug=False):
    img=Image.open(f'{root}/{img_info["file_name"]}').resize(size)
    tau=7
    sigma=7
    kpslevel=[[6,8,10],[5,7,9],[12,14,16],[11,13,15]]

    oriw=img_info['width']
    orih=img_info['height']
    inw,inh=size
    outw=inw//scale
    outh=inh//scale
    scale_x=oriw/outw
    scale_y=orih/outh
    Z=np.sqrt(outw**2+outh**2)

    centermap=np.zeros((1,outh,outw),dtype=np.float32)
    kps_offset=np.zeros((num_joints*2,outh,outw),dtype=np.float32)
    kps_count=np.zeros((num_joints*2,outh,outw),dtype=np.uint)
    kps_weight=np.zeros((num_joints*2,outh,outw),dtype=np.float32)

    def create_spm_label(bbox, kps,centermap):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w < 1 or h < 1:
            return

        if w > h:
            center_sigmay = sigma
            center_sigmax = min(sigma*1.5, center_sigmay * w / h)
        else:
            center_sigmax = sigma
            center_sigmay = min(sigma*1.5, center_sigmax * h / w)

        centers = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        try:
            centermap[0,...] = draw_ttfnet_gaussian(centermap[0,...], centers, center_sigmax, center_sigmay)
        except:
            print(f'ERROR:{centers=},{center_sigmax=},{center_sigmay}')
        body_joint_displacement_v2(centers, kps)

    def body_joint_displacement_v2(center, kps):
        '''
        if param tau is bigger, then two closed person on one image will cause confused for offset label.
        '''
        n=-1
        for single_path in kpslevel:
            start_joint = [center[0], center[1]]
            for i, index in enumerate(single_path):
                n+=1
                end_joint = kps[3*index:3*index+3]
                if end_joint[0] == 0 or end_joint[1] == 0:
                    continue
                # make new end_joint based offset
                offset_x, offset_y = end_joint[0] - start_joint[0], end_joint[1] - start_joint[1]
                next_x = center[0] + offset_x
                next_y = center[1] + offset_y

                create_dense_displacement_map(n, center, [next_x, next_y])
                start_joint[0], start_joint[1] = end_joint[0], end_joint[1]

    def create_dense_displacement_map(index, start_joint, end_joint):

        x0 = int(max(0, start_joint[0] - tau))
        y0 = int(max(0, start_joint[1] - tau))
        x1 = int(min(outw, start_joint[0] + tau))
        y1 = int(min(outh, start_joint[1] + tau))

        for x in range(x0, x1):
            for y in range(y0, y1):
                x_offset = (end_joint[0] - x) / Z
                y_offset = (end_joint[1] - y) / Z

                kps_offset[2*index,y, x] += x_offset
                kps_offset[2*index+1,y, x] += y_offset
                kps_weight[:,y, x] = 1
                if end_joint[0] != x or end_joint[1] != y:
                    kps_count[ 2 * index:2 * index + 2,y, x] += 1

    def draw_ttfnet_gaussian(heatmap, center, sigmax, sigmay, mask=None):
        # print (sigmax, sigmay)
        # center_x, center_y = int(center[0]), int(center[1])
        center_x, center_y = center[0], center[1]
        th = 4.6052
        delta = np.sqrt(th * 2)

        height = heatmap.shape[0]
        width = heatmap.shape[1]

        x0 = int(max(0, center_x - delta * sigmax + 0.5))
        y0 = int(max(0, center_y - delta * sigmay + 0.5))

        x1 = int(min(width, center_x + delta * sigmax + 0.5))
        y1 = int(min(height, center_y + delta * sigmay + 0.5))

        ## fast way
        arr_heat = heatmap[y0:y1, x0:x1]
        exp_factorx = 1 / 2.0 / sigmax / sigmax
        exp_factory = 1 / 2.0 / sigmay / sigmay
        x_vec = (np.arange(x0, x1) - center_x) ** 2
        y_vec = (np.arange(y0, y1) - center_y) ** 2
        arr_sumx = exp_factorx * x_vec
        arr_sumy = exp_factory * y_vec
        xv, yv = np.meshgrid(arr_sumx, arr_sumy)
        arr_sum = xv + yv
        arr_exp = np.exp(-arr_sum)
        arr_exp[arr_sum > th] = 0

        heatmap[y0:y1, x0:x1] = np.maximum(arr_heat, arr_exp)
        if mask is not None:
            mask[y0:y1, x0:x1] = 1
            return heatmap, mask
        return heatmap
    bboxes=[]
    kpses=[]
    for ann in annos:
        bbox = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]
        bbox = [bbox[0] / scale_x, bbox[1] / scale_y, bbox[2] / scale_x, bbox[3] / scale_y]
        kps=ann['keypoints']
        assert len(kps)==17*3
        for i in range(17):
            kps[3*i+0]/=scale_x
            kps[3*i+1]/=scale_y
        bboxes+=[bbox]
        kpses+=[kps]
    if do_data_aug:
        img,bboxes,kpses=data_aug(img,bboxes,kpses)
    for bbox ,kps in zip(bboxes,kpses):
        create_spm_label(bbox,kps,centermap)
    kps_count[kps_count==0]+=1
    kps_offset=kps_offset/kps_count
    center_mask=np.where(centermap>0,1,0)
    return img,centermap,center_mask,kps_offset,kps_weight