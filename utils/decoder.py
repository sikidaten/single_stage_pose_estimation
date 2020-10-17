import numpy as np
from utils.utils import point_nms
def decoder(factor_x,factor_y,outw,outh,centermap,kps_map,num_joints,score_thresh=.9,dis_thresh=10):
    kps_thresh=1e-10
    kpslevel=[[6,8,10],[5,7,9],[12,14,16],[11,13,15]]
    Z=np.sqrt(outh**2+outw**2)

    centers=point_nms(centermap,score_thresh,dis_thresh)[0]
    ret=[]

    for center in centers:
        single_person_joints = [0 for _ in range(num_joints * 2)]
        root_joint = [int(x) for x in center]

        if root_joint[0] >= kps_map.shape[2] or root_joint[1] >= kps_map.shape[1] or root_joint[0] < 0 or root_joint[1] < 0:
            print('find center point on wrong location')
            continue
        n=-1
        for single_path in kpslevel:
            start_joint = [root_joint[1], root_joint[0]]
            for _ in single_path:
                n+=1
                offset = kps_map[2 * n:2 * n + 2,root_joint[0], root_joint[1] ]
                if abs(offset[0]) <kps_thresh or abs(offset[1]) <kps_thresh:
                    continue
                joint = [start_joint[0] + offset[0] * Z, start_joint[1] + offset[1] * Z]
                # print('start joint {} -> end joint {}, offset {}'.format(start_joint, joint, offset))
                single_person_joints[2 * n:2 * n + 2] = joint
                start_joint = joint

        for i in range(num_joints):
            single_person_joints[2 * i] *= factor_x
            single_person_joints[2 * i + 1] *= factor_y

        ret.append({
            'center': [center[1] * factor_x, center[0] * factor_y, center[2]],
            'joints': single_person_joints
        })

    return ret