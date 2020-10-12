import numpy as np
def point_nms(inputs, score=0.1, dis=10):
    '''
    NMS function based on point value and point's euclidean distance
    Note that returns points cooridinat is (x, y) for numpy format, so actually it is (y, x) on image
    :param inputs:
    :param score:
    :param dis:
    :return:
        kept coors: [ [x,y], [x,y], ..., [x,y] ]
    '''
    # inputs=inputs.numpy()
    assert len(inputs.shape) == 3
    kept_coors = []
    for c in range(inputs.shape[0]):
        heatmap = inputs[c,...]
        x, y = np.where(heatmap > score)
        coors = list(zip(x, y))
        scores = []
        for coor in coors:
            coor_score = heatmap[coor]
            scores.append(coor_score)
        scores_index = np.asarray(scores).argsort()[::-1]
        kept = []
        kept_coor = []
        while scores_index.size > 0:
            kept.append(scores_index[0])
            coors_score = list(coors[kept[-1]])
            coors_score.append(scores[scores_index[0]])
            kept_coor.append(coors_score)
            scores_index = scores_index[1:]
            last_index = []
            for index in scores_index:
                distance = np.sqrt(np.sum(np.square(
                    np.asarray(coors[kept[-1]]) - np.asarray(coors[index])
                )))
                if distance > dis:
                    last_index.append(index)
            scores_index = np.asarray(last_index)

        kept_coors.append(kept_coor)
    return kept_coors