import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle
def addvalue(dict,key,value,epoch):
    if not key in dict.keys():
        dict[key]=[[value]]
    else:
        if epoch > len(dict[key])-1:
            dict[key].append([value])
        else:
            dict[key][epoch].append(value)
def savedic(dict,fol):
    n=1
    numgraph=len(set([i.split(':')[0] for i in dict]))
    axdic={}
    fig=plt.figure()
    for key in dict:
        for e,i in enumerate(dict[key]):
            if type(i)==type([]):
                dict[key][e]=np.mean(dict[key][e])
    for key in dict:
        graph,label=key.split(':')
        if graph in axdic:
            axdic[graph].plot(dict[key],label=f'{graph}:{label}')
        else:
            axdic[graph]=fig.add_subplot(numgraph,1,n)
            n+=1
            axdic[graph].plot(dict[key],label=f'{graph}:{label}')
    for key in axdic:
        axdic[key].legend()
    #fig.legend()
    fig.savefig(f'{fol}/graphs.png')
    plt.close()
    with open(f'{fol}/data.pkl','wb') as f:
        pickle.dump(dict,f)

def save(dic,model,fol):
    savedmodelpath=f'{fol}/model.pth'
    savedic(dic,'/'.join(savedmodelpath.split('/')[:-1]))
    torch.save(model.state_dict(), savedmodelpath)
