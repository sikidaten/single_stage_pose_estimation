from utils import core as C
writer={}
for e in range(100):
    for s in range(3):
        C.addvalue(writer,'test:test',s,e)
C.savedic(writer,'data')