import uniform as uf
import numpy as np

duration=np.arange(10).astype(np.float32)
executors=np.arange(3).astype(np.int32)
indices=np.zeros(shape=duration.shape[0],dtype=np.int32)
indices.fill(-1)
n=executors.shape[0]
permissions=np.zeros(shape=(executors.shape[0],duration.shape[0]),dtype=np.uint8)

i=0
while i<duration.shape[0]:
    if i==5:
        i+=1
        continue
    e=i%n
    permissions[e,i]=1
    i+=1

print("marked : {0}".format(permissions.reshape(-1).sum()))
uf.uniform_assignment(duration,executors,permissions,indices,tolerance=duration.mean()/2.)
print(indices)

