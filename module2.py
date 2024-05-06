import numpy as np
import pandas as pd
#import examples as ex
import time
import os
import json
import facility

path=os.getcwd()
with open(os.path.join(path,"70083315.json"),'rb') as output:
    Facility=json.load(output)
tree=facility.Tree()
tree.fit(Facility['objs'])
tree.flow(70000357,70000388)

import cortege_solution
import op
import osch

import pickle
import cortege_solution as csol
import dbsessions as db
import engineer as en
work_dir="D:\\data"
plan_id=61580341103
activefile='activitie_{0}.csv'.format(plan_id)
tpath=os.path.join(work_dir,str(plan_id))
if not os.path.isdir(tpath):
    os.mkdir(tpath)
file='pid_{0}.csv'.format(plan_id)
activefile='activitie_{0}.csv'.format(plan_id)
executorsfile='executors_{0}.csv'.format(plan_id)
tsfile='ts_{0}.npy'.format(plan_id)
wafile='wells_allowed_{0}.npy'.format(plan_id)
gafile='groups_allowed_{0}.npy'.format(plan_id)
Q0file='Q0_{0}.npy'.format(plan_id)
Q1file='Q1_{0}.npy'.format(plan_id)
trfile='tr_{0}.npy'.format(plan_id)
cufile='current_places_{0}.npy'.format(plan_id)
supfile='support_{0}.npy'.format(plan_id)
freefile='free_{0}.npy'.format(plan_id)
cifile='corteges_index_{0}.npy'.format(plan_id)
cpfile='corteges_position_{0}.npy'.format(plan_id)
cofile='corteges_{0}.sav'.format(plan_id)
uniformfile='uniform_distribution_{0}.sav'.format(plan_id)
solutionfile='solution_{0}.sav'.format(plan_id)
permutfile='permutations_{0}.sav'.format(plan_id)


with open(os.path.join(tpath,solutionfile),'rb') as output:
    solution=pickle.load(output)
fe=en.Features()
fe.fit(solution)

activities=pd.read_csv(os.path.join(tpath,activefile))
session=db.Session()
session.open()
qu=session.update(activities,plan_id=plan_id,record=True)









file='activities_new_v1.csv'
#file='expanded.csv'
active_=pd.read_csv(os.path.join(tpath,file))
ftmatrix_=np.load(os.path.join(tpath,'ftmatrix_exp.npy'),allow_pickle=True)[()]
base_active=pd.read_csv(os.path.join(tpath,'base_activities.csv'))
#ftmatrix_=ftmatrix.copy()
e=dict()
for i in np.arange(ftmatrix_.shape[1]):
    executor=cortege_solution.Executor()
    executor.weight=0
    if np.isin(i,[0,1,2,3,4]):
        executor.weight=1
    e[i]=executor
executors=cortege_solution.Executors(e)


n = active_.index.max()+1
ts = np.zeros(shape=(n, n))
solution = csol.recursion_solution()
solution.fit(active_, ftmatrix=ftmatrix_, executors=executors, ts=ts)
indices=base_active.index
with open(os.path.join(tpath,'uniform_distribution.sav'),'rb') as output:
    uniform=pickle.load(output)
for i in indices:
    try:
        fun=uniform[i]
        func=solution.debit_functions[i]
        func.executor=fun.executor
        func.master=True
        #vadf[i]=func
    except KeyError:
        continue
conditions=solution.optimize_backward()
vadf={k:solution.debit_functions[k] for k in solution.debit_functions.keys() if solution.debit_functions[k].master }
print(len(vadf.keys()))
initial=op.OrderedMap()
for k in solution.debit_functions.keys():
    func=solution.debit_functions[k]
    e=func.executor
    if func.cortege_next is None:
        initial.add(k,func.order)
s=0
for e in np.arange(5):
    m = solution.executors[e].metric(solution.debit_functions)
    s+=m

print(s)


m=solution.executors[0].metric(solution.debit_functions)
active_=pd.read_csv(os.path.join(tpath,'expanded.csv'))
base_active=pd.read_csv(os.path.join(tpath,'base_activities.csv'))
ftmatrix_=np.load(os.path.join(tpath,'ftmatrix_exp.npy'),allow_pickle=True)[()]

e=dict()
for i in np.arange(ftmatrix_.shape[1]):
    executor=cortege_solution.Executor()
    executor.weight=0
    if np.isin(i,[0,1,2,3,4]):
        executor.weight=1
    e[i]=executor
executors=cortege_solution.Executors(e)


n = active_.index.max()+1
ts = np.zeros(shape=(n, n))
solution = csol.recursion_solution()
solution.fit(active_, ftmatrix=ftmatrix_, executors=executors, ts=ts)

indices=base_active.index
for i in indices:
    try:
        func=solution.debit_functions[i]
        func.master=True
        #vadf[i]=func
    except KeyError:
        continue
conditions=solution.optimize_backward()




active_=pd.read_csv(os.path.join(tpath,'active_v1.csv'))
ftmatrix_=np.load(os.path.join(tpath,'ftmatrix_.npy'))
#ftmatrix_=ftmatrix.copy()
e=dict()
for i in np.arange(ftmatrix_.shape[1]):
    executor=cortege_solution.Executor()
    e[i]=executor
executors=cortege_solution.Executors(e)


n = active_.shape[0]
ts = np.zeros(shape=(n, n))
solution = csol.recursion_solution()
solution.fit(active_, ftmatrix=ftmatrix_, executors=executors, ts=ts)
conditions=solution.optimize_backward()
#with open(os.path.join(tpath, 'conditions.sav'), 'rb') as output:
    #Conds = pickle.load(output)
# conditions1=[{2:1,0:4,1:7}]
#solution.get_routes(Conds)










used=np.load(os.path.join(tpath,'used.npy'))
support=np.load(os.path.join(tpath,'support.npy'))
group=np.load(os.path.join(tpath,'group.npy'))
Q0=np.load(os.path.join(tpath, 'Q0.npy'))
Q1=np.load(os.path.join(tpath,'Q1.npy'))
tr=np.load(os.path.join(tpath,'tr.npy'))
ts1=np.load(os.path.join(tpath,'ts.npy'),allow_pickle=True)[()]
#pairs=np.load(path+'pairs.npy')
wells_allowed=np.load(os.path.join(tpath,'wells_allowed.npy'))
groups_allowed=np.load(os.path.join(tpath,'groups_allowed.npy'),allow_pickle=True)[()]
#current_places=np.array([used,group],dtype=np.int16)
#free_=np.arange(Q0.shape[0])
current_places=np.load(os.path.join(tpath,'current_places.npy'))
free_=np.load(os.path.join(tpath,'free.npy'))
corteges_index=np.load(os.path.join(tpath,'corteges_index.npy'))
corteges_position=np.load(os.path.join(tpath,'corteges_position.npy'))
with open(os.path.join(tpath,'corteges.sav'),'rb') as output:
    corteges=pickle.load(output)
ts1.mobility=5192816103.0
mask=~np.isnan(support[:,0])
end=support[mask].max()
activities=pd.read_csv(os.path.join(tpath,'activities.csv'))
wells_allowed=np.ones(shape=wells_allowed.shape,dtype=bool)
#wells_allowed=en.blocks
#groups_allowed=en.ftmatrix
#corteges=en.corteges
#corteges_index=en.corteges_index
#corteges_position=en.corteges_position
mask=~np.isnan(corteges_index)
support[mask]=np.nan

mask=(activities['act_type']==74917002)|(activities['act_type']==1452402773)
i=0
while i<mask.shape[0]:
    if mask[i]:
        j=0
        while j<wells_allowed.shape[1]:
            wells_allowed[i,j]=True
            j+=1

    i+=1

wsch=osch.wells_schedule()
wsch.engine='c'
#wsch.t=(tr.sum()/16)*1.1
wsch.t=360*3
wsch.tracing=False
stop=Q0.shape[0]
wsch.fun=wsch.cval
current_places_=current_places.copy()

wsch.fit(ts1,tr,Q0,Q1,current_places_,stop=stop,support=support,corteges=corteges,
         corteges_index=corteges_index,corteges_position=corteges_position,
         groups_allowed=groups_allowed,wells_allowed=wells_allowed,epsilon=1,horizon=30,shrinkage=1.5,kernel=np.arange(5))
wsch.free_all=free_
smask=~np.isnan(wsch.support[free_,0])
cmask=~np.isnan(wsch.corteges_index[free_])
mask=smask|cmask
#mask=np.isin(wsch.free,wsch.supported)
#mask=(~umask)&smask
wsch.free_all=wsch.free_all[mask]
index=np.where(wsch.kernel_mask)[0]
mask=np.isin(wsch.free_all,index)
wsch.free_all=wsch.free_all[mask]
wsch.available_activities=np.zeros(wsch.free.shape[0],dtype=bool)
free=wsch.free_all.copy()
wsch.fit_free(wsch.free_all)
t1=time.perf_counter()
trace=wsch.get_routes()
t2=time.perf_counter()
print('time=',t2-t1)
R,T,D,O=osch.get_rout_time(trace)
print('not used '+str(wsch.free.shape[0]))
#missed=wsch.free.copy()
applied=wsch.get_applied_kernel()