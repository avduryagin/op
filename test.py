work_dir='D:\\pycharm\\op'
path='D:\\Sync'
import sys
import os
sys.path.insert(0,work_dir)
fpath=os.path.join(path,'vps-mlrdp01.ois.ru\\data')
import numpy as np
import pandas as pd
import op
import osch
import pickle
import cortege_solution

#plan_id=62139538103

plan_id=61580341103

tpath=os.path.join(fpath,str(plan_id))
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

active_=pd.read_csv(os.path.join(tpath,file))
ftmatrix_=np.load(os.path.join(tpath,gafile),allow_pickle=True)[()]

array=ftmatrix_.get_boolean_array()
n = active_.index.max()+1
ts = np.zeros(shape=(n, n))
Q0=np.zeros(n)
Q1=Q0
tr=np.zeros(n)

tr_=active_['duration']
tr[tr_.index]=tr_.values
wells_allowed=np.ones(shape=(n,n),dtype=bool)
groups=np.arange(ftmatrix_.shape[1])
used=np.zeros(ftmatrix_.shape[1],dtype=bool)
current_places_=[used,groups]
support=np.empty(shape=(n,2))
support.fill(np.nan)

wsch=osch.wells_schedule()
wsch.engine='c'
#wsch.t=(tr.sum()/16)*1.1
wsch.t=360*10
wsch.tracing=False
stop=Q0.shape[0]
#wsch.fun=wsch.f15
wsch.fun=wsch.constant_func
epsilon=tr.mean()
#epsilon=np.inf


wsch.fit(ts,tr,Q0,Q1,current_places_,stop=stop,support=None,
         corteges_index=None,corteges_position=None,
         groups_allowed=ftmatrix_,wells_allowed=wells_allowed,epsilon=epsilon,horizon=np.inf,shrinkage=0,kernel=None)
for i in active_.index:
    wsch.free.add(i,0)


trace=wsch.get_routes()


active_=pd.read_csv(os.path.join(tpath,file))
ftmatrix_=np.load(os.path.join(tpath,gafile),allow_pickle=True)[()]
base_active=pd.read_csv(os.path.join(tpath,activefile))
#ftmatrix_=ftmatrix.copy()
e=dict()
for i in np.arange(ftmatrix_.shape[1]):
    executor=cortege_solution.Executor()
    executor.weight=0
    if np.isin(i,[0,1,2,3,4]):
        executor.weight=1
    e[i]=executor
executors=cortege_solution.Executors(e)
niter=1

n = active_.index.max()+1
ts = np.zeros(shape=(n, n))
solution = cortege_solution.recursion_solution()
solution.niter=niter
solution.fit(active_, ftmatrix=ftmatrix_, executors=executors, ts=ts)
indices=base_active.index
with open(os.path.join(tpath,uniformfile),'rb') as output:
    uniform=pickle.load(output)
#uniform=wsch.debit_functions
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