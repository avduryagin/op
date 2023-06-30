import os
import pickle

import cppmath
import numpy as np
import pandas as pd
import time
import op
import calendar
import dbsessions as db


path='D:\\ml\\'

def dts(x=0):
    return int(x*86400)


def reverse(x=np.array([])):
    n=int(x.shape[0]/2)
    i=0
    k=-1
    while i<n:
        a=x[i]
        b=x[k]
        x[i]=b
        x[k]=a
        i+=1
        k-=1
    return x

def hev(x):
    if x==0: return x
    if x>0: return 1
    else: return -1

class queues:
    def __init__(self,queue=np.array([])):
        self.i=1
        self.queue=queue
    def get_values(self,i=0):
        try:
            x=self.queue[:,i]
            return x
        except IndexError:
            return None
class Executor:
    def __init__(self):
        self.index=0
        self.corteges=np.ma.array(data=[],mask=[])
        self.indices=dict({})
        self.epsilon=np.inf
        self.injected=False
        self.paused=False
        self.tau_horizon=np.inf
        self.mintau_horizon=0
        self.empty=False
    def loc(self,item):
        i=self.indices[item]
        return self.iloc(i)

    def iloc(self,item):
        try:
            return self.corteges.mask[item]
        except IndexError:
            return None
    def apply(self,item):
        try:
            i = self.indices[item]
            self.corteges.mask[i]=True
        except KeyError:
            pass
    def getitems(self):
        return self.corteges[~self.corteges.mask].data
    def isempty(self):
        return self.empty
        #if self.corteges[~self.corteges.mask].shape[0]>0:
            #return False
        #else: return True

    def add(self,value):
        n=self.corteges.shape[0]
        self.indices[value]=n
        if n==0:
            data=[]
            mask=[]
        else:
            data=list(self.corteges.data)
            mask=list(self.corteges.mask)
        data.append(value)
        mask.append(False)
        self.corteges=np.ma.array(data,mask=mask)




class debit_function:
    def __init__(self,q0=1,q1=1,tau=0,t=0):
        self.q0=q0
        self.q1=q1
        self.tau=tau
        self.t=t
        self.teta=self.q1*(self.t-self.tau)
        self.dq=self.q1-self.q0
        self.b=self.teta-self.dq*self.t
        self.sign=hev(self.dq)
        self.ro=1
        self.supp=np.array((0,self.t),dtype=np.float32)
        #self.cs -compact support
        self.cs=False
        self.opened=False
        self.reserved=False
        self.index=0
        self.current_index = 0
        self.cortege=None
        self.order=0
        self.gamma=10
        self.epsilon=tau
        self.service=np.array([],dtype=np.int32)
        self.equipment = np.array([], dtype=np.int32)
        #self.busy=False
        self.t1=np.NINF
        self.t2=np.NINF
        self.x1=0 #временные координаты входа/выхода
        self.x2=0
        self.span=np.array([np.NINF,np.inf])
        self.key=None
        self.used=False
        self.executor=None
        self.usedby=None
        self.prohibits=np.array([],dtype=np.int16)
        self.bounds=dict({})
        if self.dq<0:
            self.ro=0
        self.drift=self.b*self.ro+self.teta*(1-self.ro)
        self.dt=0
        self.master=True
        self.left_slave=None
        self.right_slave=None
        self.parent=None
        self.children=None
        self.delta_children=0
        self.kernel_value=0
        self.applied=0
        self.time=0
        self.edelta=0
        self.blocked=False


    def reset_cortege(self):
        self.supp=np.array((0,self.t),dtype=np.float32)
        self.cs=False
        self.opened=False
        self.reserved=False
        self.order=0
        #self.cortege=None

    def isbusy(self,x):
        if (x>=self.t1)&(x<=self.t2):
            return True
        else:
            return False


    def value(self,x):
        return self.teta-self.dq*x

    def tail(self,x):
        b=self.t-self.tau
        a=b-self.epsilon
        w=self.scaled_v1(a)
        if (x>=a)&(x<=b):
            y=(x-a)/(b-x)
            return w+(1./np.exp(-self.gamma*y**2))-1
        elif x>b:
            return np.NINF
        else:
            return self.scaled_v1(x)


    def scaled(self,x):
        return (self.value(x)-self.drift)*self.sign

    def scaled_v1(self,x):
        return self.dq*(self.t-self.tau-x)

    def scaled_v2(self,x):

        scaled=self.scaled_v1(self.t-self.tau-self.epsilon)
        return scaled+self.tail(x)

    def update(self):
        self.teta = self.q1 * (self.t - self.tau)
        self.dq = self.q1 - self.q0
        self.b = self.teta - self.dq * self.t
        self.sign = hev(self.dq)
        self.drift = self.b * self.ro + self.teta * (1 - self.ro)
        if self.dq<0:
            self.ro=0
    def apply(self,delta,time):
        if self.parent is not None:
            if self.time<time:
                self.time=time
            if self.edelta<delta:
                self.edelta=delta
            self.applied+=1


class Engineering:
    def __init__(self,*args,**kwargs):

        self.compatibility=np.array([])
        self.pair_diction=self.PairsDict()
        self.Q0=np.array([],dtype=np.float64)
        self.Q1 = np.array([],dtype=np.float64)
        self.duration=np.array([],dtype=np.float64)
        self.support=np.array([],dtype=np.float64).reshape(-1,2)
        #self.acmatrix=np.array([],dtype=bool)
        self.ftmatrix = ListedDicts(value=False,reverse=True)
        self.blocks=np.ones([],dtype=bool)
        #self.ts=RMListedDicts(value=0)
        self.ts = RMListedDicts_py(value=0)
        self.used = np.zeros([], dtype=bool)
        self.roadmap=ListedDicts(value=False,reverse=True)
        self.session=None
        self.activities=None
        self.executors = None
        self.unique_values=dict({})
        self.ufields=['id','well','object_type','well_pad_id','act_type','field_id','shop_id','enterprise_id']
        self.executor_index = None
        self.activity_index = None
        self.time_scale=1./(24*60*60)
        self.begin=np.datetime64('2022-01-01')
        self.end=np.datetime64('2023-03-01')
        self.global_duration=(self.end-self.begin)/np.timedelta64(1,'D')
        self.corteges=dict({})
        self.corteges_index=np.array([])
        self.corteges_position=np.array([])

    def open_session(self,*args,**kwargs):
        self.session = db.Session(*args,**kwargs)
        self.session.open()

    def get_activities(self,*args,**kwargs):
        def get_unique(field='id',include_nan=False):
            if not include_nan:
                try:
                    mask=~np.isnan(self.activities[field].values)
                except TypeError:
                    mask = ~self.activities[field].isnull()
                uvalues = np.unique(self.activities.loc[mask, field].values)
            else:
                uvalues = np.unique(self.activities.loc[:,field].values)
            return tuple(uvalues)

        if self.session is None:
            self.open_session()

        self.activities=self.session.get_activities(*args,**kwargs)
        self.support=(self.activities[['supp0','supp1']].values-self.begin)/np.timedelta64(1,'D')
        mask=np.where(~np.isnan(self.support[:,0]) & np.isnan(self.support[:,1]))
        self.support[mask,1]=self.global_duration
        self.Q0=self.activities['Q0'].values
        self.Q1 = self.activities['Q1'].values
        mask=np.where(np.isnan(self.Q0))[0]
        self.Q0[mask]=0.
        mask=np.where(np.isnan(self.Q1))[0]
        self.Q1[mask]=0.
        #mask = np.where(np.isnan(self.activities['duration'].values))[0]
        self.duration=(self.activities['plan_end']-self.activities['plan_begin'])/np.timedelta64(1,'D')
        self.activities.loc[:,'duration']=self.duration
        #self.duration[mask]=0



        for c in self.ufields:
            values_= get_unique(field=c)
            self.unique_values.update({c:values_})
        #self.get_corteges()

    def get_corteges(self,available_labels=False):
        def get_pos(corteges_position,corteges_index):
            uniq=np.unique(corteges_index)
            corteges_position_ = np.empty(corteges_position.shape[0])
            corteges_position_.fill(np.nan)
            for c in uniq:
                if np.isnan(c):
                    continue
                mask = np.where(corteges_index == c)[0]
                pos = corteges_position[mask]
                minpos = pos.min()
                corteges_position_[mask] = pos - minpos
            return corteges_position_
        empty_mask=self.activities['from_end'].isnull()
        self.activities['begin_to']=self.activities['begin_to']*self.time_scale
        self.activities['begin_from'] = self.activities['begin_from'] * self.time_scale
        self.activities.loc[empty_mask,'from_end']=0
        self.corteges_index=self.activities['cortege_activity_id'].values
        corteges_position = self.activities['position'].values
        self.corteges_position=get_pos(corteges_position,self.corteges_index)
        self.activities['native_position']=self.activities['position']
        self.activities['position']=self.corteges_position
        mask = ~self.activities.loc[:,'cortege_activity_id'].isnull()

        if available_labels:
            index=self.activity_index[~self.activity_index.mask].data
            mask_=mask.copy()
            mask_[:]=False
            mask_.loc[index]=True
            mask=mask&mask_


        grouped = self.activities[mask].groupby('cortege_activity_id')


        for k in grouped.groups.keys():
            cortege = Cortege()
            cortege.fit(index=k, labels=grouped.groups[k], data=self.activities)
            self.corteges.update({cortege.index: cortege})


    def get_executors(self,*args,sdate='01.01.2022',file='executors.sql',**kwargs):
        def get_allowed():
            execu_keys=self.ftmatrix.index[1].diction.keys()
            activ_keys=self.ftmatrix.index[0].diction.keys()

            self.executor_index=np.ma.array(list(execu_keys),dtype=np.int32,fill_value=-1)
            self.executor_index.mask = np.ones(self.executor_index.shape[0],dtype=bool)
            self.activity_index = np.ma.array(list(activ_keys), dtype=np.int32, fill_value=-1)
            self.activity_index.mask = np.ones(self.activity_index.shape[0], dtype=bool)
            if (self.executor_index.shape[0]==0)|(self.activity_index.shape[0]==0):
                return

            for i,akey in enumerate(activ_keys):
                val=False
                for j,ekey in enumerate(execu_keys):
                    val_=self.ftmatrix[akey,ekey]
                    if val_& (not val):
                        val=True
                    if self.executor_index.mask[j]:
                        self.executor_index.mask[j] = False
                        break
                if val:
                    self.activity_index.mask[i] = False




        if self.activities is None:
            self.get_activities(*args,**kwargs)

        temp= self.unique_values

        self.executors = self.session.get_executors(*args,sdate=sdate,file=file,wells=temp['well'],types=temp['act_type'],
                                                  shops=temp['shop_id'],well_pads=temp['well_pad_id'],
                                                  activities=temp['id'],objtypes=temp['object_type'],
                                                    fields=temp['field_id'])

        dictionary=self.get_exec_compat(data=self.executors.loc[:,['execid','meropid']])

        exec_keys = IndexedDict(self.executors['execid'].value_counts().keys())
        act_keys = IndexedDict(self.activities['id'].values)
        self.ftmatrix= ListedDicts(dictionary, (act_keys, exec_keys), reverse=True, value=False)
        get_allowed()

    def get_executors_v1(self,sdate='01.01.2022',file='executors_obj.sql'):
        def get_allowed():
            execu_keys=self.ftmatrix.index[1].diction.keys()
            activ_keys=self.ftmatrix.index[0].diction.keys()

            self.executor_index=np.ma.array(list(execu_keys),dtype=np.int32,fill_value=-1)
            self.executor_index.mask = np.ones(self.executor_index.shape[0],dtype=bool)
            self.activity_index = np.ma.array(list(activ_keys), dtype=np.int32, fill_value=-1)
            self.activity_index.mask = np.ones(self.activity_index.shape[0], dtype=bool)
            if (self.executor_index.shape[0]==0)|(self.activity_index.shape[0]==0):
                return

            for i,akey in enumerate(activ_keys):
                val=False
                for j,ekey in enumerate(execu_keys):
                    val_=self.ftmatrix[akey,ekey]
                    if val_& (not val):
                        val=True
                    if self.executor_index.mask[j]:
                        self.executor_index.mask[j] = False
                        break
                if val:
                    self.activity_index.mask[i] = False

        temp = self.unique_values
        data=self.session.get_executors_v1(sdate=sdate,file=file,wells=temp['well'],types=temp['act_type'],
                                                  shops=temp['shop_id'],well_pads=temp['well_pad_id'],enterprises=temp['enterprise_id'],
                                                    fields=temp['field_id'])
        for c in ['well_id', 'well_pad_id', 'field_id', 'shop_id', 'enterprise_id']:
            if data[c].dtype==object:
                data[c]=data[c].astype(np.float64)

        self.executors=data
        dictionary=self.get_compat_dictionary(self.executors,self.activities)
        exec_keys = IndexedDict(self.executors['id'].value_counts().keys())
        act_keys = IndexedDict(self.activities['id'].values)
        self.ftmatrix= ListedDicts(dictionary, (act_keys, exec_keys), reverse=True, value=False)
        get_allowed()

        #return data
        #self.executor_index=self.ftmatrix.index[1].diction

    def get_roadmap(self,*args,**kwargs):
        if self.activities is None:
            self.get_activities(*args,**kwargs)

        road_map = self.session.get_roads(wells_id=self.unique_values['well_pad_id'])
        road_map['distance_time']=road_map['distance_time']*self.time_scale
        rmap = cppmath.RoadMap()
        rmap.fill_dictionary(road_map[['obj_id1', 'obj_id2', 'distance_time', 'mobility_type_id']])

        act_keys = IndexedDict(self.activities['well_pad_id'].values)
        zeros=(self.activities['act_type'].values==74917002)|(self.activities['act_type'].values==1452402773)
        act_mask=IndexedDict(zeros)
        self.ts= RMListedDicts(data=rmap, indices=(act_keys, act_mask), value=0,mobility=34712652103.0)
    def get_roadmap_py(self,*args,defval=0.,**kwargs):
        def fill_dictionary(data=pd.DataFrame([],
                                              columns=['obj_id1', 'obj_id2', 'distance_time', 'mobility_type_id'])):
            diction=dict({})
            for i in data.index:
                moby=data.at[i,'mobility_type_id']
                obj1=data.at[i,'obj_id1']
                obj2 = data.at[i, 'obj_id2']
                val=data.at[i, 'distance_time']
                if np.isnan(val):
                    val=defval

                if obj1>obj2:
                    key=(moby,obj2,obj1)
                else:
                    key = (moby, obj1, obj2)
                mask=map(lambda x: ~np.isnan(x),key)
                if all(mask):
                    diction.update({key:val})
            return diction


        if self.activities is None:
            self.get_activities(*args,**kwargs)

        road_map = self.session.get_roads(wells_id=self.unique_values['well_pad_id'])
        road_map['distance_time']=road_map['distance_time']*self.time_scale
        rmap=fill_dictionary(data=road_map[['obj_id1', 'obj_id2', 'distance_time', 'mobility_type_id']])

        act_keys = IndexedDict(self.activities['well_pad_id'].values)
        zeros=(self.activities['act_type'].values==74917002)|(self.activities['act_type'].values==1452402773)
        act_mask=IndexedDict(zeros)
        self.ts= RMListedDicts_py(data=rmap, indices=(act_keys, act_mask), value=0,mobility=34712652103.0)
    def get_blocks(self,*args,epsilon=0.,**kwargs):
        if self.activities is None:
            self.get_activities(*args,**kwargs)
        types = self.session.get_joint_by_rules(types_id=self.unique_values['act_type'])
        comp = cppmath.Compatibility()
        comp.fill_dictionary(types[['cf', 'ct', 'hours']].values)
        commatrix = np.ones(shape=(self.activities.shape[0], self.activities.shape[0]), dtype=bool)
        coord = self.activities.loc[:, ['act_type', 'latitude', 'longitude']].values
        comp.fill_commatrix(coord, commatrix, epsilon)
        self.blocks=commatrix



    def get_bounds_dict(self,array=np.array([[0,0,0]]),value=100.):
        assert isinstance(array,np.ndarray),"array is not numpy.ndarray"
        assert array.shape[1]==3,"expected 3 columns"
        diction=self.pair_diction
        for a in array:
            f1=a[0]
            f2=a[1]
            d=a[2]
            if ~np.isnan(d):
                diction.update((f1, f2),d)
        return diction

    def get_compatibility_matrix(self,coordinates=np.array([[0,0,0]]),value=100):
        bounds=self.pair_diction
        n=coordinates.shape[0]
        comp_matrix=np.ones(shape=(n,n),dtype=bool)
        i=0
        while i<n:
            x_act_type = coordinates[i, 0]
            x_latitude = coordinates[i, 1]
            x_longtitude = coordinates[i, 2]

            j=i+1
            if np.isnan(x_latitude) | np.isnan(x_longtitude):
                i+=1
                #print('i', i)
                continue
            while j<n:
                y_act_type = coordinates[j, 0]
                y_latitude = coordinates[j, 1]
                y_longtitude = coordinates[j, 2]
                if ~np.isnan(y_latitude) & ~np.isnan(y_longtitude):
                    try:
                        epsilon=bounds[(x_act_type,y_act_type)]
                    except KeyError:
                        epsilon=value
                    if np.isnan(epsilon):
                        epsilon=value

                    distance=cppmath.distance(x_latitude,x_longtitude,y_latitude,y_longtitude)
                    if i == 0:
                        print(epsilon,distance,x_act_type,y_act_type )

                    if distance<epsilon:
                        comp_matrix[i, j]=False
                        comp_matrix[j, i] = False
                j+=1

            i += 1

        return comp_matrix


    def get_executors_compatibility_matrix(self,executors=dict({}),activities=np.array([])):
        def get_mask(activs,etp=dict({}),index=np.array([],dtype=np.int32),column_index=0):
            mask = np.zeros(activs.shape[0], dtype=bool)
            for k in etp.keys():
                if k>0:
                    kmask=activs[index,column_index]==k
                    subindex=index[~kmask]
                    mask[index[kmask]]=True
                else:
                    subindex=index
                ci=column_index+1
                if ci<activs.shape[1]:
                    submask=get_mask(activs,etp=etp[k],index=subindex,column_index=ci)
                    mask=mask|submask


            return mask



        assert isinstance(executors,dict),"Executors should be a dict"
        assert isinstance(activities,
                          np.ndarray), "Activities should be a np.ndarray"
        #activities columns [0] -> activ_id;[1] ->type_id; [2]-> well_id; [3] -> pad_id;[4] -> field;[5]->shop [6] -> enterprise
        # executors columns [0] -> exec_id;[1] ->type_id; [2]-> well_id; [3] -> pad_id;[4] -> field;[5]->shop [6] -> enterprise
        matrix=np.zeros(shape=(activities.shape[0],len(executors.keys())),dtype=bool)

        i=0
        for ex in executors.keys():
            types=executors[ex].keys()
            mask=np.zeros(activities.shape[0],dtype=bool)
            for t in types:
                tmask=activities[:,1]==t
                #print(tmask[tmask==True].shape)
                index=np.arange(activities.shape[0])
                subdict=executors[ex][t]
                emask=get_mask(activities,etp=subdict,index=index[tmask],column_index=2)
                #print(emask[emask == True].shape)
                mask=mask|emask
                #print(mask[mask == True].shape)

            matrix[:,i]=mask

            i+=1
        return matrix


    def get_executors_dict(self,executors=pd.DataFrame([['execid','merop_id']]),i=0,index=None,keys=None):
        if i==0:
            keys=executors.columns
            index = executors.index

        groups=executors.loc[index].groupby(keys[i]).groups
        dictionary = dict({})

        for k in groups.keys():

            index=groups[k]
            i_=i+1
            if i_<len(keys):
                subdict=self.get_executors_dict(executors=executors,i=i_,index=index,keys=keys)
                #print('k',k)
                dictionary.update({k:subdict})
        return dictionary

    def get_exec_compat(self,data=pd.DataFrame(columns=['execid','meropid'])):
        mask=~data.loc[:,'execid'].isnull()
        gby=data.loc[mask].groupby('execid')
        dictionary=dict({})
        for e in gby.groups:
            index=gby.groups[e]
            di=dict({})
            for i in index:
                mid=data.at[i,'meropid']
                di.update({mid:True})
            dictionary.update({e:di})
        return dictionary
    def get_compat_dictionary(self,executors,activities):
        def get_act_obj(index, data):
            return data.loc[index, ['well', 'well_pad_id', 'field_id', 'shop_id', 'enterprise_id']].values

        def get_exec_obj(index, data):
            return data.loc[index, ['well_id', 'well_pad_id', 'field_id', 'shop_id', 'enterprise_id']].values

        def isvalid(exec_row, object_row):
            i = 0
            for s in exec_row:
                if ~np.isnan(s):
                    if s == object_row[i]:
                        return True
                i += 1
            return False

        grouped_exec = executors.groupby('type_id').groups
        grouped_act = activities.groupby('act_type').groups
        dictionary=dict({})
        for e in grouped_exec:
            index=grouped_exec[e]
            aindex = grouped_act[e]
            for i in index:
                exec_id=executors.at[i,'id']
                try:
                    executor=dictionary[exec_id]
                except KeyError:
                    dictionary[exec_id]=dict({})
                    executor = dictionary[exec_id]
                exec_row=get_exec_obj(i,executors)
                for a in aindex:
                    act_row=get_act_obj(a,activities)
                    if isvalid(exec_row,act_row):
                        act_id=activities.at[a,'id']
                        executor.update({act_id:True})
        return dictionary


    class PairsDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getitem__(self, key):
            key=self.get_ordered(key)
            return super().__getitem__(key)

        def get_ordered(self,pair):
            pair0=pair[0]
            pair1=pair[1]
            if pair0>pair1:
                pair=(pair1, pair0)
            return pair

        def update(self, key, value) -> None:
                super().update({self.get_ordered(key): value})

class EngineeringTosec:
    def __init__(self,*args,**kwargs):

        self.compatibility=np.array([])
        self.pair_diction=self.PairsDict()
        self.Q0=np.array([],dtype=np.float64)
        self.Q1 = np.array([],dtype=np.float64)
        self.duration=np.array([],dtype=np.int64)
        self.support=np.array([],dtype=np.int64).reshape(-1,2)
        #self.acmatrix=np.array([],dtype=bool)
        self.ftmatrix = ListedDicts(value=False,reverse=True)
        self.blocks=np.ones([],dtype=bool)
        #self.ts=RMListedDicts(value=0)
        self.ts = RMListedDicts_py(value=0)
        self.used = np.zeros([], dtype=bool)
        self.roadmap=ListedDicts(value=False,reverse=True)
        self.session=None
        self.activities=None
        self.executors = None
        self.unique_values=dict({})
        self.ufields=['id','well','object_type','well_pad_id','act_type','field_id','shop_id','enterprise_id']
        self.executor_index = None
        self.activity_index = None
        self.time_scale=1
        self.begin=np.datetime64('2022-01-01')
        self.end=np.datetime64('2023-03-01')
        self.time_items='s'
        self.global_duration=(self.end-self.begin)/np.timedelta64(1,self.time_items)

    def open_session(self,*args,**kwargs):
        self.session = db.Session()
        self.session.open()

    def get_activities(self,*args,**kwargs):
        def get_unique(field='id',include_nan=False):
            if not include_nan:
                try:
                    mask=~np.isnan(self.activities[field].values)
                except TypeError:
                    mask = ~self.activities[field].isnull()
                uvalues = np.unique(self.activities.loc[mask, field].values)
            else:
                uvalues = np.unique(self.activities.loc[:,field].values)
            return tuple(uvalues)

        if self.session is None:
            self.open_session(*args,**kwargs)

        self.activities=self.session.get_activities(*args,**kwargs)
        self.support=(self.activities[['supp0','supp1']].values-self.begin)/np.timedelta64(1,self.time_items)
        mask=np.where(~np.isnan(self.support[:,0]) & np.isnan(self.support[:,1]))
        self.support[mask,1]=self.global_duration
        self.Q0=self.activities['Q0'].values
        self.Q1 = self.activities['Q1'].values
        mask=np.where(np.isnan(self.Q0))[0]
        self.Q0[mask]=0.
        mask=np.where(np.isnan(self.Q1))[0]
        self.Q1[mask]=0.
        #mask = np.where(np.isnan(self.activities['duration'].values))[0]
        self.duration=(self.activities['plan_end']-self.activities['plan_begin'])/np.timedelta64(1,self.time_items)
        #self.duration[mask]=0



        for c in self.ufields:
            values_= get_unique(field=c)
            self.unique_values.update({c:values_})

    def get_executors(self,*args,**kwargs):
        def get_allowed():
            execu_keys=self.ftmatrix.index[1].diction.keys()
            activ_keys=self.ftmatrix.index[0].diction.keys()

            self.executor_index=np.ma.array(list(execu_keys),dtype=np.int32,fill_value=-1)
            self.executor_index.mask = np.ones(self.executor_index.shape[0],dtype=bool)
            self.activity_index = np.ma.array(list(activ_keys), dtype=np.int32, fill_value=-1)
            self.activity_index.mask = np.ones(self.activity_index.shape[0], dtype=bool)
            if (self.executor_index.shape[0]==0)|(self.activity_index.shape[0]==0):
                return

            for i,akey in enumerate(activ_keys):
                for j,ekey in enumerate(execu_keys):
                    val=self.ftmatrix[akey,ekey]
                    if val:
                        self.executor_index.mask[j] = False
                        self.activity_index.mask[i] = False
                        break




        if self.activities is None:
            self.get_activities(*args,**kwargs)

        temp= self.unique_values
        self.executors = self.session.get_executors(*args,wells=temp['well'],types=temp['act_type'],
                                                  shops=temp['shop_id'],well_pads=temp['well_pad_id'],
                                                  activities=temp['id'],objtypes=temp['object_type'],
                                                    fields=temp['field_id'])

        dictionary=self.get_exec_compat(data=self.executors.loc[:,['execid','meropid']])

        exec_keys = IndexedDict(self.executors['execid'].value_counts().keys())
        act_keys = IndexedDict(self.activities['id'].values)
        self.ftmatrix= ListedDicts(dictionary, (act_keys, exec_keys), reverse=True, value=False)
        get_allowed()
        #self.executor_index=self.ftmatrix.index[1].diction

    def get_roadmap(self,*args,**kwargs):
        if self.activities is None:
            self.get_activities(*args,**kwargs)

        road_map = self.session.get_roads(wells_id=self.unique_values['well_pad_id'])
        road_map['distance_time']=road_map['distance_time']*self.time_scale
        rmap = cppmath.RoadMap()
        rmap.fill_dictionary(road_map[['obj_id1', 'obj_id2', 'distance_time', 'mobility_type_id']])

        act_keys = IndexedDict(self.activities['well_pad_id'].values)
        self.ts= RMListedDicts(data=rmap, indices=(act_keys, act_keys), value=0,mobility=34712652103.0)
    def get_roadmap_py(self,*args,defval=0.,**kwargs):
        def fill_dictionary(data=pd.DataFrame([],
                                              columns=['obj_id1', 'obj_id2', 'distance_time', 'mobility_type_id'])):
            diction=dict({})
            for i in data.index:
                moby=data.at[i,'mobility_type_id']
                obj1=data.at[i,'obj_id1']
                obj2 = data.at[i, 'obj_id2']
                val=data.at[i, 'distance_time']
                if np.isnan(val):
                    val=defval

                if obj1>obj2:
                    key=(moby,obj2,obj1)
                else:
                    key = (moby, obj1, obj2)
                mask=map(lambda x: ~np.isnan(x),key)
                if all(mask):
                    diction.update({key:val})
            return diction


        if self.activities is None:
            self.get_activities(*args,**kwargs)

        road_map = self.session.get_roads(wells_id=self.unique_values['well_pad_id'])
        road_map['distance_time']=road_map['distance_time']*self.time_scale
        rmap=fill_dictionary(data=road_map[['obj_id1', 'obj_id2', 'distance_time', 'mobility_type_id']])

        act_keys = IndexedDict(self.activities['well_pad_id'].values)
        self.ts= RMListedDicts_py(data=rmap, indices=(act_keys, act_keys), value=0,mobility=34712652103.0)
    def get_blocks(self,*args,epsilon=0.,**kwargs):
        if self.activities is None:
            self.get_activities(*args,**kwargs)
        types = self.session.get_joint_by_rules(types_id=self.unique_values['act_type'])
        comp = cppmath.Compatibility()
        comp.fill_dictionary(types[['cf', 'ct', 'hours']].values)
        commatrix = np.ones(shape=(self.activities.shape[0], self.activities.shape[0]), dtype=bool)
        coord = self.activities.loc[:, ['act_type', 'latitude', 'longitude']].values
        comp.fill_commatrix(coord, commatrix, epsilon)
        self.blocks=commatrix



    def get_bounds_dict(self,array=np.array([[0,0,0]]),value=100.):
        assert isinstance(array,np.ndarray),"array is not numpy.ndarray"
        assert array.shape[1]==3,"expected 3 columns"
        diction=self.pair_diction
        for a in array:
            f1=a[0]
            f2=a[1]
            d=a[2]
            if ~np.isnan(d):
                diction.update((f1, f2),d)
        return diction

    def get_compatibility_matrix(self,coordinates=np.array([[0,0,0]]),value=100):
        bounds=self.pair_diction
        n=coordinates.shape[0]
        comp_matrix=np.ones(shape=(n,n),dtype=bool)
        i=0
        while i<n:
            x_act_type = coordinates[i, 0]
            x_latitude = coordinates[i, 1]
            x_longtitude = coordinates[i, 2]

            j=i+1
            if np.isnan(x_latitude) | np.isnan(x_longtitude):
                i+=1
                #print('i', i)
                continue
            while j<n:
                y_act_type = coordinates[j, 0]
                y_latitude = coordinates[j, 1]
                y_longtitude = coordinates[j, 2]
                if ~np.isnan(y_latitude) & ~np.isnan(y_longtitude):
                    try:
                        epsilon=bounds[(x_act_type,y_act_type)]
                    except KeyError:
                        epsilon=value
                    if np.isnan(epsilon):
                        epsilon=value

                    distance=cppmath.distance(x_latitude,x_longtitude,y_latitude,y_longtitude)
                    if i == 0:
                        print(epsilon,distance,x_act_type,y_act_type )

                    if distance<epsilon:
                        comp_matrix[i, j]=False
                        comp_matrix[j, i] = False
                j+=1

            i += 1

        return comp_matrix


    def get_executors_compatibility_matrix(self,executors=dict({}),activities=np.array([])):
        def get_mask(activs,etp=dict({}),index=np.array([],dtype=np.int32),column_index=0):
            mask = np.zeros(activs.shape[0], dtype=bool)
            for k in etp.keys():
                if k>0:
                    kmask=activs[index,column_index]==k
                    subindex=index[~kmask]
                    mask[index[kmask]]=True
                else:
                    subindex=index
                ci=column_index+1
                if ci<activs.shape[1]:
                    submask=get_mask(activs,etp=etp[k],index=subindex,column_index=ci)
                    mask=mask|submask


            return mask



        assert isinstance(executors,dict),"Executors should be a dict"
        assert isinstance(activities,
                          np.ndarray), "Activities should be a np.ndarray"
        #activities columns [0] -> activ_id;[1] ->type_id; [2]-> well_id; [3] -> pad_id;[4] -> field;[5]->shop [6] -> enterprise
        # executors columns [0] -> exec_id;[1] ->type_id; [2]-> well_id; [3] -> pad_id;[4] -> field;[5]->shop [6] -> enterprise
        matrix=np.zeros(shape=(activities.shape[0],len(executors.keys())),dtype=bool)

        i=0
        for ex in executors.keys():
            types=executors[ex].keys()
            mask=np.zeros(activities.shape[0],dtype=bool)
            for t in types:
                tmask=activities[:,1]==t
                #print(tmask[tmask==True].shape)
                index=np.arange(activities.shape[0])
                subdict=executors[ex][t]
                emask=get_mask(activities,etp=subdict,index=index[tmask],column_index=2)
                #print(emask[emask == True].shape)
                mask=mask|emask
                #print(mask[mask == True].shape)

            matrix[:,i]=mask

            i+=1
        return matrix


    def get_executors_dict(self,executors=pd.DataFrame([['execid','merop_id']]),i=0,index=None,keys=None):
        if i==0:
            keys=executors.columns
            index = executors.index

        groups=executors.loc[index].groupby(keys[i]).groups
        dictionary = dict({})

        for k in groups.keys():

            index=groups[k]
            i_=i+1
            if i_<len(keys):
                subdict=self.get_executors_dict(executors=executors,i=i_,index=index,keys=keys)
                #print('k',k)
                dictionary.update({k:subdict})
        return dictionary

    def get_exec_compat(self,data=pd.DataFrame(columns=['execid','meropid'])):
        mask=~data.loc[:,'execid'].isnull()
        gby=data.loc[mask].groupby('execid')
        dictionary=dict({})
        for e in gby.groups:
            index=gby.groups[e]
            di=dict({})
            for i in index:
                mid=data.at[i,'meropid']
                di.update({mid:True})
            dictionary.update({e:di})
        return dictionary


    class PairsDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getitem__(self, key):
            key=self.get_ordered(key)
            return super().__getitem__(key)

        def get_ordered(self,pair):
            pair0=pair[0]
            pair1=pair[1]
            if pair0>pair1:
                pair=(pair1, pair0)
            return pair

        def update(self, key, value) -> None:
                super().update({self.get_ordered(key): value})
class IndexedDict:
    def __init__(self, val=np.array([], dtype=np.float64)):
        self.diction = dict()
        i = 0
        while i < val.shape[0]:
            self.diction.update({i: val[i]})
            i += 1
        self.shape=val.shape[0]

    def __getitem__(self, item):
        try:
            return self.diction[item]
        except KeyError:
            return None

class ListedDicts:
    def __init__(self,data=dict(),indices=tuple(),value=None,reverse=False):
        self.index=[]
        self.count=0
        self.shape=[]
        self.value=value
        assert isinstance(data,dict),"Data must be a dict"
        self.data=data
        assert isinstance(indices,tuple),"Indices must be a tuple of IndexedDict"
        for i in indices:
            assert isinstance(i,IndexedDict),'Permitted only IndexedDict items'
            self.add(i)
        self.shape=tuple(self.shape)
        self.reverse=reverse




    def add(self,InDict=IndexedDict()):
        self.index.append(InDict)
        self.count+=1
        self.shape.append(InDict.shape)
    def __getitem__(self, *item):
        indices=item[0]
        vals=self.data
        n=len(indices)
        try:
            if self.reverse:
                i=-1
                while i>=-n:
                    index=indices[i]
                    key=self.index[i][index]
                    vals=vals[key]
                    #vals.append(val)
                    i-=1
                return vals

            else:
                i=0
                while i<len(indices):
                    index=indices[i]
                    key=self.index[i][index]
                    vals=vals[key]
                    #vals.append(val)
                    i+=1
                return vals

        except IndexError:
            return self.value
        except KeyError:
            return self.value


class RMListedDicts(ListedDicts):
    def __init__(self,*args,data=cppmath.RoadMap(),mobility=34712652103.0,**kwargs):
        super().__init__(*args,**kwargs)
        self.data=data
        self.mobility=mobility

    def __getitem__(self, *item):
        indices=item[0]

        try:
            mask=(self.index[1][indices[0]])|(self.index[1][indices[1]])
            if mask:
                return 0
            key0 = self.index[0][indices[0]]
            key1 = self.index[0][indices[1]]
            vals = self.data.time(key0, key1, self.mobility)
            return vals

        except IndexError:
            return self.value
        except KeyError:
            return self.value


class RMListedDicts_py(ListedDicts):
    def __init__(self,*args,data=dict({}),mobility=34712652103.0,**kwargs):
        super().__init__(*args,**kwargs)
        self.data=data
        self.mobility=mobility

    def __getitem__(self, *item):
        indices=item[0]

        try:
            mask=(self.index[1][indices[0]]|self.index[1][indices[1]])
            if mask:
                return 0
            key0 = self.index[0][indices[0]]
            key1 = self.index[0][indices[1]]
            if key0>key1:
                key=(self.mobility,key1, key0)
            else:
                key = (self.mobility, key0, key1)

            #print(key)
            vals = self.data[key]
            return vals

        except IndexError:
            return self.value
        except KeyError:
            return self.value













class values:
    def __init__(self):
        self.value=np.NINF
        self.index=0
        self.inf=False
        self.t1=np.NINF
        self.t2=np.NINF

class well_record:
    def __init__(self):
        self.val=np.nan
        self.eval=np.nan
        self.x=np.nan


class Record:
    def __init__(self):
        self.index=0
        self.niter=0
        self.ct=0
        self.cw=0
        self.span=np.array([])
        self.next_well=None
        self.vector=dict()
class SaftyList(list):
    def rm(self,val):
        try:
            super().remove(val)
        except ValueError:
            pass
    def remove(self, value):
        if hasattr(value, '__iter__'):
            if value is self:
                self.clear()
                return
            for v in value:
                self.rm(v)

        else:
            self.rm(value)



    #def __init__(self):super().__init__()

    #def __init__(self,t): super().__init__(t)

    #def __init__(self,t,obj): super().__init__(t,obj)


class log:
    def __init__(self,record=True):
        self.data=dict()
        self.current_record=Record()
        self.ci=0
        self.record=record
    def open(self,i):
        self.ci=i
        self.current_record=Record()
        self.current_record.index=i
        return self.current_record
    def close(self):
        try:
            self.data[self.ci].append(self.current_record)
        except KeyError:
            self.data.update({self.ci:[self.current_record]})






class wells_schedule:
    def __init__(self):
        self.trajectories=[]
        #self.wells=np.array([])#номера скважин
        self.free=np.array([]) #номера скважин
        self.unical = np.array([]) #уникальные номера скважин
        self.cd=0 #текущее значение дебита (current_debit)
        self.groups=np.array([])#начальное распределение бригад по скважинам
        self.available_executer=np.array([],dtype=bool)
        self.available_activities = np.array([], dtype=bool)
        #self.group_index=np.arange(self.group.shape[0]) # индексация бригад
        self.numbers=np.array([])# индексация бригад
        self.ct=np.array([]) #время освобождения бригад (current time)
        self.dt = np.array([])  # дополнительное время
        self.st=np.array([],dtype=np.float64) #время начала работы на скважине (start time)
        self.ts=np.array([],dtype=np.float64) #транспортная матрица
        self.service=np.array([])#номера сервиса
        self.equipment=np.array([])  # номера сервиса
        self.wells_service=np.array([])#номера сервиса для скважин
        self.wells_equipment=np.array([])  # номера оборудования для скважин
        self.tr=np.array([])#время продолжительности ремонта
        self.Q0=np.array([])#дебит до ремонта
        self.Q1=np.array([])#дебит после ремонта
        self.weights = np.array([])  # матрица рейтингов скважин
        self.fun=self.f1
        self.values=np.array([])
        self.corteges=dict({})
        self.corteges_index=np.array([])
        self.support=np.array([])
        self.logistic_values = None
        self.queue=dict({}) #словарь с значениями key - номер скважины, value -
        self.vector = np.array([],dtype=bool) #вектор состояния скважин для бригады. True - значение конечное
        self.tracing=False
        self.permutation=False
        self.t=100 #время на проведение мероприятий
        self.debit_functions=dict({})
        self.epsilon=np.inf
        self.minct=0.
        self.maxct=0.
        self.delta=1.
        self.ftmatrix = np.array([],dtype=bool) #бинарная матрица соответствия бригад скважинам по своей квалификации
        self.group_support = None
        self.transpose=False
        self.function= op.get_optim_trajectory
        self.gap_metric=lambda x: x
        self.engine='c'
        #self.function=ex.assignment
        self.routes=None
        self.start=None
        self.end=None
        self.wtime=0
        self.used=None
        self.counter=0
        self.tolerance=1e-3
        self.log=log()
        self.opened_count=0
        self.opened_corteges=0
        self.executors=dict({})
        self.horizon=1
        self.reserved_corteges=dict({})
        self.counters=dict({})
        self.added=0
        self.shrinkage=0
        self.localy_added=0
        self.source=0
        self.opened_activities=[]
        self.opened_previous=np.array([])
        self.percent=0
        self.opened=SaftyList()
        self.kernel=np.array([])
        self.kernel_mask=np.zeros([],dtype=bool)
        self.flag=False


    def fit(self,ts,tr,Q0,Q1,current_places,support=None,queue=dict({}),corteges=dict({}),corteges_index=None,corteges_position=None,wells_allowed=None,groups_allowed=None, tracing=None, permutation=None, epsilon=np.inf,horizon=np.inf,delta=3,stop=None,
            record=False,shrinkage=1,kernel=np.array([])):
        def update_debit_functions():
            if self.routes is not None:
                for i in np.arange(self.routes.shape[1]):
                    for j in np.arange(self.routes.shape[0]):
                        well=self.routes[j,i]
                        if well<0:
                            break
                        try:
                            (span)=self.set_span(j,i)
                            self.debit_functions[well].t1=self.start[j,i]
                            self.debit_functions[well].t2=self.end[j,i]
                            self.debit_functions[well].span=np.array(span)
                            self.update_bounds(self.debit_functions[well])
                        except KeyError:
                            continue


        def set_initial_time():
            if self.used is not None:
                mask = ~np.isin(self.free, self.groups[self.used])
                self.free = self.free[mask]
                ext = support[self.groups[self.used], 0]
                submask = np.isnan(ext)
                ext[submask] = 0
                tv = self.tr[self.groups[self.used]] + ext
                mask = tv < 0
                tv[mask] = 0
                self.ct[self.used] = tv
        def init_executors():

            k=0
            while k<self.groups.shape[0]:
                executor = Executor()
                executor.corteges_set = set()
                executor.index = k
                executor.epsilon = self.horizon
                self.executors[k] = executor
                k+=1

            val=np.nanmin(self.corteges_position)

            mask=self.corteges_position==val
            activities=self.free[mask]

            for a in activities:
                i=0
                while i<self.groups.shape[0]:
                    if self.ftmatrix[a,i]:
                        self.executors[i].corteges_set.add(self.corteges_index[a])
                    i+=1

            for k in self.executors.keys():
                executor=self.executors[k]
                if len(executor.corteges_set)>0:
                    for s in executor.corteges_set:
                        executor.add(s)
                #emask=np.zeros(len(executor.corteges_set),dtype=bool)
                #executor.corteges = np.ma.array(list(executor.corteges_set),mask=emask)
                del executor.corteges_set


        #used - булев массив размерности groups. True - индекс бригады, размещенной на скважине для ремонта. False - бригада на скважине, но на работу не назначена.
        if (tracing is not None) and isinstance(tracing,bool):
            self.tracing=tracing
        if (permutation is not None) and isinstance(permutation, bool):
            self.permutation = permutation

        if horizon>0:
            self.horizon=horizon

        self.Q1=Q1
        self.Q0=Q0
        self.dQ=self.Q1-self.Q0
        self.dq=np.min(self.dQ)
        self.log.record=record
        self.corteges=corteges
        self.shrinkage=shrinkage
        self.kernel=kernel
        if corteges_index is None:
            self.corteges_index=np.empty(self.Q0.shape[0])
            self.corteges_index.fill(np.nan)
        else:
            self.corteges_index=corteges_index

        if corteges_position is None:
            self.corteges_position=np.empty(self.Q0.shape[0])
            self.corteges_position.fill(np.nan)
        else:
            self.corteges_position=corteges_position
        self.kernel_mask=np.zeros(self.Q0.shape[0],dtype=bool)



        self.ts=ts
        self.tr=tr
        self.tolerance=1e-3
        self.delta=delta
        self.free=np.arange(Q0.shape[0])

        self.mask=np.ones(Q0.shape[0],dtype=bool)
        self.cd=0
        self.queue=queue


        self.epsilon=epsilon
        self.prohibits=wells_allowed
        self.ftmatrix = groups_allowed

        if groups_allowed is None:
            self.ftmatrix = np.ones(shape=(self.Q0.shape[0], self.groups.shape[0]), dtype=bool)
        if wells_allowed is None:
            self.prohibits = np.ones(shape=(self.Q0.shape[0], self.Q0.shape[0]), dtype=bool)

        self.groups=current_places[1]
        self.used=current_places[0].astype(bool)
        self.available_executer=np.ones(shape=self.groups.shape[0],dtype=bool)
        self.ct=np.zeros(self.groups.shape[0],dtype=np.float64)
        self.st=np.zeros(self.groups.shape[0],dtype=np.float64)
        self.dt = np.zeros(self.groups.shape[0],dtype=np.float64)
        self.current_index = np.zeros(self.groups.shape[0],dtype=np.int32)
        self.current_indey = np.arange(self.groups.shape[0])#координата в столбце self.routes
        self.numbers = np.arange(self.groups.shape[0])
        #устанавливаем начальное время для бригад на скважинах
        set_initial_time()
        init_executors()



        self.minct=self.ct.min()
        self.maxct=self.ct.max()

        self.values=np.empty(self.Q0.shape[0])
        self.values.fill(np.nan)

        if stop is not None:
            self.stop=stop
        else:
            self.stop=self.Q0.shape[0]

        self.maxcs=np.NINF
        self.mincs=np.inf
        self.supported=np.array([])
        self.support = support
        if support is None:
            self.support=np.empty(shape=(self.Q0.shape[0],2))
            self.support.fill(np.nan)

        #init_corteges()

        maxsc=np.NINF
        minsc=np.inf
        supported=[]

        for i in self.free:
            supp=self.support[i]
            cortege_index=self.corteges_index[i]
            wf=debit_function(self.Q0[i],self.Q1[i],self.tr[i],self.t)
            wf.index=i
            if ~np.isnan(cortege_index):
                wf.cortege=cortege_index
                wf.cs=True

            if wells_allowed is not None:
                proht=self.prohibits[i]
                wf.prohibits=np.where(proht==False)[0]
                for p in wf.prohibits:
                    wf.bounds.update({p:np.array([np.NINF,np.NINF])})


            mask=np.isnan(supp)
            if any(~mask):
                wf.cs = True
                wf.opened = True
                self.opened_count += 1

            #mark=False
            for j,m in enumerate(mask):
                if ~m:
                    wf.supp[j]=supp[j]

            if wf.cs:
                supported.append(i)
                mina=wf.scaled_v1(wf.supp[0])
                minb=wf.scaled_v1(wf.supp[1])
                mins=min(mina,minb)
                if minsc>mins:
                    minsc=mins
            self.debit_functions.update({i:wf})

            if self.tracing:
                val=wf.supp[0]
            else:
                val=wf.scaled_v1(0)
            self.values[i]=val

            if maxsc<self.values[i]:
                maxsc=self.values[i]

        self.supported=np.array(supported)
        mask=~np.isnan(self.values)
        self.values=self.values[mask]
        self.sindices=np.argsort(self.values)
        if not self.tracing:
            self.sindices=reverse(self.sindices)
        self.free=self.free[self.sindices]
        self.available_activities = np.zeros(self.stop, dtype=bool)
        self.infmask=np.zeros(shape=self.stop,dtype=bool)
        self.infindex = np.zeros(shape=self.stop, dtype=np.int32)
        self.nempty = np.arange(self.groups.shape[0])
        self.maxcs=maxsc
        self.mincs=minsc
        #self.prohib_dict = self.get_prohib_dict()
        self.counter=self.free.shape[0]
        update_debit_functions()
        self.set_kernels()

    def get_kernel(self,cortege,ftmatrix,kernel=np.array([])):
        def iskernel(a):
            for e in kernel:
                if ftmatrix[a,e]:
                    return True
            return False
        ker=dict()
        for i,k in enumerate(cortege.position.keys()):
            acts=cortege.get_activities(i)
            for ac in acts:
                if iskernel(ac):
                    self.kernel_mask[ac]=True
                    try:
                        ker[k].append(ac)
                    except KeyError:
                        ker[k]=[]
                        ker[k].append(ac)
        return ker

    def set_kernel(self,cortege,ftmatrix,kernel=np.array([]),value=0):
        def go(index,parent=None,i=0,val=0):
            if i>=len(index):
                return
            key=index[i]
            acts=ker[key]
            children=None
            distance=0
            if (i+1)<len(index):
                children=ker[index[i+1]]
                distance=self.get_activities_distance(acts[0],children[0],cortege,self.tr,self.corteges_index,self.corteges_position)

            for a in acts:
                fun=self.debit_functions[a]
                fun.children=children
                fun.parent=parent
                fun.kernel_value=val
                fun.delta_children=distance
            go(index,acts,i+1,val)



        ker=self.get_kernel(cortege,ftmatrix,kernel)
        if len(ker.keys())>0:
            index_=list(ker.keys())
            go(index_,None,0,value)
            return True
        else:return False


    def set_kernels(self):
        i=0
        for k in self.corteges.keys():
            cortege=self.corteges[k]
            if self.set_kernel(cortege,self.ftmatrix,self.kernel,i):
                i+=1

    def apply_kernel(self,fun=debit_function()):
        if fun.children is not None:
            for c in fun.children:
                func=self.debit_functions[c]
                func.apply(fun.delta_children,fun.t2)

    def fit_applied(self,applied):
        def wrap(cor,ker):
            start = list(ker.keys())[0]
            keys = filter(lambda k: k >= start, cor.position.keys())
            blocked_keys = filter(lambda k: k < start, cor.position.keys())
            t1 = None
            t2 = None
            for k in keys:
                if cor.position[k].is_done():
                    t1 = cor.position[k].mint
                    t2 = cor.position[k].maxt
                    continue
                else:
                    if cor.position[k].applied_number > 0:
                        bounds = cor.Apply(cor.position[k].mint, cor.position[k].mint, start=k, pass_used=True)
                        self.set_compact(bounds, cor.position_index)
                        break
                    else:
                        bounds = cor.Apply(t1, t2, start=k, pass_used=True)
                        self.set_compact(bounds, cor.position_index)
                        break
            if (not cor.isopened())&(t1 is not None):
                bounds = cor.Apply(t1, t2, start=k, pass_used=True)
                self.set_compact(bounds, cor.position_index)

            for i,bk in enumerate(blocked_keys):
                activities=cor.get_activities(i)
                for a in activities:
                    self.debit_functions[a].blocked=True
                    applied_activities.append(a)



        applied_activities=[]
        for cid in applied.keys():
            cortege=self.corteges[cid]
            app=applied[cid]
            for k in app.keys():
                b = cortege.apply_in_position(app[k], k)
                applied_activities.extend(app[k].keys())
                for f in app[k].keys():
                    t1=app[k][f][0]
                    t2 = app[k][f][1]
                    executor=app[k][f][2]
                    fun=self.debit_functions[f]
                    fun.used=True
                    fun.t1=t1
                    fun.t2=t2
                    fun.executor=executor
            wrap(cortege,app)
        mask=~np.isin(self.free,applied_activities)
        self.free=self.free[mask]
        self.logistic_values = self.logistic_values[mask]













    def fit_v1(self,ts,tr,Q0,Q1,groups,support=None,used=None,stop=None,queue=dict({}),epsilon=np.inf,service=np.array([]),equipment=np.array([]),wells_service=None,wells_equipment=None,prohibits=None,delta=3.,group_support=None,tracing=None):
        def update_df():
            if self.routes is not None:
                for i in np.arange(self.routes.shape[1]):
                    for j in np.arange(self.routes.shape[0]):
                        well=self.routes[j,i]
                        if well<0:
                            break
                        try:
                            (span)=self.set_span(j,i)
                            self.debit_functions[well].t1=self.start[j,i]
                            self.debit_functions[well].t2=self.end[j,i]
                            self.debit_functions[well].span=span
                            self.update_bounds(self.debit_functions[well])
                        except KeyError:
                            continue

        #used - булев массив размерности groups. True - индекс бригады, размещенной на скважине для ремонта. False - бригада на скважине, но на работу не назначена.
        if (tracing is not None) and isinstance(tracing,bool):
            self.tracing=tracing
        self.Q1=Q1
        self.Q0=Q0
        self.dQ=self.Q1-self.Q0
        self.dq=np.min(self.dQ)
        self.groups=groups
        if used is None:
            used=np.zeros(self.groups.shape[0],dtype=bool)
        self.ts=ts
        self.tr=tr
        self.delta=delta
        self.free=np.arange(Q0.shape[0])
        self.mask=np.ones(Q0.shape[0],dtype=bool)
        self.cd=0
        self.queue=queue
        self.numbers=np.arange(groups.shape[0])
        self.ct=np.zeros(self.groups.shape[0],dtype=np.float)
        self.dt = np.zeros(self.groups.shape[0],dtype=np.float)
        self.current_index = np.zeros(self.groups.shape[0],dtype=np.int32)
        self.epsilon=epsilon
        self.service=service
        self.equipment=equipment
        self.wells_service=wells_service
        self.wells_equipment=wells_equipment
        self.ftmatrix=np.ones(shape=(self.Q0.shape[0],self.groups.shape[0]),dtype=bool)
        self.prohibits=prohibits

        if group_support is not None:
            self.group_support=group_support

        if used.shape[0]>0:
            mask=~np.isin(self.free,self.groups[used])
            self.free = self.free[mask]
            ext=support[self.groups[used],0]
            submask=np.isnan(ext)
            ext[submask]=0
            tv = self.tr[self.groups[used]]+ext

            mask=tv<0
            tv[mask]=0
            self.ct[used]=tv

        self.minct=self.ct.min()
        self.maxct=self.ct.max()
        self.st=np.zeros(self.groups.shape[0],dtype=np.float)
        self.values=np.empty(self.Q0.shape[0])
        self.values.fill(np.nan)
        if stop is not None:
            self.stop=stop
        else:
            self.stop=self.Q0.shape[0]
        self.maxcs=np.NINF
        self.mincs=np.inf
        self.supported=np.array([])
        self.support = support
        if support is None:
            self.support=np.empty(shape=(self.Q0.shape[0],2))
            self.support.fill(np.nan)

        maxsc=np.NINF
        minsc=np.inf
        supported=[]

        for i in self.free:
            supp=self.support[i]
            wf=debit_function(self.Q0[i],self.Q1[i],self.tr[i],self.t)
            wf.index=i
            if self.prohibits is not None:
                proht=self.prohibits[i]
                wf.prohibits=np.where(proht==False)[0]
                for p in wf.prohibits:
                    wf.bounds.update({p:np.array([np.NINF,np.NINF])})

            if self.wells_equipment is not None:
                wf.equipment=self.wells_equipment[i]

            if self.wells_service is not None:
                wf.service=self.wells_service[i]

            mask=np.isnan(supp)
            #mark=False
            for j,m in enumerate(mask):
                if ~m:
                    wf.supp[j]=supp[j]
                    wf.cs=True
            if wf.cs:
                supported.append(i)
                mina=wf.scaled_v1(wf.supp[0])
                minb=wf.scaled_v1(wf.supp[1])
                mins=min(mina,minb)
                if minsc>mins:
                    minsc=mins
            self.debit_functions.update({i:wf})
            for n in self.numbers:
                self.ftmatrix[i,n]=self.isvalid(n,i)

            if self.tracing:
                val=wf.supp[0]
            else:
                val=wf.scaled_v1(0)
            self.values[i]=val

            if maxsc<self.values[i]:
                maxsc=self.values[i]

        self.supported=np.array(supported)
        mask=~np.isnan(self.values)
        self.values=self.values[mask]
        self.sindices=np.argsort(self.values)
        if not self.tracing:
            self.sindices=reverse(self.sindices)
        self.free=self.free[self.sindices]
        self.infmask=np.zeros(shape=self.stop,dtype=bool)
        self.infindex = np.zeros(shape=self.stop, dtype=np.int32)
        self.nempty = np.arange(self.groups.shape[0])
        self.maxcs=maxsc
        self.mincs=minsc
        #self.prohib_dict = self.get_prohib_dict()

        update_df()



    def get_group_support(self,x=0.,i=0):
        # устанавливает значение x в соответствии с ограничениями на время работы группы
        if self.group_support is None:
            return x
        hour_,day=np.modf(x)
        #hour_=hour*24
        try:
            support=self.group_support[i]
            if (hour_>=support[0])&(hour_<=support[1]):
                return x
            elif hour_<support[0]:
                return day+support[0]
            else:
                a=1+support[0]
                return day+a

        except IndexError:
            return x


    def update_debit_functions(self,debitfun_dict=dict({})):
        for i in debitfun_dict.items():
            self.debit_functions.update({i})

    def get_prohib_dict(self):
        if self.prohibits is None:
            return dict()
        index = np.arange(self.prohibits.shape[0])
        frame = dict()
        shape=max(self.prohibits.shape)
        mask = np.ones(shape=shape, dtype=bool)
        k = 0
        i = 0
        while k < index.shape[0]:
            if mask[k]:
                indices = np.where(self.prohibits[k] == False)[0]
                mask[k] = False
                if indices.shape[0] > 0:
                    indices=np.append(k, indices)
                    frame.update({i: indices })
                    mask[indices] = False

                    for s in indices:

                        try:
                            fun=self.debit_functions[s]
                            fun.key = i
                        except KeyError:
                            continue


                    i += 1

            k += 1
        return frame
    def isreserved(self,cid):
        try:
            return self.reserved_corteges[cid]
        except KeyError:
            return False

    def isprohibited_(self,x=0,well=0):

        if np.isnan(x):
            return False
        #t1 = time.perf_counter()
        try:

            fun=self.debit_functions[well]

            for i in fun.prohibits:

                try:
                    if  self.debit_functions[i].isbusy(x)|self.debit_functions[i].isbusy(x+fun.tau):

                        return False
                except KeyError:
                    continue

            return True

        except KeyError:

            return False


    def try2open(self,executers=np.array([],dtype=np.int32)):
        assigned=False
        cover=self.cover(executers)
        if len(cover.keys())==0:
            return False

        for cid in cover.keys():
            cortege=cover[cid]
            execut = cortege[0][1]
            activities = cortege[0][0]
            self.opened.extend(activities)
            cortege_ = self.corteges[cid]
            weigth = cortege[1]
            t_ = weigth.min() * -1
            mct = self.ct[execut].min()
            that = max(mct, t_)
            tvector=self.get_executers_vector(cortege_,self.tr)
            t=tvector.max()
            #t=that

            if not cortege_.isopened():
                bounds=cortege_.Apply(t,t)
                self.set_compact(bounds)
                self.opened_corteges += 1
                for e in self.executors.keys():
                    self.executors[e].apply(cid)
                    ct=self.ct[e]
                    #if (ct+self.executors[e].epsilon)<t:
                        #self.executors[e].epsilon=t-ct

            i=0
            while i<activities.shape[0]:
                ac=activities[i]
                e=execut[i]
                self.executors[e].injected=True
                #set_value(ac,e)
                i+=1
        if len(cover.keys())>0:
            assigned=True
        return assigned




    def get_optim_activities(self,corteges_id=np.array([],dtype=np.int32),executor=0):
        choise=None
        cid=None
        reserved=[]
        notreserved=[]
        for cid in corteges_id:
            if not self.isreserved(cid):
                notreserved.append(cid)
            else:
                reserved.append(cid)
        for c in notreserved:
            reserved.append(c)
        del notreserved

        for cid in reserved:
            if not self.corteges[cid].isopened():
                activities=self.corteges[cid].get_activities(0)
                mask=np.zeros(activities.shape[0],dtype=bool)
                for i,a in enumerate(activities):
                    func=self.debit_functions[a]
                    if (self.ftmatrix[a,executor])&(not func.reserved):
                        mask[i]=True
                availabel=activities[mask]
                if availabel.shape[0]>0:
                    choise=availabel[0]
                    self.debit_functions[choise].reserved=True
                    return choise,cid
        return choise,cid

    def empty_executors(self):
        empty=True
        for cid in self.executors.keys():
            if not self.executors[cid].isempty():
                empty=False
                return empty
        return empty

    def update_executors(self,val=np.inf):
        if self.empty_executors()&~(np.isinf(self.horizon)):
            for k in self.executors.keys():
                self.executors[k].epsilon=val
                self.executors[k].paused=False
            self.epsilon=np.inf
            self.horizon=val
        else:
            for k in self.executors.keys():
                self.executors[k].epsilon=self.horizon



    def get_weights(self, indices=np.array([0])):

        def set_debit_functions(executors):
            def set(executers,fun):
                other = fun.index
                for e in executers:
                    if not self.ftmatrix[other,e]:
                        continue
                    cw=self.groups[e]
                    x=self.solved_time(other,cw,e,tracing=True)
                    if x is None:
                        continue
                    xtau=x+fun.tau
                    if fun.x1>x:
                        fun.x1=x
                    if fun.x2<xtau:
                        fun.x2=xtau
                return

            if not self.tracing:
                return
            for f in self.free:
                cortege_index = self.corteges_index[f]
                if np.isnan(cortege_index):
                    continue
                fun=self.debit_functions[f]
                if (fun.blocked)|(not fun.opened):
                    continue
                fun.x1=np.inf
                fun.x2=np.NINF
                set(executors,fun)
            return

        def set_tau_horizon(executors):
            def set_corteges():
                for cid in self.corteges.keys():
                    cortege=self.corteges[cid]
                    if (not cortege.blocked) and cortege.isopened():
                        cortege.shortest_way(self.debit_functions)
                return
            conditions=lambda x: True if (not self.debit_functions[x].blocked)&(not self.debit_functions[x].opened)&(not self.debit_functions[x].used)&self.in_ocortege(x) else False
            def get_horizon(activities,ct=0):
                horizons=[]
                for a in activities:
                    fun=self.debit_functions[a]
                    if (fun.x2 is not  None) and (fun.x2>=ct):
                        horizons.append(fun.x2)
                return np.array(horizons)
            if executors.shape[0]>0:
                set_corteges()
                available_activities=self.get_available_activities(executors,conditions=conditions)
                for e in available_activities.keys():
                    activities=available_activities[e]
                    if len(activities)==0:
                        self.executors[e].tau_horizon=np.inf
                        self.executors[e].empty=True
                        continue
                    ct = self.ct[e]
                    horizons=get_horizon(activities,ct)
                    if horizons.shape[0]>0:
                        self.executors[e].tau_horizon=horizons.min()
                    else:
                        self.executors[e].tau_horizon=np.inf


        if self.stop > self.free.shape[0]:
            self.stop = self.free.shape[0]
        self.reserved_corteges=dict({})
        self.update_executors()
        self.weights=op.Safty2DArray(shape=(self.groups.shape[0], self.stop))

        tolerance=self.epsilon
        #set_tau_horizon(indices)
        step=0
        missed=[]
        #cco=0
        forw=True
        open_=self.nempty_corteges()
        horizon=True
        while forw:
            self.weights.clear()
            self.weights.fill(np.NINF)
            if horizon:
                set_debit_functions(indices)
                set_tau_horizon(indices)
                horizon=False
            #t1=time.perf_counter()
            empty=[]
            valid_=False
            for teta,i in enumerate(indices):
                ct = self.ct[i]
                if ct - self.minct < tolerance:
                    if (~np.isinf(self.executors[i].tau_horizon)) and (self.executors[i].tau_horizon<self.executors[i].mintau_horizon):
                        continue

                    if ~np.isinf(self.executors[i].tau_horizon):
                        eps=np.inf
                    else:
                        eps = self.executors[i].epsilon

                    eps+=step
                    valid=True
                    if not self.executors[i].paused:
                        valid=self.get_vector(i,eps=eps)
                        first=False
                        if valid:
                            valid_=True
                    if (self.tracing)&(not valid):
                        empty.append(i)

                else:
                    missed.append(i)

            if (len(empty)>0) &(len(self.opened)==0) & open_:
                empty_=np.array(empty,dtype=np.int32)
                open_=self.try2open(empty_)
                if open_:
                    horizon=True
                    continue

            self.nempty = self.weights.index[self.weights.mask]
            if (self.nempty.shape[0]==0)&(not self.empty_executors())&(step<self.t):
                for k in self.executors.keys():
                    self.executors[i].paused = False
                step+=10
                continue



            if self.nempty.shape[0] > 0:
                weights=self.weights.array[self.weights.mask]

                if weights.shape[0]>weights.shape[1]:
                    self.transpose=True
                    return weights.T


                return self.weights.array[self.weights.mask]

            else:
                if tolerance <= self.maxct - self.minct:
                    tolerance+=self.delta
                else:
                    forw=False
                if len(missed)>0:
                    indices=np.array(missed)
                    missed=[]
                else:
                    return None

        return None

    def get_span(self, k=0,return_next=False):
        def next_well(i,position=0):
            target_=None

            ci = self.current_index[i]+position
            yci = self.current_indey[i]
            if ci < self.routes.shape[0]:
                target_ = self.routes[ci, yci]
                if target_<0:
                    target_=None
            return target_
        def global_span(well,next_well):
            if (well is not None) & (next_well is not None):
                fun1=self.debit_functions[well]
                fun2=self.debit_functions[next_well]
                t1=fun1.t1
                t2=min(fun2.t1-fun1.tau-self.ts[well,next_well],fun1.supp[1])
                return np.array([t1,t2])
            elif (well is not None):
                fun1 = self.debit_functions[well]
                return np.array([fun1.t1,fun1.supp[1]])
            else:
                return np.array([np.NINF, np.inf])

        span = np.array([np.NINF, np.inf]).reshape(-1, 2)
        target = None
        next_target=None
        try:
            if self.routes is  None:
                raise KeyError
            target=next_well(k)
            next_target=next_well(k,position=1)
            span=global_span(target,next_target)
            fun=self.debit_functions[target]
            tau=fun.tau
            top=span[1]

            for w in fun.prohibits:
                if self.debit_functions[w].cs:
                    span=op.residual(span,np.array([self.debit_functions[w].t1,self.debit_functions[w].t2]))
            mask=np.ones(span.shape[0],dtype=bool)
            i=0
            while i<span.shape[0]:
                s=span[i]
                a=s[0]
                b=s[1]
                if b<top:
                    b_=b-tau
                    if b_>=a:
                        span[i,1]=b_
                    else:
                        mask[i]=False
                i+=1
            span=span[mask]
        except KeyError:
            pass
        finally:
            if return_next:
                return span, target
            else:
                return span

    def get_route_weigths(self) -> (np.array,np.array):
        index_=[]
        mask=np.where(self.current_index<self.routes.shape[0])[0]
        for m in mask:
            if self.routes[self.current_index[m],self.current_indey[m]]>=0:
                index_.append(m)
        index=np.array(index_)
        if index.shape[0]==0:
            return (index,index)

        #index=np.where(self.routes[self.current_index[mask],self.current_indey[mask]]>=0)[0]
        #weigth=np.empty(shape=(index.shape[0],index.shape[0]))
        weigth=op.Safty2DArray(shape=(index.shape[0],index.shape[0]))
        #mask=np.zeros(index.shape[0],dtype=bool)
        #notinf=np.zeros(index.shape[0],dtype=np.int16)
        weigth.fill(np.NINF)
        #weigth.fill(np.inf)

        i_=0

        while i_<index.shape[0]:
            i=index[i_]
            cw=self.groups[i]
            ct=self.ct[i]
            j_=0
            ninf=[]
            while j_ < index.shape[0]:
                j=index[j_]
                ci=self.current_index[j]
                yci=self.current_indey[j]
                t1_,t2_=self.get_span(j)
                if ~np.isinf(t2_):

                    target=self.routes[ci,yci]
                    free_ = self.permitted(ct, target)
                    allowed = self.ftmatrix[target, i]

                    #allowed=self.ftmatrix[target,j]
                    #free_=True
                    if (target>=0) and allowed and free_:
                        that=ct+self.ts[cw,target]
                        delta=self.tolerance
                        origin=ct
                        #if ct>t1_-delta:
                            #origin=t1_-delta

                        fun = op.GapMetrics(origin=origin, a=t1_, b=t2_, delta=0)
                        value = fun.linear(that)
                        weigth[i_, j_] = value
                        if ~np.isinf(value):
                            ninf.append(j_)
                j_+=1
            if ~weigth.mask[i_]:
                ninf=np.array(ninf, dtype=np.int32)
                weigth.swap(i_,ninf)




            i_+=1
        indices=weigth.mask
        weigth_=weigth.array[indices,:][:,indices]



        return (weigth_,index[indices])

    def get_cortege_weigths(self,cortege_id,executers=np.array([],dtype=np.int32)) -> (np.array, np.array):
        cortege=self.corteges[cortege_id]
        bounds=cortege.get_bounds()
        index=np.array(list(bounds.keys()),dtype=np.int32)

        weigth = op.Safty2DArray(shape=(index.shape[0], executers.shape[0]))

        weigth.fill(np.NINF)
        i = 0
        while i < index.shape[0]:
            activity=index[i]
            b=bounds[activity][1]
            j = 0
            ninf=[]

            while j < executers.shape[0]:
                e=executers[j]
                if not self.ftmatrix[activity,e]:
                    j+=1
                    continue

                ct = self.ct[e]
                cw=self.groups[e]
                delta=b-ct-self.ts[cw,activity]
                weigth[i,j]=delta
                ninf.append(j)
                j += 1



            if ~weigth.mask[i]:
                ninf = np.array(ninf, dtype=np.int32)
                weigth.swap(i, ninf)

            i+= 1
        return weigth,index
    def assign_corteges(self,executers=np.array([],dtype=np.int32)):
        assigned=dict()
        for cid in self.corteges.keys():
            if self.corteges[cid].isopened()|self.corteges[cid].reserved:
                continue
            weigths,activities=self.get_cortege_weigths(cortege_id=cid,executers=executers)
            mask=weigths.mask
            #nempty=executers[mask]
            if mask[mask].shape[0]==0:
                continue
            index,s=self.function(weigths[mask],criterion='max',engine=self.engine)

            weigths_=weigths[mask][index[0],index[1]]
            assigned[cid]=np.array([activities[mask],executers[index[1]]],dtype=np.int32),weigths_

        if len(assigned.keys())>0:
            soassigned = dict(sorted(assigned.items(), key=lambda item: item[1][1].shape[0], reverse=True))
            return soassigned

        return assigned
    def cover(self,executers=np.array([],dtype=np.int32)):
        go=True
        covered=dict()
        while go:
            assigned=self.assign_corteges(executers)
            #covered=[]
            count=0
            for k in assigned.keys():
                subexecuters=assigned[k][0][1]

                if subexecuters.shape[0]>executers.shape[0]:
                    continue
                mask=np.isin(executers,subexecuters)
                if mask[mask].shape[0]<subexecuters.shape[0]:
                    continue
                executers=executers[~mask]
                covered[k]=assigned[k]
                count+=1
                self.corteges[k].reserved=True
            if (count==0)|(executers.shape[0]==0):
                go=False
                break
            #covered_.extend(covered)
        for k in self.corteges.keys():
            self.corteges[k].reserved = False
        return covered

    def get_available_executers(self,activities=np.array([],dtype=np.int32)):
        d = dict()
        if not hasattr(activities, '__iter__'):
            activities=[activities]

        for a in activities:
            d[a] = []
            for e in self.executors.keys():
                if self.ftmatrix[a, e]:
                    d[a].append(e)
        return d

    def get_available_activities(self, executers=np.array([], dtype=np.int32),conditions=lambda x:True):
        d = dict()
        if not hasattr(executers, '__iter__'):
            executers=[executers]

        for e in executers:
            d[e] = []
            #for a in self.debit_functions.keys():
            for a in self.free:
                if self.ftmatrix[a, e] & conditions(a):
                    d[e].append(a)
        return d
    def in_ocortege(self,activitie):
        cortege_name=self.corteges_index[activitie]
        if np.isnan(cortege_name): return False
        if self.corteges[cortege_name].isopened()&(not self.corteges[cortege_name].blocked):return True
        return False



    def set_span(self,ci=0,i=0):

        if self.routes is None:
            return np.inf,np.inf
        if i >= self.routes.shape[1]:
            return np.NINF, np.inf

        target = self.routes[ci, i]


        if target<0:
            return np.NINF, np.inf

        t1 = self.start[ci, i]
        t2 = self.end[ci, i]
        if ci < self.routes.shape[0] - 1:
            next_target = self.routes[ci + 1, i]
            if next_target>=0:
                t3 = self.start[ci + 1, i]
                ts = self.ts[target, next_target]
            else:
                t3 = np.inf
                ts = np.inf
        else:
            t3 = np.inf
            ts = np.inf
        fun=self.debit_functions[target]
        bound = fun.supp[1]
        delta = get_delta(t1, t2, t3, ts=ts, bound=bound)

        return t1,t1+delta

    def finit(self,span=np.array([]),tolerance=0):
        def function(fun,x=0.):
            if op.inset(x,span,epsilon=tolerance):
                return self.fun(x=x,fun=fun)
            else:
                return np.NINF
        return function
    def get_opened_activities_old(self):
        return np.array(self.opened,dtype=np.int32)

    def get_opened_activities(self):
        opened = []
        for a in self.free:
            fun=self.debit_functions[a]
            if (fun.opened)&(fun.cortege is not None):
                opened.append(a)
        return np.array(opened,dtype=np.int32)

    def get_ifapply(self,executors=np.array([],dtype=np.int32),activities=np.array([],dtype=np.int32)):
        matrix=np.empty(shape=(executors.shape[0]))
        matrix.fill(np.inf)
        for i,a in enumerate(activities):
            cortege_index=self.corteges_index[a]
            tau=self.tr[a]
            if np.isnan(cortege_index):
                continue
            else:
                cortege=self.corteges[cortege_index]
            for j,e in enumerate(executors):
                if not self.ftmatrix[a,e]:
                    continue
                cw=self.groups[e]
                t=self.solved_time(other=a,current=cw,current_index=e,tracing=True)
                if t is None:
                    continue

                bounds=cortege.ifapply(a,t,t+tau)
                if bounds is None:
                    continue
                for k in bounds.keys():
                    time=bounds[k][1]
                    for s,ex in enumerate(executors):
                        if self.ftmatrix[k,ex]:
                            #cwo=self.groups[ex]
                            #t_ = self.solved_time(other=k, current=cwo, current_index=ex, tracing=True)
                            #if t_>time:
                                #continue
                            if time<matrix[s]:
                                matrix[s]=time
        return matrix








    def get_vector(self, i=0,eps=np.inf):
        cw = self.groups[i]
        k = 0
        inf = True
        # i- индекс бригады
        #span,next_well=self.get_span(i,return_next=True)
        #finit=self.finit(span=span,tolerance=self.tolerance)
        inspan=False
        next_well=None
        #if next_well is not None:
            #inspan=True
        cv = np.inf
        dt = 0.
        ck = None
        ninf=[]
        available=False
        allowed = np.zeros(shape=self.stop, dtype=bool)
        free_=False
        x=None
        y=None
        valid=False
        tau_horizon=self.executors[i].tau_horizon
        executor=self.executors[i]
        mintau_horizon=np.inf

        if self.log.record:
            log=self.log.open(i)
            log.cw=cw
            log.ct=self.ct[i]
            #log.next_well=next_well
            log.niter=self.counter
            #log.span=span


        while k < self.stop:
            valid_=False
            j = self.free[k]
            func = self.debit_functions[j]

            if func.blocked:
                k+=1
                continue
            value=self.weights[i, k]

            func.current_index = k
            allow = self.ftmatrix[j, i]
            if not allow:
                k+=1
                continue

            else:
                allowed[k]=True
                available=True

            if self.tracing:
                if not func.opened:
                    k+=1
                    continue
                if self.ct[i]>func.supp[1]:
                    k+=1
                    continue
                if (self.ct[i]<= func.supp[1]) and (self.ct[i] + eps >= func.supp[0]) and allow:
                    valid_=True
                    valid=valid or valid_
                else:
                    k+=1
                    continue


                x=self.solved_time(other=j,current=cw,
                                current_index=i,tracing=True,span_=False)
                if (x is None):
                    k+=1
                    continue
                xtau=x+func.tau

                if mintau_horizon>xtau:
                    mintau_horizon=xtau

                if xtau>tau_horizon:
                    k+=1
                    continue

            else:
                z=self.solved_time(other=j,current=cw,
                                current_index=i,tracing=self.tracing,span_=(span,next_well),return_inspan=inspan)
                if inspan:
                    x=z[0]
                    y=z[1]
                else:
                    x=z

            if self.log.record:
                wrec = well_record()
                wrec.x=x
                log.vector.update({j:wrec})

            if x is not None:
                free_=True
            else:
                k+=1
                continue


            #if next_well is not None:
                #if j == next_well:
                    #value=finit(func,x)
                #else:
            #if (not func.cs)&(y is not None):
                #value = self.fun(x=x, fun=func)

           # else:
            if self.tracing:
                penalty = self.penalty(other=j, current=cw,
                                  current_index=i, func=func)

                value = self.fun(x=x, fun=func,penalty=penalty)
            else:
                #if not func.cs:
                value = self.fun(x=x, fun=func)



            if ~np.isinf(value):
                self.weights[i, k] = value
                ninf.append(k)


            if self.log.record:
                log.vector[j].val=value

            k += 1

        if not available:
            self.available_executer[i]=False
        executor.mintau_horizon=mintau_horizon

        if ~self.weights.mask[i]:
            if (next_well is not None) and (ck is not None):
                k = ck
                self.weights.setinf(i=i, j=k, value=cv)

            else:
                if not self.tracing:
                    k=0
                    while k<allowed.shape[0]:
                        if allowed[k]:
                            j = self.free[k]
                            func = self.debit_functions[j]
                            if not func.cs:
                                tmax = self.solved_time(other=j, current=cw,current_index=i,
                                                        set_blocked=True, tracing=self.tracing)
                                value = self.fun(x=tmax, fun=func)
                                if self.log.record:
                                    log.vector[j].eval = value
                                if ~np.isinf(value):
                                    self.weights[i,k] = value
                                    ninf.append(k)
                        k+=1
                self.weights.swap(i,np.array(ninf,dtype=np.int32))
        if self.log.record:
            self.log.close()
        return  valid

    def get_vector_v1(self, i=0,eps=np.inf):
        cw = self.groups[i]
        k = 0
        inf = True
        # i- индекс бригады
        span,next_well=self.get_span(i,return_next=True)
        finit=self.finit(span=span,tolerance=self.tolerance)
        inspan=False
        if next_well is not None:
            inspan=True
        cv = np.inf
        dt = 0.
        ck = None
        ninf=[]
        available=False
        allowed = np.zeros(shape=self.stop, dtype=bool)
        free_=False
        x=np.nan
        y=np.nan
        valid=False

        if self.log.record:
            log=self.log.open(i)
            log.cw=cw
            log.ct=self.ct[i]
            log.next_well=next_well
            log.niter=self.counter
            log.span=span


        while k < self.stop:
            j = self.free[k]
            #if (i==11)&(j==260):
                #print()
            if j==next_well:
                ck=k
            func = self.debit_functions[j]
            inspan=inspan & (not func.cs)
            #func=finit(func_)
            func.current_index = k
            allow = self.ftmatrix[j, i]

            if allow:
                allowed[k]=True
                available=True

            if self.tracing:
                if allow & func.opened &(self.ct[i]<=func.supp[0]):
                    if self.ct[i] + eps < func.supp[0]:
                        self.weights[i, k] = np.NINF
                        k+=1
                        continue
                    else: valid = True


                x=self.solved_time(other=j,current=cw,
                                current_index=i,tracing=self.tracing,span_=(span,next_well))
            else:
                z=self.solved_time(other=j,current=cw,
                                current_index=i,tracing=self.tracing,span_=(span,next_well),return_inspan=inspan)
                if inspan:
                    x=z[0]
                    y=z[1]
                else:
                    x=z

            if self.log.record:
                wrec = well_record()
                wrec.x=x
                log.vector.update({j:wrec})




            if ~np.isnan(x):
                free_=True

            if allow & free_:
                if next_well is not None:
                    if j == next_well:
                        value=finit(func,x)
                        #value=self.fun(x,func)
                    else:

                        if (not func.cs)&(~np.isnan(y)):
                            value = self.fun(x=x, fun=func)

                        else:
                            value = np.NINF

                else:
                    if self.tracing:
                        penalty = self.penalty(other=j, current=cw,
                                          current_index=i, func=func)

                        value = self.fun(x=x, fun=func,penalty=penalty)
                    else:
                        if not func.cs:
                            value = self.fun(x=x, fun=func)

                        else:
                            value = np.NINF

            else:
                value = np.NINF

            if ~np.isinf(value):
                ninf.append(k)

            self.weights[i,k] = value
            if self.log.record:
                log.vector[j].val=value

            k += 1
        if not available:
            self.available_executer[i]=False

        if ~self.weights.mask[i]:
            if (next_well is not None) and (ck is not None):
                k = ck
                self.weights.setinf(i=i, j=k, value=cv)

            else:
                if not self.tracing:
                    k=0
                    while k<allowed.shape[0]:
                        if allowed[k]:
                            j = self.free[k]
                            func = self.debit_functions[j]
                            if not func.cs:
                                tmax = self.solved_time(other=j, current=cw,current_index=i,
                                                        set_blocked=True, tracing=self.tracing)
                                value = self.fun(x=tmax, fun=func)
                                if self.log.record:
                                    log.vector[j].eval = value
                                if ~np.isinf(value):
                                    self.weights[i,k] = value
                                    ninf.append(k)
                        k+=1
                self.weights.swap(i,np.array(ninf,dtype=np.int32))
        if self.log.record:
            self.log.close()
        return valid
    def get_time(self,other, current, current_index, func, tracing=False):
        x = self.ts[current, other]+self.ct[current_index]
        if tracing:
            x=self.set_bound(x, func)
        if x is None:
            return x
        x=self.get_group_support(x, current_index)
        return x

    def penalty(self,other, current, current_index, func):
        y = self.ts[current, other]+self.ct[current_index]
        x=self.set_bound(y, func)
        x=self.get_group_support(x, current_index)
        penalty=min(y-func.supp[0],0)
        return penalty+self.shrinkage*self.debit_functions[other].order

    def solved_time(self,other, current, current_index, tracing=False,set_blocked=False,span_=None,return_inspan=False):
        # expected span_=(span,target)
        def get_block(fun):
            a=[]
            for k in fun.prohibits:
                t1=self.debit_functions[k].t1
                t2 = self.debit_functions[k].t2
                a.append([t1,t2])
            return np.array(a).reshape(-1,2)

        def get_unblocked_time(x,func_):
            block_=get_block(func_)
            r=np.array([x,np.inf])
            free=op.residual(r,block_)
            amin=op.inspan(x,free,epsilon=self.tolerance,include='left',length=func_.tau,subset=True)
            if not ((amin>=func_.supp[0])&(amin<=func_.supp[1])):
                amin=None
            return amin

        x=None
        y=None
        func = self.debit_functions[other]
        if tracing:
            set_blocked=True
            return_inspan=False

        x = self.get_time(other=other, current=current, current_index=current_index, func=func, tracing=tracing)
        if x is None:
            if return_inspan:
                return x, y
            else:
                return x


        if set_blocked:
            x=get_unblocked_time(x,func)
        else:
            block = get_block(func)
            isblocked=op.inset(x,block,epsilon=self.tolerance)
            if isblocked:
                x=None
            if x is None:
                if return_inspan:
                    return x,y
                else:
                    return x

        if tracing:
            return x

        if span_ is None:
            span,target=self.get_span(current_index,return_next=True)
        else:
            span=span_[0]
            target=span_[1]

        if return_inspan:
            if target is not None:
                xhat=x+func.tau+self.ts[other,target]
                y=op.inspan(xhat,span,epsilon=self.tolerance)
            else:
                y=x
            return x,y

        return x


    def f0(self,x=0.,fun=debit_function()):
        return fun.value(x)

    def f1(self,x=0.,fun=debit_function()):
        return fun.value(x)*fun.dq

    def f2(self,x=0.,fun=debit_function()):
        # ранжирование по дебиту. Формула Елина Н.Н.
        return (fun.q0*x+fun.q1*(self.t-fun.tau-x)-(fun.q0+self.dq)*self.t)/self.t

    def f3(self,x=0.,fun=debit_function()):
        return (fun.value(x)-fun.drift)*fun.dq

    def f4(self,x=0.,fun=debit_function()):
        return (fun.value(x)-fun.drift)*fun.sign

    def f5(self,x=0.,fun=debit_function()):
        return -fun.dq*x+fun.dq*self.t-fun.q1*fun.tau-self.dq*self.t

    def f6(self,x=0.,fun=debit_function()):
        # ранжирование по дебиту
        if (x>=fun.supp[0])&(x<=fun.supp[1]):
            y=fun.dq*(self.t-fun.tau-x)
        else:
            y=np.NINF
        return y
    def f15(self,x=0.,fun=debit_function()):
        # ранжирование по дебиту
        if (x>=fun.supp[0])&(x<=fun.supp[1]):
            return fun.tail(x)
        else:
            y=np.NINF
        return y
    def cval(self,x=0.,fun=debit_function()):
        if fun.parent is None:
            return fun.kernel_value
        else:
            if fun.applied<len(fun.parent):
                return np.NINF
            else:
                if x>=fun.time+fun.edelta:
                    return fun.kernel_value
                else:
                    return np.NINF



    def f7(self,x=0.,fun=debit_function()):
        teta=fun.supp[0]-x
        if teta>=0:
            return -fun.supp[0]
        else:
            teta = x
            if teta<=fun.supp[1]:
                return -teta
            else:
                return np.NINF

    def set_bound(self,x=0.,fun=debit_function()):
        teta=fun.supp[0]-x
        if teta>=0:
            return fun.supp[0]
        else:
            teta = x
            if teta<=fun.supp[1]:
                return teta
            else:
                return None

    def f8(self,x=0.,fun=debit_function()):
        teta=fun.supp[1]-x
        if teta>=0:
            return -teta
        else:
            return np.NINF

    def f9(self,x=0.,fun=debit_function()):
        teta=-self.f7(x,fun=fun)
        d=-(teta+fun.tau)*(fun.supp[1]-teta)/(fun.supp[1]-fun.supp[0])
        return d

    def f10(self,x=0.,fun=debit_function()):
        teta=-self.f7(x,fun=fun)
        d=-(teta+fun.tau)/(fun.supp[1]-fun.supp[0])
        return d

    def f11(self,x=0.,fun=debit_function()):
        teta=-self.f7(x,fun=fun)
        d=teta+(fun.tau/(fun.supp[1]-fun.supp[0]))
        return -d


    def f12(self,x=0.,fun=debit_function()):
        teta=-self.f7(x,fun=fun)
        if teta>fun.supp[1]:
            return np.NINF
        d=teta+(fun.supp[1]-teta)/fun.tau
        return -d

    def f13(self,x=0.,fun=debit_function()):
        #ранжирование по логистике
        # i - номер группы
        teta=-self.f7(x,fun=fun)
        if np.isinf(teta):
            return np.NINF
        #teta=self.get_group_support(teta,i)
        cw=fun.index
        t=teta+fun.tau
        #t = self.get_group_support(t, i)
        d=0
        k=0
        n=0
        sum=0
        while k < self.stop:
            j = self.free[k]
            if j!=cw:
                a = self.ts[cw, j]
                teta_=t+a
                #teta_ = self.get_group_support(teta_, i)
                value=teta_-self.debit_functions[j].supp[1]
                sum+=value
                n+=1

            k += 1

        if n>0:
            return -sum/n
        else:
            return 0

    def f14(self,x=0.,fun=debit_function()):
        #ранжирование по логистике
        # i - номер группы
        def set_log_values():
            stop=self.free.shape[0]
            self.logistic_values=np.zeros(shape=stop)
            i=0
            while i<self.logistic_values.shape[0]:
                cw=self.free[i]
                k = 0
                sum=0.
                n=0
                while k < stop:
                    j = self.free[k]
                    if j != cw:
                        value =self.ts[cw, j] - self.debit_functions[j].supp[1]
                        sum += value
                        n += 1
                    k += 1
                #if n > 0:
                self.logistic_values[i]=sum
                i+=1

        if self.logistic_values is None:
            set_log_values()
            value=self.f14(x,fun)
            return value
        else:
            #teta = self.set_bound(x, fun=fun)
            #teta=x
            if np.isinf(x):
                return np.NINF
            cw = fun.current_index
            t = x + fun.tau
            n=self.free.shape[0]-1
            if n>0:
                return -(t+self.logistic_values[cw]/n)
            else:
                return -t

    def f18(self,x=0.,fun=debit_function(),penalty=0):
        #ранжирование по логистике
        # i - номер группы
        def set_log_values():
            stop=self.free.shape[0]
            self.logistic_values=np.zeros(shape=stop)
            i=0
            while i<self.logistic_values.shape[0]:
                cw=self.free[i]
                k = 0
                sum=0.
                n=0
                while k < stop:
                    j = self.free[k]
                    if j != cw:
                        value =self.ts[cw, j] - self.debit_functions[j].supp[1]
                        sum += value
                        n += 1
                    k += 1
                #if n > 0:
                self.logistic_values[i]=sum
                i+=1

        if self.logistic_values is None:
            set_log_values()
            value=self.f18(x,fun)
            return value
        else:
            #teta = self.set_bound(x, fun=fun)
            #teta=x
            if np.isinf(x):
                return np.NINF
            cw = fun.current_index
            t = x + fun.tau-penalty
            n=self.free.shape[0]-1
            if n>0:
                return -(t+self.logistic_values[cw]/n)
            else:
                return -t

    def f18c(self,x=0.,fun=debit_function(),penalty=0):
        #ранжирование по логистике
        # i - номер группы
        def get_value(fun=debit_function()):
            sum=0
            k=0
            cw=fun.index
            while k<self.free.shape[0]:
                well=self.free[k]
                func=self.debit_functions[well]
                if (cw!=well)&(func.opened):
                    value = self.ts[cw, well] - func.supp[1]
                    sum+=value
                k+=1
            return sum
        def set_log_values():
            stop=self.free.shape[0]
            self.logistic_values=np.zeros(shape=stop)
            i=0
            while i<self.logistic_values.shape[0]:
                cw=self.free[i]
                value=get_value(self.debit_functions[cw])
                self.logistic_values[i]=value
                i+=1

        if self.logistic_values is None:
            set_log_values()
            value=self.f18c(x,fun)
            return value
        else:

            if (not fun.opened)|(not op.isin2(x,fun.supp,epsilon=self.tolerance))|np.isinf(x):
                return np.NINF
            cw = fun.current_index
            t = x + fun.tau-penalty
            n=self.opened_count-1

            if n>0:
                return -(t+self.logistic_values[cw]/n)
            else:
                return -t
    def update_logistic_values(self, indices=np.array([]),incoming=dict()):
        def get_values(i=0,indices=np.array([])):
            sum=0
            for j in indices:
                if i==j:
                    return None
                value = self.ts[i, j] - self.debit_functions[j].supp[1]
                sum += value
            return sum

        if self.logistic_values is None:
            return

        if len(incoming)>0:
            self.set_compact(incoming)


        if indices.shape[0]==0:
            return

        k=0
        mask=np.ones(shape=self.logistic_values.shape[0],dtype=bool)
        while k<self.logistic_values.shape[0]:
            j=self.free[k]
            value=get_values(i=j,indices=indices)
            if value is None:
                mask[k]=False
            else:
                self.logistic_values[k]=self.logistic_values[k]-value
            k+=1
        self.logistic_values=self.logistic_values[mask]
        self.opened_count-=indices.shape[0]

    def reset_compact(self,cortege):
        def value(i=0,index=np.array([])):
            sum=0
            for j in index:
                if i==j:
                    return None
                val=self.ts[i,j]-self.debit_functions[j].supp[1]
                sum+=val
            return sum
        if not (type(cortege)==Cortege):
            return
        if not cortege.isopened():
            return

        bounds=cortege.bounds
        index = np.array(list(bounds.keys()))
        self.opened_count -= index.shape[0]
        i=0
        while i<self.free.shape[0]:
            cw=self.free[i]

            val=value(cw,index)
            if val is not None:
                if self.logistic_values is None:
                    y=self.fun(x=0,fun=self.debit_functions[cw])
                self.logistic_values[i]-=val
            i+=1
        for k in bounds.keys():
            self.debit_functions[k].reset_cortege()
            self.support[k,[0,1]]=np.nan
            self.added-=1
            self.localy_added-=1
            self.opened.remove(k)
            #self.opened_activities.remove(k)
        cortege.reset()

    def set_compact(self,bounds=dict({0:np.array([0,np.inf])}),order=0):
        if bounds is None:
            return
        def value(i=0,index=np.array([])):
            sum=0
            for j in index:
                if i==j:
                    return None
                val=self.ts[i,j]-self.debit_functions[j].supp[1]
                sum+=val
            return sum

        def realise_executors(activities=np.array([],dtype=np.int32)):
            i=0
            while i<self.ftmatrix.shape[1]:
                if self.executors[i].paused:
                    for a in activities:
                        if self.ftmatrix[a,i]:
                            self.executors[i].paused=False
                            break
                i+=1


        for k in bounds.keys():
            supp=bounds[k]
            func=self.debit_functions[k]
            func.supp=supp
            func.opened=True
            func.order=order
            self.support[k,0]=supp[0]
            self.support[k, 1] = supp[1]
            self.added+=1
            self.localy_added+=1
            self.opened_activities.append(k)
            #print('opened ',k,supp)
        index=np.array(list(bounds.keys()))
        realise_executors(activities=index)
        self.update_mintau([],index)
        i=0
        while i<self.free.shape[0]:
            cw=self.free[i]
            val=value(cw,index)
            if val is not None:
                if self.logistic_values is None:
                    y=self.fun(x=0,fun=self.debit_functions[cw])
                self.logistic_values[i]+=val
            i+=1
        self.opened_count+=index.shape[0]

    def open_compact_old(self,cw=0,t1=0,t2=0):
        try:
            cortege_name=self.corteges_index[cw]

            cortege=self.corteges[cortege_name]
            if not cortege.isopened():
                cortege.apply = cortege.get_position()
                cortege.apply(t1, t1)
                self.set_compact(cortege.bounds)
                self.opened_corteges+=1
                for e in self.executors.keys():
                    self.executors[e].apply(cortege_name)

            bounds=cortege.current.apply(cw,t1,t2)
            if bounds is not None:
                cortege.apply(bounds[0], bounds[1])
                return cortege.bounds

        except KeyError:
            return None

    def open_compact(self,cw=0,t1=0,t2=0):
        try:
            cortege_name=self.corteges_index[cw]
            cortege=self.corteges[cortege_name]
            if not cortege.isopened():
                bounds=cortege.Apply(t1,t1)
                self.set_compact(bounds)
                self.opened_corteges+=1
                for e in self.executors.keys():
                    self.executors[e].apply(cortege_name)
            bounds=cortege.Apply(t1,t2,cw)
            if bounds is not None:
                self.set_compact(bounds,cortege.position_index)
            return

        except KeyError:
            return

    def optim(self,t):
        s=0
        for k in self.debit_functions.keys():
            fun=self.debit_functions[k]
            if fun.dq>=0:
                s+=fun.value(0)
            else:
                s+=fun.value(t)
        return s

    def get_queue(self,i,j=None):
        current=0
        try:
            well=self.queue[i]
            if j is not None:
                current=j
            else:
                current=well.i
            data=well.get_values(current)
            well.i+=1
            return data,current
        except KeyError:
            return None,current


    def get_optimized_trajectories(self,indices=np.array([0])):
        #t1 = time.perf_counter()
        #i_=15
        if (not self.tracing) & self.permutation:
            weigth,sindex=self.get_route_weigths()

            if weigth.shape[0]>0:
                localswap, s = self.function(weigth, criterion='max', engine=self.engine)
                #localswap, s = self.function(weigth, criterion='min', engine=self.engine)
                swap=sindex[localswap[1]]
                self.current_index[sindex]=self.current_index[swap]
                self.current_indey[sindex] = self.current_indey[swap]


        vectors=self.get_weights(indices=indices)


        if vectors is  None:
            return None

        try:
            taken,s=self.function(vectors,criterion='max',engine=self.engine)

        except (IndexError):
            return None



        if self.transpose:
            index=self.nempty.copy()
            self.nempty=taken[1].copy()
            taken[1]=index

        #mask=self.block(taken)
        #taken=taken[:,mask]
        #self.nempty=self.nempty[mask]

        if self.transpose:
            self.transpose = False

        return taken,s

    def block(self,indices=np.array([])):

        def get_keys(indices=np.array([])):
            code = np.empty(shape=indices.shape[1], dtype=np.int32)
            code.fill(-1)
            #code=[]
            k = 0
            while k < indices.shape[1]:
                i = indices[1][k]
                w=self.free[i]
                fun = self.debit_functions[self.free[i]]
                if fun.key is not None:
                    code[k]=fun.key
                k += 1
            return code

        def get_separated(y=np.array([])):
            mask = np.ones(shape=y.shape[1], dtype=bool)
            k = 0
            index = [1, 2]
            values = y[0]
            sindex = reverse(np.argsort(values))
            x = y[:, sindex]
            i = np.arange(mask.shape[0])[sindex]
            while k < mask.shape[0]:
                if mask[k]:
                    t = x[index, k]
                    j = k + 1
                    while j < mask.shape[0]:
                        if mask[j]:
                            t_ = x[index, j]
                            interseption = op.interseption(t, t_, shape=2)
                            if interseption.shape[0] > 0:
                                mask[j] = False

                        j += 1
                k += 1
            return i[~mask], i[mask]

        mask = np.ones(shape=indices.shape[1], dtype=bool)
        keys=get_keys(indices)

        uniques,counts=np.unique(keys,return_counts=True)
        x_index = self.nempty[indices[0]]  # -номера строк матрицы self.weight
        y_index = self.groups[x_index]  # -номера скважин, на которых находятся бригады x_index

        if self.transpose:
            values = self.weights.array.T[x_index, indices[1]]
        else:
            values = self.weights.array[x_index, indices[1]]

        m_index=self.free[indices[1]]
        #m_index=self.unical[indices[1]]

        k=0
        while k<counts.shape[0]:
            c=counts[k]
            if c>1:
                key=uniques[k]
                if key>=0:
                    index=np.where(keys==key)[0]
                    vector=np.empty(shape=(3,index.shape[0]))
                    for i, j in enumerate(index):
                        val = values[j]
                        w = m_index[j]
                        #if (w==1143)|(w==1113):
                            #print()
                        cw = y_index[j]
                        ci = x_index[j]
                        func=self.debit_functions[w]

                        t1_ = self.get_time(other=w, current=cw,
                                          current_index=ci, func=func,
                                          tracing=self.tracing)
                        #t1_ = self.ct[ci] + self.ts[cw, w] + self.dt[ci]
                        t2_ = t1_ + self.tr[w]
                        vector[0, i] = val
                        vector[1, i] = t1_
                        vector[2, i] = t2_

                    get_out, leave = get_separated(vector)
                    mask[index[get_out]] = False


            k+=1
        return mask


    def update_bounds(self, fun=debit_function()):
        bounds=np.array([fun.t1,fun.t2],dtype=np.float64)
        for p in fun.prohibits:
            try:
                cfun=self.debit_functions[p]
                cfun.bounds[fun.index]=bounds
            except KeyError:
                continue

    def print_traj(self,traj):
        for i, j in enumerate(traj):
            if j < 0:
                continue
            print(i, j, sep=':', end=' ')
        print('')

    def update_mintau(self,outcome,income):
        if not self.tracing:
            return
        for e in self.executors.keys():
            mintau_horizon=self.executors[e].mintau_horizon
            cw=self.groups[e]
            continue_=True
            for other in outcome:
                if self.ftmatrix[other,e]:
                    fun=self.debit_functions[other]
                    x=self.solved_time(other,cw,e,tracing=True)
                    if x is None:
                        continue
                    xtau=x+fun.tau
                    if xtau<=mintau_horizon:
                        self.executors[e].mintau_horizon=0
                        continue_=False
                        break
            if not continue_:
                continue
            for other in income:
                if self.ftmatrix[other,e]:
                    fun=self.debit_functions[other]
                    x=self.solved_time(other,cw,e,tracing=True)
                    if x is None:
                        continue
                    xtau=x+fun.tau
                    if xtau<=mintau_horizon:
                        mintau_horizon=xtau
            self.executors[e].mintau_horizon=mintau_horizon
        return
    def update(self,indices=np.array([]),s=0):
        def mark_as_blocked():
            for f in self.free:
                fun=self.debit_functions[f]

                if (not fun.opened)|(fun.blocked):
                    continue
                cortege_index = self.corteges_index[f]
                if ~np.isnan(cortege_index) and self.corteges[cortege_index].blocked:
                    fun.blocked=True
                    continue
                blocked=True
                for e in self.executors.keys():
                    if self.ftmatrix[f,e]&self.available_executer[e]:
                        if self.ct[e]<fun.supp[1]:
                            blocked=False
                            break
                fun.blocked=blocked
                if ~np.isnan(cortege_index) and blocked:
                    self.corteges[cortege_index].blocked=blocked



        def check4reset(index):
            def check(index):
                removable=[]
                for w in index:
                    fun=self.debit_functions[w]
                    for e in self.numbers[self.available_executer]:
                        go=True
                        if self.ftmatrix[w,e]:
                            cw=self.groups[e]
                            t = self.solved_time(other=w, current=cw,
                                                 current_index=e, tracing=self.tracing)

                            if((t is not None) and (t<fun.supp[1])):
                                go=False
                                break
                    if go:
                        removable.append(w)
                return removable
            removable=check(index)
            self.opened.remove(removable)

            for i in removable:
                cortege_name=self.corteges_index[i]
                if (cortege_name is None):
                    continue
                cortege=self.corteges[cortege_name]
                if cortege.confirm:
                    continue
                self.reset_compact(cortege)
            return


        index=self.free[indices[1]]
        self.cd=self.cd+s
        mask=np.zeros(index.shape[0],dtype=bool)
        index_x=np.empty(self.groups.shape[0], dtype=int)
        index_y=np.empty(self.groups.shape[0], dtype=np.float32)
        index_z=np.empty(self.groups.shape[0], dtype=np.float32)
        index_w = np.zeros(self.groups.shape[0], dtype=int)
        index_x.fill(-1)
        index_y.fill(-1)
        index_z.fill(-1)

        #в первую очередь обходим мероприятия, выбранные в результате кастинга
        #source=self.available_executer[self.available_executer].shape[0]
        #self.source=0
        injected=0
        self.localy_added = 0

        for i,w in enumerate(index):
            j=self.nempty[i]

            if self.tracing & self.executors[j].injected:
                injected+=1
                continue

            current_well=self.groups[j]
            fun = self.debit_functions[w]
            set_blocked=True
            #if fun.cs:
                #set_blocked=False


            st = self.solved_time(other=w, current=current_well, current_index=j,
                                    set_blocked=set_blocked, tracing=self.tracing)

            if st is not None:
                if fun.cs:
                    st=max(st,fun.t1)
                self.st[j] =st
                self.ct[j] = self.st[j] + self.tr[w]
                fun.used=True
                mask[i] = True
                self.source+=1
                fun.t1=self.st[j]
                fun.t2=self.ct[j]
                fun.executor=j
                if self.tracing:
                    self.open_compact(w,fun.t1, fun.t2)
                    self.opened.remove(w)
                else:
                    self.apply_kernel(fun)
                #bounds=self.open_compact(fun.t1,fun.t2,w)
                #if bounds is not None:
                    #self.set_compact(bounds)

        # во вторую очередь обходим мероприятия, назначенные принудительно и инициирующие кортеж

        if self.tracing:

             for i,w in enumerate(index):
                j=self.nempty[i]
                if self.tracing & (not self.executors[j].injected):
                    continue
                self.executors[j].paused = False
                self.executors[j].injected=False
                current_well=self.groups[j]
                fun = self.debit_functions[w]
                set_blocked=True

                st = self.solved_time(other=w, current=current_well, current_index=j,
                                        set_blocked=set_blocked, tracing=self.tracing)

                if st is not None:
                    if fun.cs:
                        st=max(st,fun.t1)
                    self.st[j] =st
                    self.ct[j] = self.st[j] + self.tr[w]
                    fun.used=True
                    #self.localy_added-=1
                    mask[i] = True
                    fun.t1=self.st[j]
                    fun.t2=self.ct[j]

                    fun.executor = j
                    self.open_compact(w,fun.t1, fun.t2)
                    self.opened.remove(w)
                    #print('taken ',w)
                    #bounds=self.open_compact(w,fun.t1,fun.t2)
                    #if bounds is not None:
                        #self.set_compact(bounds)



        index_y[:]=self.st
        index_z[:]=self.ct
        #index_y=self.st.copy()
        #index_z=self.ct.copy()
        #index_x=index[self.nempty]



        assert mask[mask==True].shape[0]>0,"No one activities has been taken!"

        index=index[mask]
        #print('recorded=',index.shape[0])
        opened=np.array(self.opened_activities,dtype=np.int32)
        self.percent=0
        if self.opened_previous.shape[0]>0:
            ma=np.isin(self.opened_previous,index)
            self.percent = ma[ma].shape[0] / ma.shape[0]
        self.opened_previous=opened


        self.opened_activities=[]
        self.nempty=self.nempty[mask]
        indices=indices[:,mask]
        self.update_schedule(index=index)
        self.update_logistic_values(indices=index)
        index_x[self.nempty] = index
        self.groups[self.nempty]=index
        self.mask=np.ones(self.free.shape[0],dtype=bool)
        self.mask[indices[1]]=False
        self.free=self.free[self.mask]
        self.available_activities=np.zeros(shape=self.stop,dtype=bool)
        self.infmask = np.zeros(shape=self.stop, dtype=bool)
        self.infindex = np.zeros(shape=self.stop, dtype=np.int32)
        self.minct = self.ct[self.available_executer].min()
        self.maxct=self.ct[self.available_executer].max()
        self.trajectories.append([index_x,(index_y,index_z,index_w)])
        check4reset(self.opened)
        mark_as_blocked()
        self.update_mintau(outcome=index,income=[])
        #self.print_traj(index_x)

    def update_schedule(self,index=np.array([])):
        if self.routes is not None:
            k=0
            for i in self.nempty:

                ci=self.current_index[i]
                yci = self.current_indey[i]
                if ci<self.routes.shape[0]:
                    well=self.routes[ci,yci]
                    taken=index[k]

                    if taken==well:
                        #if well==1796:
                            #print()
                        self.current_index[i]=ci+1
                k+=1
    def get_corteges_vector(self,cortege,tau_vector):
        def get_max(position,tau_vector,ahat):
            maxa = 0
            for k in position.keys():
                a = position[k][0]
                alpha=position[k][2]
                tau = tau_vector[k]
                ahat_=a+tau+ahat*alpha
                if (ahat_) > maxa:
                    maxa = ahat_
            return maxa

        #vector=dict()
        def go(cortege,tau_vector,index=0):
            position=cortege.get_bounds(index)
            if index==0:
                maxa=get_max(position,tau_vector,0)
                vector[index]=maxa
                return maxa
            else:
                ahat=go(cortege,tau_vector,index-1)
                maxa=get_max(position,tau_vector,ahat)
                vector[index]=maxa
                return maxa
        i=cortege.count()

        if i<1:
            return np.array([])
        vector = np.empty(shape=i)
        go(cortege,tau_vector,i-1)
        return vector
    def get_executers_vector(self,cortege,tau_vector):
        def get_executers_ct(activities,criterion='mean'):
            def get_value(vector):
                if criterion=='mean':
                    return vector.mean()
                if criterion=='max':
                    return vector.max()
                else:
                    return vector.min()

            if len(activities)==0:
                return np.array([])
            time=np.empty(len(activities))
            for i,a in enumerate(activities):
                t=[]
                for e in self.executors.keys():
                    if self.ftmatrix[a,e]:
                        #if self.ct[e]<t:
                        t.append(self.ct[e])
                time[i]=get_value(np.array(t))
            return time


        def get_time(position,executers_time):
            activities=list(position.keys())
            time=get_executers_ct(activities,'max')
            return time.max()-executers_time


        vector=self.get_corteges_vector(cortege,tau_vector)
        n=cortege.count()
        tvector=np.empty(n)
        i=0
        etime=0

        while i<n:
            position=cortege.get_bounds(i)
            if i>0:
                etime=vector[i-1]
            time=get_time(position,etime)
            tvector[i]=time
            i+=1

        return tvector

    def get_activities_distance(self,activitie1,activitie2,cortege,tau_vector,corteges_index,corteges_position):
        def get_distance(a,b,cortege):
            distance=0
            start=False
            for k in cortege.position.keys():
                if (k!=a)&(not start):
                    continue
                if k==a:
                    start=True
                if k==b:
                    return distance
                acts=list(cortege.position[k].diction.keys())
                d=tau_vector[acts].max()
                distance+=d
            return 0
        cid1=corteges_index[activitie1]
        cid2 = corteges_index[activitie2]
        distance=0
        if cid1!=cid2:
            return 0
        if cid1!=cortege.index:
            return 0
        p1=corteges_position[activitie1]
        p2 = corteges_position[activitie2]
        if p1==p2:
            return 0
        if p1<p2:
            distance=get_distance(p1,p2,cortege)
        else:
            distance = get_distance(p2, p1, cortege)

        return distance

    def get_applied_kernel(self):
        ci=dict()
        for cid in self.corteges.keys():
            cortege=self.corteges[cid]
            ker=self.get_kernel(cortege,self.ftmatrix,self.kernel)
            if len(ker.keys())>0:
                d = dict()
                for k in ker.keys():
                    acts = ker[k]
                    d_ = dict()
                    for a in acts:
                        fun = self.debit_functions[a]
                        if fun.used:
                            d_[a] = (fun.t1, fun.t2, fun.executor)
                    d[k] = d_
                if len(d.keys())>0:
                    ci[cid]=d
        return ci





    def update_queue(self,w=0,i=None):
        queue,current = self.get_queue(w,i)
        if queue is not None:
            fun = self.debit_functions[w]
            fun.supp = queue[[0, 1]]
            fun.tau = queue[2]
            fun.q0 = queue[3]
            fun.q1 = queue[4]
            fun.update()
            self.tr[w] = queue[2]
            self.Q0[w] = queue[3]
            self.Q1[w] = queue[4]
            self.dQ[w] = queue[4] - queue[3]
            return True, current
        else:
            return False,current
    def nempty_corteges(self):
        empty=False
        for cid in self.corteges.keys():
            if not self.corteges[cid].isopened():
                return True
        return empty

    def isvalid(self,igroup,well):
        fun=self.debit_functions[well]
        try:
            service=self.service[igroup]
            equipment=self.equipment[igroup]
        except IndexError:
            return True
        serv=True
        equip=True
        for j in fun.service:
            val=service[j]
            if val==0:
                serv=False
                break
        for e in fun.equipment:
            val=equipment[e]
            if val==0:
                equip=False
                break

        return serv&equip



    def get_routes(self):
        self.counter=0
        self.counters.update({"niter":[],"free":[],"opened":[],"executers":[],"opened_act":[],
                              "added":[],"percent":[],"time":[]})
        try:

            while self.free.shape[0]>0:
                self.opened_activities = []
                res=self.get_optimized_trajectories(self.numbers[self.available_executer])
                self.counter+=1
                self.counters["niter"].append(self.counter)
                self.counters["free"].append(self.free.shape[0])
                self.counters["opened"].append(self.opened_corteges)
                self.counters["opened_act"].append(self.opened_count)
                self.counters["executers"].append(self.available_executer[self.available_executer].shape[0])
                self.counters["added"].append(self.added)
                self.counters["percent"].append(self.percent)
                self.counters['time'].append([self.ct.copy(),self.opened_previous])
                self.added=0


                #if self.counter==2:
                    #self.counter=self.counter


                #self.flag=True
                #if self.flag:
                    #print(self.counter,self.executors[22].tau_horizon,self.ct[22])
                print(self.counter,self.free.shape[0])
                #if self.counter==52:
                    #self.counter=self.counter
                #print(self.counter,self.free.shape,self.opened_corteges,self.available_executer[self.available_executer].shape)
                if res is None:
                    break
                indices=res[0]
                s = res[1]
                self.update(indices,s)


            return self.trajectories
        except AssertionError:
            if self.log.record:
                with open(os.path.join(os.getcwd(),'log.sav'),'wb') as file:
                    pickle.dump(self.log,file)

    def update_restrictions(self,index=np.array([],dtype=np.int32)):
        if index.shape[0]==0:
            return
        for k in index:
            i=0
            while i<self.routes.shape[1]:
                if not self.ftmatrix[k,i]:
                    i+=1
                    continue
                j=0
                while j<self.routes.shape[0]:
                    well=self.routes[j,i]
                    self.prohibits[k,well]=False
                    if well<0:
                        break
                    j+=1
                i+=1

    def to_date_old(self,sdate=np.datetime64('2023-01-01'),R=np.array([]),T=np.array([]),D=np.array([])):
        diction=dict({})
        i=0
        while i<R.shape[1]:
            j=0
            while j<R.shape[0]:
                act=R[j,i]
                if act<0:
                    break
                t1=D[j,i]
                t2=T[j,i]
                t1_=dts(t1)
                t2_=dts(t2)
                sd=sdate+pd.DateOffset(seconds=t1_)
                ed = sdate + pd.DateOffset(seconds=t2_)

                try:
                    q0=self.Q0[act]
                    q1=self.Q1[act]
                except IndexError:
                    q0 = 0
                    q1 = 0
                diction[act]=dict({'exec':i,"begin":sd,"end":ed,"q0":q0,"q1":q1})

                j+=1

            i+=1
        return diction
    def to_date(self,debit_functions,sdate=np.datetime64('2023-01-01')):
        diction=dict({})
        i=0
        for f in debit_functions.keys():
            fun=debit_functions[f]
            if fun.used:
                t1=fun.t1
                t2=fun.t2
                q0=fun.q0
                q1=fun.q1
                t1_=dts(t1)
                t2_=dts(t2)
                i=fun.executor
                sd=sdate+pd.DateOffset(seconds=t1_)
                ed = sdate + pd.DateOffset(seconds=t2_)
                diction[f]=dict({'exec':i,"begin":sd,"end":ed,"q0":q0,"q1":q1})

        return diction



def issubset (A=np.array([]),B=np.array([])):
    ax=A[0]
    ay=A[1]
    bx=B[0]
    by=B[1]
    mask=(bx>=ax)&(by<=ay)
    if mask:
        return True
    else:
        return False
def getsubsetmask(A=np.array([]),B=np.array([])):
    index=[]
    for i in np.arange(B.shape[1]):
        m=issubset(A[:,i],B[:,i])
        #index.append(m)
        if m:
            index.append(i)
    return np.array(index, dtype=int)

def get_sub_routes(w=0,t0=0,ts=np.array([]),ti=np.array([]), bounds=np.array([]),index=np.array([])):
    #print(index)
    left=ts[w,index[1]]+t0
    right=left+ti[index[1]]
    bnd=np.array([left,right])
    #print(bnd)
    #print(index[0])
    #print(bounds)
    indices=getsubsetmask(bounds[:,index[0]],bnd)
    return index[:,indices]

def getsub(w=np.array([]),t0=0,ts=np.array([]),ti=np.array([]), bounds=np.array([]),index=np.array([], dtype=int),mask=np.array([], dtype=bool),INDICES=[],trajectories=[], maxn=0, maxiter=np.inf):
    x0=t0
    indices=get_sub_routes(w=w,t0=x0,ts=ts,ti=ti, bounds=bounds,index=index[:,mask])
    if (indices.shape[1]>0)&(maxn<maxiter):
        maxn=maxn+1
    else:
        trajectories.append((INDICES,maxn,x0))
        return
    for i in np.arange(indices.shape[1]):
        t0=x0+ts[w,indices[:,i][1]]+ti[indices[:,i][1]]
        j=INDICES.copy()
        j.append(indices[:,i][0])
        mask[indices[:,i][0]]=False
        getsub(w=indices[:,i][1],t0=t0,ts=ts,ti=ti, bounds=bounds,index=index,INDICES=j,mask=mask, maxn=maxn,trajectories=trajectories, maxiter=maxiter)
        mask[indices[:,i][0]]=True
    return trajectories

def get_debit(wells=np.array([]),time_=np.array([]),wsch=wells_schedule(),mt=100):
    #mt -горизонт расчета
    s=np.empty(shape=wells.shape)
    s.fill(0.)
    for i in np.arange(wells.shape[1]):
        for j in np.arange(wells.shape[0]):
                w=wells[j,i]
                t=time_[j,i]
                if w>=0:
                    fun=wsch.debit_functions[w]
                    fun.t=mt
                    fun.teta = fun.q1 * (fun.t - fun.tau)
                    s_=fun.value(t)
                    s[j,i]=s_
    return s

def get_rout_time(routes):
    shape0=len(routes)
    shape1=routes[0][0].shape[0]
    route=np.ones(shape=(shape0,shape1),dtype=np.int32)*-1
    time = np.ones(shape=(shape0, shape1), dtype=np.float)*-1
    delta = np.ones(shape=(shape0, shape1), dtype=np.float)*-1
    order = np.ones(shape=(shape0, shape1), dtype=np.int32)*-1
    for i in np.arange(shape1):
        k=0
        for j in np.arange(shape0):
            ro = routes[j][0][i]
            if ro>=0:
                route[k,i]=ro
                time[k,i]=routes[j][1][1][i]
                delta[k,i]=routes[j][1][0][i]
                order[k, i] = routes[j][1][2][i]
                k+=1
    return route,time,delta,order


def debit(group=np.array([]),R=np.array([]),tr=np.array([]),ts=np.array([]),wsch=wells_schedule()):
    X=np.vstack((group,R))
    s=np.empty(shape=R.shape)
    for i in np.arange(X.shape[1]):
        x=X[:,i]
        t=0
        j=1
        while j<x.shape[0]:
            t=t+ts[x[j-1],x[j]]
            s_=wsch.debit_functions[x[j]].value(t)
            s[j-1,i]=s_
            t=t+tr[x[j]]
            j+=1

    return s

def get_month_range(month,year=2017):
    month=int(month)
    if month>9:
        month_=str(month)
    else:
        month_='0'+str(month)

    start=np.datetime64(str(year)+'-'+str(month_)+'-'+'01')
    last=calendar.monthrange(year,month)[1]
    end=np.datetime64(str(year)+'-'+str(month_)+'-'+str(last))
    return start,end

def get_quarter_range(quarter,year=2017):
    if quarter<1:
        return np.nan,np.nan
    quarter=int(quarter)
    a=int(quarter/5)
    b=np.fmod(quarter-1,4)
    c=np.array([1,2,3])
    #q=c[int(b)]-1
    year=year+a
    monthes=c+b*3
    #print(monthes)
    #print(year)
    start=get_month_range(monthes[0],year)[0]
    end=get_month_range(monthes[-1],year)[1]
    return start,end

def get_distances(data,fields,well='ID',field='Месторождение'):
    agg=data[[field,well]].groupby(well)
    first=agg.first()
    shape=first.shape[0]
    zeros=np.zeros(shape=(shape,shape))
    index=first.index

    for i in np.arange(first.shape[0]):
        f=first.iloc[i][0]
        #w=first.index[i]
        for j in np.arange(i+1,first.shape[0]):
            #w_=first.index[j]
            f_=first.iloc[j][0]
            fd=fields.loc[f,f_]
            zeros[i,j]=fd+6
            zeros[j,i]=fd+6
    return pd.DataFrame(data=zeros,index=index,columns=index)


def get_distances_by_coordinates(data=np.array([[0,0]])):
    if data.shape[0]<=1:
        return np.array([0])
    distances=np.zeros(shape=(data.shape[0],data.shape[0]))
    i=0
    while i<data.shape[0]:
        latitude=data[i,0]
        longtitude=data[i,1]
        j=i+1
        while j<data.shape[0]:
            latitude_ = data[j, 0]
            longtitude_ = data[j, 1]
            d=cppmath.haversine(latitude,longtitude,latitude_,longtitude_)
            #d = cppmath.distance(latitude,longtitude,latitude_, longtitude_)
            distances[i, j] = d
            distances[j, i] = d
            j+=1
        i+=1
    return distances


def set_unical_well_index(wells=np.array([]),index=np.array([])):
    indices=np.zeros(wells.shape[0],dtype=np.int32)
    for i,j in enumerate(index):
        mask=np.where(wells==j)[0]
        indices[mask]=i
    return indices



def try_swap(a=np.array([]),b=np.array([]),funa=debit_function(),funb=debit_function(),l=0,bound=0,delta=0,eps=1e-3):
    suppa=funa.supp
    suppb=funb.supp
    interseption= op.interseption(suppa, suppb, shape=2)
    if interseption.shape[0]==0:
        return False
    s=funa.scaled_v1(a[0])+funb.scaled_v1(b[0])
    print(s)
    t1=interseption[0]
    t2=t1+b[1]-b[0]
    t3=t2+l
    t4=t3+a[1]-a[0]
    print(interseption)
    print(t1,t2,t3,t4)
    if abs(t4+delta-bound)<eps:
        s_ = funb.scaled_v1(t1) + funa.scaled_v1(t3)
        print(s_)
        if s_>s:
            return np.array([t1,t2]),np.array([t3,t4])
    return False

def get_delta(t1=0,t2=0,t3=0,ts=0,bound=0):
    d1=t3-t2-ts
    d2=bound-t1
    delta=min(d2,d1)
    return delta


def get_queue(data,field='Скважина'):
    aggdata = data.groupby(field)
    first = aggdata.first()
    queue = dict({})
    for i, group in enumerate(aggdata):
        ID = group[0]
        p = group[1]
        # mask=p['Фиксация на месяц проведения']=='фиксация'
        mask = (~np.isnan(p['Start'])) & (p['Начатые операции'] != 'Начатое')
        q = p[mask]

        start = []
        end = []
        tau = []
        q0 = []
        q1 = []
        t = 0
        if q[mask].shape[0] > 1:
            for k in q.index:
                row = q.loc[k]
                #if t > 0:
                    # if row['Фиксация на месяц проведения']=='фиксация':
                start.append(row['Start'])
                end.append(row['End'])
                tau.append(row['Продолжительность ремонта'])
                q0.append(row['Stop oil rate'])
                q1.append(row['Start oil rate'])
                t += 1
        if t > 0:
            index = first.index.get_loc(ID)
            array = queues(np.array([start, end, tau, q0, q1]))
            queue.update({index: array})
    return first, queue


def get_otm_binary(otm_wells=np.array([]), otm_services=np.array([]), wells=pd.Index([]), services=np.array([]),
                   otm_prohibits=np.array([])):
    k = otm_wells.shape[0]
    n = k + wells.shape[0]
    binary = np.ones(shape=(otm_wells.shape[0], n), dtype=bool)

    i = 0
    while i < k:
        index = otm_wells[i]
        serv = otm_services[i]
        try:
            well = wells.get_loc(index)
            serv1 = services[well]
            for s in serv:
                for s_ in serv1:
                    if not otm_prohibits[s_, s]:
                        binary[i, k + well] = False

        except KeyError:
            i += 1
            continue
        i += 1
    return binary

def get_debitfun_dict(bounds=np.array([]),start=0):
    debit_functions=dict({})
    k=0
    while k<bounds.shape[0]:
        fun=debit_function()
        t=bounds[k]
        fun.t1=t[0]
        fun.t2=t[1]
        debit_functions.update({k+start:fun})
        k+=1
    return debit_functions

def set_cathegory(x,columns=np.array([])):
    cathegory=[]
    for i,c in enumerate(columns):
        if c==x:
            cathegory.append(i)
            break
    return np.array(cathegory)

def set_loc_index(index=np.array([]), R=np.array([]), T=np.array([]), D=np.array([])):
    i = 0
    res = np.empty(shape=(index.shape[0], 2))
    res.fill(np.NINF)
    while i < R.shape[1]:
        j = 0
        while j < R.shape[0]:
            ind = R[j, i]
            if ind < 0:
                break
            t2 = T[j, i]
            t1 = D[j, i]
            res[ind, 0] = t1
            res[ind, 1] = t2

            j += 1
        i += 1
    return res

class Item:
    def __init__(self,index=0,a=0,b=0,alpha=0):
        assert ((type(index)==int)|(type(index)==np.int16)|(type(index)==np.int32)|(type(index)==np.int64)),\
            "index must be an integer"
        self.index=index
        assert (a>=0)&(b>=0)&(alpha>=0),"a,b and alpha must be not negative. error on  index "+str(self.index)
        self.a=a
        self.b=b
        self.alpha=alpha
        self.t1=0.
        self.t2=0.
        self.used=False
    def set_bounds(self,x=0,xtau=0):
        assert (x<=xtau),"left bound must be less or equal than right one "

        if self.alpha==0:
            t=x
        else:
            t=xtau

        self.t1=t+self.a
        self.t2=t+self.b

        if self.t1>self.t2:
            self.t1=self.t2

        return self.t1,self.t2
    def get_bounds(self,x=0,xtau=0):
        assert (x<=xtau),"left bound must be less or equal than right one "

        if self.alpha==0:
            t=x
        else:
            t=xtau
        t1=t+self.a
        t2=t+self.b
        if t1>t2:
            t1=t2
        return t1,t2

class ListItems:
    def __init__(self):
        self.diction=dict()
        self.index=dict()
        self.count=0
        self.current_index=0
        self.applied_number=0
        self.mint=np.inf
        self.maxt=np.NINF

    def insert(self,item=Item()):
        assert isinstance(item,Item),"allowed only Item type"
        try:
            self.diction[item.index]
        except KeyError:
            self.diction[item.index]=item
            self.index[self.get_count()-1]=item.index

    def apply(self,index,t1=0,t2=0):
        try:
            item=self.diction[index]
            if not item.used:
                self.applied_number+=1
                item.used=True
            if t1<self.mint:
                self.mint=t1
            if t2>self.maxt:
                self.maxt=t2
            if self.is_done():
                return np.array([self.mint,self.maxt])
            else:
                return None
        except IndexError:
            pass

    def __iter__(self):
        self.count=len(self.index)
        self.current_index=0
        return self

    def __next__(self):
        if self.current_index<self.count:
            j=self.index[self.current_index]
            val=self.diction[j]
            self.current_index+=1
            return val
        else:
            raise StopIteration

    def __getitem__(self, item):
        try:
            val=self.diction[item]
            return val
        except KeyError:
            return None

    def iloc(self, index=0):
        assert ((type(index) == int) | (type(index) == np.int16) | (type(index) == np.int32) | (type(index) == np.int64)),\
            "index must be an integer"
        try:
            i=self.index[index]
            return self.diction[i]
        except KeyError:
            return None


    def get_count(self):
        return len(self.diction)

    def set_bounds(self,x=0,xtau=0,pass_used=True):
        assert (x <= xtau), "left bound must be less or equal than right one "
        res=dict()
        for key in self.diction.keys():
            item=self.diction[key]
            if pass_used:
                if item.used:
                    continue

            item.set_bounds(x=x, xtau=xtau)
            res.update({key: np.array([item.t1, item.t2])})
            res.update({key: np.array([item.t1, item.t2])})
        return res

    def get_bounds(self,x=0,xtau=0):
        assert (x <= xtau), "left bound must be less or equal than right one "
        res=dict()
        for key in self.diction.keys():
            item=self.diction[key]
            t1,t2=item.get_bounds(x=x,xtau=xtau)
            res.update({key:np.array([t1,t2])})
        return res
    def is_done(self):
        if self.applied_number<self.get_count():
            return False
        else:
            return True



class Cortege:
    def __init__(self):
        self.position=dict()
        self.position_index=-1
        self.iloc=dict()
        self.groupby='position'
        self.columns=['cortege_activity_id', 'position',
                      'begin_from', 'begin_to','from_end']
        self.index=0
        self.bounds=dict()
        self.current=dict()
        self.apply=dict()
        self.dQsum=0
        self.dTausum=0
        self.n=0
        self.reserved=False
        self.confirm=False
        self.blocked=False

    def val(self,t):
        return (t-self.dTausum)*self.dQsum
    def reset(self):
        self.bounds=dict()
        self.position_index=-1
        self.current=dict()
        self.apply=dict()
        self.queue=dict()
        self.reserved=False
        self.confirm=False
    def insert(self,key=0,val=ListItems()):
        assert isinstance(val,ListItems)
        self.position[key]=val

    def fit(self,index=0,labels=np.array([],dtype=np.int32),data=pd.DataFrame([])):
        self.index=index
        val = data.loc[labels,['Q1','Q0','duration']].values
        self.dTausum=val[:,2].sum()
        dq=val[:,0]-val[:,1]
        self.dQsum=dq.sum()
        self.n=labels.shape[0]
        groups=data.loc[labels,self.columns].groupby(self.groupby).groups
        for iloc,key in enumerate(groups.keys()):
            Li = ListItems()
            indices=groups[key]
            for i in indices:
                a=data.at[i,'begin_from']
                b = data.at[i,'begin_to']
                alpha = data.at[i,'from_end']
                item = Item(index=i,a=a,b=b,alpha=alpha)
                Li.insert(item)
            self.position[int(key)]=Li
            self.iloc[iloc]=int(key)

    def Apply(self,x=0,xtau=0,cw=None,start=0,pass_used=False):
        if  cw is None:
            self.queue=self.get_position(start=start,pass_used=pass_used)
            self.queue(x,xtau)
            return self.bounds
        else:
            self.confirm=True
            bounds=self.current.apply(cw,x,xtau)
            if bounds is not None:
                self.queue(bounds[0],bounds[1])
                return self.bounds
            else:
                return None
    def apply_in_position(self,bounds,position_index,pass_used=True):
        bounds_=None
        for k in bounds.keys():
            x0=bounds[k][0]
            xtau=bounds[k][1]
            bounds_=self.position[position_index].apply(k,x0,xtau)
        if bounds_ is None:
            t=self.position[position_index].mint
            bounds_=self.position[position_index].set_bounds(t,t,pass_used=pass_used)
        return bounds_

    def get_position(self,start=0,pass_used=False):
        for i,k in enumerate(self.position.keys()):
            if k==start:
                break
        mapped = map(lambda x: self.position[x], filter(lambda y: y>=start,self.position.keys()))
        self.position_index=i-1
        def value(x=0,xtau=0):
            done=True
            while done:
                try:
                    fun=next(mapped)
                    done=fun.is_done()
                    self.position_index+=1
                    if done:
                        x=fun.mint
                        xtau=fun.maxt
                    else:
                        self.bounds=fun.set_bounds(x=x,xtau=xtau,pass_used=pass_used)
                    self.current=fun

                except StopIteration:
                    self.bounds=None
                    done=False
        return value
    def count(self):
        return len(self.position)
    def isopened(self):
        if type(self.current)==ListItems:
            return True
        else:
            return False
    def get_bounds(self,iloc=0):
        def init():
            self.iloc=dict()
            for i,k in enumerate(self.position.keys()):
                self.iloc[i]=k
        res=dict()
        try:
            try:
                key=self.iloc[iloc]
            except AttributeError:
                init()
                key = self.iloc[iloc]
            for k in self.position[key].diction.keys():
                res[k]=(self.position[key].diction[k].a,self.position[key].diction[k].b,self.position[key].diction[k].alpha)

            return res
        except KeyError:
            return res
        except IndexError:
            return res


    def get_activities(self,iloc=0):
        def init():
            self.iloc=dict()
            for i,k in enumerate(self.position.keys()):
                self.iloc[i]=k
        try:
            try:
                key=self.iloc[iloc]
            except AttributeError:
                init()
                key = self.iloc[iloc]

            activities=list(self.position[key].diction.keys())
            return np.array(activities,dtype=np.int32)
        except KeyError:
            return np.array([],dtype=np.int32)
        except IndexError:
            return np.array([],dtype=np.int32)
    def ifapply(self,activity=0,stime=0,etime=0):
        if not self.isopened():
            return None
        activities=self.current.diction.keys()
        if ~np.isin(activity,list(activities)):
            return None
        if self.position_index+1<self.count():
            next_index=self.iloc[self.position_index+1]
            next_position=self.position[next_index]
            bounds=next_position.get_bounds(stime,etime)
            return bounds
        else:
            return None

    def shortest_way(self,debit_functions):
        def begin(debit_functions):
            activities=self.get_activities(self.position_index)
            t1 = position.mint
            t2 = position.maxt

            for a in activities:
                fun=debit_functions[a]
                if fun.used|fun.blocked:
                    continue
                x=fun.x1
                xtau=fun.x2
                if x<t1:
                    t1=x
                if xtau> t2:
                    t2=xtau
                #supp=fun.supp
                #if supp[0]<t1:
                    #t1=supp[0]
                #if (supp[0]+fun.tau)>t2:
                    #t2=supp[0]+fun.tau
                #fun.x1=fun.supp[0]
                #fun.x2 = fun.x1+fun.tau
            return t1,t2
        def shortest(t1_,t2_,position,debit_functions):
            t1=position.mint
            t2=position.maxt
            bounds = position.get_bounds(t1_, t2_)
            for k in bounds.keys():
                fun=debit_functions[k]
                b=bounds[k]
                if fun.used:
                    continue
                if b[0]<t1:
                    t1=b[0]
                if b[0]+fun.tau>t2:
                    t2=b[0]+fun.tau
                fun.x1=b[0]
                fun.x2=fun.x1+fun.tau
            return t1,t2

        i=self.position_index
        key = self.iloc[i]
        position = self.position[key]
        #вычисляем границы закрытия текущей очереди кортежа
        t1,t2=begin(debit_functions)
        j=i+1
        while j<self.count():
            key=self.iloc[j]
            position=self.position[key]
            t1_,t2_=shortest(t1,t2,position,debit_functions)
            t1=t1_
            t2=t2_
            j+=1
        return



class Node:
    def __init__(self):
        self.index=0
        self.type=-1
        self.main_obj =0
        self.value=0
        self.tvalue = 0
        self.qreserv=0
        self.qout=0
        self.flow_delay = 0
        self.parents=[]
        self.xoc_vals=np.array([])
        self.dates=np.array([])
        self.used=False
        self.flow = False
        self.inflow = False
        self.impure= False
        self.is_leaf=False
        self.is_root = False

class Tree:
    def __init__(self,graph=dict()):
        self.graph=dict()
        self.leafs=[]
    def get_leafs(self):
        def go(node_id):
            node = self.graph[node_id]
            if len(node.parents) > 0:
                for p in node.parents:
                    try:
                        self.graph[p].used = True
                        #if not self.graph[p].used:
                        go(p)


                    except KeyError:
                        pass
            else:
                node.is_root=True
                return

        for k in self.graph.keys():
            if not self.graph[k].used:
                go(k)
        for k in self.graph.keys():
            if not self.graph[k].used:
                self.graph[k].is_leaf=True
                self.leafs.append(k)

    def get_flows(self):
        flows=dict()
        for leaf in self.leafs:
            flow=self.get_flow(leaf)
            flows.update({leaf:flow})
        return flows

    def get_flow(self,leaf):
        def go(node_id):
            nodes=[]
            try:
                node=self.graph[node_id]

            except KeyError:
                print("error ",node_id)
                nodes.append(node_id)
                return nodes

            if (len(node.parents)>0) and (not node.used):
                node.used = True
                for p in node.parents:
                    nodes_=go(p)
                    if len(nodes_)>0:
                        for n in nodes_:
                            l=[node_id]
                            if type(n)==list:
                                l.extend(n)
                            else:
                                l.append(n)

                            nodes.append(l)
                    else:
                        nodes.append(node_id)
            else:
                nodes.append(node_id)

            return nodes
        flow=go(leaf)
        return flow

class Events:
    def __init__(self,events=np.array([]),size=30):
        self.events=events
        self.size=size
        self.dtz=None
        self.distr=np.zeros(shape=(2,self.size))
        self.index=dict()
        self.mean=0.
        self.var=0.
        self.var_log=[]
    def fit(self):
        distribution = np.histogram(self.events, bins=self.size,range=(self.events.min()-1,self.events.max()+1))
        self.dtz = np.digitize(self.events, distribution[1],right=True)
        k=0
        while k<self.size:
            self.index[k]=[]
            k+=1
        i=0
        while i<self.dtz.shape[0]:
            j=self.dtz[i]-1
            s=self.events[i]
            self.distr[0,j]+=s
            self.index[j].append(i)
            i+=1
        self.mean=self.distr[0].mean()
        self.distr[1,:]=self.distr[0,:]-self.mean
        self.distr[1,:]=np.power(self.distr[1,:],2)
        self.distr[1, :]=self.distr[1,:]/self.distr[1].shape[0]
        self.var=self.distr[0].var()
    def get_var(self,out,into,item=0):
        sout=self.distr[0,out]
        sin=self.distr[0,into]
        return self.var+(2*item*(sin-sout)+2*item**2)/self.size

    def get_optim_var(self,out,into,item=0):
        var=self.get_var(out,into,item)
        index=self.index[into]
        minvar=var
        move=None
        for i in index:
            val=self.events[i]
            sout = self.distr[0, into]+item
            sin = self.distr[0, out]-item
            var_=var + (2 * val * (sin - sout) + 2 * val ** 2) / self.size
            if var_<minvar:
                minvar=var_
                move=i
        return minvar,move


    def apply(self, out, into, index,backward=None):
        item=self.events[index]
        bitem=0
        if backward is not None:
            bitem = self.events[backward]
            self.index[into].remove(backward)
            self.index[out].append(backward)

        item=item-bitem
        self.distr[0, out]-=item
        self.distr[0, into]+= item
        self.index[out].remove(index)
        self.index[into].append(index)



    def eval(self):
        go=True
        mask=np.ones(shape=self.distr[0].shape[0],dtype=bool)
        forward=True
        j = -1
        si=np.arange(self.size)
        while go:

            if forward:
                mask.fill(True)
                si = np.argsort(self.distr[0])
                j=-1
                forward=False
            else:
                j-=1

            out = si[j]
            mask[j]=False

            possible=self.index[out]

            i=0
            while i<len(possible):
                index=possible[i]
                i+=1
                item=self.events[index]
                for k,into in enumerate(si):
                    if ~mask[k]:
                        continue
                    var,backward=self.get_optim_var(out,into,item)
                    if (var<self.var)&(abs(var-self.var)>1e-3):
                        self.var=var
                        #print(self.var)
                        self.var_log.append(self.var)
                        self.apply(out,into,index,backward)
                        forward=True
                        mask.fill(True)
                        break

            go=any(mask)


    def eval_old(self):
        go=True
        mask=np.ones(shape=self.distr[0].shape[0],dtype=bool)
        while go:
            si = np.argsort(self.distr[0])
            smask=mask[si]
            out=si[smask][-1]
            #out=np.argmax(self.distr[0,mask])
            forward=False
            possible=self.index[out]

            i=0
            while i<len(possible):
                index=possible[i]
                i+=1
                item=self.events[index]
                for into in si:
                    if into==out:
                        continue
                    var,backward=self.get_optim_var(out,into,item)
                    if (var<self.var)&(abs(var-self.var)>1e-3):
                        self.var=var
                        #print(self.var)
                        self.var_log.append(self.var)
                        self.apply(out,into,index,backward)
                        forward=True
                        mask.fill(True)
                        break
            if not forward:
                mask[out]=False
                #print(mask[mask].shape[0])

            go=any(mask)







#from importlib import reload
#used=np.load(path+'task\\used.npy')
#support=np.load(path+'task\\support.npy')
#group=np.load(path+'task\\groups.npy')
#Q0=np.load(path+'task\\Q0.npy')
#Q1=np.load(path+'task\\Q1.npy')
#tr=np.load(path+'task\\tr.npy')
#ts1=np.load(path+'task\\ts.npy')
#pairs=np.load(path+'pairs.npy')
#reload(optim)
#index=[234,325,326]
#tr[index]=5.
#Q0[index]=10.
#Q1[index]=15.
#queue=np.load(path+'task\\queue.npy',allow_pickle=True)[()]
#support=np.empty(shape=(Q0.shape[0],2))
#support.fill(np.nan)

#service=np.load(path+'service.npy')
#equipment=np.load(path+'equipment.npy')
#wells_service=np.load(path+'wells_service.npy',allow_pickle=True)
#wells_equipment=np.load(path+'wells_equipment.npy',allow_pickle=True)

#R=np.load(path+'task\\R.npy')
#T=np.load(path+'task\\T.npy')
#D=np.load(path+'task\\D.npy')

#wsch=wells_schedule()
#wsch.t=360
#wsch.tracing=True
#wsch.fun=wsch.f14
#stop=Q0.shape[0]
#wsch.routes=R
#wsch.start=D
#wsch.end=T
#wsch.function=optim.get_optim_trajectory
#epsilon=tr.mean()
#epsilon=np.inf
#wsch.fit(ts1,tr,Q0,Q1,group,support=support,used=used,stop=stop,epsilon=epsilon,service=service,equipment=equipment,wells_service=wells_service,wells_equipment=wells_equipment,prohibits=pairs)

#mask=np.isin(wsch.free,wsch.supported)
#wsch.free=wsch.free[mask]

#trace=wsch.get_routes()
#R_,T_,D_,O_=get_rout_time(trace)
#print()
#wsch.stop=group.shape[0]
#wsch.fun=wsch.f6
#wsch.routes=R
#wsch.start=D
#wsch.end=T

#Q0=np.array([1,1.5,0.5,1,1,1])
#Q1=np.array([10,5,1,np.NINF,np.NINF,np.NINF])
#tr1=np.ones(Q0.shape[0])
#ts1=np.zeros(shape=(Q0.shape[0],Q0.shape[0]))
#n=ts1.shape[0]
#for i in np.arange(ts1.shape[0]):
   # k=i+1
   # while k<ts1.shape[0]:
        #ts1[i,k]=n-k
        #ts1[k, i] = n-k
        #k+=1
#support=np.array([[1.5,2],[1,2],[1,1.5]])
#support=None
#tr1=np.array([0.1,0.1,0.5])
#groups=np.array([0,1,2,3],dtype=int)
#ts1=np.array([[0,1,1],[1,0,1],[1,1,0]])/10
#t=tr1.sum()
#wsch=wells_schedule()
#wsch.t=10
#wsch.fit(ts1,tr1,Q0,Q1,groups,support=support)
#wsch.weights=np.array([[1,2,3,np.NINF,np.NINF,np.NINF],[3,2,1,np.NINF,np.NINF,np.NINF],[np.NINF,2,1,np.NINF,np.NINF,np.NINF]])
#wsch.infindex=np.array([0,1,2,0,0,0],dtype=np.int32)
#wsch.infmask=np.array([1,1,1,0,0,0],dtype=bool)
#v=np.array([5,5,5,np.NINF,np.NINF,np.NINF])
#wsch.vector=np.array([1,1,1,0,0,0],dtype=bool)
#wsch.check_infty(v,4)
#print('')
#wsch.fun=wsch.f6
#wsch.stop=6
#trace=wsch.get_routes_v1(tracing=False)
#R,T,D=get_rout_time(trace)
#print(R)


#subpath=path+'task\\otm\\'
#otm_Q0=np.load(subpath+'Q0.npy')
#otm_Q1=np.load(subpath+'Q1.npy')
#otm_distances=np.load(subpath+'ts.npy')
#otm_group=np.load(subpath+'groups.npy')
#otm_support=np.load(subpath+'support.npy')
#service=np.load(subpath+'service.npy')
#equipment=np.load(subpath+'equipment.npy')
#otm_service=np.load(subpath+'well_service.npy',allow_pickle=True)
#otm_tr=np.load(subpath+'tr.npy')
#prohibits=np.load(subpath+'otm_binary.npy')
#expanded=np.load(subpath+'expanded.npy',allow_pickle=True)[()]
#otm_group_support=np.array([[0,24],[9,19],[0,24]])/24

#wsch=wells_schedule()
#wsch.t=360
#wsch.tracing=True
#stop=otm_Q0.shape[0]
#wsch.fun=wsch.f14
#wsch.fit(otm_distances,otm_tr,otm_Q0,otm_Q1,otm_group,support=otm_support,tracing=True,stop=stop,epsilon=np.inf,service=service, equipment=equipment,wells_service=otm_service,prohibits=prohibits,group_support=otm_group_support)
#wsch.fit(otm_distances,otm_tr,otm_Q0,otm_Q1,otm_group,support=otm_support,stop=stop,epsilon=np.inf,service=service, equipment=equipment,wells_service=otm_service,prohibits=prohibits,group_support=otm_group_support)
#wsch.update_debit_functions(expanded)
#mask=np.isin(wsch.free,wsch.supported)
#wsch.free=wsch.free[mask]
#t1=wsch.get_group_support(0.2,1)
#t2=wsch.get_group_support(0.8,1)
#trace=wsch.get_routes_v1(tracing=True)
#trace=wsch.get_routes()
#R,T,D,O=get_rout_time(trace)
#print('not used '+str(wsch.free.shape[0]))
#otm_service=np.load(subpath+'otm_service.npy',allow_pickle=True)
#otm_wells=np.load(subpath+'otm_wells.npy',allow_pickle=True)
#wells=pd.Index(np.load(subpath+'wells.npy',allow_pickle=True))
#wells_service=np.load(subpath+'wells_service.npy',allow_pickle=True)
#otm_prohibits=np.load(subpath+'otm_prohibits.npy',allow_pickle=True)
#binary=get_otm_binary(otm_wells,otm_service,wells,wells_service,otm_prohibits)