import op
import osch
import numpy as np
from osch import debit_function as dfun

class Executor:
    def __init__(self):
        self.current_time=0
        self.current_place=-1
        self.first=None
        self.last=None
        self.tau_horizon=np.inf
        self.schedule=op.OrderedMap()
        self.spaces=None
        self.weight=1
        self.used=False
        self.work_time=0
    def reset(self):
        weight=self.weight
        self.__init__()
        self.weight=weight


    def metric(self,debit_functions=None):
        if debit_functions is None:
            return 0
        if self.weight==0:
            return 0
        if len(self.schedule)==0:
            return 0
        def go(index):
            current=self.schedule[index]
            func=debit_functions[current]
            t2=func.t2
            if index+1 >=self.schedule.shape[0]:
                return 0
            next_=self.schedule[index+1]
            t1=debit_functions[next_].t1
            that=go(index+1)
            that+=(t1-t2)
            return that
        ro_=go(0)
        f1=self.schedule[-1]
        f0=self.schedule[0]
        fun1=debit_functions[f1]
        fun0 = debit_functions[f0]
        t1_=fun1.t2
        t0_=fun0.t1
        #t1_=debit_functions[f1].t2
        #t0_ = debit_functions[f0].t1
        t1=max(t1_,self.work_time)
        t0=min(0,t0_)
        ro=ro_+t0_+t1-t1_

        tmax=t1-t0
        if tmax>0:
            return ro/tmax
        else:
            return np.inf

class Executors:
    def __init__(self,executors):
        self.items=executors
    def __getitem__(self, item):
        try:
            return self.items[item]
        except KeyError:
            return None
    def keys(self):
        return self.items.keys()
    def shape(self):
        return len(self.items.keys())

    def metric(self,debit_functions):
        sum_=0
        for k in self.items.keys():
            m=self.items[k].metric(debit_functions)
            sum_+=m
        return sum_

    def calc_wt(self,debit_functions):
        for f in debit_functions.keys():
            func=debit_functions[f]
            e=func.executor
            self.items[e].work_time+=func.tau
class recursion_solution:
    def __init__(self):
        self.debit_functions=dict()
        self.optimizer=op.get_optim_trajectory
        self.free=op.OrderedMap()
        self.ftmatrix=None
        self.current=np.array([])
        self.executors=Executors({})
        self.Transpose=False
        self.weights = op.Safty2DArray(shape=(0,0))
        self.nempty = np.array([],dtype=np.int32)
        self.conditions=Conditions()
        self.applied=op.OrderedMap()
        self.log_=dict()
        self.applied_count=0
        self.niter=100
        self.permutations=None
        self.iteration_order=dict()
        self.current_iter=0
        self.record_mode=True


    def fit(self,data_,ftmatrix=None,executors=np.array([]),ts=np.array([])):
        data=data_.sort_values(by='position', ascending=False)
        grouped = data.groupby('cortege_activity_id')
        for k in grouped.groups.keys():
            index=grouped.groups[k]
            n=index.shape[0]
            for j,i in enumerate(index):
                fun = dfun()
                fun.tau=data.at[i,'duration']
                fun.cortege_a = data.at[i,'begin_from']
                fun.cortege_b = data.at[i,'begin_to']
                fun.cortege_alpha = data.at[i,'from_end']
                fun.atype=data.at[i,'act_type']
                fun.index = i
                fun.cortege=k
                if j+1<n:
                    next_index=index[j+1]
                    fun.cortege_next=next_index
                if j-1>=0:
                    previous_index=index[j-1]
                    fun.cortege_previous=previous_index

                self.debit_functions[i] = fun

        self.executors=executors
        self.nempty = np.array(list(self.executors.keys()), dtype=np.int32)
        self.ts=ts
        if ftmatrix is None:
            self.ftmatrix=np.ones(self.free.shape[0],self.executors.shape[0])
        else:
            self.ftmatrix=ftmatrix
        index=np.array(list(self.debit_functions.keys()),dtype=np.int32)

        self.rmatrix=op.Indexed2DArray(shape=(index.shape[0],index.shape[0]))
        self.rmatrix.set_index(index)
    def get_next(self,index=None):
        if index is None:
            next_index=[k for k in self.debit_functions.keys() if self.debit_functions[k].cortege_previous is None]
            return next_index
        next_index = [self.debit_functions[k].cortege_next for k in index if self.debit_functions[k].cortege_next is not None]
        return next_index

    def lval(self,x=0.,fun=osch.debit_function(),penalty=0):
        #if (not fun.opened)|np.isinf(x):
            #return -np.inf
        if np.isinf(x):
            return -np.inf
        t = x + fun.tau-penalty
        n=self.free.shape[0]-1

        if n>0:
            return -(t+fun.lvalue/n)
        else:
            return -t

    def add_logistic_value(self,member_):
        def add(member):
            fun=self.debit_functions[member]
            val=fun.supp[1]
            value=0
            for f in self.free:
                func=self.debit_functions[f]
                ts=self.ts[member,f]
                v=ts-func.supp[1]
                value+=v
                func.lvalue+=(ts-val)
            fun.lvalue=value
            self.free.add(member,val)
            return

        if hasattr(member_, '__iter__'):
            for m in member_:
                add(m)
        else:
            add(member_)
    def remove_logistic_value(self,member_):
        def remove(member):
            fun=self.debit_functions[member]
            fun.lvalue = 0
            val=fun.supp[1]
            self.free.remove(member)
            for f in self.free:
                func=self.debit_functions[f]
                ts=self.ts[member,f]
                func.lvalue-=(ts-val)

            return
        if hasattr(member_,'__iter__'):
            for m in member_:
                remove(m)
        else:
            remove(member_)
    def func_distance(self,a,b):
        delta=np.nan
        try:
            func1=self.debit_functions[a]
            func2 = self.debit_functions[b]
            delta=func2.t1-func1.t2
            return delta
        except KeyError:
            return delta


    def solved_time(self,current_well,well,i):
        executor=self.executors[i]
        ct = executor.current_time
        current_well = executor.current_place
        time=0
        if current_well>=0:
            ts = self.ts[current_well, well]
            time=ct+ts
            if time>self.debit_functions[well].supp[1] and abs(time-self.debit_functions[well].supp[1])>1e-3:
                return None
            else:
                return time
        return time

    def set_bounds(self,conditions):
        self.free=op.OrderedMap()
        for k in conditions.keys():
            if self.debit_functions[k].used:
                continue
            func=self.debit_functions[k]
            cond=conditions[k]
            mint = np.inf
            current_well=None
            for e in self.executors.keys():
                if self.ftmatrix[k,e]:
                    t=self.executors[e].current_time

                    if t<mint:
                        mint=t
                        current_well=self.executors[e].current_place

            if ~np.isinf(mint):
                tau=max(mint,cond)
                dt=self.ts[current_well,k]
                t=tau+dt
                s1,s2=func.solve_bounds(t,t)
                func.supp[0] = s1
                func.supp[1] = s2
                self.add_logistic_value(k)
    def get_weight(self):
        self.Transpose=False
        def set_values(i=0):
            k=0
            executor=self.executors[i]
            tau_horizon = self.executors[i].tau_horizon
            current_well=executor.current_place
            while k<self.free.shape[0]:
                well=self.free[k]
                func=self.debit_functions[well]
                if not self.ftmatrix[well,i]:
                    k+=1
                    continue
                t=self.solved_time(current_well,well,i)
                if t is None:
                    k+=1
                    continue
                xtau=t+func.tau
                if xtau>tau_horizon:
                    k+=1
                    continue
                penalty=min(t-func.supp[0],0)
                value=self.lval(t,func,penalty)
                if ~np.isinf(value):
                    self.weights[i, k] = value
                k+=1

        self.weights=op.Safty2DArray(shape=(self.executors.shape(),self.free.shape[0]))
        self.weights.fill(-np.inf)

        for e in self.executors.keys():
            set_values(e)
        self.nempty = self.weights.index[self.weights.mask]
        if self.nempty.shape[0] > 0:
            weights = self.weights.array[self.weights.mask]
            if weights.shape[0] > weights.shape[1]:
                self.Transpose = True
                return weights.T
            return weights
        else:
            self.weights=None

    def set_corteges(self):
        def go(next_well,t1,t2):
            nfunc=self.debit_functions[next_well]
            t=nfunc.cortege_alpha*t2+(1-nfunc.cortege_alpha)*t1
            s1=nfunc.cortege_a+t
            s2=nfunc.cortege_b+t
            nfunc.x1=s1
            nfunc.x2=s2
            if nfunc.cortege_previous is None:
                return
            go(nfunc.cortege_previous,nfunc.x1,nfunc.x1++nfunc.tau)

        for k in self.debit_functions.keys():
            func=self.debit_functions[k]
            if func.used or (not func.opened):
                continue
            t1=func.supp[0]
            t2=t1+func.tau
            func.x1=t1
            func.x2=t2
            if func.cortege_previous is None:
                continue
            go(func.cortege_previous,func.x1,func.x2)

    def set_tau_horizon(self):
        conditions = lambda x: False if self.debit_functions[x].used or self.debit_functions[x].opened else True
        self.set_corteges()
        for e in self.executors.keys():
            executor=self.executors[e]
            ct=executor.current_time
            horizon = np.inf
            for f in self.debit_functions.keys():
                if (not self.ftmatrix[f, e]) or (not conditions(f)):
                    continue
                fun = self.debit_functions[f]
                if (fun.x2 is not None) and (fun.x2 >= ct):
                    if fun.x2 < horizon:
                        horizon = fun.x2
            self.executors[e].tau_horizon = horizon



    def assign(self):
        vectors=self.get_weight()
        if vectors is None:
            return None
        taken, s = self.optimizer(vectors, criterion='max', engine='c')
        if self.Transpose:
            index=self.nempty.copy()
            self.nempty=taken[1].copy()
            taken[1]=index
            self.Transpose=False
        return taken
    def routes(self,activities,conditions=None):
        def fit_empty(activities):
            conditions=Conditions()
            for k in activities:
                cond=Condition()
                conditions.add(k,cond)
            return conditions
        def begin(condition):
            go=True
            while go:
                go_=True
                while go_:
                    self.set_bounds(condition)
                    indices=self.assign()
                    if indices is None:
                        go_ =False
                        continue
                    if not self.update(indices):
                        return
                l=np.array(list(condition.keys()),dtype=np.int32)
                condition=conditions.get_previous(l)
                if len(condition.keys())==0:
                    go=False
                    continue
                for c in condition.keys():
                    prev=conditions[c].previous
                    fun=self.debit_functions[prev]
                    s=fun.supp[0]
                    drift=condition[c]
                    condition[c]=drift+s
            return self.conditions
        def init(condition):
            go_=True
            while go_:
                self.set_bounds(condition)
                indices=self.assign()
                if indices is None:
                    go_ =False
                    continue
                if not self.update_initial(indices):
                    go_ = False
                    continue
            conditions=Conditions()
            s0=0
            prev=None
            for i,a in enumerate(self.applied):
                cond=Condition()
                s=self.applied.loc(a)
                if i==0:
                    conditions.add(a,cond)
                    prev=a
                    s0=s
                    continue
                pcond=conditions[prev]
                pcond.next=a
                cond.previous=prev
                cond.drift=s-s0
                conditions.add(a,cond)
                prev=a
                s0=s
            self.conditions=conditions
            return self.conditions
        def go(index,start=0,new_conditions=Conditions()):
            if len(index)==0:
                return start
            for k in index:
                current_well=-1
                func=self.debit_functions[k]
                assigned=None
                tmin=np.inf
                that=0
                delta=self.conditions[k].drift
                for e in self.executors.keys():
                    if self.ftmatrix[k,e]:
                        t=self.executors[e].current_time
                        current_well=self.executors[e].current_place
                        ts=0
                        if current_well >=0:
                            ts=self.ts[k,current_well]
                        if t+ts<tmin:
                            tmin=t+ts
                            assigned=e

                if assigned is not None:
                    s0=0
                    prev=self.conditions[k].previous
                    if prev is not None:
                        s0=self.debit_functions[prev].supp[0]
                    #current_well = self.executors[assigned].current_place
                    that=max(tmin,s0+delta)
                    #drift=that-start
                    #that=start+delta_
                    s1,s2=func.solve_bounds(that,that)

                    func.supp[0]=s1
                    func.supp[1]=s2
                    self.update_single(k,assigned,new_conditions)
                    if start<s1:
                        start=s1
                    #start=s1
                    #next_well=self.conditions[k].next
            next_index=list(conditions.get_previous(index).keys())
            start_=go(next_index,start,new_conditions)
            return start_
                    #return start_
                #else:
                    #return start

        if conditions is None:
            self.conditions=fit_empty(activities)
            conditions =self.conditions

            condition = conditions.get_previous()
            new_conditions=init(condition)
            return new_conditions

        condition=list(conditions.get_previous().keys())
        new_conditions=Conditions()
        start=0
        start_ = go(condition, start, new_conditions)
        #for c in condition:
            #start_=go(c,start,new_conditions)
            #start=start_
        return new_conditions
    def get_routes(self,activities,conditions=None):
        if conditions is None:
            conditions=Conditions()
            for k in activities:
                cond=Condition()
                conditions.add(k,cond)
        condition=conditions.get_previous()
        t=0
        if len(condition.keys())>0:
            go=True
            while go:
                go_=True
                while go_:
                    self.set_bounds(condition)
                    indices=self.assign()
                    if indices is None:
                        go_ =False
                        continue
                    if not self.update(indices):
                        return
                l=np.array(list(condition.keys()),dtype=np.int32)
                condition=conditions.get_previous(l)
                if len(condition.keys())==0:
                    go=False
                    continue
                for c in condition.keys():
                    prev=conditions[c].previous
                    fun=self.debit_functions[prev]
                    s=fun.supp[0]
                    drift=condition[c]
                    condition[c]=drift+s
            return self.conditions
    def set_routes(self,activities):
        def open_activities(activities):
            opened=[]
            for a in activities:
                func=self.debit_functions[a]
                next_index=func.cortege_previous
                if next_index is not None:
                    nfunc=self.debit_functions[next_index]
                    t=func.t2*(func.cortege_alpha)+func.t1*(1-func.cortege_alpha)
                    nfunc.supp[0]= t+nfunc.cortege_a
                    nfunc.supp[1] = t + nfunc.cortege_a
                    self.add_logistic_value(next_index)
                    opened.append(next_index)
            return opened

        if self.free.shape[0]==0:
            return True
        go_=True
        while go_:
            indices=self.assign()
            if indices is None:
                go_ =False
                continue
            if not self.update(indices):
                return False
        if self.free.shape[0]==0:
            opened = open_activities(activities)
        else:
            return False
        if self.set_routes(opened):
            return True
        return False

    def log(self,index, t0, t1, s0, s1, sp0, sp1, b0, b1, start, y):
        lo = {'index':index,'income':[t0,t1], 'bounds':[s0,s1],
             'space':[sp0,sp1], 'bo':[b0,b1],'start':start, 'y':y}
        return lo
    def plog(self,index, t0, t1, s0, s1, sp0, sp1, b0, b1, start, y):
        lo = 'index {:.0f},income:{:.2f},{:.2f}, bounds:' \
             ' {:.2f}, {:.2f}, space {:.2f}, {:.2f}, bo:{:.2f},{:.2f},' \
             ' start {:.2f}, y {:.2f}'.format(index, t0, t1, s0, s1, sp0, sp1, b0, b1, start, y)
    def set_init(self,i,t):
        func = self.debit_functions[i]
        s1, s2 = func.solve_bounds(t, t)
        #self.executors[executor].schedule.add(i, func.t1)
        func.supp[0] = s1
        func.supp[1] = s2
        func.t1 = t
        func.t2 = t + func.tau
        func.used=True
        self.applied_count+=1
    def go_forward(self,index,t=0):
        def cometric(t0, t1,sol_index,curr_metric, curr_time):
            m0 = self.co_metric(sol_index, t0)
            m1 = self.co_metric(sol_index, t1)
            t_ = t0
            if m0 < m1:
                sol_metric_ = m0
            else:
                sol_metric_ = m1
                t_ = t1
            if sol_metric_ < curr_metric:
                curr_metric = sol_metric_
                #curr_index = sol_index
                curr_time = t_
            return curr_metric,curr_time
        def mapping(a_,b_,c_,d_):
            finit=True
            if np.isinf(b_) | np.isinf(d_):
                finit=False
            else:
                shrinkage=(d_-c_)/(b_-a_)
            def value(x):
                if not finit:
                    return x-a_+c_
                return (x-a_)*shrinkage+c_
            return value

        def get_bounds(t0,t1,s0,s1):
            if (t0>=s0) & (t1<=s1):
                teta1=t0
                teta2=teta1+s1-t1
                return teta1,teta2
            if (s0>=t0)&((t1-t0)<=(s1-s0)):
                teta1=s0
                teta2=s1-(t1-t0)
                #teta2=teta1+s1-(t1-t0)
                return teta1, teta2
            return None

        def get_bounds_(span,space,tau):
            if tau>space[1]-space[0]:
                return np.array([])
            def intersection(a, b):
                if (a[0] > b[1]) | (b[0] > a[1]):
                    return np.array([])
                return np.array([max(a[0], b[0]), min(a[1], b[1])])
            tarr=np.array([space[0],space[1]-tau])
            bounds=intersection(tarr,span)
            return bounds


        def go(index,t0=0,t1=0,htau=0):
            if index is None:
                return t0,t1,0

            func=self.debit_functions[index]
            cortege_previous_ = func.cortege_previous
            if not func.master:
                start0,start1, delta_ = go(cortege_previous_, t0, t1, htau)
                return start0,start1,delta_

            s0=func.cortege_alpha*htau+func.cortege_a+t0
            s1 = func.cortege_alpha * htau + func.cortege_b + t1
            tau=func.tau
            executor_=func.executor
            alpha=self.executors[executor_].weight

            spaces = self.get_executer_row(executor_)
            t=s0
            delta=0

            for space in spaces:
                if space[0]>t:
                    t=space[0]
                if t>s1:
                    break
                hs1=space[0]
                hs2=min(s1,space[1])
                #hs2=space[1]
                bounds = get_bounds(t, t + tau, hs1, hs2)
                #new_bounds=get_bounds_(np.array([s0,s1]),space,tau)
                if bounds is None:
                    continue

                start0,start1,delta_ = go(cortege_previous_,bounds[0], bounds[1],func.tau )

                if start0 is None:
                    continue
                func.t1=start0
                func.t2=func.t1+func.tau
                func.span[0]=start0
                func.span[1]=start1
                if t0==t1:
                    y=t0
                    y0=y
                else:
                    fu=mapping(s0,s1,t0,t1)
                    y=fu(start0)
                    y0=fu(start1)
                tilda=(start0-s0)**2
                delta=delta_+alpha*tilda

                return y,y0,delta
            return None,None,None

        i=index
        Log=[]
        fun=self.debit_functions[i]
        executor = fun.executor
        spaces=self.get_executer_row(executor)
        cortege_previous = fun.cortege_previous
        min_metric=np.inf
        start0_=None
        start1_=np.inf
        k=0
        #for k,space in enumerate(spaces):
        while k<len(spaces):
            space=spaces[k]
            if space[0]>t:
                t=space[0]
            bounds=get_bounds(t,t+fun.tau,space[0],space[1])
            if bounds is None:
                k+=1
                continue
            start,start1,metric= go(cortege_previous,bounds[0],bounds[1],fun.tau)
            #lo=self.log(index,t,t+fun.tau,t,t+fun.tau,space[0],space[1],bounds[0],bounds[1],start,start)
            #Log.append(lo)
            if start is not None:
                metric_ = metric ** 0.5
                j=k
                #valid=False
                while j<len(spaces):
                    space_=spaces[j]
                    if op.isin2(start,space_):
                        #metric_,s0=cometric(start,start1,index,min_metric,start0_)
                        if metric_<min_metric:
                            min_metric=metric_
                            start0_=start
                            start1_=start1
                            j+=1
                        fun.t1 = start0_,
                        fun.t2 = fun.t1 + fun.tau
                        fun.span[0] = start0_
                        fun.span[1] = start1_
                            #break
                        return min_metric,start0_,start1_
                    j+=1
                k=j
                continue
                #if not valid:
                    #k+=1
                    #continue
                #return start,start1,metric_
            else:
                k+=1
                continue
        #fun.t1=start0_,
        #fun.t2=fun.t1+fun.tau
        #fun.span[0]=start0_
        #fun.span[1] = start1_
        return min_metric,start0_,start1_

    def go_forward_old(self, index, t=0):
        def mapping(a_, b_, c_, d_):
            finit = True
            if np.isinf(b_) | np.isinf(d_):
                finit = False
            else:
                shrinkage = (d_ - c_) / (b_ - a_)

            def value(x):
                if not finit:
                    return x - a_ + c_
                return (x - a_) * shrinkage + c_

            return value

        def get_bounds(t0, t1, s0, s1):
            if (t0 >= s0) & (t1 <= s1):
                teta1 = t0
                teta2 = teta1 + s1 - t1
                return teta1, teta2
            if (s0 >= t0) & ((t1 - t0) <= (s1 - s0)):
                teta1 = s0
                teta2 = s1 - (t1 - t0)
                # teta2=teta1+s1-(t1-t0)
                return teta1, teta2
            return None

        def go(index, t0=0, t1=0, htau=0):
            if index is None:
                return t0, 0

            func = self.debit_functions[index]
            cortege_previous_ = func.cortege_previous

            s0 = func.cortege_alpha * htau + func.cortege_a + t0
            s1 = func.cortege_alpha * htau + func.cortege_b + t1
            tau = func.tau
            executor_ = func.executor
            alpha = self.executors[executor_].weight

            spaces = self.get_executer_row(executor_)
            t = s0
            delta = 0
            for space in spaces:
                if space[0] > t:
                    t = space[0]
                if t > s1:
                    break
                hs1 = space[0]
                hs2 = min(s1, space[1])
                # hs2=space[1]
                bounds = get_bounds(t, t + tau, hs1, hs2)
                if bounds is None:
                    continue

                start_, delta_ = go(cortege_previous_, bounds[0], bounds[1], func.tau)

                if start_ is None:
                    continue
                if t0 == t1:
                    y = t0
                else:
                    fu = mapping(s0, s1, t0, t1)
                    y = fu(start_)
                delta = delta_ + alpha * (t0 - y) ** 2
                # lo=self.log(index,t0,t1,s0,s1,hs1,hs2,bounds[0],bounds[1],start_,y)
                # Log.append(lo)
                return y, delta
            return None, None

        i = index
        Log = []
        fun = self.debit_functions[i]
        executor = fun.executor
        spaces = self.get_executer_row(executor)
        cortege_previous = fun.cortege_previous
        for k, space in enumerate(spaces):
            bounds = get_bounds(t, t + fun.tau, space[0], space[1])
            if bounds is None:
                continue
            start, metric = go(cortege_previous, bounds[0], bounds[1], fun.tau)

            # lo=self.log(index,t,t+fun.tau,t,t+fun.tau,space[0],space[1],bounds[0],bounds[1],start,start)
            # Log.append(lo)

            if start is not None:
                metric_ = metric ** 0.5
                j = k
                valid = False
                while j < len(spaces):
                    space_ = spaces[j]
                    if op.isin2(start, space_):
                        return start, metric_
                    j += 1
                if not valid:
                    continue
                return start, metric_
        return None
    def go_backward_old(self,index):
        if len(index)==0:
            return
        next_index=[]
        for i in index:
            fun=self.debit_functions[i]
            executor = fun.executor
            if fun.master:
                self.executors[executor].schedule.add(i, fun.t1+fun.tau/2)
                self.executors[executor].spaces=None
            cortege_previous=fun.cortege_previous
            if cortege_previous is None:
                continue
            func=self.debit_functions[cortege_previous]
            func.order=fun.order
            s1,s2=func.solve_bounds(fun.t1,fun.t2)
            func.supp[0]=s1
            func.supp[1]=s2
            func.t1=s1
            func.t2=s1+func.tau
            func.used=True
            self.applied_count+=1
            next_index.append(cortege_previous)
        self.go_backward_old(next_index)
    def go_backward(self,index):
        if len(index)==0:
            return
        next_index=[]
        for i in index:
            fun=self.debit_functions[i]
            executor = fun.executor
            if fun.master:
                self.executors[executor].schedule.add(i, fun.t1+fun.tau/2)
                self.executors[executor].spaces=None
            cortege_previous=fun.cortege_previous
            if cortege_previous is None:
                continue
            func=self.debit_functions[cortege_previous]
            func.order=fun.order
            s1,s2=func.solve_bounds(fun.t1,fun.t2)
            func.supp[0]=s1
            func.supp[1]=s2
            #func.t1=s1
            #func.t2=s1+func.tau
            func.used=True
            self.applied_count+=1
            next_index.append(cortege_previous)
        self.go_backward(next_index)
    def co_metric(self,index,t):
        def metric(index, t0,t1):
            if index is None:
                return 0
            i=index
            fun=self.debit_functions[i]
            cortege_previous=fun.cortege_previous
            executor = fun.executor
            if not fun.master:
                ro=metric(cortege_previous,t0,t1)
                return ro
            s1, s2 = fun.solve_bounds(t0, t1)

            fun.t1=s1
            fun.t2=s1+fun.tau
            self.executors[executor].schedule.add(i, fun.t1 + fun.tau / 2)
            ro_ = metric(cortege_previous,fun.t1,fun.t2)
            rhat=0
            if not self.executors[executor].used:
                rhat = self.executors[executor].metric(self.debit_functions)
            self.executors[executor].schedule.remove(i)
            return ro_+rhat

        def clear():
            for e in self.executors.keys():
                self.executors[e].used=False
        clear()
        if np.isinf(t):
            return t
        #func=self.debit_functions[index]
        #func.t1=t
        #func.t2=func.t1+func.tau
        me=metric(index,t,t)
        return me
    def co_metric_old(self,index,t):
        def metric(index):
            if index is None:
                return 0
            i=index
            fun=self.debit_functions[i]
            executor = fun.executor
            if fun.master:
                self.executors[executor].schedule.add(i, fun.t1+fun.tau/2)
                #self.executors[executor].spaces=None
            cortege_previous=fun.cortege_previous
            if cortege_previous is None:
                self.executors[executor].used=True
                ro=self.executors[executor].metric(self.debit_functions)
                if fun.master:
                    self.executors[executor].schedule.remove(i)
                return ro
            func=self.debit_functions[cortege_previous]
            s1,s2=func.solve_bounds(fun.t1,fun.t2)
            func.t1=s1
            func.t2=s1+func.tau
            ro_=metric(cortege_previous)
            rhat=0
            if not self.executors[executor].used:
                self.executors[executor].used=True
                rhat=self.executors[executor].metric(self.debit_functions)
            if fun.master:
                self.executors[executor].schedule.remove(i)
            return ro_+rhat
        def clear():
            for e in self.executors.keys():
                self.executors[e].used=False
        clear()
        func=self.debit_functions[index]
        func.t1=t
        func.t2=func.t1+func.tau
        me=metric(index)
        return me
    def update_single(self,activitie,executor,conditions,smax=0):
        st=0
        w=activitie
        j=executor
        current_well=self.executors[j].current_place
        fun = self.debit_functions[w]
        #last = self.executors[j].last
        #fun.sh_next = last
        st=self.solved_time(current_well,w,j)
        if st is not None:
            st=max(st,fun.supp[0])
            cond=Condition()
            cond_=self.conditions[w]
            cond.next=cond_.next
            cond.previous=cond_.previous
            ps=0
            if cond.previous is not None:
                ps=self.debit_functions[cond.previous].supp[0]
            drift=fun.supp[0]-ps
            #self.debit_functions[current_well].next=w
            fun.cortege_delta=drift
            cond.drift = drift
            conditions.add(w, cond)
            #self.rmatrix[current_well,w]=drift
            fun.used=True
            fun.t1=st
            fun.t2=st+fun.tau
            fun.executor=j

            if self.executors[j].first is None:
                self.executors[j].first=w

            self.executors[j].current_place=w
            self.executors[j].current_time = st+fun.tau
            #if last is not None and fun.sh_next==last:
                #self.debit_functions[last].sh_previous=w


    def update_initial(self,indices=np.array([])):
        index=self.free[indices[1]]
        mask=np.zeros(index.shape[0],dtype=bool)
        st=0
        for i,w in enumerate(index):
            j=self.nempty[i]
            current_well=self.executors[j].current_place
            #last=self.executors[j].last
            fun = self.debit_functions[w]
            #fun.sh_next=last

            st=self.solved_time(current_well,w,j)
            if st is not None:
                st=max(st,fun.supp[0])
                fun.used=True
                fun.t1=st
                fun.t2=st+fun.tau
                fun.executor=j
                mask[i]=True
                self.remove_logistic_value(w)
                self.applied.add(w,fun.supp[0])
                self.executors[j].current_place=w
                self.executors[j].current_time = st+fun.tau
                if self.executors[j].first is None:
                    self.executors[j].first = w
                if current_well>=0:
                    #self.debit_functions[current_well].sh_next = w
                    self.debit_functions[w].sh_previous = current_well
                #if last is not None and fun.sh_next == last:
                    #self.debit_functions[last].sh_previous = w

        if mask[mask].shape[0]==0:
            return False
        else:
            return True
    def update(self,indices=np.array([])):
        self.applied=[]
        index=self.free[indices[1]]
        mask=np.zeros(index.shape[0],dtype=bool)
        st=0
        for i,w in enumerate(index):
            j=self.nempty[i]
            current_well=self.executors[j].current_place
            fun = self.debit_functions[w]

            st=self.solved_time(current_well,w,j)
            if st is not None:
                st=max(st,fun.supp[0])
                cond=Condition()
                if current_well >= 0:
                    ps = self.debit_functions[current_well].supp[0]
                    cond.previous = current_well
                    try:
                        self.conditions[current_well].next=w
                    except AttributeError:
                        pass
                else:
                    ps=0
                    current_well=w
                drift=fun.supp[0]-ps
                self.debit_functions[current_well].next=w
                fun.cortege_delta=drift
                cond.drift = drift
                self.conditions.add(w, cond)
                #self.rmatrix[current_well,w]=drift
                fun.used=True
                fun.t1=st
                fun.t2=st+fun.tau
                fun.executor=j
                mask[i]=True
                self.remove_logistic_value(w)
                self.applied.append(w)

                if self.executors[j].first is None:
                    self.executors[j].first=w

                self.executors[j].current_place=w
                self.executors[j].current_time = st+fun.tau


        if mask[mask].shape[0]==0:
            return False
        else:
            return True
    def reset_executors(self,index=None):
        if index is None:
            index=self.executors.keys()
        for e in index:
            self.executors[e].current_place = -1
            self.executors[e].current_time = 0
            #self.executors[e].last=self.executors[e].first
            self.executors[e].first = None
            self.executors[e].last=None
    def wrap_conditions(self,conditions):
        new_conditions = Conditions()
        for c in conditions.keys():
            cond = conditions[c]
            previous=self.debit_functions[c].cortege_next

            if previous is None:
                continue
            ts1=self.ts[c,previous]
            tau1=self.debit_functions[previous].tau
            #ts2=0
            #tau2=0

            prev_ = cond.previous
            next_ = cond.next
            cond_ = Condition()

            #delta=-self.debit_functions[c].tau
            delta=0
            if prev_ is not None:
                cond_.previous = self.debit_functions[prev_].cortege_next
                if cond_.previous is not None:
                    ts2 = self.ts[prev_, cond_.previous]
                    tau2 = self.debit_functions[cond_.previous].tau
                    #delta=ts2+tau2
                    delta=cond.drift-(tau1-tau2)-(ts1-ts2)

                #delta=self.debit_functions[prev_].tau-self.debit_functions[c].tau
                #if delta<0:
                    #delta=0

            if next_ is not None:
                cond_.next = self.debit_functions[next_].cortege_next
            #cond_.drift = max(cond.drift,delta)
            cond_.drift = delta
            #cond_.drift = cond.drift+ delta
            new_conditions.add(previous, cond_)
        return new_conditions
    def get_executer_row(self,executor):
        def go(index):

            current=self.executors[executor].schedule[index]
            func=self.debit_functions[current]
            t2=func.t2
            if index+1 >=shape:
                return [[t2,np.inf]]
            next_=self.executors[executor].schedule[index+1]
            t1=self.debit_functions[next_].t1
            that=go(index+1)
            if (t1-t2)>1e-3:
                that.insert(0,[t2,t1])
            return that

        try:
            if self.executors[executor].spaces is not None:
                return self.executors[executor].spaces
            #first=self.executors[executor].schedule[0]
            first=0
            shape=self.executors[executor].schedule.shape[0]
            if shape==0:
                return [[0,np.inf]]
            current=self.executors[executor].schedule[first]
            func=self.debit_functions[current]
            l=go(first)
            l.insert(0, [0, func.t1])
            self.executors[executor].spaces=l
            return self.executors[executor].spaces

        except KeyError:
            return None
    def depth(self,index):
        def go(index):
            if index is None:
                return 0
            func=self.debit_functions[index]
            cortege_next=func.cortege_next
            n=go(cortege_next)
            return n+1
        le=go(index)
        return le

    def optimize_backward(self):
        conditions_list=[]
        def clear_cortege(index):
            if index is None:
                return
            fun=self.debit_functions[index]
            if fun.master:
                executor=fun.executor
                self.executors[executor].schedule.remove(index)

            cortege_previous=fun.cortege_previous
            clear_cortege(cortege_previous)
            return
        def fit_by_metric(conditions_):
            for f in self.debit_functions.keys():
                self.debit_functions[f].used = False
            i=0

            #first = [k for k in conditions_.keys() if conditions_[k].previous is None]
            t = 0
            count = 0
            go_=True

            while go_:
                index_=None
                go_=False
                metric=np.inf
                that=0
                for index in conditions_.keys():
                    if self.debit_functions[index].used:
                        continue

                    t = 0
                    t_,t1, metric_ = self.go_forward(index, t)
                    if metric_<metric:
                        metric=metric_
                        index_=index
                        that=t_
                    if metric_==0:
                        break

                if index_ is None:
                    break
                count+=1
                print(count,index_)
                self.set_init(index_, that)
                self.go_backward([index_])
                go_=True

        def fit_by_cometric(conditions_,kind='Euclidian'):
            def euclidian(t0,t1,sol_metric,sol_index,curr_index,curr_metric,curr_time):
                if sol_metric < curr_metric:
                    curr_metric = sol_metric
                    curr_index = sol_index
                    curr_time = t0
                return curr_time,curr_index,curr_metric
            def cometric(t0,t1,sol_metric,sol_index,curr_index,curr_metric,curr_time):
                m0 = self.co_metric(sol_index, t0)
                m1 = self.co_metric(sol_index, t1)
                t_=t0
                if m0 < m1:
                    sol_metric_ = m0
                else:
                    sol_metric_ = m1
                    t_ = t1
                if sol_metric_ < curr_metric:
                    curr_metric = sol_metric_
                    curr_index = sol_index
                    curr_time=t_
                return curr_time,curr_index,curr_metric

            def go(indices=np.array([],dtype=np.int32),count=0):
                #index = 0
                go_ = True
                while go_:
                    index_ = None
                    go_ = False
                    metric = np.inf
                    metric_=0
                    that = 0
                    for index in indices:
                        if self.debit_functions[index].used:
                            continue

                        t = 0
                        emetric_,t_, t1 = self.go_forward(index, t)
                        #that,index_,metric=cometric(t_,t1,emetric_,index,index_,metric,that)
                        that, index_, metric = euclidian(t_, t1, emetric_, index, index_, metric, that)
                        if metric == 0:
                            break

                    if index_ is None:
                        break

                    self.debit_functions[index_].order = count
                    emetric_, t_, t1 = self.go_forward(index_, 0)
                    #that,index_,metric=cometric(t_,t1,emetric_,index_,index_,metric,that)
                    #that, index_, metric = euclidian(t_, t1, emetric_, index_, index_, metric, that)
                    self.set_init(index_, t_)
                    self.go_backward([index_])
                    if self.record_mode:
                        self.iteration_order[self.current_iter][count]=index_

                    count += 1
                    go_ = True
                return count
            def split(indices):
                def go(index):
                    fun=self.debit_functions[index]
                    e=fun.executor
                    weight=self.executors[e].weight
                    if (weight>0) & fun.master:
                        return True
                    cortege_previous=fun.cortege_previous
                    if cortege_previous is None:
                        return False
                    sign=go(cortege_previous)
                    return sign

                kernel=[]
                other=[]
                for k in indices:
                    ker=go(k)
                    if ker:
                        kernel.append(k)
                    else:
                        other.append(k)
                return np.array(kernel,dtype=np.int32),np.array(other,dtype=np.int32)
            def optimize(indices=np.array([],dtype=np.int32)):
                for index in indices:
                    func=self.debit_functions[index]
                    if self.debit_functions[index].used:
                        continue
                    clear_cortege(index)
                    t = 0
                    emetric_,t_, t1 = self.go_forward(index, t)
                    that=t_
                    if that!=func.t:
                        that=that
                    #print(index,func.t,that)
                    #metric=emetric_
                    index_=index
                    #that,index_,metric=cometric(t_,t1,emetric_,index,index_,metric,that)
                    #that, index_, metric = euclidian(t_, t1, emetric_, index, index_, metric, that)
                    #if metric == 0:
                        #break
                    self.set_init(index_, that)
                    self.go_backward([index_])




            for f in self.debit_functions.keys():
                self.debit_functions[f].used = False
                self.debit_functions[f].order = -1
            self.executors.calc_wt(self.debit_functions)

            indices=[k for k in self.debit_functions.keys() if self.debit_functions[k].cortege_next is None]
            kernel, other = split(indices)
            trajectories=[]
            #indices=conditions_.keys()
            optimal_index=kernel
            #min_me=np.inf
            metric_=np.inf
            #niter=self.niter
            i=0

            while i <self.niter:
                shuffled=np.random.permutation(kernel)
                self.iteration_order[i]=np.zeros(shuffled.shape[0],dtype=np.int32)
                self.current_iter=i
                count=go(shuffled)
                me = self.executors.metric(self.debit_functions)
                trajectories.append([shuffled,me])
                #print(i,me)
                if me<metric_:
                    metric_=me
                    optimal_index=shuffled
                for f in self.debit_functions.keys():
                    self.debit_functions[f].used = False
                    self.debit_functions[f].order = -1
                for e in self.executors.keys():
                    self.executors[e].reset()
                i+=1
            self.record_mode=False
            count = go(optimal_index)
            print('kernel',count)
            me = self.executors.metric(self.debit_functions)
            print(me)
            self.permutations=trajectories


            i=0
            while abs(me-metric_)>1e-2:
                if i==0:
                    break
                me=metric_
                for k in kernel:
                    self.debit_functions[k].used = False
                #optimize(kernel)
                metric_ = self.executors.metric(self.debit_functions)
                i += 1

                if i>0:
                    print(metric_)
                    break


            count=go(other,count)
            print('other',count)

        def go(activities,conditions):

            if len(activities)==0:
                return True
            self.reset_executors()
            conditions_=self.routes(activities,conditions)
            conditions_list.append(conditions_)
            activities_=self.get_next(conditions_.keys())
            self.conditions=self.wrap_conditions(conditions_)



            if go(activities_,self.conditions):
                fit_by_cometric(conditions_)
                return False
            return False



        index=None
        conditions=None
        fit_by_cometric(self.debit_functions.keys())

    def copy_executors(self):
        new_dict=dict()
        for k in self.executors.keys():
            executor=Executor()
            executor_=self.executors[k]
            executor.current_place=executor_.current_place
            executor.current_time=executor_.current_time
            executor.first=executor_.first
            new_dict[k]=executor
        new_executors=Executors(new_dict)
        return new_executors

    def optimize(self):
        def go(activities,conditions):
            def wrap():
                new_conditions=Conditions()
                dictionary={ c1:c2 for c1,c2 in zip(conditions_.keys(),activities_)}
                for c in dictionary.keys():
                    a=dictionary[c]
                    cond=conditions_[c]
                    cond_=Condition()
                    prev_=cond.previous
                    next_=cond.next
                    if prev_ is not None:
                        cond_.previous=dictionary[prev_]
                    if next_ is not None:
                        cond_.next=dictionary[next_]
                    cond_.drift=cond.drift
                    new_conditions.add(a,cond_)
                return new_conditions

            if len(activities)==0:
                return True
            self.reset_executers()

            conditions_=self.get_routes(activities,conditions)
            if conditions_ is None:
                conditions_=conditions_
            activities_=self.get_next(conditions_.keys())
            self.conditions=wrap()

            if go(activities_,self.conditions):
                self.reset_executers()
                for f in self.debit_functions.keys():
                    self.debit_functions[f].used=False
                for a in activities:
                    self.debit_functions[a].opened=True
                    self.add_logistic_value(a)
                self.go_forward()
                return False
            return False



        index=None
        conditions=None
        activities=self.get_next(index)
        go(activities,conditions)
        return


class Condition:
    def __init__(self):
        self.drift=0
        self.previous=None
        self.next=None
class Conditions:
    def __init__(self):
        self.conditions=dict()
    def add(self,key,val=Condition()):
        self.conditions[key]=val
    def get_previous(self,indices=None):
        previous=dict()
        if indices is None:
            previous={k:self.conditions[k].drift for k in self.conditions.keys() if self.conditions[k].previous is None }
        else:
            previous = {k: self.conditions[k].drift for k in self.conditions.keys() if np.isin(self.conditions[k].previous,indices)}
        return previous
    def keys(self):
        return self.conditions.keys()
    def __getitem__(self, item):
        try:
            return self.conditions[item]
        except KeyError:
            return None









