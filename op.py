import numpy as np
import cppmath as ex
import time

#приближенное решение задачи о назначениях. Быстрый алгоритм
def get_optim_trajectory_fast(A, criterion='max'):

    def get_max_indices(svalues=np.array([]), values=np.array([]), indices=np.array([])):
        # indices[0] -бригады
        # indices[1] -скважины
        def get_max(svalues=np.array([]), values=np.array([]), indices=np.array([])):
            if indices[0].shape[0] <= 1:
                return indices[0]

            index = indices[1]
            i = 0
            n = index[0]
            if n > 0:
                size = n + 1
            else:
                size = svalues.shape[1] + 1 + n
            am = 0

            while i < size:
                x_index = svalues[indices[0], index]
                x_values = values[indices[0], x_index]

                if np.any(x_values[0] != x_values):
                    am = np.argmin(x_values)
                    return indices[0][am]
                else:
                    index = index - 1

                i += 1
                # print('i',i)
            return indices[0][am]

        arg_x = []
        arg_y = []
        row = svalues[indices[0], indices[1]]
        # print(row)
        unique = np.unique(row)
        for j in unique:
            indices_x = np.where(row == j)
            indices_y = row[indices_x]
            i_x = indices[0][indices_x]
            i_y = indices[1][indices_x]
            arg = np.argmax(values[i_x, indices_y])
            mvalue = values[i_x, indices_y][arg]
            equal = np.where(values[i_x, indices_y] == mvalue)[0]
            # print(indices_x)

            if equal.shape[0] > 1:
                index = np.array([i_x[equal], i_y[equal]])
                am = get_max(svalues, values, index)
            else:
                am = i_x[arg]

            arg_x.append(am)
            arg_y.append(j)
        return np.array([arg_x, arg_y], dtype=np.int32)

    def get_row(A=np.array([]), index=np.array([])):
        row = []
        col = []
        for j in index:
            mask = np.where(A[j, :] >= 0)
            if mask[0].shape[0] > 0:
                row.append(mask[0][-1])
                col.append(j)
        return np.array([col, row], dtype=np.int32)


    sA = A.argsort()
    trajectory_x = []
    trajectory_y = []
    s = 0
    #function = get_max_indices
    index_x = np.arange(sA.shape[0])
    index_y = np.ones(sA.shape[0]) * -1
    index = np.array([index_x, index_y], dtype=np.int32)


    while index.shape[1] > 0:
        indices = get_max_indices(sA, A, index)
        trajectory_x.extend(indices[0])
        trajectory_y.extend(indices[1])
        s_ = A[indices[0], indices[1]].sum()
        s += s_
        mask = np.isin(index[0], indices[0])
        index_x = index[0][~mask]
        mask1 = np.isin(sA, indices[1])
        sA[mask1] = -1
        index = get_row(sA, index=index_x)

        if index.shape[0] == 0:
            break
    array = np.array([trajectory_x, trajectory_y])
    sorted = array[0].argsort()
    return array[:, sorted], s

#точное решение задачи о назначениях
def get_optim_trajectory(A, criterion='max',engine='c'):

    if criterion == 'max':
        A = A * -1
    if engine=='c':
        numbers=np.zeros(shape=A.shape[0],dtype=np.int32)
        #t1 = time.perf_counter()
        s_=ex.assignment(A,numbers)
        #t2 = time.perf_counter()
        #s_=t2-t1

        indices=np.vstack((np.arange(numbers.shape[0]),numbers)).astype(np.int32)
    else:
        #t1 = time.perf_counter()
        indices, s_ = assignment(A)
        #t2 = time.perf_counter()
        #s_=t2-t1
    return indices, s_


def assignment(a=np.array([]), add=True):
    trsp = False
    if a.shape[0] > a.shape[1]:
        a = a.T
        trsp = True
    if add:
        rzeros = np.zeros(a.shape[0]).reshape(-1, 1)
        a = np.hstack((rzeros, a))
        czeros = np.zeros(a.shape[1]).reshape(1, -1)
        a = np.vstack((czeros, a))
    u = np.zeros(a.shape[0], dtype=np.float)
    v = np.zeros(a.shape[1], dtype=np.float)
    p = np.zeros(a.shape[1], dtype=np.int32)
    way = np.zeros(a.shape[1], dtype=np.int32)

    for i in np.arange(1, a.shape[0]):
        p[0] = i
        j0 = 0
        minv = np.empty(shape=a.shape[1])
        minv.fill(np.inf)
        used = np.zeros(shape=a.shape[1], dtype=bool)
        mark = True
        while mark:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in np.arange(1, a.shape[1]):
                if not used[j]:
                    cur = a[i0, j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in np.arange(a.shape[1]):
                if used[j]:
                    # print('used ',j)
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] != 0:
                mark = True
            else:
                mark = False

        mark1 = True
        while mark1:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 > 0:
                mark1 = True
            else:
                mark1 = False
    c = np.empty(a.shape[0], dtype=np.int32)
    for i in np.arange(p.shape[0]):
        c[p[i] - 1] = i - 1
    if trsp:
        return np.array([c[:-1], np.arange(a.shape[0] - 1)]), v[0]

    return np.array([np.arange(a.shape[0] - 1), c[:-1]]), v[0]

def interseption(C, D,shape=3):
    if shape==3:
        A = np.array(C)
        X = np.array(D)
    else:
        A = np.array(C,dtype=float)
        X = np.array(D,dtype=float)
    a = A[0]
    b = A[1]
    x = X[0]
    y = X[1]
    mask1 = (a < x) & (x < b)
    mask2 = (a < y) & (y < b)
    mask3 = ((x <= a) & (a <= y)) & ((x <= b) & (b <= y))
    # print(A)
    # print(X)
    if mask1 & mask2:
        A[0] = x
        A[1] = y
        # print('returned ',A)
        return A
    if mask1:
        A[0] = x
        # print('returned ',A)
        return A
    if mask2:
        A[1] = y
        # print('returned ',A)
        return A
    if mask3:
        # print('returned ',A)
        return A.reshape(-1, shape)
    return np.array([],dtype=float)

def in2int(A=np.array([]), X=np.array([])):
    a=interseption2(A,X)
    if a.shape[0]==0:
        return False
    if (a[1]-a[0])==0:
        b=interseption2(X,A)
        if b.shape[0]==0:
            return False
    return True


def interseption2(A=np.array([]), X=np.array([])):
    if (A.shape[0]==0)|(X.shape[0]==0):
        return np.array([])
    a = A[0]
    b = A[1]
    x = X[0]
    y = X[1]
    mask1 = (a < x) & (x < b)
    mask2 = (a < y) & (y < b)
    mask3 = ((x <= a) & (a <= y)) & ((x <= b) & (b <= y))
    # print(A)
    # print(X)
    if mask1 & mask2:
        return X

    if mask1:
        return np.array([x,A[1]])

    if mask2:
        return np.array([A[0],y])
    if mask3:
        return A
    return np.array([],dtype=float)
def interseptions(A=np.array([])):
    # returns interseption of 2shaped sets of A
    def go(a,b):
        if b.shape[0]<=1:
            return interseption2(a,b.reshape(-1))
        isp=go(b[0],b[1:])
        return interseption2(a, isp)

    if (A.shape[0]<=1)|len(A.shape)<=1:
        return A
    a=A[0]
    b=A[1:]
    result=go(a,b)
    return result
def interseptions_2S(A=np.array([])):
    # returns interseption of 2shaped sets of A
    def go(a,b):
        if len(b)<=1:
            return interseption_2S(a,b[0].reshape(-1,2))
        isp=go(b[0],b[1:])
        return interseption_2S(a, isp)

    if (len(A)<=1):
        return A
    a=A[0]
    b=A[1:]
    result=go(a,b)
    return result

def residual(A=np.array([]),B=np.array([])):
    #return A\B
    def UnionWoSet(Union, X):
        # returns residual of union 2shaped sets and 2shaped set X
        Y = np.array([]).reshape(-1, 2)
        for l in Union:
            y = residual2(l, X)
            if len(y) > 0:
                Y = np.vstack((Y, y))
        return Y
    X=A.reshape(-1,2).copy()
    for b in B.reshape(-1,2):
        X=UnionWoSet(X,b).reshape(-1,2)
    return X
def residual2(A_=np.array([]), X_=np.array([])):
    #returns A/X
    A=np.array(A_,dtype=np.float64)
    X = np.array(X_, dtype=np.float64)
    a = A[0]
    b = A[1]
    x = X[0]
    y = X[1]
    mask1 = (a < x) & (x < b)
    mask2 = (a < y) & (y < b)
    mask3 = ((x <= a) & (a <= y)) & ((x <= b) & (b <= y))
    if mask1 & mask2:
        A[1] = x
        B = np.array([y, b],dtype=float)

        if (A[1]-A[0]>0)&(B[1]-B[0]>0):
            return np.array([A, B])

        elif (A[1]-A[0]>0):
            return np.array([A])

        elif (B[1]-B[0]>0):
            return np.array([B])

        else:
            return np.array([],dtype=float)


    if mask1:
        A[1] = x
        if A[1]-A[0]>0:
            return A
        else: return np.array([],dtype=float)
    if mask2:
        A[0] = y
        if A[1]-A[0]>0:
            return A
        else: return np.array([],dtype=float)

    if mask3:
        if b-a>0:
            return np.array([],dtype=float)
        else:
            return A
    return A
def isin2(x=0,a=np.array([0,0]),epsilon=0):
    if (x>=a[0])&(x<=a[1]):
        return True
    else:
        if (abs(x-a[0])<epsilon)|(abs(x-a[1])<epsilon):
            return True
    return False
def inset(x=0,A=np.array([]),epsilon=0):
    # returns True if x in any subset of A
    def isin(x,a):
        if (x>=a[0])&(x<=a[1]):
            return True
        else:
            if (abs(x-a[0])<epsilon)|(abs(x-a[1])<epsilon):
                return True
        return False

    for a in A.reshape(-1,2):
        if isin(x,a):
            return True
    return False

def inspan(x=0,A=np.array([]),epsilon=0,include='both',length=0.,subset=False):
    # returns x if x in any subset (a,b) of A
    #or nearest a to x : x<a
    def isin(x,a,include='both') -> bool:
        if include=='both':
            mask=(x>=a[0])&(x<=a[1])
            if mask: return True
            else:
                if (abs(x - a[0]) < epsilon) | (abs(x - a[1]) < epsilon):
                    return True
        elif include=='left':
            mask=(x>=a[0])&(x<a[1])
            if mask: return True
            else:
                if (abs(x - a[0]) < epsilon):
                    return True
        elif include=='right':
            mask = (x >a[0]) & (x <= a[1])
            if mask: return True
            else:
                if (abs(x - a[1]) < epsilon):
                    return True
        else:
            mask = (x > a[0]) & (x <a[1])
            if mask: return True
            else:
                if (abs(x - a[0]) < epsilon) | (abs(x - a[1]) < epsilon):
                    return True
        return False

    def value(x,a,include='both'):
        if not subset:
            return isin(x,a,include=include)
        else:
            return isin(x,a,include=include)&isin(x+length,a,include='both')

    amin=np.inf
    for a in A.reshape(-1,2):
        if (a[1]-a[0])>=length:
            if value(x,a,include=include):
                return x
            else:
                if (x<a[0])&(amin>a[0]):
                    amin=a[0]
    return amin

def interseption_2S(A=np.array([]),B=np.array([])):
    # returns interseption arrays of 2shaped arrays
    isp=[]
    for a in A.reshape(-1,2):
        for b in B.reshape(-1,2):
            res=interseption2(a,b)
            if res.shape[0]>0:
                isp.append(res)
    return np.array(isp)


class GapMetrics:
    def __init__(self, origin=0, a=1, b=2,delta=0):
        self.origin = origin
        self.a = a
        self.b = b
        self.delta=delta
        self.c=self.a-self.delta

    def linear(self, x):
        if (x > self.b) | (x < self.origin):
            return np.NINF
        if (x <= self.a)&(x>=self.origin):
            return self.a-x
        if (x > self.a) & (x <= self.b):
            return x-self.a

    def lin(self,x):
        value=x-self.a
        if (value>self.b)|(x>self.b):
            value= np.NINF
        return value

    def fihat(self,x):
        def fi(a=0, b=1, gamma=0.5):
            teta = b - a
            drift = a + teta / 2
            def value(x):
                y = -(x - a) * (b - x) * np.exp(-((x - drift) ** 2) / (teta * gamma))
                return y

            return value

        def ksi(a=0, b=1):
            def value(x):
                if a == b:
                    return 0
                if x == a:
                    return np.inf
                y=(b-x)/(x-a)
                return y
            return value

        def eta(a=0, b=1):
            def value(x):
                if a == b:
                    return 0
                if x == b:
                    return np.inf
                y=(x-a)/(b-x)
                return y
            return value

        if (x >= self.origin) & (x <= self.c):
            fun = ksi(self.origin, self.c)
            return fun(x)
        if (x > self.c) & (x < self.a):
            fun = fi(a=self.c, b=self. a)
            return fun(x)
        if (x >= self.a) & (x <= self.b):
            fun = eta(a=self.a, b=self.b)
            return fun(x)
        else:
            return np.NINF


    def ksihat(self,x):
        def fi(a=0, b=1, gamma=0.5):
            teta = b - a
            drift = a + teta / 2
            def value(x):
                y = -(x - a) * (b - x) * np.exp(-((x - drift) ** 2) / (teta * gamma))
                return y

            return value

        def ksi(a=0, b=1):
            def value(x):
                if a == b:
                    return 0
                if x == a:
                    return np.inf
                y=(b-x)
                return y
            return value

        def eta(a=0, b=1):
            def value(x):
                if a == b:
                    return 0
                if x == b:
                    return np.inf
                y=(x-a)
                return y
            return value

        if (x >= self.origin) & (x <= self.c):
            fun = ksi(self.origin, self.c)
            return fun(x)
        if (x > self.c) & (x < self.a):
            fun = fi(a=self.c, b=self. a)
            return fun(x)
        if (x >= self.a) & (x <= self.b):
            fun = eta(a=self.a, b=self.b)
            return fun(x)
        else:
            return np.NINF

class Safty2DArray:
    def __init__(self, *args, **kwargs):
        self.array = np.empty(*args, **kwargs)
        dim = len(self.array.shape)
        assert dim <=2, "Only 2D numpy arrays!"

        if dim <= 1:
            self.array.reshape(-1,1)

        self.infmask = np.zeros(self.array.shape[1], dtype=bool)
        self.infindex = np.zeros(self.array.shape[1], dtype=np.int32)
        self.mask = np.zeros(self.array.shape[0], dtype=bool)
        self.index = np.arange(self.mask.shape[0])

    def __getitem__(self, *args):
        return self.array.__getitem__(*args)

    def __setitem__(self, *args):
        value = args[1]
        index = args[0]

        if (~np.isinf(value) & ~self.infmask[index[1]]) & ~self.mask[index[0]]:
            self.mask[index[0]] = True
            self.infmask[index[1]] = True
            self.infindex[index[1]] = index[0]

        self.array.__setitem__(*args)

    def __repr__(self):
        return self.array.__repr__()

    def clear(self):
        self.mask.fill(False)
        self.infmask.fill(False)
        self.infindex.fill(0)

    def fill(self, *args):
        self.array.fill(*args)

    def setinf(self,i=0,j=0,value=np.inf):
        if not self.release_index(j):
            i_=self.infindex[j]
            self.mask[i_]=False
        self.infmask[j] = True
        self.infindex[j] = i
        self.mask[i] = True
        self.array[i,j]=value

    def swap(self,row=0,index=np.array([],dtype=np.int32)):
        k=0
        while k<index.shape[0]:
            j=index[k]
            if self.release_index(j):
                self.infmask[j]=True
                self.infindex[j]=row
                self.mask[row]=True
                break
            k+=1

    def try_swap(self,row=0,index=0):
        j=index
        try:
            if self.release_index(j):
                self.infmask[j]=True
                self.infindex[j]=row
                self.mask[row]=True
                return True
            else:
                return False
        except IndexError:
            return False


    def release_index(self,index=0):
        def swap(index=0,busy=0):
            i=0
            while i<self.infindex.shape[0]:
                val=self.array[index,i]
                if ~np.isinf(val) & ~self.infmask[i]:
                    self.infmask[busy]=False
                    self.infindex[i]=index
                    self.infmask[i] = True
                    #self.mask[index]=True
                    return True
                i+=1
            return False

        if ~self.infmask[index]:
            return True

        success=False
        used = self.infindex[index]

        j=0

        while j<self.mask.shape[0]:
            if swap(index=used,busy=index):
                return True
            k=0
            while k<self.infmask.shape[0]:
                if (k!=index)&(self.infmask[k]):
                    if swap(index=self.infindex[k], busy=k):
                        break
                k+=1
            j+=1

        return False














