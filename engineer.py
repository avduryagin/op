import numpy as np
from cortege_solution import recursion_solution as RS

class Features:
    def __init__(self):
        self.y=np.array([])
        self.X=np.array([])
    def fit(self,solution=RS(),dtype=np.float16):
        def get_index(index):
            if index is None:
                return []
            fun = solution.debit_functions[index]
            next_fun = fun.cortege_previous
            L = get_index(next_fun)
            L.insert(0, index)
            return L

        def get_vec(index, taumin_scale=0, taumax_scale=1, deltamin_scale=0, deltamax_scale=1):
            if index is None:
                return []
            fun = solution.debit_functions[index]
            ex = fun.executor
            weight = solution.executors[ex].weight
            tau = fun.tau
            delta = fun.cortege_b - fun.cortege_a
            next_fun = fun.cortege_previous
            va = [(tau - taumin_scale) / (taumax_scale - taumin_scale),
                  (delta - deltamin_scale) / (deltamax_scale - deltamin_scale), weight,ex]
            L = get_vec(next_fun,taumin_scale=taumin_scale,taumax_scale=taumax_scale,deltamin_scale=deltamin_scale,deltamax_scale=deltamax_scale)
            L.insert(0, va)
            return L
        def get_weights(nmax,order,vector,ncols=3,nexec=1,dtype=np.float16):
            def valid_rows(order):
                weights_mask = np.ones(len(order), dtype=bool)
                i = 0
                while i < len(order):
                    row = order[i]
                    j = 0
                    while j < len(row):
                        a = row[j]
                        try:
                            v = vector[a]
                        except KeyError:
                            weights_mask[i] = False
                            break
                        j+=1
                    i+=1

                order_=[order[i] for i in np.arange(len(order)) if weights_mask[i]]
                del order
                self.y=val[weights_mask]
                return order_

            order=valid_rows(order)
            weights = np.zeros(shape=(len(order), len(order[0]), nmax, ncols+nexec-1),dtype=dtype)

            i = 0
            while i < len(order):
                row = order[i]
                j = 0
                while j < len(row):
                    a = row[j]
                    v = vector[a]
                    k = 0
                    while k < len(v):
                        col = v[k]
                        w = 0
                        while w < ncols:
                            weights[i, j, k, w] = col[w]
                            w += 1
                        cell=w-1
                        executer=weights[i, j, k, cell]
                        ecell=cell+int(executer)
                        weights[i, j, k, w - 1]=0
                        weights[i, j, k, ecell] = 1

                        k += 1
                    j += 1
                i += 1
            return weights
        def scale_var(indices):
            mintau=np.inf
            maxtau=np.NINF
            mindelta=np.inf
            maxdelta=np.NINF
            for k in indices:
                fun=solution.debit_functions[k]
                tau=fun.tau
                delta=fun.cortege_b-fun.cortege_a
                if tau<mintau:
                    mintau=tau
                if tau>maxtau:
                    maxtau=tau
                if delta<mindelta:
                    mindelta=delta
                if delta>maxdelta:
                    maxdelta=delta
            return mintau,maxtau,mindelta,maxdelta

        # разделяем перестановки на значения метрики и индексы
        val_ = [solution.permutations[k][1] for k in solution.iteration_order.keys()]
        order_ = [solution.iteration_order[k] for k in solution.iteration_order.keys()]
        val = np.array(val_,dtype=dtype)
        self.y=val
        order = np.array(order_, dtype=np.int32)
        del val_
        del order_
        # получаем индексы всех мероприятий, входящих в кортежи order
        indices = []
        for k in order[0]:
            L = get_index(k)
            indices.extend(L)
        indices = np.array(indices, dtype=np.int32)
        taumin_scale_, taumax_scale_, deltamin_scale_, deltamax_scale_=scale_var(indices)
        # вычисляем масштабированные значения tau, delta для произвольной перестановки
        vec = dict()
        for k in order[0]:
            L = get_vec(k,taumin_scale=taumin_scale_,taumax_scale=taumax_scale_,deltamin_scale=deltamin_scale_,deltamax_scale=deltamax_scale_)
            vec[k] = L
        # вычисляем максимальную длину кортежа
        nmax = np.NINF
        for k in vec.keys():
            s = len(vec[k])
            if s > nmax:
                nmax = s
        nexec=len(solution.executors.keys())
        weight=get_weights(nmax,order,vec,4,nexec,dtype=dtype)
        self.X=weight
        del weight

