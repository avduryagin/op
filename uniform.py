import numpy as np
import numba as nb

def uniform_assignment(duration,executors,executor_permissions,indices,delta=1.,tolerance=np.inf):
    assert (type(duration)==np.ndarray)&(type(executors)==np.ndarray)&(type(executors)==np.ndarray)&(type(executor_permissions)==np.ndarray)&(type(indices)==np.ndarray),"The method excepts every parameters as array numpy."

    assert (duration.shape[0]==indices.shape[0])&(duration.shape[0]==executor_permissions.shape[1]),"The method excepts matching arrays shape: "
    "(duration.shape[0]==indices.shape[0])&(duration.shape[0]==executors_permission.shape[1])"
    assert (executors.shape[0]==executor_permissions.shape[0]),"The method excepts matching arrays shape: "
    "(executors.shape[0]==executors_permission.shape[0])"
    assert (duration.dtype==np.float32)|(duration.dtype==np.float64),"Expect duration as float ndarray."
    assert (indices.dtype==np.int32)|(indices.dtype==np.int64),"Expect indices as integer ndarray."
    assert (executors.dtype==np.int32)|(executors.dtype==np.int64),"Expect executors as integer ndarray."
    assert (executor_permissions.dtype==np.uint8)|(executor_permissions.dtype==np.bool),"Expect executors_permission as boolean or np.uint8  ndarray."
    if executor_permissions.dtype==np.bool:
        executor_permissions=executor_permissions.astype(np.uint8)
    __uniform_assignment_numba(duration,executors,executor_permissions,indices,abs(delta),abs(tolerance))


@nb.jit(fastmath=True,cache=True)
def __uniform_assignment_numba(duration,executors,executor_permissions,indices,delta,tolerance):
    
    executors_mask=np.ones(shape=executors.shape[0],dtype=np.uint8)
    activities_mask=np.ones(shape=duration.shape[0],dtype=np.uint8)
    executors_position=np.zeros(shape=executors.shape[0],dtype=duration.dtype)
    indices.fill(-1)
    __check(duration,executors,executor_permissions,activities_mask)
    activities_number=activities_mask.sum()
    tmin=0.
    epsilon=tolerance    
    counter=0
    while counter<activities_number:
        j=0
        moved=False
        
        while j<executors.shape[0]:
            executor=executors[j]
            if not executors_mask[j]:
                j+=1
                continue
            executors_mask[j]=0
            i=0
            while i<duration.shape[0]:
                if (activities_mask[i])&(executor_permissions[executor,i]):
                    executors_mask[j]=1
                    if executors_position[j]-tmin<epsilon:
                        executors_position[j]+=duration[i]
                        indices[i]=executor
                        activities_mask[i]=0
                        counter+=1                        
                        moved=True
                        break
                i+=1
            j+=1
        
        if not (counter<activities_number):
            break
        
        if moved:        
            tmin=executors_position[np.where(executors_mask>0)[0]].min()
            epsilon=tolerance       

            
        else:
            epsilon+=delta
            

                

@nb.jit(fastmath=True,cache=True)
def __check_nb(duration,executors,executor_permissions,activities_mask):
    activities_mask.fill(0)    
    j=0
    while j<executors.shape[0]:
        executor=executors[j]
        i=0
        while i<duration.shape[0]:
            if activities_mask[i]:
                i+=1
                continue
            if executor_permissions[executor,i]:
                activities_mask[i]=1
            i+=1
        j+=1

@nb.jit(fastmath=True,cache=True,parallel=True)
def __check(duration,executors,executor_permissions,activities_mask):
    activities_mask.fill(0)

    for i in nb.prange(duration.shape[0]):
        j=0
        while j<executors.shape[0]:
            executor=executors[j]            
            if executor_permissions[executor,i]:
                activities_mask[i]=1
                break
            j+=1

        




