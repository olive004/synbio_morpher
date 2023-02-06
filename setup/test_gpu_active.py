import jax
import jax.numpy as jnp


## Jax test
jax.config.update('jax_platform_name', 'gpu')


def test_vmap(m):
    return jnp.sin(m)


jax.vmap(test_vmap)(jnp.ones(4))


## Numba test
from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer   
  
# normal function to run on cpu
def func(a):                                
    for i in range(10000000):
        a[i]+= 1      
  
# function optimized to run on gpu 
@jit(target_backend='cuda')                         
def func2(a):
    for i in range(10000000):
        a[i]+= 1
if __name__=="__main__":
    n = 10000000                            
    a = np.ones(n, dtype = np.float64)
      
    start = timer()
    func(a)
    print("without GPU:", timer()-start)    
      
    start = timer()
    func2(a)
    print("with GPU:", timer()-start)
