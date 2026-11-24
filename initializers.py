from math import floor, sqrt

import numpy as np 

def chaining(n): 
    '''
        If n=5, k=2, produces
        [
            [0,1,1,0,0],
            [0,0,1,1,0],
            [0,0,0,1,1],
            [1,0,0,0,1],
            [1,1,0,0,0]
        ]
        A k-linked chain in order of node id
        '''
    
    # TODO: fiddle with this. Effectively the budget, so may be important
    k = floor(sqrt(n))
    i = np.arange(n)
    adj = (((i[None, :] - i[:, None]) % n) <= k).astype(float)

    # No need for self-loops
    np.fill_diagonal(adj, 0)
    return adj 

def rand_init(n): 
    '''
    Each agent initiated with k random neighbors 
    '''

    k = floor(sqrt(n))
    r = np.random.rand(n,n)
    np.fill_diagonal(r, np.inf) # Ensure no self-loops 

    idx = np.argpartition(r, k - 1, axis=1)[:, :k]
    a = np.zeros((n,n), dtype=float)
    np.put_along_axis(a, idx, 1, axis=1)

    return a 