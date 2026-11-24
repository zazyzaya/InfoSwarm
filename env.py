import numpy as np 

from initializers import *

class SecretSpread: 
    def __init__(self, n, budget, init_strat=chaining): 
        self.n = n 
        self.budget = budget 