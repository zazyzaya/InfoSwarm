import torch
from torch.nn import Module
from torch_geometric.utils import scatter

'''
Implements functions described by:
https://www.seas.ucla.edu/coopcontrol/papers/flocking_tac04b.pdf
'''

# Hyperparams from original paper on pg. 24 of the pdf
D = 7
R = 1.2*D
D_PRIME = 0.6*D
R_PRIME = 1.2*D_PRIME
EPS = 0.1
A = B = 5; C=0
H_ALPHA = 0.2
H_BETA = 0.9

DT = 0.01 # Paper says "Between 0.01 and 0.03"

def sigma_norm(z):
    '''
    1/eps (sqrt(1-eps ||x||^2 ) - 1)
    '''
    sq = torch.sum(z**2, dim=-1, keepdim=True)

    return (1/EPS) * (
        torch.sqrt(
            1 + EPS*sq
        ) - 1
    )

def sigma_norm_gradient(z):
    return z / (1 + EPS*sigma_norm(z))

# More hyperparams, but need to define sigma norm first
R_ALPHA = sigma_norm(torch.tensor(R)).item()
D_ALPHA = sigma_norm(torch.tensor(D)).item()

def _bump(z):
    '''
                          cos(pi (z-h))
        p(z) = 1/2 * (1 + -------------  )
                              (1-h)

    For input z and some threshold h
    '''
    return 0.5 * (
        1 + torch.cos( (torch.pi * (z-H_ALPHA)) / (1-H_ALPHA) )
    )

def bump(z):
    '''
    Smooth pairwise function Eq. 10
    '''
    # TODO more efficient way to do this?
    z[(z > 0) & (z < H_ALPHA)] = 1.
    z[(z > 1)] = 0.
    z[(z >= H_ALPHA) & (z <= 1)] = _bump(z[(z >= H_ALPHA) & (z <= 1)])

    return z

def phi(z):
    ''' Eq. 15 '''
    sig1 = lambda x : x / torch.sqrt(1 + torch.pow(x,2))

    return 0.5 * (
        (A+B) * sig1(z + C) + (A - B)
    )

def action_function(z):
    bump(z/R) * phi(z - D_ALPHA)

def _spatial_adjecency(qi, qj):
    return bump(
        sigma_norm(qi - qj) / R_ALPHA
    )

def spatial_adjacency(q):
    a_ij = []
    src,dst = [],[]

    # TODO do this with matmuls
    for i in range(q.shape[0]):
        for j in range(q.shape[0]):
            if i == j:
                continue
            ij = _spatial_adjecency(q[i], q[j])

            if ij != 0:
                a_ij.append(ij)
                src.append(i)
                dst.append(j)

    a_ij = torch.tensor(a_ij)
    ei = torch.tensor([src,dst])
    return ei, a_ij

class Boids(Module):
    def forward(self, q,v, ei=None, dim_size=None):
        '''
            x: tuple of (q,p) velocity and position
            ei: edge index (need both src and dst)
        '''
        # Need to calculate this for term 2 anyway, so just do it and don't
        # require ei explicitly input
        ei, a_ij = spatial_adjacency(q)  # V x V
        src,dst = ei

        # Attraction/repulsion term
        phi_a = phi( sigma_norm(q[dst] - q[src]) )
        n_ij = sigma_norm_gradient(q[dst] - q[src])
        term1 = phi_a * n_ij
        term1 = scatter(term1, dst, dim=0, dim_size=dim_size, reduce='sum') # V x d

        # Alignment term
        term2 = a_ij[:,None] * (v[dst] - v[src])
        term2 = scatter(term2, dst, dim=0, dim_size=dim_size, reduce='sum')

        return term1 + term2

class Flock():
    def __init__(self, n, device='cpu', dt=DT):
        self.q = torch.randn(n,2, device=device) * D
        self.v = torch.zeros(n,2, device=device)
        self.device = device

        self.boid = Boids()
        self.dt = dt

    def step(self):
        dv = self.boid.forward(self.q, self.v)
        self.v += dv * self.dt
        self.q += self.v * self.dt

if __name__ == '__main__':
    # Tests
    q = torch.randn(100,2) * D
    v = torch.zeros(100,2)
    b = Boids()

    b.forward(q,v)