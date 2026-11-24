import numpy as np
import torch
from torch.nn import Module

'''
Implements the alpha-lattice flocking protocol described by:
https://www.seas.ucla.edu/coopcontrol/papers/flocking_tac04b.pdf
(Olfati-Saber, "Flocking for Multi-Agent Dynamic Systems", TAC 2006)

Everything is computed densely with matrix multiplies. For N agents the
cost is O(N^2 d) time and O(N^2) memory, which is fine up to a few
thousand agents on CPU (tens of thousands on GPU).
'''

# Hyperparams from original paper on pg. 24 of the pdf
D = 7
R = 1.2*D
D_PRIME = 0.6*D
R_PRIME = 1.2*D_PRIME
EPS = 0.1
A = B = 5; C = abs(A-B) / (4*A*B)**0.5   # Eq. 15: c = |a-b| / sqrt(4ab)
H_ALPHA = 0.2
H_BETA = 0.9

DT = 0.01 # Paper says "Between 0.01 and 0.03"


def sigma_norm_sq(sq):
    '''
    sigma-norm computed from a *squared* Euclidean norm:
        ||z||_s = 1/eps (sqrt(1 + eps ||z||^2) - 1)
    Lets us go straight from a pairwise squared-distance matrix.
    '''
    return (1/EPS) * (torch.sqrt(1 + EPS*sq) - 1)

def sigma_norm(z):
    ''' ||z||_s = 1/eps (sqrt(1 + eps ||z||^2) - 1) '''
    sq = torch.sum(z**2, dim=-1, keepdim=True)
    return sigma_norm_sq(sq)

def sigma_norm_gradient(z):
    ''' sigma_eps(z) = z / sqrt(1 + eps ||z||^2) = z / (1 + eps ||z||_s) '''
    return z / (1 + EPS*sigma_norm(z))

# More hyperparams, but need to define sigma norm first
R_ALPHA = sigma_norm_sq(torch.tensor(R**2)).item()
D_ALPHA = sigma_norm_sq(torch.tensor(D**2)).item()


def bump(z, h=H_ALPHA):
    '''
    Smooth bump function rho_h, Eq. 10:
        1                                  z in [0, h)
        1/2 (1 + cos(pi (z-h)/(1-h)))      z in [h, 1]
        0                                  otherwise
    Pure function (does not modify z) and branch-free.
    '''
    mid = 0.5 * (1 + torch.cos(torch.pi * (z - h) / (1 - h)))
    out = torch.where(z < h, torch.ones_like(z), mid)
    return torch.where((z < 0) | (z > 1), torch.zeros_like(z), out)

def phi(z):
    ''' Uneven sigmoid, Eq. 15 '''
    sig1 = z + C
    sig1 = sig1 / torch.sqrt(1 + sig1**2)
    return 0.5 * ((A+B) * sig1 + (A-B))

def action_function(z):
    ''' phi_alpha(z) = rho_h(z / r_alpha) * phi(z - d_alpha), Eq. 16 '''
    return bump(z / R_ALPHA) * phi(z - D_ALPHA)


def pairwise_sq_dist(q):
    '''
    ||q_i - q_j||^2 = ||q_i||^2 + ||q_j||^2 - 2 q_i . q_j  -> one matmul.
    Clamped at 0 to absorb floating-point cancellation.
    '''
    sq = (q*q).sum(-1)
    d2 = sq[:, None] + sq[None, :] - 2 * (q @ q.T)
    return d2.clamp_(min=0)

def spatial_adjacency_dense(q, d2=None):
    '''
    Dense N x N spatial adjacency a_ij = rho_h(||q_j - q_i||_s / r_alpha), Eq. 11.
    Diagonal is zero.
    '''
    if d2 is None:
        d2 = pairwise_sq_dist(q)
    a = bump(sigma_norm_sq(d2) / R_ALPHA)
    a.fill_diagonal_(0)
    return a

def spatial_adjacency(q):
    '''
    Sparse (edge list) version, same output format as before:
        ei   : 2 x E LongTensor (src, dst)
        a_ij : E weights
    Handy for passing the proximity graph to a GNN.
    '''
    a = spatial_adjacency_dense(q)
    src, dst = a.nonzero(as_tuple=True)
    return torch.stack([src, dst]), a[src, dst]


class Boids(Module):
    def forward(self, q, v):
        '''
            q: N x d positions
            v: N x d velocities
        Returns u_alpha (Eq. 23), the N x d control input.

        Both terms have the form  sum_j W_ij (x_j - x_i) = (W @ x) - rowsum(W) * x,
        so each is a single matmul over the dense weight matrix.
        '''
        d2 = pairwise_sq_dist(q)
        s = sigma_norm_sq(d2)                        # ||q_j - q_i||_s

        # Gradient (attraction/repulsion) term:
        #   sum_j phi_alpha(||q_j-q_i||_s) * (q_j - q_i) / sqrt(1 + eps ||q_j-q_i||^2)
        w = action_function(s) / torch.sqrt(1 + EPS*d2)
        w.fill_diagonal_(0)
        term1 = w @ q - w.sum(1, keepdim=True) * q

        # Consensus (velocity alignment) term:
        #   sum_j a_ij (p_j - p_i) = -L p
        a = bump(s / R_ALPHA)
        a.fill_diagonal_(0)
        term2 = a @ v - a.sum(1, keepdim=True) * v

        return term1 + term2


def hex_disk(n, spacing, jitter=0.15, device='cpu'):
    ''' n points of a hexagonal lattice closest to the origin, with small random jitter '''
    k = int(np.ceil(np.sqrt(n))) + 2
    i, j = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    pts = np.stack([i + 0.5*j, j * np.sqrt(3)/2], -1).reshape(-1, 2) * spacing
    pts = pts[np.argsort((pts**2).sum(1), kind='stable')[:n]]
    q = torch.tensor(pts.astype(np.float32), device=device)
    return q + jitter * spacing * torch.randn(n, 2, device=device)


TARGET_INFO_MODES = ('global', 'local', 'relay', 'consensus')


class TargetKnowledge():
    """
    What each agent knows about the gamma agent (target), for three regimes:

      'global'  every agent knows (q_r, p_r) at all times (the paper's Algorithm 2)
      'local'   an agent knows the target only while it is within its sensing
                range ("informed"); otherwise it knows nothing about it
      'relay'   informed agents copy the true target state and timestamp it.
                Every agent keeps an estimate (q_hat_i, p_hat_i, stamp_i) and
                dead-reckons it (q_hat += p_hat dt). Every `hop_every` steps it
                looks at its neighbours on the proximity graph and copies the
                estimate with the newest stamp, if that is newer than its own.
                Information therefore moves one hop per relay round and is never
                averaged with stale estimates.
      'consensus'  same, but instead of copying the freshest estimate, agents
                update ONLY from neighbours by pinned consensus:

                    d/dt q_hat_i = p_hat_i + k_relay * (avg_j q_hat_j - q_hat_i)
                    d/dt p_hat_i =           k_relay * (avg_j p_hat_j - p_hat_i)

                where avg_j is the a_ij-weighted mean over neighbours that already
                have an estimate (degree-normalised, so it is stable for any
                neighbourhood size as long as k_relay * dt < 1). Between
                updates an agent dead-reckons the target with its velocity estimate.
                An agent with no estimate yet ("unknown") adopts the weighted mean
                of its knowing neighbours' estimates the first time it has one.
                Once known, an agent never forgets.

    Everything is dense N x N matmuls on the same adjacency the flock uses.

    Faults (relay mode only; all default off, all can be changed between steps):
      obs_noise   std of Gaussian noise on an informed agent's observation of the
                  target (applied to position, and to velocity with the same std)
      hop_noise   std of Gaussian noise added to an estimate every time it is
                  copied from a neighbour (errors accumulate with hop count)
      drop        probability that any single neighbour-to-neighbour message is lost
                  in a given gossip round
      liar_frac   fraction of agents that are liars. A liar ignores the real target
                  and every incoming message, always announces the decoy (position
                  `decoy`, velocity 0) and steers toward it itself. Liars are chosen
                  by a fixed random priority, so raising liar_frac adds liars
                  without reshuffling the existing ones.
      forge       seconds a liar adds to the timestamp it announces. 0 = it claims a
                  fresh observation "now", which already beats any estimate that
                  has travelled a hop; > 0 = a future stamp, which no honest
                  estimate can ever beat until real time catches up.
      reject_future  defence: honest agents ignore messages stamped later than
                  their own clock (stops forge > 0, not forge = 0)
      decoy       position the liars announce (defaults to 80 units below the target)
    """
    def __init__(self, flock, mode='relay', k_relay=5.0, hop_every=5):
        assert mode in TARGET_INFO_MODES, mode
        self.f, self.mode, self.k_relay, self.hop_every = flock, mode, k_relay, hop_every
        self.t, self.steps = 0.0, 0
        n, dim = flock.q.shape
        self.q_hat = flock.q.new_zeros(n, dim) + flock.target
        self.p_hat = flock.q.new_zeros(n, dim) + flock.target_v
        if mode == 'global':
            self.known = torch.ones(n, dtype=torch.bool, device=flock.q.device)
        else:
            self.known = flock.sees_target_now()
        # time at which each agent's estimate was observed at the source (-inf = none)
        self.stamp = torch.where(self.known, torch.zeros_like(flock.q[:, 0]),
                                 torch.full_like(flock.q[:, 0], float('-inf')))

        # relay faults (see class docstring)
        self.obs_noise = self.hop_noise = self.drop = 0.0
        self.liar_frac, self.forge, self.reject_future = 0.0, 0.0, False
        self.decoy = flock.target.clone() + flock.target.new_zeros(dim)
        self.decoy[1] = self.decoy[1] - 80.0
        self._liar_rank = torch.rand(n, device=flock.q.device)
        # provenance: True if an agent's current estimate traces back to a liar
        self.taint = torch.zeros(n, dtype=torch.bool, device=flock.q.device)

    def copy_faults(self, other):
        for k in ('obs_noise', 'hop_noise', 'drop', 'liar_frac', 'forge', 'reject_future', '_liar_rank'):
            setattr(self, k, getattr(other, k))
        self.decoy = other.decoy.clone()

    def liars(self):
        """ N bool: which agents are lying (relay mode only) """
        if self.mode != 'relay' or self.liar_frac <= 0:
            return torch.zeros_like(self._liar_rank, dtype=torch.bool)
        return self._liar_rank < self.liar_frac

    def update(self, adj):
        """ Advance estimates one dt. adj: N x N non-negative neighbour weights. """
        f = self.f
        if self.mode == 'global':
            return
        informed = f.sees_target_now()
        if self.mode == 'local':
            self.known = informed
            self.q_hat[informed] = f.target
            self.p_hat[informed] = f.target_v
            return

        if self.mode == 'relay':
            return self._relay(adj, informed)

        kf = self.known.to(adj.dtype)
        # first contact: unknown agents with a knowing neighbour adopt the neighbours' mean
        ak = adj * kf[None, :]
        deg = ak.sum(1, keepdim=True)
        newly = (~self.known) & (deg[:, 0] > 0)
        if newly.any():
            mq = (ak @ self.q_hat) / deg.clamp(min=1e-12)
            mp = (ak @ self.p_hat) / deg.clamp(min=1e-12)
            self.q_hat = torch.where(newly[:, None], mq, self.q_hat)
            self.p_hat = torch.where(newly[:, None], mp, self.p_hat)
            self.known = self.known | newly
            kf = self.known.to(adj.dtype)
            ak = adj * kf[None, :]
            deg = ak.sum(1, keepdim=True)

        # pinned consensus among knowing agents (dead reckoning + neighbour averaging)
        has = (deg > 0).to(adj.dtype)
        inv = has / deg.clamp(min=1e-12)
        dq = self.p_hat + self.k_relay * has * ((ak @ self.q_hat) * inv - self.q_hat)
        dp = self.k_relay * has * ((ak @ self.p_hat) * inv - self.p_hat)
        m = kf[:, None]
        self.q_hat = self.q_hat + f.dt * m * dq
        self.p_hat = self.p_hat + f.dt * m * dp

        # informed agents see the truth
        self.known = self.known | informed
        self.q_hat = torch.where(informed[:, None], f.target.expand_as(self.q_hat), self.q_hat)
        self.p_hat = torch.where(informed[:, None], f.target_v.expand_as(self.p_hat), self.p_hat)

    def _relay(self, adj, informed):
        f = self.f
        self.t += f.dt
        self.steps += 1
        liar = self.liars()
        honest = ~liar
        informed = informed & honest

        # dead reckoning
        self.q_hat = self.q_hat + f.dt * self.p_hat

        # liars hold the decoy and keep re-stamping it
        self.q_hat = torch.where(liar[:, None], self.decoy.expand_as(self.q_hat), self.q_hat)
        self.p_hat = torch.where(liar[:, None], torch.zeros_like(self.p_hat), self.p_hat)
        self.stamp = torch.where(liar, torch.full_like(self.stamp, self.t + self.forge), self.stamp)
        self.taint = self.taint | liar

        # gossip round: honest agents copy the freshest neighbouring message if newer than theirs
        if self.steps % self.hop_every == 0:
            link = adj > 0
            if self.drop > 0:
                link = link & (torch.rand_like(adj) >= self.drop)
            ninf = torch.full_like(adj, float('-inf'))
            cand = torch.where(link, self.stamp[None, :].expand_as(adj), ninf)
            if self.reject_future:
                cand = torch.where(cand > self.t, ninf, cand)
            best, j = cand.max(dim=1)
            take = (best > self.stamp) & honest
            if take.any():
                nq, npv = self.q_hat[j], self.p_hat[j]
                if self.hop_noise > 0:
                    nq = nq + self.hop_noise * torch.randn_like(nq)
                    npv = npv + self.hop_noise * torch.randn_like(npv)
                self.q_hat = torch.where(take[:, None], nq, self.q_hat)
                self.p_hat = torch.where(take[:, None], npv, self.p_hat)
                self.stamp = torch.where(take, best, self.stamp)
                self.taint = torch.where(take, self.taint[j], self.taint)

        # honest informed agents observe the (noisy) truth now
        oq, op = f.target.expand_as(self.q_hat), f.target_v.expand_as(self.p_hat)
        if self.obs_noise > 0:
            oq = oq + self.obs_noise * torch.randn_like(self.q_hat)
            op = op + self.obs_noise * torch.randn_like(self.p_hat)
        self.q_hat = torch.where(informed[:, None], oq, self.q_hat)
        self.p_hat = torch.where(informed[:, None], op, self.p_hat)
        self.stamp = torch.where(informed, torch.full_like(self.stamp, self.t), self.stamp)
        self.taint = self.taint & ~informed          # a direct sighting replaces any lie
        self.known = self.stamp > float('-inf')

    def poisoned(self):
        """ N bool: honest agents whose current estimate originated with a liar
        (tracked by provenance, so it is always 0 when there have been no liars) """
        return self.known & ~self.liars() & self.taint

    def age(self):
        ''' N: seconds since each agent's information left the target (relay only) '''
        return self.t - self.stamp

    def reference(self):
        """ (knows N x 1 float, q_ref N x d, p_ref N x d) used by the control law """
        f = self.f
        if self.mode == 'global':
            k = torch.ones_like(f.q[:, :1])
            return k, f.target.expand_as(f.q), f.target_v.expand_as(f.q)
        return self.known.to(f.q.dtype)[:, None], self.q_hat, self.p_hat

    def error(self):
        """ N: distance between each agent's estimate and the true target (nan if unknown) """
        f = self.f
        if self.mode == 'global':
            return torch.zeros_like(f.q[:, 0])
        e = (self.q_hat - f.target).norm(dim=1)
        ok = self.known & ~self.liars()          # liars' "estimates" are their lie
        return torch.where(ok, e, torch.full_like(e, float('nan')))


class Flock():
    def __init__(self, n, device='cpu', dt=DT, dim=2, spread=None,
                 c1=0., c2=0., target=None, target_v=None, noise=0.,
                 target_info='relay', k_relay=5.0, hop_every=5, init='grid'):
        '''
        c1, c2   : gains on the optional navigational-feedback ("gamma agent")
                   term of Algorithm 2. Leave at 0 for pure Algorithm 1
                   (which, per the paper, tends to fragment the flock).
        target   : gamma-agent position (defaults to origin)
        target_v : gamma-agent velocity (defaults to 0)
        noise    : std of a random acceleration added every step (0 = deterministic)
        target_info : how agents learn the gamma agent's state; see TargetKnowledge.
                   'global' = the paper (everyone knows it), 'local' = only agents
                   within r of the target know it, 'relay' (default) = informed agents
                   pass the freshest estimate to neighbours over the proximity graph,
                   'consensus' = neighbours average their estimates instead.
                   The gamma term uses each agent's own estimate:
                       knowing agent:  -c1 (q_i - q_hat_i) - c2 (p_i - p_hat_i)
                       unknowing:      -c2 p_i   (drag only; an agent always knows
                                                  its own velocity)
                   An agent counts as "seeing" the target when it is inside the same
                   bump cutoff used for a_ij (Euclidean distance < r).
        k_relay  : averaging gain for 'consensus' (needs k_relay * dt < 1)
        hop_every: 'relay' gossip period in steps (5 steps = 0.05 s per hop)
        init     : 'grid'     jittered hexagonal disk at spacing 0.9 d (2-D only): starts
                              connected and relaxes gently to spacing d, so the flock
                              does not blow apart (default)
                   'gaussian' q ~ N(0, spread^2), the original start; when dense
                              the initial repulsion can fragment the flock
        '''
        spread = D if spread is None else spread
        if init == 'grid' and dim == 2:
            self.q = hex_disk(n, 0.9*D, device=device)
        else:
            self.q = torch.randn(n, dim, device=device) * spread
        self.v = torch.zeros(n, dim, device=device)
        self.device = device

        self.boid = Boids()
        self.dt = dt

        self.c1, self.c2 = c1, c2
        self.target = torch.zeros(dim, device=device) if target is None else target
        self.target_v = torch.zeros(dim, device=device) if target_v is None else target_v
        self.noise = noise
        self.info = TargetKnowledge(self, target_info, k_relay, hop_every)

    @property
    def target_info(self):
        return self.info.mode

    @target_info.setter
    def target_info(self, mode):
        if mode != self.info.mode:
            old = self.info
            self.info = TargetKnowledge(self, mode, old.k_relay, old.hop_every)
            self.info.copy_faults(old)

    @torch.no_grad()
    def step(self):
        u = self.boid(self.q, self.v)
        self.info.update(spatial_adjacency_dense(self.q))
        if self.c1 or self.c2:
            # Knowing agents track their estimate of the target. Unknowing agents keep
            # the c2 drag on their own velocity: without it they are frictionless and
            # the flock fragments and flies apart (Algorithm 1 behaviour).
            k, q_ref, p_ref = self.info.reference()
            u = u - k * (self.c1 * (self.q - q_ref) + self.c2 * (self.v - p_ref)) \
                  - (1 - k) * self.c2 * self.v
        if self.noise:
            u = u + self.noise * torch.randn_like(self.q)

        self.v += u * self.dt
        self.q += self.v * self.dt
        self.target += self.target_v * self.dt

    def sees_target_now(self):
        ''' N bool: agents whose sensing range (same bump cutoff as a_ij) contains the target '''
        return bump(sigma_norm(self.q - self.target) / R_ALPHA)[:, 0] > 0

    def sees_target(self):
        ''' N bool: agents that directly sense the target (all of them in 'global' mode) '''
        if self.info.mode == 'global':
            return torch.ones_like(self.q[:, 0], dtype=torch.bool)
        return self.sees_target_now()

    def knows_target(self):
        ''' N bool: agents with an estimate of the target (directly or relayed) '''
        return self.info.known

    def adjacency(self):
        return spatial_adjacency_dense(self.q)


class ReynoldsFlock():
    '''
    Classic Reynolds (1987) boids: separation, alignment, cohesion, with a
    cruising-speed band, limited field of view and steering noise. Unlike the
    alpha-lattice there is no potential being minimised and agents are
    self-propelled (|v| never drops below vmin), so the flock keeps churning
    rather than freezing into a crystal.

    Same interface as Flock (q, v, dt, c1, c2, target, step(), adjacency()),
    so the GUI can drive either. c1 pulls every boid toward `target`, which
    keeps the flock on screen and, when strong, makes it swirl/mill.
    c2 is unused (speed is set by vmin/vmax instead).

    All neighbour sums are dense N x N masks times (N x d) matrices.
    '''
    def __init__(self, n, device='cpu', dt=DT, dim=2, spread=None,
                 c1=0.2, c2=0., target=None,
                 r=2*D, r_sep=D, fov_deg=270.,
                 w_coh=1.0, w_ali=2.0, w_sep=60.,
                 vmin=8., vmax=16., a_max=40., noise=20., target_info='global',
                 k_relay=5.0, hop_every=5):
        spread = D if spread is None else spread
        self.q = torch.randn(n, dim, device=device) * spread
        v = torch.randn(n, dim, device=device)
        self.v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-9) * vmax
        self.device, self.dt = device, dt

        self.c1, self.c2 = c1, c2
        self.target = torch.zeros(dim, device=device) if target is None else target
        self.target_v = torch.zeros(dim, device=device)

        self.r, self.r_sep, self.fov_deg = r, r_sep, fov_deg
        self.w_coh, self.w_ali, self.w_sep = w_coh, w_ali, w_sep
        self.vmin, self.vmax, self.a_max, self.noise = vmin, vmax, a_max, noise
        self.info = TargetKnowledge(self, target_info, k_relay, hop_every)

    target_info = Flock.target_info

    def sees_target_now(self):
        return ((self.q - self.target)**2).sum(1) < self.r**2

    def sees_target(self):
        if self.info.mode == 'global':
            return torch.ones_like(self.q[:, 0], dtype=torch.bool)
        return self.sees_target_now()

    def knows_target(self):
        return self.info.known

    def _neighbours(self, d2):
        ''' Boolean N x N: j is within r of i and inside i's field of view '''
        q, v = self.q, self.v
        vh = v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)
        # vh_i . (q_j - q_i) = (vh @ q^T)_ij - (vh_i . q_i)
        ahead = vh @ q.T - (vh * q).sum(1, keepdim=True)
        cos_half = torch.cos(torch.tensor(self.fov_deg / 2 * torch.pi / 180)).item()
        seen = ahead > cos_half * torch.sqrt(d2)
        nb = (d2 < self.r**2) & seen
        nb.fill_diagonal_(False)
        return nb

    @torch.no_grad()
    def step(self):
        q, v = self.q, self.v
        d2 = pairwise_sq_dist(q)

        nb = self._neighbours(d2).to(q.dtype)
        cnt = nb.sum(1, keepdim=True)
        has = (cnt > 0).to(q.dtype)
        cnt = cnt.clamp(min=1)

        coh = (nb @ q) / cnt - q                  # toward local centroid
        ali = (nb @ v) / cnt - v                  # toward local mean velocity

        # separation: sum_j (q_i - q_j) / ||q_i - q_j||^2 over close boids (any direction)
        s = (d2 < self.r_sep**2).to(q.dtype) / d2.clamp(min=1e-2)
        s.fill_diagonal_(0)
        sep = q * s.sum(1, keepdim=True) - s @ q

        a = has * (self.w_coh*coh + self.w_ali*ali) + self.w_sep*sep
        self.info.update(self.adjacency())
        if self.c1:
            k, q_ref, _ = self.info.reference()
            a = a - self.c1 * k * (q - q_ref)
        if self.noise:
            a = a + self.noise * torch.randn_like(q)

        # steering limit, then keep speed inside [vmin, vmax]
        an = a.norm(dim=1, keepdim=True)
        a = a * (self.a_max / an.clamp(min=1e-9)).clamp(max=1)
        v = v + a * self.dt
        sp = v.norm(dim=1, keepdim=True)
        v = v * sp.clamp(self.vmin, self.vmax) / sp.clamp(min=1e-9)

        self.v = v
        self.q = q + v * self.dt
        self.target = self.target + self.target_v * self.dt

    def adjacency(self):
        ''' Undirected proximity graph (distance < r) as 0/1 weights, for metrics/drawing '''
        a = (pairwise_sq_dist(self.q) < self.r**2).to(self.q.dtype)
        a.fill_diagonal_(0)
        return a


MODELS = {
    'Reynolds': ReynoldsFlock,
    'Olfati-Saber': Flock,
}


if __name__ == '__main__':
    # Tests
    import time
    q = torch.randn(1000,2) * D
    v = torch.randn(1000,2)
    b = Boids()

    st = time.time()
    u = b(q,v)
    print(f'1000 agents: {(time.time()-st)*1000:0.1f} ms, out shape {tuple(u.shape)}')