'''
Real-time viewer for the alpha-lattice flock in boid.py (pyqtgraph + Qt).

    python gui.py --n 1000 --device cuda                        # Olfati-Saber (default)
    python gui.py --n 1000 --device cuda --model Reynolds       # classic boids

How it is put together
    * The simulation runs in its own thread (Sim) and publishes CPU snapshots.
    * The window only ever draws the newest snapshot, from one timer.
    * Buttons and checkboxes just post a request or set a flag; none of them
      steps the simulation or redraws directly. A slow step therefore cannot
      lock the window, and Reset returns immediately.
    * A watchdog prints every thread's stack to the console if the window
      stops updating for WATCHDOG_S seconds, so a stall reports where it is.

Models (dropdown; switching resets with that model's defaults)
    Reynolds              separation / alignment / cohesion, cruising speed,
                          270 deg field of view, steering noise: keeps churning
    Olfati-Saber          the paper's alpha-lattice: settles into a stable mesh

Controls (left panel)
    Play/Pause, Step, Reset
    agents (applies on Reset), c1 pull toward target, c2 (lattice only),
    noise (random steering), sim steps per frame
    target info: global (everyone knows it, the paper) / local (only agents within r)
             / relay (informed agents gossip the freshest estimate to neighbours)
             / consensus (neighbours average their estimates)
             orange = sees the target, green = knows it by relay, blue = knows nothing;
    relay faults (relay mode only): observation noise, per-hop noise, message drop
             probability, liar fraction, forged-timestamp lead, and a
             "reject future stamps" defence. Liars (purple squares) announce the
             decoy (purple x; right-click to move it); honest agents whose estimate
             is closer to the decoy than to the target are drawn pink ("fooled")
             the red ring is the target's visibility radius
    target speed: a click sends the target travelling to the clicked point at this
             speed (0 = jump there instantly)
    toggles: proximity-graph edges, velocity vectors, follow flock
    Click on the flock view to send the gamma-agent target there (only matters if c1 > 0)
    Mouse wheel / drag pans and zooms (untick "follow flock" first)

Right-hand plots track cohesion:
    - mean age of target information over honest agents that know it (relay:
      seconds since the estimate left the target; negative = forged future stamp)
    - connected components of the proximity graph (1 = cohesive flock)
'''
import argparse
import faulthandler
import sys
import threading
import time
import traceback
import warnings
from collections import deque

import numpy as np
import torch
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from boid import MODELS, D, R, TARGET_INFO_MODES

GUI_VERSION = '13'
WATCHDOG_S = 5            # seconds without a GUI tick before stacks are dumped
EMPTY = np.zeros(0)

# Per-model defaults applied when you switch models in the dropdown
MODEL_DEFAULTS = {
    'Reynolds':          dict(c1=0.2,  c2=0.0, noise=20.0, info='global'),
    'Olfati-Saber':  dict(c1=0.02, c2=0.3, noise=50.0, info='relay'),
}
DEFAULT_MODEL = 'Olfati-Saber'

# pyqtgraph compares view ranges against +/-1e307 limits; if a float32 ever
# reaches it NumPy 2 warns on every frame. Harmless, but slow on a console.
warnings.filterwarnings('ignore', message='overflow encountered in cast',
                        category=RuntimeWarning, module=r'pyqtgraph\.')


def n_components(i, j, n):
    '''
    Number of connected components given an undirected edge list (i[k], j[k])
    over n nodes. Min-label propagation with pointer jumping; every pass is
    vectorised, so there is no per-node Python loop.
    '''
    if len(i) == 0:
        return n
    lab = np.arange(n)
    while True:
        new = lab.copy()
        np.minimum.at(new, i, lab[j])
        np.minimum.at(new, j, lab[i])
        new = new[new]
        if np.array_equal(new, lab):
            return int((lab == np.arange(n)).sum())
        lab = new


def segments(a, b):
    ''' Interleave endpoints for PlotCurveItem(connect="pairs") '''
    seg = np.empty((2 * len(a), 2))
    seg[0::2], seg[1::2] = a, b
    return seg[:, 0], seg[:, 1]


class Sim:
    '''
    Owns the Flock. Steps it in the background and publishes snapshots made
    of plain float64 NumPy arrays, which are never modified after publishing.
    '''
    HISTORY = 2000

    def __init__(self, n, device, model, c1, c2, noise, spf, rate=60):
        self.thread = threading.Thread(target=self._run, name='flock-sim', daemon=True)
        self.device = device
        self.rate = rate                  # loop iterations per second (each does spf steps)
        self.error = None
        self._lock = threading.Lock()
        self._quit = threading.Event()
        self._params = {'c1': float(c1), 'c2': float(c2), 'noise': float(noise),
                        'info': MODEL_DEFAULTS[model]['info'], 'tspeed': 5.0,
                        'obs_noise': 0.0, 'hop_noise': 0.0, 'drop': 0.0,
                        'liar_frac': 0.0, 'forge': 0.0, 'reject_future': False,
                        'spf': int(spf), 'running': True}
        self._reset_n = (int(n), model)   # pending requests, consumed by the sim thread
        self._step = False
        self._target = None
        self._decoy = None
        self._snap = None

    # ---------- called from the GUI thread; none of these block ----------
    def set_params(self, **kw):
        with self._lock:
            self._params.update(kw)

    @property
    def running(self):
        with self._lock:
            return self._params['running']

    def request_reset(self, n, model):
        with self._lock:
            self._reset_n = (int(n), model)

    def request_step(self):
        with self._lock:
            self._step = True

    def set_target(self, x, y):
        with self._lock:
            self._target = (float(x), float(y))

    def set_decoy(self, x, y):
        with self._lock:
            self._decoy = (float(x), float(y))

    def snapshot(self):
        with self._lock:
            return self._snap

    def start(self):
        self.thread.start()

    def stop(self, timeout=2.0):
        self._quit.set()
        if self.thread.is_alive():
            self.thread.join(timeout)

    # ---------- sim thread ----------
    def _run(self):
        try:
            self._loop()
        except Exception:
            self.error = traceback.format_exc()
            print(self.error, file=sys.stderr, flush=True)

    def _loop(self):
        flock, last, gen, steps, rev = None, None, 0, 0, 0
        ht, hp, hc, hk, hpz = (deque(maxlen=self.HISTORY) for _ in range(5))
        goal = None
        rate = 0.0

        while not self._quit.is_set():
            t0 = time.perf_counter()
            with self._lock:
                p = dict(self._params)
                reset_n, self._reset_n = self._reset_n, None
                step, self._step = self._step, False
                target, self._target = self._target, None
                decoy, self._decoy = self._decoy, None
                if reset_n is not None and step:
                    self._step = True            # keep a Step that arrived with a Reset

            moved = False
            if reset_n is not None:
                n, model = reset_n
                # spread so initial density roughly matches the paper's setup
                spread = max(float(D), 2.5 * D * n ** 0.5 / 10)
                flock = MODELS[model](n, device=self.device, spread=float(spread))
                flock.target_info = p['info']
                for k in ('obs_noise', 'hop_noise', 'drop', 'liar_frac', 'forge', 'reject_future'):
                    setattr(flock.info, k, p[k])
                gen, steps, goal = gen + 1, 0, None
                ht.clear(); hp.clear(); hc.clear(); hk.clear(); hpz.clear()
                moved = True

            if decoy is not None:
                flock.info.decoy = torch.tensor(decoy, dtype=flock.q.dtype, device=self.device)
                rev += 1

            if target is not None:
                goal = torch.tensor(target, dtype=flock.q.dtype, device=self.device)
                if p['tspeed'] <= 0:                 # jump straight there
                    flock.target = goal
                    flock.target_v = torch.zeros_like(goal)
                    goal = None
                rev += 1

            # On the pass that resets, publish the initial state before stepping.
            if reset_n is None and (p['running'] or step):
                flock.c1, flock.c2, flock.noise = p['c1'], p['c2'], p['noise']
                flock.target_info = p['info']
                for k in ('obs_noise', 'hop_noise', 'drop', 'liar_frac', 'forge', 'reject_future'):
                    setattr(flock.info, k, p[k])
                if goal is not None:
                    # move the target toward the clicked point at tspeed, arriving exactly
                    off = goal - flock.target
                    dist = float(off.norm())
                    span = p['spf'] * flock.dt
                    if dist < 1e-6 or p['tspeed'] <= 0:
                        flock.target = goal
                        flock.target_v = torch.zeros_like(goal)
                        goal = None
                    else:
                        flock.target_v = off / dist * min(p['tspeed'], dist / span)
                for _ in range(p['spf']):
                    flock.step()
                steps += p['spf']
                moved = True

            if moved or target is not None or decoy is not None:
                snap = self._measure(flock)
                if snap is None:                     # non-finite state: keep last good frame
                    if last is None:
                        raise RuntimeError('flock state is non-finite at initialisation')
                    self.set_params(running=False)
                    rev += 1
                    snap = dict(last, diverged=True, rev=rev)
                else:
                    if moved:
                        ht.append(steps * flock.dt)
                        hp.append(snap['age_mean'])
                        hc.append(snap['comps'])
                        hk.append(snap['n_known'] / len(snap['q']))
                        hpz.append(snap['poisoned'].sum() / len(snap['q']))
                    snap.update(gen=gen, steps=steps, rev=rev, t=steps * flock.dt,
                                rate=rate, diverged=False,
                                hist_t=np.array(ht, dtype=float),
                                hist_age=np.array(hp, dtype=float),
                                hist_c=np.array(hc, dtype=float),
                                hist_known=np.array(hk, dtype=float),
                                hist_poisoned=np.array(hpz, dtype=float))
                    last = snap
                with self._lock:
                    self._snap = snap

            # pace the loop, and always yield so the GUI thread gets the GIL
            busy = time.perf_counter() - t0
            time.sleep(max(0.002, 1.0 / self.rate - busy))
            if moved and reset_n is None:
                inst = p['spf'] / (time.perf_counter() - t0)
                rate = inst if rate == 0 else 0.9 * rate + 0.1 * inst

    @staticmethod
    def _mean_age(flock):
        '''
        Mean age (s) of target information over honest agents that know the target.
        relay: now - stamp (negative = holding a forged future stamp)
        global / local: 0 (knowing means seeing it now); consensus: nan (no stamps)
        '''
        mode = flock.target_info
        if mode in ('global', 'local'):
            return 0.0
        if mode != 'relay':
            return float('nan')
        a = flock.info.age().detach().cpu().double().numpy()
        ok = (flock.knows_target() & ~flock.info.liars()).detach().cpu().numpy().astype(bool)
        ok &= np.isfinite(a)
        return float(a[ok].mean()) if ok.any() else float('nan')

    def _measure(self, flock):
        q = flock.q.detach().cpu().double().numpy()
        v = flock.v.detach().cpu().double().numpy()
        if not (np.isfinite(q).all() and np.isfinite(v).all()):
            return None

        # upper triangle of the proximity graph -> one (i, j) per undirected edge
        ei, ej = torch.triu(flock.adjacency(), diagonal=1).nonzero(as_tuple=True)
        ei, ej = ei.cpu().numpy(), ej.cpu().numpy()

        return {
            'q': q, 'v': v, 'ei': ei, 'ej': ej,
            'target': flock.target.detach().cpu().double().numpy(),
            'informed': flock.sees_target().detach().cpu().numpy().astype(bool),
            'known': flock.knows_target().detach().cpu().numpy().astype(bool),
            'mode': flock.target_info,
            'liar': flock.info.liars().detach().cpu().numpy().astype(bool),
            'poisoned': (flock.info.poisoned().detach().cpu().numpy().astype(bool)
                         if flock.target_info == 'relay' else np.zeros(len(q), bool)),
            'decoy': flock.info.decoy.detach().cpu().double().numpy(),
            'err': flock.info.error().detach().cpu().double().numpy(),
            'age': (flock.info.age().detach().cpu().double().numpy()
                    if flock.target_info == 'relay' else None),
            'r_vis': float(R if hasattr(flock, 'boid') else flock.r),
            'k': 0.5 * float(((v - v.mean(0)) ** 2).sum()),
            # polarization: |mean unit heading|, 1 = all aligned, ~0 = disordered/milling
            'pol': float(np.linalg.norm((v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)).mean(0))),
            'comps': n_components(ei, ej, len(q)),
            'n_known': int(flock.knows_target().sum().item()),
            'age_mean': self._mean_age(flock),
        }


class FlockGUI(QtWidgets.QMainWindow):

    def __init__(self, n=100, device='cpu', model=DEFAULT_MODEL, c1=None, c2=None,
                 noise=None, spf=10, fps=60):
        super().__init__()
        d = MODEL_DEFAULTS[model]
        c1 = d['c1'] if c1 is None else c1
        c2 = d['c2'] if c2 is None else c2
        noise = d['noise'] if noise is None else noise
        self.setWindowTitle(f'InfoSwarm: flocking (viewer v{GUI_VERSION})')

        self._seen = None        # (gen, steps, rev) of the snapshot on screen
        self._gen = None         # reset generation on screen
        self._dirty = False      # a display toggle changed; redraw current snapshot
        self._cam = None         # (cx, cy, half) the follow camera last set
        self._play_label = None
        self._draw_ms = 0.0
        self._draw_error = None
        self._armed = 0.0

        # ---------- controls ----------
        self.sp_n = QtWidgets.QSpinBox(); self.sp_n.setRange(2, 5000); self.sp_n.setValue(n)
        self.sp_c1 = QtWidgets.QDoubleSpinBox(); self.sp_c1.setRange(0, 5); self.sp_c1.setDecimals(3)
        self.sp_c1.setSingleStep(0.01); self.sp_c1.setValue(c1)
        self.sp_c2 = QtWidgets.QDoubleSpinBox(); self.sp_c2.setRange(0, 5); self.sp_c2.setDecimals(3)
        self.sp_c2.setSingleStep(0.05); self.sp_c2.setValue(c2)
        self.sp_noise = QtWidgets.QDoubleSpinBox(); self.sp_noise.setRange(0, 200); self.sp_noise.setDecimals(1)
        self.sp_noise.setSingleStep(2.0); self.sp_noise.setValue(noise)
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(list(MODELS))
        self.cmb_model.setCurrentText(model)
        self.sp_spf = QtWidgets.QSpinBox(); self.sp_spf.setRange(1, 500); self.sp_spf.setValue(spf)

        self.b_play = QtWidgets.QPushButton('Pause')
        self.b_step = QtWidgets.QPushButton('Step')
        self.b_reset = QtWidgets.QPushButton('Reset')

        self.cb_edges = QtWidgets.QCheckBox('edges'); self.cb_edges.setChecked(True)
        self.cb_vel = QtWidgets.QCheckBox('velocity tails'); self.cb_vel.setChecked(False)
        self.cb_follow = QtWidgets.QCheckBox('follow flock'); self.cb_follow.setChecked(True)
        self.cmb_info = QtWidgets.QComboBox()
        self.cmb_info.addItems(list(TARGET_INFO_MODES))
        self.cmb_info.setCurrentText(MODEL_DEFAULTS[model]['info'])
        self.sp_tspeed = QtWidgets.QDoubleSpinBox(); self.sp_tspeed.setRange(0, 100); self.sp_tspeed.setDecimals(1)
        self.sp_tspeed.setSingleStep(1.0); self.sp_tspeed.setValue(5.0)

        def dspin(lo, hi, step, dec, val=0.0):
            w = QtWidgets.QDoubleSpinBox(); w.setRange(lo, hi); w.setSingleStep(step)
            w.setDecimals(dec); w.setValue(val)
            return w
        self.sp_obs = dspin(0, 50, 0.5, 2)
        self.sp_hop = dspin(0, 50, 0.25, 2)
        self.sp_drop = dspin(0, 1, 0.05, 2)
        self.sp_liar = dspin(0, 1, 0.01, 2)
        self.sp_forge = dspin(0, 60, 0.5, 2)
        self.cb_reject = QtWidgets.QCheckBox('reject future stamps')
        faults = QtWidgets.QFormLayout()
        faults.addRow('obs noise', self.sp_obs)
        faults.addRow('hop noise', self.sp_hop)
        faults.addRow('drop prob', self.sp_drop)
        faults.addRow('liar frac', self.sp_liar)
        faults.addRow('forge (s)', self.sp_forge)
        faults.addRow(self.cb_reject)
        self.grp_faults = QtWidgets.QGroupBox('relay faults (right-click = decoy)')
        self.grp_faults.setLayout(faults)

        self.status = QtWidgets.QLabel()

        form = QtWidgets.QFormLayout()
        form.addRow('model', self.cmb_model)
        form.addRow('agents', self.sp_n)
        form.addRow('c1 (pos)', self.sp_c1)
        form.addRow('c2 (vel)', self.sp_c2)
        form.addRow('noise', self.sp_noise)
        form.addRow('target info', self.cmb_info)
        form.addRow('target speed', self.sp_tspeed)
        form.addRow('steps/frame', self.sp_spf)
        btns = QtWidgets.QHBoxLayout()
        for b in (self.b_play, self.b_step, self.b_reset):
            btns.addWidget(b)
        panel = QtWidgets.QVBoxLayout()
        panel.addLayout(btns)
        panel.addLayout(form)
        panel.addWidget(self.grp_faults)
        for cb in (self.cb_edges, self.cb_vel, self.cb_follow):
            panel.addWidget(cb)
        panel.addWidget(self.status)
        panel.addStretch()
        side = QtWidgets.QWidget(); side.setLayout(panel); side.setFixedWidth(230)

        # ---------- plots ----------
        # Lines are drawn 1 px wide without antialiasing: that is Qt's fast path.
        pg.setConfigOptions(antialias=False)
        self.view = pg.PlotWidget()
        self.view.setAspectLocked(True)
        self.view.setMenuEnabled(False)          # right-click places the decoy instead
        self.edge_item = pg.PlotCurveItem(pen=pg.mkPen((110, 110, 110), width=1), connect='pairs')
        self.vel_item = pg.PlotCurveItem(pen=pg.mkPen((80, 200, 120), width=1), connect='pairs')
        self.pts = pg.ScatterPlotItem(size=7, pen=None, brush=pg.mkBrush(70, 150, 255))
        self.pts_rel = pg.ScatterPlotItem(size=7, pen=None, brush=pg.mkBrush(90, 200, 110))
        self.pts_inf = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush(255, 150, 40))
        self.pts_poi = pg.ScatterPlotItem(size=7, pen=None, brush=pg.mkBrush(230, 80, 200))
        self.pts_liar = pg.ScatterPlotItem(size=10, symbol='s', pen=pg.mkPen('w', width=1),
                                           brush=pg.mkBrush(150, 40, 200))
        self.dcy = pg.ScatterPlotItem(size=16, symbol='x', pen=pg.mkPen((180, 60, 230), width=2),
                                      brush=pg.mkBrush(180, 60, 230))
        self.tgt = pg.ScatterPlotItem(size=16, symbol='+', pen=pg.mkPen('r', width=2), brush='r')
        self.tgt_ring = pg.PlotCurveItem(pen=pg.mkPen((255, 80, 80, 140), width=1))
        self._ring = np.linspace(0, 2 * np.pi, 65)
        for it in (self.edge_item, self.vel_item, self.tgt_ring, self.pts, self.pts_rel,
                   self.pts_poi, self.pts_inf, self.pts_liar, self.dcy, self.tgt):
            self.view.addItem(it)
        self.view.scene().sigMouseClicked.connect(self.on_click)

        self.plot_k = pg.PlotWidget(title='mean age of target info (s), honest agents')
        self.k_curve = self.plot_k.plot(pen=pg.mkPen((70, 150, 255), width=1), connect='finite')
        self.plot_c = pg.PlotWidget(title='connected components')
        self.plot_c.setLabel('bottom', 'sim time (s)')
        self.c_curve = self.plot_c.plot(pen=pg.mkPen((255, 160, 60), width=1))
        self.plot_known = pg.PlotWidget(title='fraction knowing target (green) / fooled by decoy (pink)')
        self.plot_known.setYRange(0, 1.02, padding=0)
        self.plot_known.enableAutoRange(axis='y', enable=False)
        self.known_curve = self.plot_known.plot(pen=pg.mkPen((90, 200, 110), width=1))
        self.poison_curve = self.plot_known.plot(pen=pg.mkPen((230, 80, 200), width=1))
        right = QtWidgets.QVBoxLayout()
        right.addWidget(self.plot_k); right.addWidget(self.plot_c); right.addWidget(self.plot_known)
        rightw = QtWidgets.QWidget(); rightw.setLayout(right); rightw.setFixedWidth(380)

        root = QtWidgets.QHBoxLayout()
        root.addWidget(side); root.addWidget(self.view, stretch=1); root.addWidget(rightw)
        central = QtWidgets.QWidget(); central.setLayout(root)
        self.setCentralWidget(central)
        self.resize(1400, 800)

        # ---------- simulation thread ----------
        self.sim = Sim(n, device, model, c1, c2, noise, spf, rate=fps)
        self.sim.start()

        # ---------- wiring: every handler only posts a request or sets a flag ----------
        self.b_play.clicked.connect(self.toggle)
        self.b_step.clicked.connect(self.step)
        self.b_reset.clicked.connect(self.reset)
        self.cmb_model.currentTextChanged.connect(self.model_changed)
        self._sync_c2_enabled()
        for sp in (self.sp_c1, self.sp_c2, self.sp_noise, self.sp_spf):
            sp.valueChanged.connect(self.push_params)
        for cb in (self.cb_edges, self.cb_vel):
            cb.toggled.connect(self.mark_dirty)
        self.cb_follow.toggled.connect(self.follow_toggled)
        self.cmb_info.currentTextChanged.connect(self.push_params)
        self.sp_tspeed.valueChanged.connect(self.push_params)
        for sp in (self.sp_obs, self.sp_hop, self.sp_drop, self.sp_liar, self.sp_forge):
            sp.valueChanged.connect(self.push_params)
        self.cb_reject.toggled.connect(self.push_params)
        self.cmb_info.currentTextChanged.connect(self._sync_faults_enabled)
        self._sync_faults_enabled()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(max(1, int(1000 / fps)))

    # ---------- handlers (cheap; no simulation, no drawing) ----------
    def toggle(self, *_):
        self.sim.set_params(running=not self.sim.running)

    def step(self, *_):
        self.sim.request_step()

    def reset(self, *_):
        self.sim.request_reset(self.sp_n.value(), self.cmb_model.currentText())

    def model_changed(self, name):
        d = MODEL_DEFAULTS[name]
        self.sp_c1.setValue(d['c1']); self.sp_c2.setValue(d['c2']); self.sp_noise.setValue(d['noise'])
        self.cmb_info.setCurrentText(d['info'])
        self._sync_c2_enabled()
        self.push_params()
        self.reset()

    def _sync_c2_enabled(self):
        # Reynolds boids hold speed in [vmin, vmax]; c2 has no effect there
        self.sp_c2.setEnabled(self.cmb_model.currentText() != 'Reynolds')

    def _sync_faults_enabled(self, *_):
        self.grp_faults.setEnabled(self.cmb_info.currentText() == 'relay')

    def push_params(self, *_):
        self.sim.set_params(c1=float(self.sp_c1.value()), c2=float(self.sp_c2.value()),
                            noise=float(self.sp_noise.value()), spf=int(self.sp_spf.value()),
                            info=self.cmb_info.currentText(),
                            tspeed=float(self.sp_tspeed.value()),
                            obs_noise=float(self.sp_obs.value()), hop_noise=float(self.sp_hop.value()),
                            drop=float(self.sp_drop.value()), liar_frac=float(self.sp_liar.value()),
                            forge=float(self.sp_forge.value()),
                            reject_future=bool(self.cb_reject.isChecked()))
        self._dirty = True

    def mark_dirty(self, *_):
        self._dirty = True

    def follow_toggled(self, *_):
        self._cam = None          # re-fit from scratch next time follow is on
        self._dirty = True

    def on_click(self, ev):
        left = ev.button() == QtCore.Qt.MouseButton.LeftButton
        right = ev.button() == QtCore.Qt.MouseButton.RightButton
        if not (left or right):
            return
        vb = self.view.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = vb.mapSceneToView(ev.scenePos())
        if left:
            self.sim.set_target(p.x(), p.y())
        else:
            self.sim.set_decoy(p.x(), p.y())

    # ---------- the only place anything is drawn ----------
    def tick(self):
        now = time.perf_counter()
        if now - self._armed > 1.0:
            # Re-armed once a second; it only fires if this thread stops getting here.
            faulthandler.dump_traceback_later(WATCHDOG_S, repeat=False)
            self._armed = now

        if self.sim.error:
            self.timer.stop()
            faulthandler.cancel_dump_traceback_later()
            self.status.setText('Simulation thread stopped with an error.\n'
                                'The traceback is in the console.')
            return

        label = 'Pause' if self.sim.running else 'Play'
        if label != self._play_label:
            self.b_play.setText(label)
            self._play_label = label

        snap = self.sim.snapshot()
        if snap is None:
            return
        key = (snap['gen'], snap['steps'], snap['rev'])
        if key == self._seen and not self._dirty:
            return

        t0 = time.perf_counter()
        try:
            self.draw(snap, refit=(snap['gen'] != self._gen))
        except Exception as e:
            # Show it in the window and print the traceback once, not once per frame.
            msg = f'{type(e).__name__}: {e}'
            if msg != self._draw_error:
                self._draw_error = msg
                traceback.print_exc()
            self.status.setText('Draw error (traceback in console):\n' + msg[:200])
        self._seen, self._gen, self._dirty = key, snap['gen'], False
        ms = (time.perf_counter() - t0) * 1000
        self._draw_ms = ms if self._draw_ms == 0 else 0.9 * self._draw_ms + 0.1 * ms

    def draw(self, s, refit=False):
        q, v = s['q'], s['v']

        # colour by what each agent knows about the target (not shown in 'global' mode)
        none = np.zeros(len(q), bool)
        partial = s['mode'] != 'global'
        liar = s['liar'] if partial else none
        poi = s['poisoned'] & ~liar if partial else none
        inf = s['informed'] & ~liar if partial else none
        rel = (s['known'] & ~s['informed'] & ~liar & ~poi) if partial else none
        unk = ~(inf | rel | poi | liar)
        for item, m in ((self.pts, unk), (self.pts_rel, rel), (self.pts_poi, poi),
                        (self.pts_inf, inf), (self.pts_liar, liar)):
            item.setData(pos=q[m] if m.any() else np.zeros((0, 2)))
        show_decoy = bool(s['mode'] == 'relay' and liar.any())   # PyQt6 wants a Python bool
        self.dcy.setVisible(show_decoy)
        if show_decoy:
            self.dcy.setData(pos=s['decoy'][None, :2])

        if self.cb_edges.isChecked() and len(s['ei']):
            self.edge_item.setData(*segments(q[s['ei']], q[s['ej']]))
        else:
            self.edge_item.setData(EMPTY, EMPTY)

        if self.cb_vel.isChecked():
            self.vel_item.setData(*segments(q, q - 0.5 * v))    # tail trails behind: tadpoles
        else:
            self.vel_item.setData(EMPTY, EMPTY)

        show_target = bool(self.sp_c1.value() > 0)
        self.tgt.setVisible(show_target)
        ring = bool(show_target and s['mode'] != 'global')
        self.tgt_ring.setVisible(ring)
        if show_target:
            self.tgt.setData(pos=s['target'][None, :2])
        if ring:
            tx, ty = s['target'][:2]
            self.tgt_ring.setData(tx + s['r_vis'] * np.cos(self._ring), ty + s['r_vis'] * np.sin(self._ring))

        if refit or self.cb_follow.isChecked():
            self.fit(q, force=refit)

        self.k_curve.setData(s['hist_t'], s['hist_age'])
        self.c_curve.setData(s['hist_t'], s['hist_c'])
        self.known_curve.setData(s['hist_t'], s['hist_known'])
        self.poison_curve.setData(s['hist_t'], s['hist_poisoned'])

        lines = [f'N = {len(q)}',
                 f"t = {s['t']:0.2f} s",
                 f"edges = {len(s['ei'])}",
                 f"components = {s['comps']}",
                 f"polarization = {s['pol']:0.2f}",
                 f"target info: {s['mode']}",
                 f"  see {int(s['informed'].sum())} / know {s['n_known']}",
                 f"  liars {int(s['liar'].sum())} / fooled {int(s['poisoned'].sum())}",
                 f"K = {s['k']:0.3g}",
                 f"sim: {s['rate']:0.0f} steps/s",
                 f'draw: {self._draw_ms:0.1f} ms']
        if s['mode'] != 'global' and s['n_known']:
            e = s['err'][np.isfinite(s['err'])]
            if len(e):
                lines.insert(8, f"  est. error mean {e.mean():0.1f} max {e.max():0.1f}")
            if s['age'] is not None:
                a = s['age'][np.isfinite(s['age'])]
                if len(a):
                    lines.insert(9, f"  info age mean {s['age_mean']:0.2f} max {a.max():0.2f} s")
        if s['diverged']:
            lines += ['DIVERGED (non-finite state)', 'Paused: lower c1/c2, Reset.']
        self.status.setText('\n'.join(lines))

    def fit(self, q, force=False):
        '''
        Follow camera. The view range is only touched when the flock has
        drifted or outgrown / shrunk well inside the current view, rather
        than on every frame.
        '''
        c = q.mean(0)
        cx, cy = float(c[0]), float(c[1])
        need = max(float(np.abs(q - c).max()) * 1.15 + D, 3.0 * D)
        half = need * 1.1
        if not force and self._cam is not None:
            ox, oy, oh = self._cam
            if 0.7 * oh <= need <= oh:                       # zoom still fits: keep it
                half = oh
                if max(abs(cx - ox), abs(cy - oy)) < 0.02 * oh:
                    return                                   # and it is still centred
        self._cam = (cx, cy, half)
        self.view.setRange(xRange=(cx - half, cx + half),
                           yRange=(cy - half, cy + half), padding=0)

    def closeEvent(self, ev):
        self.timer.stop()
        faulthandler.cancel_dump_traceback_later()
        self.sim.stop()
        super().closeEvent(ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--model', choices=list(MODELS), default=DEFAULT_MODEL,
                    help='Reynolds = classic boids (chaotic), Olfati-Saber = alpha-lattice')
    ap.add_argument('--c1', type=float, default=None, help='pull toward target (default depends on model)')
    ap.add_argument('--c2', type=float, default=None, help='lattice only: gamma-agent velocity gain')
    ap.add_argument('--noise', type=float, default=None, help='random steering acceleration (std)')
    ap.add_argument('--spf', type=int, default=10, help='sim steps per frame')
    ap.add_argument('--fps', type=int, default=60)
    args = ap.parse_args()

    # Diagnostics: report native crashes with a Python stack.
    try:
        faulthandler.enable()
    except Exception:
        pass

    print(f'[gui v{GUI_VERSION}] pyqtgraph {pg.__version__} | {pg.Qt.QT_LIB} {pg.Qt.QtVersion} | '
          f'numpy {np.__version__} | torch {torch.__version__} | device {args.device}\n'
          f'[gui v{GUI_VERSION}] watchdog on: if the window stalls for {WATCHDOG_S}s, '
          f'a "Timeout" stack trace is printed here.', flush=True)

    app = pg.mkQApp('InfoSwarm')
    win = FlockGUI(args.n, args.device, args.model, args.c1, args.c2, args.noise, args.spf, args.fps)
    win.show()
    code = app.exec()

    faulthandler.cancel_dump_traceback_later()
    win.sim.stop()
    sys.exit(code)


if __name__ == '__main__':
    main()