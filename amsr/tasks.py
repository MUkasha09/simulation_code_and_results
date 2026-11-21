"""Benchmarks: synthetic Task A / Task B and the ECG5000 (UCR) real-world task.

All generators are seeded; nothing is hand-tuned per model.
"""
import os
import numpy as np

DT = 0.1          # ms
T_SEQ = 100.0     # ms (synthetic tasks)
V_SIM = 1.5       # V
W_SIM = 1.0       # ms


# --------------------------------------------------------------- event times
def _periodic(rng, isi, jit, T=T_SEQ):
    t0 = rng.uniform(0, isi)
    t = np.arange(t0, T, isi)
    return t + rng.normal(0, jit, t.size)


def _bursting(rng, jit, T=T_SEQ, intra=2.0, n_in=4, silent=15.0):
    t, out = rng.uniform(0, silent), []
    while t < T:
        for j in range(n_in):
            out.append(t + j * intra)
        t += (n_in - 1) * intra + silent
    out = np.array(out)
    return out + rng.normal(0, jit, out.size)


def _random_sparse(rng, jit, T=T_SEQ, lo=5.0, hi=20.0):
    t, out = rng.uniform(0, lo), []
    while t < T:
        out.append(t)
        t += rng.uniform(lo, hi)
    return np.array(out)


def _increasing(rng, jit, T=T_SEQ, i0=20.0, i1=5.0):
    t, out = rng.uniform(0, 3), []
    while t < T:
        out.append(t)
        t += i0 + (i1 - i0) * (t / T)
    out = np.array(out)
    return out + rng.normal(0, jit, out.size)


def _alternating(rng, jit, T=T_SEQ):
    a = _bursting(rng, jit, T=T / 2)
    b = _periodic(rng, 10.0, jit, T=T / 2) + T / 2
    return np.concatenate([a, b]) if rng.random() < 0.5 else np.concatenate([_periodic(rng, 10.0, jit, T / 2), _bursting(rng, jit, T / 2) + T / 2])


def _multifreq(rng, jit, T=T_SEQ):
    seg = T / 3
    isis = [6.0, 12.0, 18.0]
    rng.shuffle(isis)
    return np.concatenate([_periodic(rng, isis[i], jit, seg) + i * seg for i in range(3)])


def _perturb(rng, t, p_del, p_ins, T=T_SEQ):
    t = t[(t >= 0) & (t < T)]
    keep = rng.random(t.size) >= p_del
    t = t[keep]
    n_ins = rng.binomial(max(len(t), 1), p_ins)
    return np.sort(np.concatenate([t, rng.uniform(0, T, n_ins)]))


def events_to_voltage(ev_list, T=T_SEQ, dt=DT, amp=V_SIM, width=W_SIM):
    K = int(round(T / dt))
    V = np.zeros((len(ev_list), K))
    w = int(round(width / dt))
    for i, ev in enumerate(ev_list):
        for t in ev:
            k = int(t / dt)
            if 0 <= k < K:
                V[i, k:k + w] = amp
    return V


def make_task(name, seed, n_total=400):
    """Task A: 4 classes, small jitter (0.5 ms). Task B: 6 classes, 1.5 ms jitter,
    5% spike deletion and 5% spurious insertion, +-10% rate variability."""
    rng = np.random.default_rng(seed)
    if name == "A":
        jit, pdel, pins, rv = 0.5, 0.0, 0.0, 0.0
        gens = [lambda r, j: _periodic(r, 10.0, j), _bursting, _random_sparse, _increasing]
    elif name == "B":
        jit, pdel, pins, rv = 1.5, 0.05, 0.05, 0.10
        gens = [lambda r, j: _periodic(r, 10.0 * (1 + r.uniform(-rv, rv)), j), _bursting, _random_sparse,
                _increasing, _alternating, _multifreq]
    else:
        raise ValueError(name)
    C = len(gens)
    counts = [n_total // C + (1 if c < n_total % C else 0) for c in range(C)]
    ev, y = [], []
    for c, gfun in enumerate(gens):
        for _ in range(counts[c]):
            ev.append(_perturb(rng, gfun(rng, jit), pdel, pins))
            y.append(c)
    y = np.array(y)
    perm = rng.permutation(len(y))
    ev = [ev[i] for i in perm]
    y = y[perm]
    return ev, y, C


def stratified_split(y, rng, fr=(0.70, 0.15, 0.15)):
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n = len(idx)
        a = int(round(fr[0] * n)); b = int(round((fr[0] + fr[1]) * n))
        tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
    return np.array(tr), np.array(va), np.array(te)


# ------------------------------------------------------------------ ECG5000
ECG_DIR = os.environ.get("ECG5000_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def load_ecg5000():
    """UCR ECG5000, official split: 500 train / 4500 test, 140 samples, 5 classes.
    (The mirror used stores the files with swapped names; we assign by size.)"""
    a = np.loadtxt(os.path.join(ECG_DIR, "ECG5000_TRAIN.txt"))
    b = np.loadtxt(os.path.join(ECG_DIR, "ECG5000_TEST.txt"))
    tr, te = (a, b) if len(a) < len(b) else (b, a)
    Xtr, ytr = tr[:, 1:], tr[:, 0].astype(int) - 1
    Xte, yte = te[:, 1:], te[:, 0].astype(int) - 1
    lo, hi = Xtr.min(), Xtr.max()                      # global min-max from training data only
    f = lambda X: np.clip((X - lo) / (hi - lo), 0, 1)
    return f(Xtr), ytr, f(Xte), yte


def analog_to_voltage(X, dt=DT, ms_per_sample=1.0, amp=V_SIM):
    """Zero-order hold: each sample is held for 1 ms; V_in = V_sim * x."""
    rep = int(round(ms_per_sample / dt))
    return np.repeat(X, rep, axis=1) * amp


def voltage_to_ms_grid(V, dt=DT, amp=V_SIM):
    """For the ESN baseline: average the input over each 1 ms, normalised to [0, 1]."""
    rep = int(round(1.0 / dt))
    B, K = V.shape
    return V[:, : (K // rep) * rep].reshape(B, K // rep, rep).mean(axis=2) / amp
