"""Adaptive stochastic memristive LIF reservoir (AMSR) - reference implementation.

Every quantity used in the manuscript is defined here. Units: time in ms,
voltage in V, conductance in S, resistance in Ohm. The normalised device
state g is dimensionless in [0, 1].

Model (per reservoir neuron i, Euler / Euler-Maruyama with step dt):
  device  : g_i <- g_i + dt/tau_g (g_eq - g_i) + beta V_in dt + sigma sqrt(dt) z
            G_i  = G_min + g_i (G_max - G_min)
  input   : I_in,i = w_in,i G_i V_in                         (A)
  synapse : r_j  <- r_j (1 - dt/tau_s) + s_j                 (dimensionless trace)
  membrane: V_i  <- V_i + dt/tau_m ( -V_i + R I_in,i + J sum_j W_ij r_j )
  spike   : s_i = [V_i >= theta_i] and not refractory; V_i <- V_reset
  thresh. : theta_i <- theta_i + dt/tau_th (theta_0 - theta_i) + alpha s_i   (adaptive only)
"""
from dataclasses import dataclass, field, asdict, replace
import numpy as np


@dataclass
class DeviceParams:
    G_min: float = 1e-6        # S
    G_max: float = 100e-6      # S
    tau_g: float = 40.0        # ms, conductance relaxation time constant
    g_eq: float = 0.2          # normalised equilibrium conductance
    g0: float = 0.2            # normalised initial conductance (= g_eq)
    beta: float = 0.05         # V^-1 ms^-1, programming coefficient (dg per pulse = beta*V*W)
    sigma: float = 0.02        # ms^-1/2, diffusion coefficient (0 = ideal deterministic device)


@dataclass
class NeuronParams:
    tau_m: float = 10.0        # ms
    R: float = 100e3           # Ohm  (R*G_max*V_sim = 15 V)
    V_reset: float = 0.0       # V
    V0: float = 0.0            # V, initial membrane potential
    theta0: float = 0.5        # V, baseline threshold
    t_ref: float = 2.0         # ms, absolute refractory period
    adaptive: bool = False
    alpha: float = 0.05        # V, spike-triggered threshold increment
    tau_th: float = 50.0       # ms, threshold relaxation time constant


@dataclass
class ReservoirParams:
    N: int = 100
    density: float = 0.10
    rho: float = 0.90          # spectral radius of W before gain J
    J: float = 0.5             # V, recurrent coupling gain
    tau_s: float = 5.0         # ms, synaptic trace time constant
    w_in_low: float = 0.5      # |w_in| ~ U(w_in_low, 1) * kappa
    kappa: float = 1.0         # input gain
    frac_inhib_in: float = 0.2 # fraction of neurons with negative input weight
    dt: float = 0.1            # ms
    memristive: bool = True    # False -> conventional LIF reservoir (I_in = w_in G_ref V_in)


@dataclass
class Config:
    name: str
    device: DeviceParams = field(default_factory=DeviceParams)
    neuron: NeuronParams = field(default_factory=NeuronParams)
    res: ReservoirParams = field(default_factory=ReservoirParams)

    def to_dict(self):
        return asdict(self)


def make_weights(res: ReservoirParams, rng):
    N = res.N
    mask = rng.random((N, N)) < res.density
    np.fill_diagonal(mask, False)
    W = rng.standard_normal((N, N)) * mask
    ev = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (res.rho / ev)
    mag = rng.uniform(res.w_in_low, 1.0, N)
    sign = np.where(rng.random(N) < res.frac_inhib_in, -1.0, 1.0)
    return W, mag * sign


def simulate(cfg: Config, V_in, W, w_in, rng, n_bins=10, record=None, return_traces=False):
    """Run the reservoir on a batch of input voltage sequences.

    V_in : (B, K) input voltage at each simulation step (V).
    Returns dict with 'counts' (B, N, n_bins) spike counts and diagnostics.
    record: optional (b_idx, list of neuron idx) to store full traces.
    """
    d, n, r = cfg.device, cfg.neuron, cfg.res
    B, K = V_in.shape
    N = r.N
    dt = r.dt
    g = np.full((B, N), d.g0)
    V = np.full((B, N), n.V0)
    th = np.full((B, N), n.theta0)
    tr = np.zeros((B, N))
    ref = np.zeros((B, N), dtype=np.int32)
    ref_steps = int(round(n.t_ref / dt))
    counts = np.zeros((B, N, n_bins), dtype=np.int32)
    bin_of = np.minimum((np.arange(K) * n_bins) // K, n_bins - 1)
    kin = r.kappa * w_in[None, :]
    WT = (r.J * W).T
    dG = d.G_max - d.G_min
    G_ref = d.G_min + d.g_eq * dG       # fixed conductance for conventional reservoir
    sq = d.sigma * np.sqrt(dt)
    decay_s = 1.0 - dt / r.tau_s
    e_dev = np.zeros(B)                 # device read energy, J (sum G V^2 dt)
    traces = None
    if return_traces:
        traces = {k: np.zeros((K, N)) for k in ("g", "V", "th", "s")}
    for k in range(K):
        v = V_in[:, k][:, None]
        if r.memristive:
            g += (dt / d.tau_g) * (d.g_eq - g) + d.beta * v * dt
            if sq > 0:
                g += sq * rng.standard_normal((B, N))
            np.clip(g, 0.0, 1.0, out=g)
            G = d.G_min + g * dG
        else:
            G = G_ref
        I_in = kin * G * v
        e_dev += (np.abs(G * v * v) * dt * 1e-3).sum(axis=1) if r.memristive else 0.0
        V += (dt / n.tau_m) * (-V + n.R * I_in + tr @ WT)
        refr = ref > 0
        V[refr] = n.V_reset
        ref[refr] -= 1
        s = V >= th
        V[s] = n.V_reset
        ref[s] = ref_steps
        tr = tr * decay_s + s
        if n.adaptive:
            th += (dt / n.tau_th) * (n.theta0 - th) + n.alpha * s
        counts[:, :, bin_of[k]] += s
        if return_traces:
            traces["g"][k] = g[0]; traces["V"][k] = V[0]; traces["th"][k] = th[0]; traces["s"][k] = s[0]
    out = {"counts": counts, "e_dev": e_dev, "spikes": counts.sum(axis=(1, 2))}
    if return_traces:
        out["traces"] = traces
    return out


def simulate_states(cfg: Config, V_in, W, w_in, rng, sample_every=10):
    """Same dynamics, returns synaptic traces sampled every `sample_every` steps
    (used for the memory-capacity task). V_in: (1, K)."""
    d, n, r = cfg.device, cfg.neuron, cfg.res
    B, K = V_in.shape
    N, dt = r.N, r.dt
    g = np.full((B, N), d.g0); V = np.full((B, N), n.V0); th = np.full((B, N), n.theta0)
    tr = np.zeros((B, N)); ref = np.zeros((B, N), dtype=np.int32)
    ref_steps = int(round(n.t_ref / dt))
    kin = r.kappa * w_in[None, :]; WT = (r.J * W).T; dG = d.G_max - d.G_min
    G_ref = d.G_min + d.g_eq * dG; sq = d.sigma * np.sqrt(dt); decay_s = 1.0 - dt / r.tau_s
    out = np.zeros((K // sample_every, B, N))
    for k in range(K):
        v = V_in[:, k][:, None]
        if r.memristive:
            g += (dt / d.tau_g) * (d.g_eq - g) + d.beta * v * dt
            if sq > 0:
                g += sq * rng.standard_normal((B, N))
            np.clip(g, 0.0, 1.0, out=g)
            G = d.G_min + g * dG
        else:
            G = G_ref
        V += (dt / n.tau_m) * (-V + n.R * kin * G * v + tr @ WT)
        refr = ref > 0; V[refr] = n.V_reset; ref[refr] -= 1
        s = V >= th; V[s] = n.V_reset; ref[s] = ref_steps
        tr = tr * decay_s + s
        if n.adaptive:
            th += (dt / n.tau_th) * (n.theta0 - th) + n.alpha * s
        if (k + 1) % sample_every == 0:
            out[(k + 1) // sample_every - 1] = tr
    return out


# ---------------------------------------------------------------- ESN baseline
@dataclass
class ESNParams:
    N: int = 100
    density: float = 0.10
    rho: float = 0.90
    leak: float = 0.3
    kappa: float = 1.0
    bias: float = 0.1


def esn_run(p: ESNParams, U, rng_w, n_bins=10, return_states=False):
    """Leaky-integrator ESN on a 1 ms grid. U: (B, T) inputs in [0, 1]."""
    N = p.N
    mask = rng_w.random((N, N)) < p.density
    np.fill_diagonal(mask, False)
    W = rng_w.standard_normal((N, N)) * mask
    W *= p.rho / np.max(np.abs(np.linalg.eigvals(W)))
    w_in = rng_w.uniform(-1, 1, N)
    b = rng_w.uniform(-1, 1, N) * p.bias
    B, T = U.shape
    x = np.zeros((B, N))
    feats = np.zeros((B, N, n_bins))
    cnt = np.zeros(n_bins)
    bin_of = np.minimum((np.arange(T) * n_bins) // T, n_bins - 1)
    states = np.zeros((T, B, N)) if return_states else None
    for t in range(T):
        x = (1 - p.leak) * x + p.leak * np.tanh(x @ W.T + p.kappa * U[:, t][:, None] * w_in + b)
        feats[:, :, bin_of[t]] += x
        cnt[bin_of[t]] += 1
        if return_states:
            states[t] = x
    feats /= cnt[None, None, :]
    return (feats, states) if return_states else feats
