"""Model registry, feature extraction and hyper-parameter search space.

Each model receives the same number of random-search trials on the same
validation partition (matched tuning budget)."""
import copy
import numpy as np
from .model import Config, DeviceParams, NeuronParams, ReservoirParams, ESNParams, make_weights, simulate, esn_run
from .tasks import voltage_to_ms_grid

SIGMA_DEFAULT = 0.02

MODELS = {
    # name: (kind, memristive, sigma, adaptive)
    "LIF":      ("snn", True, 0.0, False),            # ideal (noise-free) memristive LIF
    "A-LIF":    ("snn", True, 0.0, True),             # ideal, adaptive threshold
    "S-LIF":    ("snn", True, SIGMA_DEFAULT, False),  # realistic device noise
    "AS-LIF":   ("snn", True, SIGMA_DEFAULT, True),   # proposed
    "LSM":      ("snn", False, 0.0, False),           # conventional LIF reservoir, no memristive input
    "ESN":      ("esn", None, None, None),
    "Linear":   ("lin", None, None, None),            # ridge on the raw (1 ms) input
}
LABEL = {"LIF": "Deterministic LIF", "A-LIF": "Adaptive deterministic LIF", "S-LIF": "Stochastic LIF",
         "AS-LIF": "Adaptive stochastic LIF (studied)", "LSM": "Conventional LIF reservoir (LSM)",
         "ESN": "Echo state network", "Linear": "Linear readout on raw input"}


def build_config(name, hp, sigma=None):
    kind, mem, sig, ad = MODELS[name]
    cfg = Config(name=name, device=DeviceParams(sigma=sig if sigma is None else sigma),
                 neuron=NeuronParams(adaptive=ad), res=ReservoirParams(memristive=mem))
    cfg.res.kappa = hp["kappa"]; cfg.res.J = hp["J"]
    if ad:
        cfg.neuron.alpha = hp["alpha"]; cfg.neuron.tau_th = hp["tau_th"]
    return cfg


def sample_hp(name, rng):
    kind = MODELS[name][0]
    if kind == "snn":
        hp = dict(kappa=float(np.exp(rng.uniform(np.log(0.3), np.log(6.0)))), J=float(rng.uniform(0.0, 2.0)))
        if MODELS[name][3]:
            hp["alpha"] = float(np.exp(rng.uniform(np.log(0.01), np.log(0.5))))
            hp["tau_th"] = float(np.exp(rng.uniform(np.log(10.0), np.log(200.0))))
        return hp
    if kind == "esn":
        return dict(rho=float(rng.uniform(0.5, 1.2)), leak=float(np.exp(rng.uniform(np.log(0.05), np.log(1.0)))),
                    kappa=float(np.exp(rng.uniform(np.log(0.1), np.log(5.0)))), bias=float(rng.uniform(0.0, 1.0)))
    return {}


def features(name, hp, V_in, seed, gain=1.0, sigma=None, n_bins=10, return_aux=False):
    """V_in: (B, K) input voltage on the 0.1 ms grid. `gain` rescales the input at
    inference time (distribution-shift test). Reservoir weights depend only on `seed`."""
    kind = MODELS[name][0]
    rng_w = np.random.default_rng(10_000 + seed)
    if kind == "snn":
        cfg = build_config(name, hp, sigma)
        W, w_in = make_weights(cfg.res, rng_w)
        out = simulate(cfg, V_in * gain, W, w_in, np.random.default_rng(20_000 + seed), n_bins=n_bins)
        X = out["counts"].reshape(len(V_in), -1).astype(float)
        aux = dict(spikes=out["spikes"], e_dev=out["e_dev"])
    elif kind == "esn":
        U = voltage_to_ms_grid(V_in) * gain
        X = esn_run(ESNParams(**hp), U, rng_w, n_bins=n_bins).reshape(len(V_in), -1)
        aux = {}
    else:
        X = voltage_to_ms_grid(V_in) * gain
        aux = {}
    return (X, aux) if return_aux else X
