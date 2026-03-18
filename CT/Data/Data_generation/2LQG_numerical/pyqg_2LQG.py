import os
import yaml
import numpy as np
import torch
import pyqg


# ----------------------
# Utilities
# ----------------------
def coerce_number(x):
    """Convert numeric strings to int/float; leave other types untouched."""
    if isinstance(x, (int, float, np.integer, np.floating)):
        return x
    if isinstance(x, str):
        s = x.strip()
        # Try int first (handles "256")
        try:
            return int(s)
        except ValueError:
            pass
        # Try float (handles "1.0e6", "1e-11", etc.)
        try:
            return float(s)
        except ValueError:
            return x
    return x
    
def block_average_2d(field2d: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return field2d
    ny, nx = field2d.shape
    if (ny % factor) != 0 or (nx % factor) != 0:
        raise ValueError(f"Downsample factor {factor} must divide (ny, nx)=({ny}, {nx}).")
    return field2d.reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))


def block_average_q(q: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return q
    out = np.empty((q.shape[0], q.shape[1] // factor, q.shape[2] // factor), dtype=q.dtype)
    for k in range(q.shape[0]):
        out[k] = block_average_2d(q[k], factor)
    return out


def eddy_pv(q: np.ndarray) -> np.ndarray:
    return q - q.mean(axis=-1, keepdims=True)


def ensure_multiple(name: str, big: float, small: float, tol: float = 1e-9) -> int:
    r = big / small
    if abs(r - round(r)) > tol:
        raise ValueError(
            f"{name} must be an integer multiple of dt. Got {name}={big}, dt={small}, ratio={r}."
        )
    return int(round(r))


def get_stepper(m):
    if hasattr(m, "step_forward") and callable(m.step_forward):
        return m.step_forward
    if hasattr(m, "_step_forward") and callable(m._step_forward):
        return m._step_forward
    raise AttributeError("Could not find step method.")


def seed_baroclinic_perturbation(m, seed: int, amp: float):
    if amp == 0.0:
        return
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(m.q.shape)
    noise -= noise.mean(axis=(-2, -1), keepdims=True)
    m.q[0] += amp * noise[0]
    m.q[1] -= amp * noise[1]


def compute_tmax(tavestart, n_samples, rollout_len, Dt, decor_t):
    return tavestart + n_samples * (rollout_len * Dt + decor_t)


# ----------------------
# Main
# ----------------------
def main(yml_path: str, i: int, n_samples_override: int | None):
    with open(yml_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    model_cfg = {k: coerce_number(v) for k, v in model_cfg.items()}
    tavestart = float(cfg["run"]["tavestart"])

    data_cfg = cfg["dataset"]
    n_samples = int(data_cfg["n_samples_from_one_traj"])
    if n_samples_override is not None:
        n_samples = int(n_samples_override)

    ds_factor = int(data_cfg["spatial_downsample_factor"])
    Dt = float(data_cfg["Dt"])
    rollout_len = int(data_cfg["rollout_length"])
    decor_t = float(data_cfg["decorrelation_time"])

    amp = float(cfg["perturbation"].get("amplitude", 0.0))

    base_outpath = cfg["output"]["outpath"]
    out_dir = os.path.dirname(base_outpath)
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, f"qg_jet_long_{i}.pt")

    # outpath = cfg["output"]["outpath"]
    dtype = cfg["output"].get("dtype", "float32")
    torch_dtype = torch.float32 if dtype == "float32" else torch.float64
    np_dtype = np.float32 if dtype == "float32" else np.float64

    # Seed = shard index
    seed = int(i)

    # Infer grid and strides
    m0 = pyqg.QGModel(**model_cfg)
    phys_stride = ensure_multiple("Dt", Dt, m0.dt)
    decor_stride = ensure_multiple("decorrelation_time", decor_t, m0.dt)

    ny, nx = m0.q.shape[-2], m0.q.shape[-1]
    ny_ds, nx_ds = ny // ds_factor, nx // ds_factor

    tmax_auto = compute_tmax(tavestart, n_samples, rollout_len, Dt, decor_t)

    data = torch.empty(
        (n_samples, rollout_len, 2, ny_ds, nx_ds),
        dtype=torch_dtype,
        device="cpu"
    )

    print(f"INFO: Logger initialized | shard {i}")
    print(f"Saving to {outpath}")
    print(f"Seed = {seed}")
    print(f"tmax = {tmax_auto:.3e}")

    m = pyqg.QGModel(**model_cfg)
    m.tmax = tmax_auto
    step_forward = get_stepper(m)

    seed_baroclinic_perturbation(m, seed=seed, amp=amp)

    while m.t < tavestart:
        step_forward()

    for s in range(n_samples):
        for k in range(rollout_len):
            if k > 0:
                for _ in range(phys_stride):
                    step_forward()
            q = eddy_pv(m.q.copy())
            q = block_average_q(q, ds_factor)
            data[s, k] = torch.from_numpy(q.astype(np_dtype, copy=False))

        for _ in range(decor_stride):
            step_forward()

        if (s + 1) % max(1, n_samples // 10) == 0:
            print(f"  {s+1}/{n_samples} samples")

    data_config = {
        "shard_index": i,
        "seed": seed,
        "shape": list(data.shape),
        "dtype": str(data.dtype),

        "model": model_cfg,
        "run": {
            "tavestart": tavestart,
            "tmax": tmax_auto,
        },
        "dataset": {
            "n_samples_from_one_traj": n_samples,
            "rollout_length": rollout_len,
            "Dt": Dt,
            "decorrelation_time": decor_t,
            "spatial_downsample_factor": ds_factor,
        },
        "grid": {
            "nx": nx,
            "ny": ny,
            "nx_ds": nx_ds,
            "ny_ds": ny_ds,
        },
        "notes": {
            "layout": "[samples, time, 2, ny_ds, nx_ds]",
            "pv": "eddy PV = q - zonal mean in x",
            "downsampling": "block average",
        },
    }

    torch.save(
        {
            "data": data,
            "data_config": data_config,
        },
        outpath,
    )

    print(f"Saved {outpath}")
    print("Final tensor shape:", tuple(data.shape))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--i", type=int, required=True)
    p.add_argument("--n_samples", type=int, default=None)
    args = p.parse_args()
    main(args.config, args.i, args.n_samples)
