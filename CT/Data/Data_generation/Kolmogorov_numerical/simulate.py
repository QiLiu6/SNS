import jax
import xarray
import jax.numpy as jnp
import numpy as np

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
from jax_cfd.base import resize
from jax_cfd.spectral import utils as spectral_utils

def run_kolmogorov_sim(nsteps, dt, Dt, spinup = 5000, decorr_steps = 1995, viscosity = 1e-4, gridsize = 512, downsample = 8, n_traj = 20, chunk_size = 1000):
    """ 
    Jump_size = nsteps + decorr_steps
    Batch_size_per_sim = n_traj * (traj_T - spinup)/Jump_size
    L = gridsize / downsample
        Output:
            
            Sim_stack (Batch_size_per_sim  x  nsteps  x  L  x  L   tensor): training data

        Inputs:
            nsteps:     number of steps in a rollout
            dt:         numerical timestep
            Dt:         physical timestep (must be >numerical timestep)
            spinup:     number of numerical timesteps to drop from output
            viscosity:  viscosity for NS PDE
            chunk_size: number of steps to process at once (reduce if OOM)
    
    """
    ratio = int(Dt/dt)
    
    ## These cuts split the simulation into short training trajectories
    cuts=[]
    for i in range(n_traj):
        for j in range(nsteps):
            cuts.append(spinup + i * (decorr_steps + nsteps) + j * (ratio))

    traj_T = cuts[-1]
    max_velocity = 7
    grid = grids.Grid((gridsize, gridsize), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    
    # Setup step function using crank-nicolson runge-kutta order 4
    smooth = True
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt)
    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, 1), traj_T)
    
    # Initialize
    rand_key = np.random.randint(0, 100000000)
    v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(rand_key), grid, max_velocity, 4)
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)
    
    if nsteps < 1280:
        ## Trajectory here is in Fourier space
        _, trajectory = trajectory_fn(vorticity_hat0)
        
        trajectory=trajectory[cuts,:,:]
        traj_real=np.fft.irfftn(trajectory, axes=(1,2))

        if downsample is not None:
            traj_real=np.empty((traj_real.shape[0],int(traj_real.shape[1]/downsample),int(traj_real.shape[1]/downsample)))
            ## Overwrite grid object
            grid = grids.Grid(((int(gridsize/downsample), int(gridsize/downsample))), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
            for aa in range(len(trajectory)):
                coarse_h = resize.downsample_spectral(None, grid, trajectory[aa])
                traj_real[aa]=np.fft.irfftn(coarse_h)
        
        spatial_coord = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0]
        coords = {
            'time': Dt * jnp.arange(len(traj_real)),
            'x': spatial_coord,
            'y': spatial_coord,
        }
        
        return xarray.DataArray(traj_real, dims=["time", "x", "y"], coords=coords)
        
    else:
        # First, run spinup phase and discard
        if spinup > 0:
            print(f"Running spinup phase: {spinup} steps...")
            spinup_trajectory_fn = cfd.funcutils.trajectory(
                cfd.funcutils.repeated(step_fn, 1), spinup)
            vorticity_hat_current, _ = spinup_trajectory_fn(vorticity_hat0)
            # Clear memory
            del spinup_trajectory_fn
        else:
            vorticity_hat_current = vorticity_hat0
        
        # Now process the main simulation in chunks
        total_chunks = (nsteps + chunk_size - 1) // chunk_size
        all_trajectories = []
        
        print(f"Processing main simulation in {total_chunks} chunks of {chunk_size} steps...")
        
        for chunk_idx in range(total_chunks):
            start_step = chunk_idx * chunk_size
            end_step = min((chunk_idx + 1) * chunk_size, nsteps)
            actual_chunk_size = end_step - start_step
            
            print(f"Processing chunk {chunk_idx + 1}/{total_chunks}: steps {start_step}-{end_step} ({actual_chunk_size} steps)")
            
            # Create trajectory function for this chunk
            chunk_trajectory_fn = cfd.funcutils.trajectory(
                cfd.funcutils.repeated(step_fn, 1), actual_chunk_size)
            
            # Run this chunk
            vorticity_hat_current, chunk_trajectory = chunk_trajectory_fn(vorticity_hat_current)
            
            # Subsample to physical timesteps
            chunk_trajectory_subsampled=chunk_trajectory[::ratio]
            
            # Convert to real space
            chunk_traj_real = np.fft.irfftn(chunk_trajectory_subsampled, axes=(1, 2))
            
            # Downsample if needed
            if downsample is not None:
                chunk_traj_downsampled=np.empty((chunk_traj_real.shape[0],int(chunk_traj_real.shape[1]/downsample),int(chunk_traj_real.shape[1]/downsample)))
                ## Overwrite grid object
                grid = grids.Grid(((int(gridsize/downsample), int(gridsize/downsample))), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
                for aa in range(len(chunk_traj_real)):
                    coarse_h = resize.downsample_spectral(None, grid, chunk_trajectory_subsampled[aa])
                    chunk_traj_downsampled[aa]=np.fft.irfftn(coarse_h) ## Using numpy here as jnp won't allow for loops.. but this is gross
                
            # Store this chunk
            all_trajectories.append(chunk_traj_downsampled)
            
            # Clean up memory
            del chunk_trajectory_fn, chunk_trajectory, chunk_trajectory_subsampled, chunk_traj_real
            if downsample is not None:
                del chunk_traj_downsampled
            
            print(f"Chunk {chunk_idx + 1} completed")
        
        # Concatenate all chunks
        print("Concatenating all chunks...")
        full_trajectory = np.concatenate(all_trajectories, axis=0)
        
        # Create coordinates
        spatial_coord = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0]
        coords = {
            'time': Dt * jnp.arange(len(full_trajectory)),
            'x': spatial_coord,
            'y': spatial_coord,
        }
        
        return xarray.DataArray(full_trajectory, dims=["time", "x", "y"], coords=coords)

def run_kolmogorov_sim_from_snapshot(
    ics,
    nsteps,
    dt,
    Dt,
    viscosity=1e-4,
    gridsize=512,
    downsample=8,
    chunk_size=1000,
):
    """
    Evolves a batch of coarse vorticity snapshots forward in time with the same
    spectral CN-RK4 integrator used above, returning trajectories on the
    *coarse* grid as an xarray.DataArray.

    Args:
        ics: initial vorticity snapshots (torch.Tensor, np.ndarray, or jnp.ndarray)
             of shape [B, 1, Hc, Wc] where Hc=Wc=gridsize//downsample (e.g. 64).
        nsteps: number of *numerical* steps to integrate for each batch item.
        dt: numerical time step used by the solver.
        Dt: physical output interval. Trajectory is subsampled every `ratio = int(Dt/dt)`.
        viscosity, gridsize, downsample, chunk_size: as in run_kolmogorov_sim.

    Returns:
        xarray.DataArray with shape [B, T, Hc, Wc] where
        T = ceil(nsteps / ratio). Dims are ["batch", "time", "x", "y"].
    """
    ratio = int(Dt / dt)
    if ratio < 1:
        raise ValueError("Dt must be >= dt and an integer multiple is expected.")

    # --- Normalize/unwrap input ---
    try:
        import torch
        is_torch = isinstance(ics, torch.Tensor)
    except Exception:
        is_torch = False

    if is_torch:
        ics_np = ics.detach().cpu().numpy()
    elif isinstance(ics, jnp.ndarray):
        ics_np = np.asarray(ics)
    else:
        ics_np = np.array(ics)

    if ics_np.ndim != 4 or ics_np.shape[1] != 1:
        raise ValueError(
            f"`ics` must have shape [B, 1, Hc, Wc]; got {ics_np.shape}"
        )

    B, _, Hc, Wc = ics_np.shape
    if Hc != Wc:
        raise ValueError("Coarse snapshots must be square: Hc must equal Wc.")

    # Expected coarse size from downsample & fine gridsize
    expected_c = gridsize // downsample if downsample is not None else gridsize
    if Hc != expected_c:
        raise ValueError(
            f"Coarse resolution {Hc} does not match gridsize//downsample={expected_c}."
        )

    # --- Grids & stepper ---
    fine_grid = grids.Grid((gridsize, gridsize), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    coarse_grid = grids.Grid((Hc, Wc),       domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

    smooth = True
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        spectral.equations.ForcedNavierStokes2D(viscosity, fine_grid, smooth=smooth), dt
    )

    # trajectory over "actual_chunk_size" steps
    def make_traj_fn(n):
        return cfd.funcutils.trajectory(cfd.funcutils.repeated(step_fn, 1), n)

    # --- Helper: one trajectory from one coarse IC ---
    def evolve_one(vorticity_coarse):
        """
        vorticity_coarse: (Hc, Wc) real array at coarse grid.
        Returns: (T, Hc, Wc) real array on coarse grid, subsampled in time by `ratio`.
        """
        # (1) FFT on coarse, then upsample Fourier coefficients to fine grid
        vort_hat_coarse = np.fft.rfftn(vorticity_coarse)
        vort_hat_fine = resize.upsample_spectral(None, fine_grid, vort_hat_coarse)

        # (2) Integrate in chunks on fine grid (Fourier space state)
        total_chunks = (nsteps + chunk_size - 1) // chunk_size
        collected = []

        vhat = vort_hat_fine  # current Fourier state
        for chunk_idx in range(total_chunks):
            start_step = chunk_idx * chunk_size
            end_step = min((chunk_idx + 1) * chunk_size, nsteps)
            actual_chunk_size = end_step - start_step

            chunk_traj_fn = make_traj_fn(actual_chunk_size)
            vhat, chunk_traj = chunk_traj_fn(vhat)  # chunk_traj in Fourier space: (actual_chunk_size, Nx, Ny/2+1)

            # (3) Subsample to physical times
            chunk_traj_sub = chunk_traj[::ratio]

            if downsample is not None:
                # (4a) Downsample spectrally back to coarse grid, then iRFFT
                # We operate in Fourier space for downsampling stability.
                # chunk_traj_sub is an array of Fourier fields; loop (small) along time axis.
                out_real = np.empty((chunk_traj_sub.shape[0], Hc, Wc), dtype=np.float64)
                for t in range(chunk_traj_sub.shape[0]):
                    coarse_hat = resize.downsample_spectral(None, coarse_grid, chunk_traj_sub[t])
                    out_real[t] = np.fft.irfftn(coarse_hat)
            else:
                # (4b) No downsample requested; just go to real space on the fine grid
                out_real = np.fft.irfftn(chunk_traj_sub, axes=(1, 2))

            collected.append(out_real)

            # Cleanup
            del chunk_traj_fn, chunk_traj, chunk_traj_sub, out_real

        # (5) Concatenate all subsampled outputs along time
        return np.concatenate(collected, axis=0)

    # --- Evolve each batch item (loop in Python to keep memory reasonable) ---
    batch_outputs = []
    for b in range(B):
        v_coarse = ics_np[b, 0]  # (Hc, Wc)
        traj_b = evolve_one(v_coarse)  # (T, Hc, Wc)
        batch_outputs.append(traj_b)

    full = np.stack(batch_outputs, axis=0)  # (B, T, Hc, Wc)

    # Compute time coordinate based on how many subsampled frames we got
    T = full.shape[1]
    time_coord = Dt * jnp.arange(T)

    spatial_coord = jnp.arange(Hc) * 2 * jnp.pi / Hc
    coords = {
        "batch": np.arange(B),
        "time": time_coord,
        "x": spatial_coord,
        "y": spatial_coord,
    }

    return xarray.DataArray(full, dims=["batch", "time", "x", "y"], coords=coords)
