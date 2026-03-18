'''
Python3
Author: Qi Liu
This script is for generating data for the flow matching model's training. See detailed in Qi Liu's notion workspace: flow matching implementation for emulation
Modified to process data in batches to avoid GPU memory issues.
'''
import sys
import torch
import CT.Inference.Kolmogorov.performance as performance
import Emulator.Models.misc as Emulator_misc
import CT.Models.misc as CT_misc


def Sim_therm_kol(
    path_to_ics, path_to_emu_checkpoint, path_to_CT_checkpoint,
    n_steps = [10_000, 10_000],
    run_by_batch = False, total_batch_num = 1, this_batch_num = 1,
    emu_chunk_size = 1000, subsample_in_time_factor: int = 100,
    short_lag = torch.tensor([2]), long_lag = torch.tensor([100]), s = 10, 
    silence = False, Regression = False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading data and emulator...")
    data_dict = torch.load(path_to_ics)
    emulator = Emulator_misc.load_model(path_to_emu_checkpoint).to(device)
    CT = CT_misc.load_diffusion_model(path_to_CT_checkpoint).to(device)
    data = data_dict['data'] / 4.44

    if run_by_batch:
        assert data.shape[0] % total_batch_num == 0
        batch_size = int(data.shape[0] / total_batch_num)
        start_idx = (this_batch_num - 1) * batch_size
        end_idx = start_idx + batch_size
        ics = data[start_idx:end_idx, 0, :, :].unsqueeze(1)
        print(f"Initial conditions shape: {ics.shape}")
    else:
        ics = data[:, 0:1]

    del data_dict, data

    total_samples = ics.shape[0]
    num_chunks = (total_samples + emu_chunk_size - 1) // emu_chunk_size
    emu_rollouts = []
    short_lag = short_lag.to(device)
    long_lag = long_lag.to(device)

    print(f"Will process {total_samples} samples in {num_chunks} chunks of size {emu_chunk_size}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        for i in range(num_chunks):
            start_idx = i * emu_chunk_size
            end_idx = min(start_idx + emu_chunk_size, total_samples)
            actual_chunk_size = end_idx - start_idx

            print(f"Processing chunk {i+1}/{num_chunks} (samples {start_idx}:{end_idx}, size: {actual_chunk_size})...")

            ics_chunk = ics[start_idx:end_idx].to(device)
            print(f"Chunk {i+1} shape: {ics_chunk.shape}")

            try:
                subsampled_parts = []
                current_state = ics_chunk

                for j in range(len(n_steps)):
                    emu_rollout_chunk, _ = performance.run_conditional_emu(
                        current_state, emulator, therm=CT, n_steps=n_steps[j],
                        short_lag = short_lag, long_lag = long_lag, s = s, freq=25,
                        silence=True, sigma=None, Regression=Regression
                    )
                    print(f"Chunk {i+1} rollout shape (part {j+1}): {emu_rollout_chunk.shape}")

                    # Sub-sample to match the numerical trajectories' time step
                    emu_rollout_chunk_subsampled = emu_rollout_chunk[:, ::subsample_in_time_factor, :, :]
                    print(f"Chunk {i+1} subsampled shape (part {j+1}): {emu_rollout_chunk_subsampled.shape}")

                    subsampled_parts.append(emu_rollout_chunk_subsampled)

                    # Prepare next part to continue from last frame
                    current_state = emu_rollout_chunk_subsampled[:, -1:, :, :]

                    del emu_rollout_chunk, emu_rollout_chunk_subsampled

                # Concatenate all subsampled parts along time
                emu_rollout_chunk_all = torch.cat(subsampled_parts, dim=1)

                # Move to CPU and store
                emu_rollouts.append(emu_rollout_chunk_all.cpu())
                del subsampled_parts, emu_rollout_chunk_all

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Chunk {i+1} completed and moved to CPU")

            except Exception as e:
                import traceback
                print(f"Chunk {i+1} failed: {e}")
                traceback.print_exc()

    print("Concatenating all chunks...")

    emu_rollout_subsampled = torch.cat(emu_rollouts, dim=0)
    return emu_rollout_subsampled
