import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.animation as animation
from IPython.display import HTML
import os
import sys
from tqdm import tqdm

#Rolling and Anchoring
def LSDM_NS_inference(ics, emu, LSDM, n_steps=1000, short_lag_int = 1, freq = 5, lag_ratio = 5, s_init = 10, max_long_lag = 99, starting_time = 1000, sigma=None, silence=False,  forward_diff = True, fix_short = False, device = "cuda"):
    B, _, X, Y = ics.shape
    state_vector=torch.zeros((B,n_steps,X,Y),device=device)

    state_vector[:,0:1]=ics.to(device)
    short_lag = torch.full((B,), short_lag_int, dtype=torch.int64, device=device)
    with torch.no_grad():
        last_therm_step = torch.full((B,), 0, dtype=torch.long, device=device)
        for aa in tqdm(range(1, n_steps), disable=silence):
            prev = state_vector[:, aa - 1]
            state_vector[:, aa] = emu(prev.unsqueeze(1)).squeeze() + prev

            if sigma is not None:
                state_vector[:, aa] += sigma * torch.randn_like(state_vector[:, aa],device=state_vector[:, aa].device,)
            if aa % freq != 0 or aa - starting_time < 0:
                continue 

            batch_idx = torch.arange(B, device=device)
            x_t_noised = state_vector[:, aa].unsqueeze(1)
            x_t_minus_short_lag = state_vector[:, aa - lag_ratio * short_lag_int].unsqueeze(1)

            long_lag_frame_number_before_t = aa - last_therm_step
            mask_1 = (long_lag_frame_number_before_t > max_long_lag * lag_ratio)
            long_lag_frame_number_before_t[mask_1] = max_long_lag * lag_ratio
            
            x_t_minus_long_lag = state_vector[batch_idx, aa - long_lag_frame_number_before_t].unsqueeze(1)
            x_conditional = torch.cat((x_t_minus_long_lag, x_t_minus_short_lag, x_t_noised), dim = 1)
            long_lag = torch.round(long_lag_frame_number_before_t.float() / lag_ratio).long()
            pred_noise_levels = LSDM.model.ratio_class_both(x_conditional, None, None, long_lag).to(device)
            pred_s_t = pred_noise_levels[:,-1:]
            pred_s_short = pred_noise_levels[:,-2:-1]

            mask = (pred_s_t < s_init)
            pred_noise_levels[mask.squeeze(), -2:] = 0
            
            if pred_s_t.max() > s_init:
                if fix_short:
                    state_vector[:, aa:aa+1] = LSDM.denoising_with_fixed_short(x_conditional, long_lag, pred_noise_levels, forward_diff)[:,-1:,:,:]
                else:
                    combined_vector = LSDM.denoising(x_conditional, long_lag, pred_noise_levels, forward_diff)
                    state_vector[:, aa:aa+1] = combined_vector[:,-1:,:,:]
                    state_vector[:,aa-1:aa] = combined_vector[:,-2:-1]
                test_x_conditional = torch.cat((x_t_minus_long_lag, state_vector[:,aa-1:aa], state_vector[:,aa:aa+1]),dim=1)
                test_s_t = LSDM.model.ratio_class(test_x_conditional, 2, None, None, long_lag).to(device)
                mask_6 = (test_s_t <= s_init)
                last_therm_step[mask_6] = aa
    return state_vector

def SNS_NS_refinement(ics, emu, SNS, n_steps=1000, sigma=None, silence=False, forward_diff = True, denoise_mode = "fix_short", device = "cuda"):
    B, _, X, Y = ics.shape
    state_vector=torch.zeros((B,n_steps,X,Y),device=device)
    s_t_history = []
    s_short_history = []

    state_vector[:,0:1]=ics.to(device)
    with torch.no_grad():
        for aa in tqdm(range(1, n_steps), disable=silence):
            prev = state_vector[:, aa - 1]
            state_vector[:, aa] = emu(prev.unsqueeze(1)).squeeze() + prev

            if sigma is not None:
                state_vector[:, aa] += sigma * torch.randn_like(state_vector[:, aa],device=state_vector[:, aa].device,)

            x_t_noised = state_vector[:, aa].unsqueeze(1)
            x_t_minus_short_lag = state_vector[:, aa - 1].unsqueeze(1)
            
            x_conditional = torch.cat((ics, x_t_minus_short_lag, x_t_noised), dim = 1)
            pred_noise_levels = SNS.model.ratio_class_both(x_conditional, None, None, None).to(device)
            pred_s_t = pred_noise_levels[:,-1:]
            pred_s_short = pred_noise_levels[:,-2:-1]
            s_t_history.append(pred_s_t)
            s_short_history.append(pred_s_short)

            if denoise_mode == "fix_short":
                state_vector[:, aa:aa+1] = SNS.denoising_with_fixed_short(x_conditional, pred_noise_levels, forward_diff)[:,-1:,:,:]
            elif denoise_mode == "down_left":
                state_vector[:, aa:aa+1] = SNS.denoising_down_left(x_conditional, pred_noise_levels, forward_diff)[:,-1:,:,:]
            elif denoise_mode == "left_down":
                state_vector[:, aa:aa+1] = SNS.denoising_left_down(x_conditional, pred_noise_levels, forward_diff)[:,-1:,:,:]
            elif denoise_mode == "SNS":
                combined_vector = SNS.denoising_SNS(x_conditional, pred_noise_levels, forward_diff)
                state_vector[:, aa:aa+1] = combined_vector[:,-1:,:,:]
                state_vector[:,aa-1:aa] = combined_vector[:,-2:-1]
            else:
                print("aaaaa")
    return state_vector, s_short_history, s_t_history

#Rolling and Anchoring
def roll_and_anchor(ics, emu, CT, n_steps=1000, short_lag_int = 1, freq = 5, lag_ratio = 5, s_init = 10, max_long_lag = 99, starting_time = 1000, sigma=None, silence=False, device = "cuda"):
    B, _, X, Y = ics.shape
    state_vector=torch.zeros((B,n_steps,X,Y),device=device)

    state_vector[:,0:1]=ics.to(device)
    short_lag = torch.full((B,), short_lag_int, dtype=torch.int64, device=device)
    with torch.no_grad():
        last_therm_step = torch.full((B,), 0, dtype=torch.long, device=device)
        for aa in tqdm(range(1, n_steps), disable=silence):
            prev = state_vector[:, aa - 1]
            state_vector[:, aa] = emu(prev.unsqueeze(1)).squeeze() + prev

            if sigma is not None:
                state_vector[:, aa] += sigma * torch.randn_like(state_vector[:, aa],device=state_vector[:, aa].device,)
            if aa % freq != 0 or aa - starting_time < 0:
                continue 

            batch_idx = torch.arange(B, device=device)
            x_t_noised = state_vector[:, aa].unsqueeze(1)
            x_t_minus_short_lag = state_vector[:, aa - lag_ratio * short_lag_int].unsqueeze(1)

            long_lag_frame_number_before_t = aa - last_therm_step
            mask_1 = (long_lag_frame_number_before_t > max_long_lag * lag_ratio)
            long_lag_frame_number_before_t[mask_1] = max_long_lag * lag_ratio
            
            x_t_minus_long_lag = state_vector[batch_idx, aa - long_lag_frame_number_before_t].unsqueeze(1)
            x_conditional = torch.cat((x_t_noised, x_t_minus_short_lag, x_t_minus_long_lag), dim = 1)
            long_lag = torch.round(long_lag_frame_number_before_t.float() / lag_ratio).long()
            pred_noise_level = CT.model.ratio_class(x_conditional, short_lag, long_lag).to(device)
            print(pred_noise_level)
            
            if pred_noise_level.max() > s_init:
                state_vector[:, aa:aa+1] = CT.denoising(x_conditional, short_lag, long_lag, pred_noise_level)
                test_x_conditional = torch.cat((state_vector[:,aa:aa+1],x_t_minus_short_lag,x_t_minus_long_lag),dim=1)
                test_s = CT.model.ratio_class(test_x_conditional, short_lag, long_lag).to(device)
                mask_6 = (test_s < s_init)
                last_therm_step[mask_6] = aa
    return state_vector

def SAD_only_rollout(ics, SAD, n_steps=1000, refinement = True, sigma=None, silence=False, device = "cuda"):
    B, _, X, Y = ics.shape
    state_vector = torch.zeros((B,n_steps,X,Y),device=device)
    state_vector[:,0:1]=ics.to(device)
    with torch.no_grad():
        if refinement:
            for aa in tqdm(range(1,n_steps),disable=silence):
                x_t_short_lag = state_vector[:,aa-1].unsqueeze(1)
                x_new = SAD.sampling_with_refinement(x_short_lag)
                state_vector[:,aa-1:aa+1]=x_new
        else:
            for aa in tqdm(range(1,n_steps),disable=silence):
                x_t_short_lag = state_vector[:,aa-1].unsqueeze(1)
                x_new = SAD.sampling_with_fix_short_lag(x_short_lag)
                state_vector[:,aa:aa+1]=x_new
        
    return state_vector




    

def run_emu(ics,emu,n_steps=1270,silent=False,sigma=None):
    """ Run an emuluator on some ICs
        ics:     initial conditions for emulator
        emu:     torch emulator model
        n_steps: how many emulator steps to run
        silent:  silence tqdm progress bar (for slurm scripts)
        sigma:   noise std level if we have a stochastic rollout """
    ## Set up state tensors
    state_vector=torch.zeros((len(ics),n_steps,64,64),device="cuda")
    
    ## Set ICs
    state_vector[:,0:1]=ics
    state_vector=state_vector.to("cuda")

    with torch.no_grad(): 
        for aa in tqdm(range(1,n_steps),disable=silent):
            state_vector[:,aa]=emu(state_vector[:,aa-1].unsqueeze(1)).squeeze()+state_vector[:,aa-1]
            if sigma:
                state_vector[:,aa]+=sigma*torch.randn_like(state_vector[:,aa],device=state_vector[:,aa].device)
    return state_vector