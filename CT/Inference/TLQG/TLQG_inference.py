import torch
import CT.Models.misc as misc
import CT.Inference.TLQG.performance as performance
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lag_ratio = 1
LSDM_QG_file_path = "/scratch/ql2221/CT_models/wandb_data/wandb/run-20260119_122954-1cxsh1jy/files/checkpoint_last.p"

LSDM_QG = misc.load_LSDM(LSDM_QG_file_path).to(device)

#load data
data_dict = torch.load("/scratch/ql2221/PDE_data/2LQG/jet/qg_jet_long_ICML_NUM.pt")
data = data_dict["data"]
print(data.shape)

upper_std=7.6382e-06
lower_std=3.0120e-07
data[:,:,0,:,:]/=upper_std
data[:,:,1,:,:]/=lower_std

x = data.to(device)
Emu_file_path = file_string = "/scratch/ql2221/thermalizer_data/wandb_data/wandb/run-20260128_025723-qbx6gf4w/files/checkpoint_best.p"
Emu = misc.load_Emulator(Emu_file_path).to(device)
Emu.eval()
LSDM_QG.model.eval()
rollout= performance.LSDM_QG_inference(data[:,0:1], Emu, LSDM_QG, n_steps=50000, short_lag_int = 1, freq = lag_ratio, lag_ratio = lag_ratio, s_init = -1, max_long_lag = 99, starting_time = 100, sigma=None, silence=False, forward_diff = True, denoise_mode = "left_down", device = "cuda")

torch.save(rollout, "/scratch/ql2221/PDE_data/2LQG/jet/LSDM_inference_ICML.p")