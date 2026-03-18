import CT.Data.Data_generation.Kolmogorov_thermalized.simulate_thermalized_kol as simulate

import torch
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="name of config file")
parser.add_argument("--batch_number", required=True, type=int, help="batch_number")
args = parser.parse_args()

config = yaml.safe_load(Path(args.config).read_text())
config["batch_number"] = args.batch_number
config["save_path"] = "/scratch/ql2221/thermalizer_data/kolmogorov/reynold10k/CTR_online_inference" + str(config["batch_number"]) + ".p"


sim_stack=torch.tensor([],dtype=torch.float32)
short_lag = torch.tensor([config["short_lag"]])
long_lag = torch.tensor([config["long_lag"]])

sim = simulate.Sim_therm_kol(
    config["path_to_ics"], config["path_to_emu_checkpoint"], config["path_to_CT_checkpoint"],
    n_steps = config["n_steps"],
    run_by_batch = config["run_by_batch"], total_batch_num = config["total_batch_num"], this_batch_num = config["batch_number"],
    emu_chunk_size = config["emu_chunk_size"], subsample_in_time_factor = config["subsample_in_time_factor"],
    short_lag = short_lag, long_lag = long_lag, s = config["s"], silence = config["silence"], Regression = config["Regression"])


save_dict={"data_config":config,
           "data":sim}

torch.save(save_dict, config["save_path"])
