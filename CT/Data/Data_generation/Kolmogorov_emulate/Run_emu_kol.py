import CT.Data.Data_generation.Kolmogorov_emulate.simulate_emu_kol as simulate

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
config["save_path"] = "/scratch/ql2221/thermalizer_data/kolmogorov/reynold10k/coupling_training_emu_data_matched" + str(config["batch_number"]) + ".p"


sim_stack=torch.tensor([],dtype=torch.float32)

sim = simulate.Sim_emu_kol(
    config["path_to_ics"], config["path_to_emu_checkpoint"],
    n_steps = config["n_steps"],
    run_by_batch = config["run_by_batch"], total_batch_num = config["total_batch_num"], this_batch_num = config["batch_number"],
    emu_chunk_size = config["emu_chunk_size"], subsample_in_time_factor = config["subsample_in_time_factor"])


save_dict={"emu_data_config":config,
           "data":sim}

torch.save(save_dict, config["save_path"])
