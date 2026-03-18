import torch
import sys
import os
import CT.Data.Data_generation.Kolmogorov_numerical.simulate as simulate

data_dict = torch.load(sys.argv[1])
data_config = data_dict["data_config"]
data = data_dict["data"]
ics = data[:,0:1]
del data_dict, data

data_config["n_steps"] = int(sys.argv[2])
data_config["chunk_size"] = 1000
sim = simulate.run_kolmogorov_sim_from_snapshot(ics, data_config["n_steps"], data_config["dt"], data_config["Dt"], data_config["viscosity"], data_config["gridsize"], data_config["downsample"],data_config["chunk_size"])

sim = torch.tensor(sim.values, dtype=torch.float32)

save_dict = {
        "data_config": data_config,
        "data": sim,
}
output_path = sys.argv[3]

torch.save(save_dict, output_path)