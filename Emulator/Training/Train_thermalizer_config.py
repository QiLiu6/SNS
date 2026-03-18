# import os
# import wandb

# import Emulator.Training.Training_algo as Training

# ## Stop jax hoovering up GPU memory
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# config={}
# config["input_channels"]=1
# config["output_channels"]=1
# config["model_type"]="ModernUnetRegressor"
# config["dim_mults"]=[2,2,2] 
# config["n_blocks"] = 2
# config["hidden_channels"]=64
# config["activation"]="gelu"
# config["loader_workers"]=1
# config["image_width"]=64
# config["image_height"]=64
# config["image_size"] = 64
# ## Thermalizer stuff
# config["regression_loss_weight"]=1
# config["denoise_time"]=400
# config["valid_samps"]=500
# config["timesteps"]=1000
# config["ema_decay"]=0.995
# config["mid_attn"]=False
# config["is_attn"] = (False, False, False, False)

# config["project"]="ICML26"
# config["wandb_run_name"] = "QG_thermalizer"
# config["norm"]=False
# config["ddp"]=False
# config["PDE"]="qg"
# config["qg"] = "jet"
# config["file_path"]="/scratch/ql2221/PDE_data/Kolmogorov_flow/Reynolds10k/Thermalizer_training_data_combined.p"
# config["subsample"]=None
# config["train_ratio"]=0.95
# config["save_name"]="model_weights.pt"

# #config["short_rollout"]=1
# #config["add_noise"]=1e-4
# config["optimization"]={}
# config["optimization"]["epochs"]=50
# config["optimization"]["lr"]=0.00002
# config["optimization"]["wd"]=0.05
# config["optimization"]["batch_size"]=64
# config["optimization"]["gradient_clipping"]=1.
# config["optimization"]["scheduler_step"]=100000
# config["optimization"]["scheduler_gamma"]=0.5

# trainer = Training.ThermalizerTrainer(config)
# print(trainer.config["cnn learnable parameters"])
# trainer.run()

import os
import wandb
import Emulator.Training.Training_algo as Training

## Stop jax hoovering up GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

config={}
config["input_channels"]=2
config["output_channels"]=2
config["model_type"]="ModernUnetRegressor"
config["dim_mults"]=[2,2,2] 
config["hidden_channels"]=64
config["activation"]="gelu"
config["loader_workers"]=3
config["image_size"]=64
## Thermalizer stuff
config["regression_loss_weight"]=1
config["denoise_time"]=400
config["valid_samps"]=500
config["timesteps"]=1000
config["ema_decay"]=0.995

config["project"]="ICML2026"
config["norm"]=False
config["ddp"]=False
config["PDE"]="QG"
config["wandb_run_name"] = "QG_thermalizer"
config["qg"] = "jet"
config["file_path"]="/scratch/ql2221/PDE_data/2LQG/jet/Thermalizer_training_data.pt"
config["subsample"]=None
config["train_ratio"]=0.95
config["save_name"]="model_weights.pt"

#config["short_rollout"]=1
#config["add_noise"]=1e-4
config["optimization"]={}
config["optimization"]["epochs"]=100
config["optimization"]["lr"]=0.00002
config["optimization"]["wd"]=0.05
config["optimization"]["batch_size"]=64
config["optimization"]["gradient_clipping"]=1.
config["optimization"]["scheduler_step"]=100000
config["optimization"]["scheduler_gamma"]=0.5


trainer=Training.ThermalizerTrainer(config)
print(trainer.config["cnn learnable parameters"])
trainer.run()