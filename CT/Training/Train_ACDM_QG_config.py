import os
import wandb
import sys
import CT.Training.Train_conditional_thermalizer_algo as Train_CT

## Stop jax hoovering up GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

config={}
config["model_type"]="ModernUnet"

#Configuration for the baseline Unet
config["input_channels"]=4
config["output_channels"]=2
config["hidden_channels"]=64
config["norm"]=False
config["dim_mults"]=[2,2,2] 
config["n_blocks"] = 2
config["mid_attn"]=False
config["is_attn"] = (False, False, False, False)
config["activation"]="gelu"
config["image_width"]=64
config["image_height"]=64
#lag time embedding for first lag
config["first_embedding_dim"] = None
config["noise_short_lag"] = True
config["different_noise"] = True
#lag time embedding for second lag
config["second_embedding_dim"] = None
#noise timestep embedding ( non if doing regression )
config["third_embedding_dim"] = 512
config["norm"]=False
config["save_path"]="/scratch/ql2221/CT_models/wandb_data"
config["save_name"]="model_weights.pt"

config["diffusion_steps"] = 100
config["regression_output_dim"] = 100
config["noise_schedule"] = "Linear"

#Wandb info
config["project"]="CT"
config["wandb_run_name"] = "12/15_QG_baseline_ACDM"
config["wandb_log_freq"] = 100

#Trainer Info
config["ddp"]=False
config["PDE"]="QG"
config["qg"]="jet"
config["file_path"]="/scratch/ql2221/PDE_data/2LQG/jet/ACDM_training_data.pt"
config["subsample"]=None
config["train_ratio"]=0.95
# config["ema_decay"] = 0.995
config["loader_workers"] = 3

#optimization parameters
config["optimization"]={}
config["optimization"]["epochs"]=200
config["optimization"]["lr"]=0.0002
config["optimization"]["wd"]=0.05
config["optimization"]["batch_size"]=64
config["optimization"]["gradient_clip"]=1.
config["optimization"]["scheduler_step"]=100000
config["optimization"]["scheduler_gamma"]=0.5

trainer = Train_CT.ACDM_QG_Trainer(config)
print(trainer.config["cnn learnable parameters"])
trainer.run()
