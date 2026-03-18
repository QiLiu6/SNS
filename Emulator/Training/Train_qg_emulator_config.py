import os
import sys
import wandb

import Emulator.Training.Training_algo as Training

## Stop jax hoovering up GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

config={}
config["model_type"]="ModernUnet"

# Configuration for the baseline Unet
config["input_channels"]=2
config["output_channels"]=2
config["hidden_channels"]=64
config["norm"]=False
config["dim_mults"]=[2,2,2]
config["n_blocks"] = 2
config["mid_attn"]=False
config["is_attn"] = (False, False, False, False)
config["activation"]="gelu"
config["first_embedding_dim"] = None
config["second_embedding_dim"] = None
config["third_embedding_dim"] = None
config["save_path"]="/scratch/ql2221/CT_models/wandb_data"

# Wandb info
config["project"]="ICML26"
config["wandb_run_name"] = "QG_emulator_bounded"

config["ddp"]=False
config["loader_workers"]=5
config["ema_decay"] = 0.995

config["image_width"]=64
config["image_height"]=64

config["rollout_scheduler"]=20000
config["max_rollout"]=4
config["sigma"]=1e-4

config["PDE"]="QG"
config["qg"] = "jet"
config["file_path"]="/scratch/ql2221/PDE_data/2LQG/jet/Baseline_emulator_training_data.pt"
config["subsample"]=None
config["train_ratio"]=0.95
config["save_name"]="model_weights.pt"
config["residual_loss"]="Residual"

# ---------------------------
# NEW: next-step norm bound (stability)
# ---------------------------
config["next_step_norm_mode"] = "rms"          # recommended for grid data
config["next_step_norm_max"] = 1            # B (since values typically in [-4,4])
config["next_step_norm_weight"] = 1        # lambda (tune 1e-4 to 1e-2)
config["next_step_norm_apply_each_rollout"] = True
config["bound_increment_instead"] = False     # bound ||x_{t+1}|| not ||Δx||
# ---------------------------

config["optimization"]={}
config["optimization"]["epochs"]=20
config["optimization"]["lr"]=0.00005
config["optimization"]["wd"]=0.05
config["optimization"]["batch_size"]=32

# NOTE: your Trainer expects "gradient_clip", not "gradient_clipping"
config["optimization"]["gradient_clip"]=1.0

config["optimization"]["scheduler_step"]=100000
config["optimization"]["scheduler_gamma"]=0.5

trainer=Training.ResidualEmulatorTrainer(config)
trainer.run()
wandb.finish()
