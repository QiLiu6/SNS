import os
import wandb
import sys
import CT.Training.Train_conditional_thermalizer_algo as Train_CT

## Stop jax hoovering up GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

config={}
config["model_type"]="ModernUnetDoubleRegressor_plus"

#Configuration for the baseline Unet
config["input_channels"]=3
config["output_channels"]=2
config["hidden_channels"]=64
config["norm"]=False
config["dim_mults"]=[2,2,2] 
config["n_blocks"] = 2
config["mid_attn"]=True
config["is_attn"] = (False, False, True)
config["activation"]="gelu"
config["image_width"]=64
config["image_height"]=64
config["first_embedding_dim"] = 1000
config["second_embedding_dim"] = 1000
config["third_embedding_dim"] = None
config["norm"]=False
config["save_path"]="/scratch/ql2221/CT_models/wandb_data"
config["save_name"]="model_weights.pt"

config["diffusion_steps"] = 1000
config["regression_output_dim"] = 1000
config["regression_target"] = "pred_both_noise_level"
config["noise_schedule"] = "Cosine"
config["sampling_bias"] = 0.2

config["project"]="SNS"
config["wandb_run_name"] = "SNS_1000_bias_sampling"
config["wandb_log_freq"] = 100

#Trainer Info
config["ddp"]=False
config["PDE"]="Kolmogorov"
config["file_path"]="/scratch/ql2221/PDE_data/Kolmogorov_flow/Reynolds10k/SNS_training_data.p"
config["subsample"]=None
config["train_ratio"]=0.95
config["ema_decay"] = 0.999
config["loader_workers"] = 1
config["regression_loss_weight"] = 0.1

#optimization parameters
config["optimization"]={}
config["optimization"]["epochs"]=200
config["optimization"]["lr"]=0.00005
config["optimization"]["wd"]=0.05
config["optimization"]["batch_size"]=64
config["optimization"]["gradient_clip"]=1.0
config["optimization"]["scheduler_step"]=600000
config["optimization"]["scheduler_gamma"]=0.9

trainer = Train_CT.SNS_NS_Trainer(config)
trainer.load_checkpoint("/scratch/ql2221/CT_models/wandb_data/wandb/run-20260206_070307-boa0c79d/files/checkpoint_last.p")
print(trainer.config["cnn learnable parameters"])
trainer.run()