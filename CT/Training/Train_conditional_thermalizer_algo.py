import os
import pickle
import wandb
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import CT.Models.diffusion_regression as diffusion_regression
import CT.Models.MNLCD as MNLCD
import CT.Models.LSDM_NS as LSDM_NS
import CT.Models.LSDM_QG as LSDM_QG
import CT.Models.ACDM_NS as ACDM_NS
import CT.Models.ACDM_QG as ACDM_QG
import CT.Models.SAD_NS as SAD_NS
import CT.Models.misc as misc
import CT.Data.Dataset as datasets
import CT.Models.SNS_NS as SNS_NS
import CT.Models.SNS_QG as SNS_QG

def setup():
    """Sets up the process group for distributed training.
       We are using torchrun so not using rank and world size arguments """
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")

def cleanup():
    """Cleans up the process group."""
    dist.destroy_process_group()

def trainer_from_checkpoint(checkpoint_string):
    if model_dict["config"]["model_type"]=='ModernUnetDoubleRegressor_plus':
        trainer = LSDM_QG_Trainer(model_dict["config"])
    elif model_dict["config"]["model_type"]=='ModernUnetRegressor':
        trainer = CTR_Trainer(model_dict["config"])
    else:
        raise ValueError("Invalid base model for diffusion model.")
    trainer.load_checkpoint(checkpoint_string)
    return trainer
        
class Trainer:
    """ Base trainer class """
    def __init__(self,config):
        self.config  = config
        self.epoch = 1
        self.training_step = 0
        self.wandb_init = False 
        self.ema = None
        self.best_val_loss = float("inf")
        
        if self.config.get("wandb_log_freq"):
            self.log_freq = self.config.get("wandb_log_freq")
        else:
            self.log_freq = 50

        self.gradient_clip = self.config["optimization"].get("gradient_clip")

        '''
        Don't know what this part does.
        '''
        if self.config["ddp"]:
            setup()
            self.gpu_id=int(os.environ["LOCAL_RANK"])
            self.ddp=True
            self.world_size=dist.get_world_size()
            self.config["world_size"]=self.world_size
            if self.gpu_id==0:
                self.logging=True
            else:
                self.logging=False
        else:
            self.gpu_id="cuda"
            self.ddp=False
            self.logging=True

        print("Prep data")
        self._prep_data()
        print("Prep model")
        self._prep_model()
        print("Prep optimizer")
        self._prep_optimizer()

    #I also don't know how the wandb stuff work
    def init_wandb(self):
        ## Set up wandb stuff
        wandb.init(entity="qiliu2221",
                   project=self.config["project"],
                   dir="/scratch/ql2221/CT_models/wandb_data",
                   name=self.config["wandb_run_name"],
                   config=self.config,
                   )
        self.config["save_path"]=wandb.run.dir
        self.config["wandb_url"]=wandb.run.get_url()
        self.wandb_init=True 
        ## Sync all configs
        wandb.config.update(self.config, allow_val_change=True)
        self.model.config = self.config

    def resume_wandb(self):
        """ Resume a wandb run from the self.config wandb url. """
        wandb.init(entity="qiliu2221",project=self.config["project"],
                            id=self.config["wandb_url"][-8:],dir="/scratch/ql2221/CT_models/wandb_data", resume="must")
        self.wandb_init=True
        return

    def _prep_data(self):
        if self.config["PDE"]=="Kolmogorov":
            train_data,valid_data,config=datasets.parse_kol_data(self.config)
        elif self.config["PDE"]=="QG":
            train_data,valid_data,config=datasets.parse_data_file_qg(self.config)
        
        else:
            print("Need to know what PDE system we are working with")
            quit()
    
        ds_train = datasets.FluidDataset(train_data)
        ds_valid = datasets.FluidDataset(valid_data)
        self.config = config ## Update config dict

        if self.ddp:
            train_sampler = DistributedSampler(ds_train)
            valid_sampler = DistributedSampler(ds_valid)
        else:
            train_sampler=RandomSampler(ds_train)
            valid_sampler=RandomSampler(ds_valid)
    
        self.train_loader = DataLoader(
                ds_train,
                num_workers = self.config["loader_workers"],
                batch_size = self.config["optimization"]["batch_size"],
                sampler = train_sampler,
            )
    
        self.valid_loader = DataLoader(
                ds_valid,
                num_workers = self.config["loader_workers"],
                batch_size = self.config["optimization"]["batch_size"],
                sampler = valid_sampler,
            )

    def _prep_model(self):
        self.model=misc.model_factory(self.config).to(self.gpu_id)
        self.config["cnn learnable parameters"]=sum(p.numel() for p in self.model.parameters())

        if self.config.get("ema_decay"):
            self.ema=misc.ExponentialMovingAverage(self.model,decay=self.config.get("ema_decay"))
            self.ema.register()

        if self.ddp:
            self.model = DDP(self.model,device_ids=[self.gpu_id])

    def _prep_optimizer(self):
        self.criterion=nn.MSELoss()
        self.optimizer=torch.optim.AdamW(self.model.parameters(),
                            lr=self.config["optimization"]["lr"],
                            weight_decay=self.config["optimization"]["wd"])
        if self.config["optimization"].get("scheduler_step"):
            self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,
                    self.config["optimization"]["scheduler_step"],
                    gamma=self.config["optimization"]["scheduler_gamma"], last_epoch=-1)
        else:
            self.scheduler=None

    def save_checkpoint(self, checkpoint_path):
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        save_dict = {
            "epoch": self.epoch,
            "training_step": self.training_step,
            "state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(save_dict, checkpoint_path)

    def training_loop(self):
        raise NotImplementedError("Implemented by subclass")

    def valid_loop(self):
        raise NotImplementedError("Implemented by subclass")

    def run(self, epochs=None):
        if self.logging and self.wandb_init == False:
            self.init_wandb()
            self.model.model.config = self.config
        if epochs:
            max_epochs = epochs
        else:
            max_epochs = self.config["optimization"]["epochs"]
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            print("Training at epoch", self.epoch)
            if self.ddp and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.epoch)
            self.training_loop()
            val_loss = self.valid_loop()
            if self.logging:  # rank 0 only
                self.save_checkpoint(self.config["save_path"] + "/checkpoint_last.p")
            
            if self.ema:
                self.ema.apply_shadow()
                if self.logging:
                    self.save_checkpoint(self.config["save_path"] + "/checkpoint_last_ema.p")
                self.ema.restore()
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            
                if self.logging:
                    self.save_checkpoint(self.config["save_path"] + "/checkpoint_best.p")
            
                if self.ema:
                    self.ema.apply_shadow()
                    if self.logging:
                        self.save_checkpoint(self.config["save_path"] + "/checkpoint_best_ema.p")
                    self.ema.restore()
        print("DONE on rank", self.gpu_id)
        if self.ddp:
            cleanup()

    def load_checkpoint(
        self,
        checkpoint_path,
        map_location="cuda",
        load_optimizer=True,
        load_scheduler=True,
        resume_wandb=True,
    ):
        ckpt = torch.load(checkpoint_path, map_location=map_location)
    
        # ---- model / optimizer / scheduler ----
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_load.load_state_dict(ckpt["state_dict"], strict=True)
    
        if load_optimizer and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    
        if load_scheduler and self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    
        self.epoch = int(ckpt.get("epoch", 1))
        self.training_step = int(ckpt.get("training_step", 0))
        self.best_val_loss = float(ckpt.get("best_val_loss", self.best_val_loss))
    
        # ---- wandb resume ----
        if resume_wandb and self.logging:
            if "wandb_url" not in ckpt.get("config", {}):
                raise RuntimeError("Checkpoint has no wandb_url; cannot resume wandb run.")
            self.config["wandb_url"] = ckpt["config"]["wandb_url"]
            self.resume_wandb()
    
        if self.logging:
            print(
                f"Loaded checkpoint: {checkpoint_path} | "
                f"epoch={self.epoch}, step={self.training_step}"
            )


class CT_Trainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        if self.config["ddp"]==True:
            raise NotImplementedError

    def _prep_model(self):
        model_unet=misc.model_factory(self.config).to(self.gpu_id)
        self.model=Conditional_diffusion.Diffusion(self.config, model=model_unet).to(self.gpu_id)
        self.config["cnn learnable parameters"]=sum(p.numel() for p in self.model.parameters())
        if self.config.get("ema_decay"):
            self.ema=misc.ExponentialMovingAverage(self.model,decay=self.config.get("ema_decay"))
            self.ema.register()

    def load_checkpoint(self,file_string,resume_wandb=True):
        model_dict = torch.load(file_string)
        self.model=misc.load_diffusion_model(file_string).to(self.gpu_id)
        self._prep_optimizer()
        self.optimizer.load_state_dict(model_dict['optimizer_state_dict'])
        self.epoch=model_dict["epoch"]
        self.training_step=model_dict["training_step"]

        if self.wandb_init==False and resume_wandb:
            self.resume_wandb()
        return

    def training_loop(self):
        self.model.train()
        for j,x_train in enumerate(self.train_loader):
            x_train = x_train.to(self.gpu_id)
            self.optimizer.zero_grad()
            noise_t = torch.randn((x_train.shape[0],1,x_train.shape[2],x_train.shape[3]),device = x_train.device)
            lags = torch.randint(1, self.config["lagsteps"] - 1, (2,))
            lag_short, lag_long = torch.sort(lags)[0]
            pred_noise_t = self.model(x_train, noise_t , lag_short, lag_long)
            loss = self.criterion(pred_noise_t, noise_t)
            loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            if self.ema:               
                self.ema.update()       
            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss"]=loss.item()
                log_dic["training_step"]=self.training_step
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1
        return loss
    def valid_loop(self):
        log_dic = {}
        self.model.eval()
        if self.ema:                      
            self.ema.apply_shadow()       
        epoch_loss = 0
        nsamp = 0
        with torch.no_grad():
            for x_valid in self.valid_loader:
                x_valid = x_valid.to(self.gpu_id)
                nsamp += x_valid.shape[0]
                loss = 0
                noise_t = torch.randn((x_valid.shape[0],1,x_valid.shape[2],x_valid.shape[3]),device = x_valid.device)
                lags = torch.randint(1, self.config["lagsteps"] - 1, (2,))
                lag_short, lag_long = torch.sort(lags)[0]
                pred_noise_t = self.model(x_valid, noise_t, lag_short, lag_long)
                loss = self.criterion(pred_noise_t, noise_t)
                epoch_loss+=loss.detach()*x_valid.shape[0]
        epoch_loss/=nsamp
        if self.ddp:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            self.val_loss = epoch_loss.item()/self.world_size
        else:
            self.val_loss = epoch_loss.item()
        if self.logging:
            log_dic={}
            log_dic["valid_loss"]=self.val_loss ## Average over full epoch
            log_dic["training_step"]=self.training_step
            wandb.log(log_dic)
        if self.ema:
            self.ema.restore()
        return self.val_loss

    def run(self, epochs=None):
        if self.logging and self.wandb_init == False:
            self.init_wandb()
            self.model.model.config = self.config
        if epochs:
            max_epochs = epochs
        else:
            max_epochs = self.config["optimization"]["epochs"]
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            print("Training at epoch", self.epoch)
            self.training_loop()
            val_loss = self.valid_loop()
            self.save_checkpoint(self.config["save_path"] + "/checkpoint_last.p")
            if self.ema:
                self.ema.apply_shadow()
                self.save_checkpoint(self.config["save_path"] + "/checkpoint_last_ema.p")
                self.ema.restore()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(self.config["save_path"] + "/checkpoint_best.p")
                if self.ema:
                    self.ema.apply_shadow()
                    self.save_checkpoint(self.config["save_path"] + "/checkpoint_best_ema.p")
                    self.ema.restore()
                if self.logging:
                    wandb.log({"best_valid_loss": self.best_val_loss})
        print("DONE on rank", self.gpu_id)
        
class MNLCD_Trainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        if self.config["ddp"]==True:
            raise NotImplementedError

    def _prep_model(self):
        model_unet=misc.model_factory(self.config).to(self.gpu_id)
        self.model=MNLCD.MNLCD(self.config, model=model_unet).to(self.gpu_id)
        self.config["cnn learnable parameters"]=sum(p.numel() for p in self.model.parameters())
        if self.config.get("ema_decay"):
            self.ema=misc.ExponentialMovingAverage(self.model,decay=self.config.get("ema_decay"))
            self.ema.register()
            
    def training_loop(self):
        self.model.train()
        for j,x_train in enumerate(self.train_loader):
            x_train = x_train.to(self.gpu_id)
            self.optimizer.zero_grad()
            
            noise = torch.randn((x_train.shape[0],2,x_train.shape[2],x_train.shape[3]),device = x_train.device)
            pred_noise = self.model(x_train, noise)
            
            loss = self.criterion(pred_noise,noise)
            loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss"]=loss.item()
                log_dic["training_step"]=self.training_step
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1

            if self.ema:
                self.ema.update()

        return loss

    def valid_loop(self):
        log_dic = {}
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        epoch_loss = 0
        nsamp = 0
        with torch.no_grad():
            for x_valid in self.valid_loader:
                x_valid = x_valid.to(self.gpu_id)
                nsamp += x_valid.shape[0]
                loss = 0
                
                noise = torch.randn((x_valid.shape[0],2,x_valid.shape[2],x_valid.shape[3]),device = x_valid.device)
               
                pred_noise = self.model(x_valid, noise)
                
                loss = self.criterion(pred_noise,noise)

                epoch_loss+=loss.detach()*x_valid.shape[0]
               
        epoch_loss/=nsamp
        if self.ddp:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            self.val_loss = epoch_loss.item()/self.world_size
        else:
            self.val_loss = epoch_loss.item()
        if self.logging:
            log_dic={}
            log_dic["valid_loss"]=self.val_loss ## Average over full epoch
            log_dic["training_step"]=self.training_step
            wandb.log(log_dic)
        if self.ema:
            self.ema.restore()
        return self.val_loss

class ACDM_NS_Trainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        if self.config["ddp"]==True:
            raise NotImplementedError

    def _prep_model(self):
        model_unet=misc.model_factory(self.config).to(self.gpu_id)
        self.model=ACDM_NS.ACDM_NS(self.config, model=model_unet).to(self.gpu_id)
        self.config["cnn learnable parameters"]=sum(p.numel() for p in self.model.parameters())
        if self.config.get("ema_decay"):
            self.ema=misc.ExponentialMovingAverage(self.model,decay=self.config.get("ema_decay"))
            self.ema.register()
            
    def training_loop(self):
        self.model.train()
        for j,x_train in enumerate(self.train_loader):
            x_train = x_train.to(self.gpu_id)
            self.optimizer.zero_grad()
            noise = torch.randn((x_train.shape[0], 2 ,x_train.shape[2],x_train.shape[3]),device = x_train.device)
            pred_noise = self.model(x_train, noise)
            loss = self.criterion(pred_noise,noise)

            loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss"]=loss.item()
                log_dic["training_step"]=self.training_step
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1
            if self.ema:
                self.ema.update()
        return loss

    def valid_loop(self):
        log_dic = {}
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        epoch_loss = 0
        nsamp = 0
        with torch.no_grad():
            for x_valid in self.valid_loader:
                x_valid = x_valid.to(self.gpu_id)
                nsamp += x_valid.shape[0]
                loss = 0
                noise = torch.randn((x_valid.shape[0],2,x_valid.shape[2],x_valid.shape[3]),device = x_valid.device)
                pred_noise = self.model(x_valid, noise)
                loss = self.criterion(pred_noise,noise)
                epoch_loss+=loss.detach()*x_valid.shape[0]
        epoch_loss/=nsamp
        if self.ddp:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            self.val_loss = epoch_loss.item()/self.world_size
        else:
            self.val_loss = epoch_loss.item()
        if self.logging:
            log_dic={}
            log_dic["valid_loss"]=self.val_loss
            log_dic["training_step"]=self.training_step
            wandb.log(log_dic)
        if self.ema:
            self.ema.restore()
        return self.val_loss

class ACDM_QG_Trainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        if self.config["ddp"]==True:
            raise NotImplementedError

    def _prep_model(self):
        model_unet=misc.model_factory(self.config).to(self.gpu_id)
        self.model=ACDM_QG.ACDM_QG(self.config, model=model_unet).to(self.gpu_id)
        self.config["cnn learnable parameters"]=sum(p.numel() for p in self.model.parameters())
        if self.config.get("ema_decay"):
            self.ema=misc.ExponentialMovingAverage(self.model,decay=self.config.get("ema_decay"))
            self.ema.register()
            
    def training_loop(self):
        self.model.train()
        for j,x_train in enumerate(self.train_loader):
            x_train = x_train.to(self.gpu_id)
            self.optimizer.zero_grad()
            noise_t = torch.randn((x_train.shape[0], 2, x_train.shape[3], x_train.shape[4]),device = x_train.device)
            pred_noise = self.model(x_train, noise_t)
            loss = self.criterion(pred_noise,noise_t)
            loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss"]=loss.item()
                log_dic["training_step"]=self.training_step
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1
            if self.ema:
                self.ema.update()
        return loss
    def valid_loop(self):
        log_dic = {}
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        epoch_loss = 0
        nsamp = 0
        with torch.no_grad():
            for x_valid in self.valid_loader:
                x_valid = x_valid.to(self.gpu_id)
                nsamp += x_valid.shape[0]
                loss = 0
                noise_t = torch.randn((x_valid.shape[0], 2, x_valid.shape[3], x_valid.shape[4]),device = x_valid.device)
                pred_noise = self.model(x_valid, noise_t)
                loss = self.criterion(pred_noise,noise_t)
                epoch_loss+=loss.detach()*x_valid.shape[0]
        epoch_loss/=nsamp
        if self.ddp:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            self.val_loss = epoch_loss.item()/self.world_size
        else:
            self.val_loss = epoch_loss.item()
        if self.logging:
            log_dic={}
            log_dic["valid_loss"]=self.val_loss
            log_dic["training_step"]=self.training_step
            wandb.log(log_dic)
        if self.ema:
            self.ema.restore()
        return self.val_loss

class LSDM_NS_Trainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        self.lambda_c=config["regression_loss_weight"]
        self.softmax = nn.Softmax(dim=1)
        if self.config["ddp"]==True:
            raise NotImplementedError

    def _prep_model(self):
        model_unet=misc.model_factory(self.config).to(self.gpu_id)
        self.model=LSDM_NS.LSDM_NS(self.config, model=model_unet).to(self.gpu_id)
        self.config["cnn learnable parameters"]=sum(p.numel() for p in self.model.parameters())
        if self.config.get("ema_decay"):
            self.ema=misc.ExponentialMovingAverage(self.model,decay=self.config.get("ema_decay"))
            self.ema.register()
            
    def training_loop(self):
        self.model.train()
        for j,x_train in enumerate(self.train_loader):
            x_train = x_train.to(self.gpu_id)
            self.optimizer.zero_grad()
            
            noise = torch.randn((x_train.shape[0], 2, x_train.shape[2],x_train.shape[3]),device = x_train.device)
            if self.config.get("fix_short_lag") is not None:
                short_lag = torch.tensor(self.config["fix_short_lag"],dtype=torch.int64, device=x_train.device)
                long_lag = torch.randint(1, self.config["lagsteps"], (), device=x_train.device)
            else:
                lags = torch.randint(1, self.config["lagsteps"], (2,),device = x_train.device)
                short_lag, long_lag = torch.sort(lags)[0]
            #decide what noise levels to predict:
            if self.config.get("regression_target") == "pred_x_t_noise_level":
                pred_noise, pred_s, true_s , true_s_prime = self.model(x_train, noise, short_lag, long_lag)
                loss = self.criterion(pred_noise,noise_t)
                loss += self.lambda_c * F.cross_entropy(pred_s, true_s)
            elif self.config.get("regression_target") == "pred_x_t_short_lag_noise_level":
                pred_noise, pred_s_prime, true_s , true_s_prime = self.model(x_train, noise, short_lag, long_lag)
                loss = self.criterion(pred_noise,noise_t)
                loss += self.lambda_c * F.cross_entropy(pred_s_prime, true_s_prime)
            elif self.config.get("regression_target") == "pred_both_noise_level":
                pred_noise, pred_s_short, pred_s_t, true_s_short , true_s_t = self.model(x_train, noise, short_lag, long_lag)
                loss = self.criterion(pred_noise,noise)
                loss += self.lambda_c * F.cross_entropy(pred_s_t, true_s_t)
                loss += self.lambda_c * F.cross_entropy(pred_s_short, true_s_short)
            loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss"]=loss.item()
                log_dic["training_step"]=self.training_step
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1
            if self.ema:
                self.ema.update()
        return loss
    def valid_loop(self):
        log_dic = {}
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        epoch_loss = 0
        nsamp = 0
        with torch.no_grad():
            for x_valid in self.valid_loader:
                x_valid = x_valid.to(self.gpu_id)
                nsamp += x_valid.shape[0]
                loss = 0
                
                noise = torch.randn((x_valid.shape[0], 2, x_valid.shape[2],x_valid.shape[3]),device = x_valid.device)
                if self.config.get("fix_short_lag") is not None:
                    short_lag = torch.tensor(self.config["fix_short_lag"],dtype=torch.int64, device=x_valid.device)
                    long_lag = torch.randint(1, self.config["lagsteps"], (), device=x_valid.device)
                else:
                    lags = torch.randint(1, self.config["lagsteps"] - 1, (2,),device = x_valid.device)
                    short_lag, long_lag = torch.sort(lags)[0]
                
                if self.config.get("regression_target") == "pred_x_t_noise_level":
                    pred_noise, pred_s, true_s , true_s_prime = self.model(x_valid, noise_t, short_lag, long_lag)
                    loss = self.criterion(pred_noise,noise_t)
                    loss += self.lambda_c * F.cross_entropy(pred_s, true_s)
                elif self.config.get("regression_target") == "pred_x_t_short_lag_noise_level":
                    pred_noise, pred_s_prime, true_s , true_s_prime = self.model(x_valid, noise_t, short_lag, long_lag)
                    loss = self.criterion(pred_noise,noise_t)
                    loss += self.lambda_c * F.cross_entropy(pred_s_prime, true_s_prime)
                elif self.config.get("regression_target") == "pred_both_noise_level":
                    pred_noise, pred_s_short, pred_s_t, true_s_short , true_s_t = self.model(x_valid, noise, short_lag, long_lag)
                    loss = self.criterion(pred_noise,noise)
                    loss += self.lambda_c * F.cross_entropy(pred_s_short, true_s_short)
                    loss += self.lambda_c * F.cross_entropy(pred_s_t, true_s_t)
                epoch_loss+=loss.detach()*x_valid.shape[0]
        epoch_loss/=nsamp
        if self.ddp:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            self.val_loss = epoch_loss.item()/self.world_size
        else:
            self.val_loss = epoch_loss.item()
        if self.logging:
            log_dic={}
            log_dic["valid_loss"]=self.val_loss ## Average over full epoch
            log_dic["training_step"]=self.training_step
            wandb.log(log_dic)
        if self.ema:
            self.ema.restore()
        return self.val_loss

class LSDM_QG_Trainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        self.lambda_c=config["regression_loss_weight"]
        self.softmax = nn.Softmax(dim=1)
        if self.config["ddp"]==True:
            raise NotImplementedError

    def _prep_model(self):
        model_unet=misc.model_factory(self.config).to(self.gpu_id)
        self.model=LSDM_QG.LSDM_QG(self.config, model=model_unet).to(self.gpu_id)
        self.config["cnn learnable parameters"]=sum(p.numel() for p in self.model.parameters())
        if self.config.get("ema_decay"):
            self.ema=misc.ExponentialMovingAverage(self.model,decay=self.config.get("ema_decay"))
            self.ema.register()
            
    def training_loop(self):
        self.model.train()
        for j,x_train in enumerate(self.train_loader):
            x_train = x_train.to(self.gpu_id)
            self.optimizer.zero_grad()
            
            noise = torch.randn((x_train.shape[0], 4, x_train.shape[3],x_train.shape[4]),device = x_train.device)
            if self.config.get("fix_short_lag") is not None:
                short_lag = torch.tensor(self.config["fix_short_lag"],dtype=torch.int64, device=x_train.device)
                long_lag = torch.randint(1, self.config["lagsteps"], (), device=x_train.device)
            else:
                lags = torch.randint(1, self.config["lagsteps"], (2,),device = x_train.device)
                short_lag, long_lag = torch.sort(lags)[0]
            #decide what noise levels to predict:
            if self.config.get("regression_target") == "pred_x_t_noise_level":
                pred_noise, pred_s, true_s , true_s_prime = self.model(x_train, noise, short_lag, long_lag)
                loss = self.criterion(pred_noise,noise_t)
                loss += self.lambda_c * F.cross_entropy(pred_s, true_s)
            elif self.config.get("regression_target") == "pred_x_t_short_lag_noise_level":
                pred_noise, pred_s_prime, true_s , true_s_prime = self.model(x_train, noise, short_lag, long_lag)
                loss = self.criterion(pred_noise,noise_t)
                loss += self.lambda_c * F.cross_entropy(pred_s_prime, true_s_prime)
            elif self.config.get("regression_target") == "pred_both_noise_level":
                pred_noise, pred_s_short, pred_s_t, true_s_short , true_s_t = self.model(x_train, noise, short_lag, long_lag)
                loss = self.criterion(pred_noise,noise)
                loss += self.lambda_c * F.cross_entropy(pred_s_t, true_s_t)
                loss += self.lambda_c * F.cross_entropy(pred_s_short, true_s_short)
            loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss"]=loss.item()
                log_dic["training_step"]=self.training_step
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1
            if self.ema:
                self.ema.update()
        return loss
    def valid_loop(self):
        log_dic = {}
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        epoch_loss = 0
        nsamp = 0
        with torch.no_grad():
            for x_valid in self.valid_loader:
                x_valid = x_valid.to(self.gpu_id)
                nsamp += x_valid.shape[0]
                loss = 0
                
                noise = torch.randn((x_valid.shape[0],4, x_valid.shape[3],x_valid.shape[4]),device = x_valid.device)
                if self.config.get("fix_short_lag") is not None:
                    short_lag = torch.tensor(self.config["fix_short_lag"],dtype=torch.int64, device=x_valid.device)
                    long_lag = torch.randint(1, self.config["lagsteps"], (), device=x_valid.device)
                else:
                    lags = torch.randint(1, self.config["lagsteps"] - 1, (2,),device = x_valid.device)
                    short_lag, long_lag = torch.sort(lags)[0]
                
                if self.config.get("regression_target") == "pred_x_t_noise_level":
                    pred_noise, pred_s, true_s , true_s_prime = self.model(x_valid, noise_t, short_lag, long_lag)
                    loss = self.criterion(pred_noise,noise_t)
                    loss += self.lambda_c * F.cross_entropy(pred_s, true_s)
                elif self.config.get("regression_target") == "pred_x_t_short_lag_noise_level":
                    pred_noise, pred_s_prime, true_s , true_s_prime = self.model(x_valid, noise_t, short_lag, long_lag)
                    loss = self.criterion(pred_noise,noise_t)
                    loss += self.lambda_c * F.cross_entropy(pred_s_prime, true_s_prime)
                elif self.config.get("regression_target") == "pred_both_noise_level":
                    pred_noise, pred_s_short, pred_s_t, true_s_short , true_s_t = self.model(x_valid, noise, short_lag, long_lag)
                    loss = self.criterion(pred_noise,noise)
                    loss += self.lambda_c * F.cross_entropy(pred_s_short, true_s_short)
                    loss += self.lambda_c * F.cross_entropy(pred_s_t, true_s_t)
                epoch_loss+=loss.detach()*x_valid.shape[0]
        epoch_loss/=nsamp
        if self.ddp:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            self.val_loss = epoch_loss.item()/self.world_size
        else:
            self.val_loss = epoch_loss.item()
        if self.logging:
            log_dic={}
            log_dic["valid_loss"]=self.val_loss ## Average over full epoch
            log_dic["training_step"]=self.training_step
            wandb.log(log_dic)
        if self.ema:
            self.ema.restore()
        return self.val_loss
        
class SAD_NS_Trainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        self.lambda_c=config["regression_loss_weight"]
        self.softmax = nn.Softmax(dim=1)

    def _prep_model(self):
        model_unet = misc.model_factory(self.config).to(self.gpu_id)
        self.model = SAD_NS.SAD_NS(self.config, model=model_unet).to(self.gpu_id)
    
        self.config["cnn learnable parameters"] = sum(p.numel() for p in self.model.parameters())
    
        if self.config.get("ema_decay"):
            self.ema = misc.ExponentialMovingAverage(self.model, decay=self.config.get("ema_decay"))
            self.ema.register()
    
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.gpu_id], output_device=self.gpu_id, find_unused_parameters=False)
            
    def training_loop(self):
        self.model.train()
        for j,x_train in enumerate(self.train_loader):
            x_train = x_train.to(self.gpu_id)
            self.optimizer.zero_grad()
            
            noise = torch.randn((x_train.shape[0],2,x_train.shape[2],x_train.shape[3]),device = x_train.device)
            pred_noise, pred_s_short_lag, pred_s, true_s_short_lag , true_s = self.model(x_train, noise)
            loss = self.criterion(pred_noise,noise)
            loss += self.lambda_c * F.cross_entropy(pred_s_short_lag, true_s_short_lag)
            loss += self.lambda_c * F.cross_entropy(pred_s, true_s)
            
            loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss"]=loss.item()
                log_dic["training_step"]=self.training_step
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1
            if self.ema:
                self.ema.update()
        return loss
    def valid_loop(self):
        log_dic = {}
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        epoch_loss = 0
        nsamp = 0
        with torch.no_grad():
            for x_valid in self.valid_loader:
                x_valid = x_valid.to(self.gpu_id)
                nsamp += x_valid.shape[0]
                loss = 0
                
                noise = torch.randn((x_valid.shape[0],2,x_valid.shape[2],x_valid.shape[3]),device = x_valid.device)
                pred_noise, pred_s_short_lag, pred_s, true_s_short_lag , true_s = self.model(x_valid, noise)
                loss = self.criterion(pred_noise,noise)
                loss += self.lambda_c * F.cross_entropy(pred_s_short_lag, true_s_short_lag)
                loss += self.lambda_c * F.cross_entropy(pred_s, true_s)
                epoch_loss+=loss.detach()*x_valid.shape[0]
            epoch_loss/=nsamp
            if self.ddp:
                dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
                self.val_loss = (epoch_loss.item() / self.world_size)
            else:
                self.val_loss = epoch_loss.item()
            if self.logging:
                log_dic={}
                log_dic["valid_loss"]=self.val_loss ## Average over full epoch
                log_dic["training_step"]=self.training_step
                wandb.log(log_dic)
            if self.ema:
                self.ema.restore()
        return self.val_loss


class SNS_NS_Trainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        self.lambda_c=config["regression_loss_weight"]
        self.softmax = nn.Softmax(dim=1)

    def _prep_model(self):
        model_unet = misc.model_factory(self.config).to(self.gpu_id)
        self.model = SNS_NS.SNS_NS(self.config, model=model_unet).to(self.gpu_id)
    
        self.config["cnn learnable parameters"] = sum(p.numel() for p in self.model.parameters())
    
        if self.config.get("ema_decay"):
            self.ema = misc.ExponentialMovingAverage(self.model, decay=self.config.get("ema_decay"))
            self.ema.register()
    
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.gpu_id], output_device=self.gpu_id, find_unused_parameters=False)
            
    def training_loop(self):
        self.model.train()
        for j,x_train in enumerate(self.train_loader):
            x_train = x_train.to(self.gpu_id)
            self.optimizer.zero_grad()
            
            noise = torch.randn((x_train.shape[0],2,x_train.shape[2],x_train.shape[3]),device = x_train.device)
            pred_noise, pred_s_short_lag, pred_s, true_s_short_lag , true_s = self.model(x_train, noise)
            loss = self.criterion(pred_noise,noise)
            loss += self.lambda_c * F.cross_entropy(pred_s_short_lag, true_s_short_lag)
            loss += self.lambda_c * F.cross_entropy(pred_s, true_s)
            
            loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss"]=loss.item()
                log_dic["training_step"]=self.training_step
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1
            if self.ema:
                self.ema.update()
        return loss
    def valid_loop(self):
        log_dic = {}
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        epoch_loss = 0
        nsamp = 0
        with torch.no_grad():
            for x_valid in self.valid_loader:
                x_valid = x_valid.to(self.gpu_id)
                nsamp += x_valid.shape[0]
                loss = 0
                
                noise = torch.randn((x_valid.shape[0],2,x_valid.shape[2],x_valid.shape[3]),device = x_valid.device)
                pred_noise, pred_s_short_lag, pred_s, true_s_short_lag , true_s = self.model(x_valid, noise)
                loss = self.criterion(pred_noise,noise)
                loss += self.lambda_c * F.cross_entropy(pred_s_short_lag, true_s_short_lag)
                loss += self.lambda_c * F.cross_entropy(pred_s, true_s)
                epoch_loss+=loss.detach()*x_valid.shape[0]
            epoch_loss/=nsamp
            if self.ddp:
                dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
                self.val_loss = (epoch_loss.item() / self.world_size)
            else:
                self.val_loss = epoch_loss.item()
            if self.logging:
                log_dic={}
                log_dic["valid_loss"]=self.val_loss ## Average over full epoch
                log_dic["training_step"]=self.training_step
                wandb.log(log_dic)
            if self.ema:
                self.ema.restore()
        return self.val_loss

class Classifier_Trainer(Trainer):
    """
    Trainer for the double-head noise-level classifier.

    Assumptions:
      - The dataset returns x with shape (B, 2, 64, 64) for Kolmogorov
        or (B, 2, H, W) generally, where:
          x[:, 0:1] = x_short
          x[:, 1:2] = x_t
      - The model created by misc.model_factory(config) returns:
          logits_short, logits_t = model(x_cond)
        where:
          logits_short shape: (B, diffusion_steps) (or num_classes_1)
          logits_t     shape: (B, diffusion_steps) (or num_classes_2)
      - config contains:
          config["diffusion_steps"]
        and optionally:
          config["regression_output_dim_1"], config["regression_output_dim_2"]
        (otherwise both heads assumed to be diffusion_steps classes)
    """
    def __init__(self, config):
        super().__init__(config)
        if self.config["ddp"] is True:
            raise NotImplementedError("DDP not implemented for this trainer (matches your other trainers).")

        self.diffusion_steps = int(self.config["diffusion_steps"])
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=1)

        # Precompute cosine schedule buffers on GPU
        betas = self._cosine_variance_schedule(self.diffusion_steps).to(self.gpu_id)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def _prep_model(self):
        self.model = misc.model_factory(self.config).to(self.gpu_id)
        self.config["cnn learnable parameters"] = sum(p.numel() for p in self.model.parameters())

        if self.config.get("ema_decay"):
            self.ema = misc.ExponentialMovingAverage(self.model, decay=self.config.get("ema_decay"))
            self.ema.register()

    def _prep_optimizer(self):
        # CrossEntropy for classification of noise level indices
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["optimization"]["lr"],
            weight_decay=self.config["optimization"]["wd"],
        )
        if self.config["optimization"].get("scheduler_step"):
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                self.config["optimization"]["scheduler_step"],
                gamma=self.config["optimization"]["scheduler_gamma"],
                last_epoch=-1,
            )
        else:
            self.scheduler = None

    @staticmethod
    def _cosine_variance_schedule(timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * torch.pi * 0.5) ** 2
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas

    def _forward_diffusion(self, x_0, t, noise=None):
        """
        x_0:   (B,1,H,W)
        t:     (B,) int64
        noise: (B,1,H,W)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        s1 = self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1)
        return s1 * x_0 + s2 * noise

    def training_loop(self):
        self.model.train()
        for j, x_train in enumerate(self.train_loader):
            x_train = x_train.to(self.gpu_id)
            self.optimizer.zero_grad()

            # Expect (B,2,H,W)
            x_short = x_train[:, 0:1]
            x_t = x_train[:, 1:2]

            B = x_train.shape[0]
            s_short = torch.randint(0, self.diffusion_steps, (B,), device=x_train.device, dtype=torch.int64)
            s_t = torch.randint(0, self.diffusion_steps, (B,), device=x_train.device, dtype=torch.int64)

            noise_short = torch.randn_like(x_short)
            noise_t = torch.randn_like(x_t)

            x_short_noised = self._forward_diffusion(x_short, s_short, noise_short)
            x_t_noised = self._forward_diffusion(x_t, s_t, noise_t)

            # (B,2,H,W)
            x_cond = torch.cat((x_short_noised, x_t_noised), dim=1)

            # Model outputs logits for both heads
            logits_short, logits_t = self.model(x_cond)

            loss_short = self.criterion(logits_short, s_short)
            loss_t = self.criterion(logits_t, s_t)
            loss = (loss_short + loss_t)

            loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            if self.ema:
                self.ema.update()

            if self.logging and (self.training_step % self.log_freq == 0):
                log_dic = {
                    "train_loss": loss.item(),
                    "train_loss_short": loss_short.item(),
                    "train_loss_t": loss_t.item(),
                    "training_step": self.training_step,
                }
                if self.scheduler:
                    log_dic["lr"] = self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)

            self.training_step += 1

        return loss

    def valid_loop(self):
        log_dic = {}
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()

        epoch_loss = 0.0
        epoch_loss_short = 0.0
        epoch_loss_t = 0.0
        nsamp = 0

        with torch.no_grad():
            for x_valid in self.valid_loader:
                x_valid = x_valid.to(self.gpu_id)
                nsamp += x_valid.shape[0]

                x_short = x_valid[:, 0:1]
                x_t = x_valid[:, 1:2]

                B = x_valid.shape[0]
                s_short = torch.randint(0, self.diffusion_steps, (B,), device=x_valid.device, dtype=torch.int64)
                s_t = torch.randint(0, self.diffusion_steps, (B,), device=x_valid.device, dtype=torch.int64)

                noise_short = torch.randn_like(x_short)
                noise_t = torch.randn_like(x_t)

                x_short_noised = self._forward_diffusion(x_short, s_short, noise_short)
                x_t_noised = self._forward_diffusion(x_t, s_t, noise_t)

                x_cond = torch.cat((x_short_noised, x_t_noised), dim=1)

                logits_short, logits_t = self.model(x_cond)

                loss_short = self.criterion(logits_short, s_short)
                loss_t = self.criterion(logits_t, s_t)
                loss = (loss_short + loss_t)

                epoch_loss += loss.detach() * B
                epoch_loss_short += loss_short.detach() * B
                epoch_loss_t += loss_t.detach() * B

        epoch_loss /= nsamp
        epoch_loss_short /= nsamp
        epoch_loss_t /= nsamp

        if self.ddp:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_loss_short, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_loss_t, op=dist.ReduceOp.SUM)
            self.val_loss = epoch_loss.item() / self.world_size
            val_loss_short = epoch_loss_short.item() / self.world_size
            val_loss_t = epoch_loss_t.item() / self.world_size
        else:
            self.val_loss = epoch_loss.item()
            val_loss_short = epoch_loss_short.item()
            val_loss_t = epoch_loss_t.item()

        if self.logging:
            log_dic = {
                "valid_loss": self.val_loss,
                "valid_loss_short": val_loss_short,
                "valid_loss_t": val_loss_t,
                "training_step": self.training_step,
            }
            wandb.log(log_dic)

        if self.ema:
            self.ema.restore()

        return self.val_loss
    def save_checkpoint(self, checkpoint_path):
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        save_dict = {
            "epoch": self.epoch,
            "training_step": self.training_step,
            "best_val_loss": self.best_val_loss,   # <-- add this
            "state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        if self.scheduler is not None:
            save_dict["scheduler_state_dict"] = self.scheduler.state_dict()  # optional but useful
    
        torch.save(save_dict, checkpoint_path)

    def run(self, epochs=None):
        if self.logging and self.wandb_init == False:
            self.init_wandb()
            # NOTE: some of your trainers have self.model.model.config; classifier likely does not.
            # Keep only if your model has .config or .model.config
            if hasattr(self.model, "config"):
                self.model.config = self.config
    
        if epochs:
            max_epochs = epochs
        else:
            max_epochs = self.config["optimization"]["epochs"]
    
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            print("Training at epoch", self.epoch)
    
            # (DDP epoch seeding if you ever enable it)
            if self.ddp and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.epoch)
    
            # ---- train / validate ----
            self.training_loop()
            val_loss = self.valid_loop()
    
            # ---- always save "last" ----
            if self.logging:
                last_path = self.config["save_path"] + "/checkpoint_last.p"
                self.save_checkpoint(last_path)
    
            # ---- EMA "last" ----
            if self.ema:
                self.ema.apply_shadow()
                if self.logging:
                    last_ema_path = self.config["save_path"] + "/checkpoint_last_ema.p"
                    self.save_checkpoint(last_ema_path)
                self.ema.restore()
    
            # ---- save "best" if improved ----
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
    
                if self.logging:
                    best_path = self.config["save_path"] + "/checkpoint_best.p"
                    self.save_checkpoint(best_path)
                    wandb.log({"best_valid_loss": self.best_val_loss, "epoch": self.epoch})
    
                # ---- EMA "best" ----
                if self.ema:
                    self.ema.apply_shadow()
                    if self.logging:
                        best_ema_path = self.config["save_path"] + "/checkpoint_best_ema.p"
                        self.save_checkpoint(best_ema_path)
                    self.ema.restore()
    
        print("DONE on rank", self.gpu_id)
        if self.ddp:
            cleanup()

class SNS_QG_Trainer(Trainer):
    def __init__(self,config):
        super().__init__(config)
        self.lambda_c=config["regression_loss_weight"]
        self.softmax = nn.Softmax(dim=1)

    def _prep_model(self):
        model_unet = misc.model_factory(self.config).to(self.gpu_id)
        self.model = SNS_QG.SNS_QG(self.config, model=model_unet).to(self.gpu_id)
        self.config["cnn learnable parameters"] = sum(p.numel() for p in self.model.parameters())
        if self.config.get("ema_decay"):
            self.ema = misc.ExponentialMovingAverage(self.model, decay=self.config.get("ema_decay"))
            self.ema.register()
        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.gpu_id], output_device=self.gpu_id, find_unused_parameters=False)
            
    def training_loop(self):
        self.model.train()
        for j,x_train in enumerate(self.train_loader):
            x_train = x_train.to(self.gpu_id)
            self.optimizer.zero_grad()
            
            noise = torch.randn((x_train.shape[0], 4, x_train.shape[3],x_train.shape[4]),device = x_train.device)
            pred_noise, pred_s_short_lag, pred_s, true_s_short_lag , true_s = self.model(x_train, noise)
            loss = self.criterion(pred_noise,noise)
            loss += self.lambda_c * F.cross_entropy(pred_s_short_lag, true_s_short_lag)
            loss += self.lambda_c * F.cross_entropy(pred_s, true_s)
            
            loss.backward()
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.gradient_clip)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            if self.logging and (self.training_step%self.log_freq==0):
                log_dic={}
                log_dic["train_loss"]=loss.item()
                log_dic["training_step"]=self.training_step
                if self.scheduler:
                    log_dic["lr"]=self.scheduler.get_last_lr()[-1]
                wandb.log(log_dic)
            self.training_step+=1
            if self.ema:
                self.ema.update()
        return loss
    def valid_loop(self):
        log_dic = {}
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        epoch_loss = 0
        nsamp = 0
        with torch.no_grad():
            for x_valid in self.valid_loader:
                x_valid = x_valid.to(self.gpu_id)
                nsamp += x_valid.shape[0]
                loss = 0
                
                noise = torch.randn((x_valid.shape[0],4, x_valid.shape[3],x_valid.shape[4]),device = x_valid.device)
                pred_noise, pred_s_short_lag, pred_s, true_s_short_lag , true_s = self.model(x_valid, noise)
                loss = self.criterion(pred_noise,noise)
                loss += self.lambda_c * F.cross_entropy(pred_s_short_lag, true_s_short_lag)
                loss += self.lambda_c * F.cross_entropy(pred_s, true_s)
                epoch_loss+=loss.detach()*x_valid.shape[0]
            epoch_loss/=nsamp
            if self.ddp:
                dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
                self.val_loss = (epoch_loss.item() / self.world_size)
            else:
                self.val_loss = epoch_loss.item()
            if self.logging:
                log_dic={}
                log_dic["valid_loss"]=self.val_loss ## Average over full epoch
                log_dic["training_step"]=self.training_step
                wandb.log(log_dic)
            if self.ema:
                self.ema.restore()
        return self.val_loss
