import numpy as np
import io
import pickle
import xarray as xr
import math
import torch
from torch import nn
from tqdm import tqdm
import string

from torch.utils.data import DataLoader

import sys
import CT.Models.diffusion_regression as diffusion_regression
import CT.Models.Conditional_diffusion as Conditional_diffusion
import CT.Models.LSDM_NS as LSDM_NS
import CT.Models.LSDM_QG as LSDM_QG
import CT.Models.ACDM_NS as ACDM_NS
import CT.Models.ACDM_QG as ACDM_QG
import CT.Models.SAD_NS as SAD_NS
import CT.Models.Unet as Unet
import CT.Models.SNS_NS as SNS_NS
import CT.Models.SNS_QG as SNS_QG
import CT.Models.Classifier as Classifier
import CT.Models.MNLCD as MNLCD

### Model factory
def model_factory(config):
    """ Function to take a config dict, and return one of our nn.Modules """
    if config["model_type"]=="ModernUnet":
        return Unet.ModernUnet(config)
    elif config["model_type"]=="ModernUnetRegressor":
        return Unet.ModernUnetRegressor(config)
    elif config["model_type"]=="ModernUnetDoubleRegressor":
        return Unet.ModernUnetDoubleRegressor(config)
    elif config["model_type"]=="ModernUnetDoubleRegressor_plus":
        return Unet.ModernUnetDoubleRegressor_plus(config)
    elif config["model_type"]=="Classifier":
        return Classifier.Classifier(config)
    else:
        print("Model type not recognised")
        quit()

## Activation registry for resnet modules
ACTIVATION_REGISTRY = {
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
    "gelu": nn.GELU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
}


## Noise timestep embedding
def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    Retrieved from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py#LL90C1-L109C13
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.type(torch.float32)[:, None] * emb[None, :]
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)

    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    assert emb.shape == (timesteps.shape[0], embedding_dim), f"{emb.shape}"
    return emb

def load_MNLCD(file_string):
    model_dict = torch.load(file_string)
    model_cnn=Unet.ModernUnet(model_dict["config"])
    if model_dict["config"]["PDE"] == "Kolmogorov":
        diffusion_model=MNLCD.MNLCD(model_dict["config"], model=model_cnn)
    elif model_dict["config"]["PDE"] == "QG":
        diffusion_model=ACDM_QG.ACDM_QG(model_dict["config"], model=model_cnn)
    else:
        raise ValueError("PDE Type not recognized")
    diffusion_model.load_state_dict(model_dict["state_dict"])
    return diffusion_model


def load_Emulator(file_string):
    model_dict = torch.load(file_string)
    if model_dict["config"]["model_type"]=="ModernUnet":
        model=Unet.ModernUnet(model_dict["config"])
    model.load_state_dict(model_dict["state_dict"])
    return model
    
def load_LSDM(file_string):
    model_dict = torch.load(file_string)
    if model_dict["config"]["model_type"]=="ModernUnetRegressor":
        model_cnn=Unet.ModernUnetRegressor(model_dict["config"])
        if model_dict["config"]["PDE"] == "Kolmogorov":
            diffusion_model=LSDM_NS.LSDM_NS(model_dict["config"], model=model_cnn)
        elif model_dict["config"]["PDE"] == "QG":
            diffusion_model=LSDM_QG.LSDM_QG(model_dict["config"], model=model_cnn)
        else:
            raise ValueError("PDE Type not recognized")
    elif model_dict["config"]["model_type"]=="ModernUnetDoubleRegressor":
        model_cnn=Unet.ModernUnetDoubleRegressor(model_dict["config"])
        if model_dict["config"]["PDE"] == "Kolmogorov":
            diffusion_model=LSDM_NS.LSDM_NS(model_dict["config"], model=model_cnn)
        elif model_dict["config"]["PDE"] == "QG":
            diffusion_model=LSDM_QG.LSDM_QG(model_dict["config"], model=model_cnn)
        else:
            raise ValueError("PDE Type not recognized")
    diffusion_model.load_state_dict(model_dict["state_dict"])
    return diffusion_model

def load_SNS(file_string):
    model_dict = torch.load(file_string)
    if model_dict["config"]["model_type"]=="ModernUnetDoubleRegressor_plus":
        model_cnn=Unet.ModernUnetDoubleRegressor_plus(model_dict["config"])
        if model_dict["config"]["PDE"] == "Kolmogorov":
            diffusion_model=SNS_NS.SNS_NS(model_dict["config"], model=model_cnn)
        elif model_dict["config"]["PDE"] == "QG":
            diffusion_model=SNS_QG.SNS_QG(model_dict["config"], model=model_cnn)
        else:
            raise ValueError("PDE Type not recognized")
    elif model_dict["config"]["model_type"]=="ModernUnetDoubleRegressor":
        model_cnn=Unet.ModernUnetDoubleRegressor(model_dict["config"])
        if model_dict["config"]["PDE"] == "Kolmogorov":
            diffusion_model=SNS_NS.SNS_NS(model_dict["config"], model=model_cnn)
        elif model_dict["config"]["PDE"] == "QG":
            diffusion_model=SNS_QG.SNS_QG(model_dict["config"], model=model_cnn)
        else:
            raise ValueError("PDE Type not recognized")
    diffusion_model.load_state_dict(model_dict["state_dict"])
    return diffusion_model

def load_ACDM(file_string):
    model_dict = torch.load(file_string)
    model_cnn=Unet.ModernUnet(model_dict["config"])
    model_cnn.load_state_dict(model_dict["state_dict"])
    if model_dict["config"]["PDE"] == "Kolmogorov":
        diffusion_model=ACDM_NS.ACDM_NS(model_dict["config"], model=model_cnn)
    elif model_dict["config"]["PDE"] == "QG":
        diffusion_model=ACDM_QG.ACDM_QG(model_dict["config"], model=model_cnn)
    else:
        raise ValueError("PDE Type not recognized")
    return diffusion_model

def load_SAD(file_string):
    model_dict = torch.load(file_string)
    model_cnn=Unet.ModernUnetDoubleRegressor_plus(model_dict["config"])
    if model_dict["config"]["PDE"] == "Kolmogorov":
        diffusion_model=SAD_NS.SAD_NS(model_dict["config"], model=model_cnn)
    elif model_dict["config"]["PDE"] == "QG":
        diffusion_model=SAD_QG.SAD_QG(model_dict["config"], model=model_cnn)
    else:
        raise ValueError("PDE Type not recognized")
    diffusion_model.load_state_dict(model_dict["state_dict"])
    return diffusion_model


    
class ExponentialMovingAverage:
    def __init__(self, model, decay=0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self, overwrite=False):
        if len(self.shadow) > 0 and not overwrite:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data.detach() + self.decay * self.shadow[name]
                self.shadow[name] = new_average

    def apply_shadow(self):
        if len(self.shadow) == 0:
            print("Warning: EMA shadow is empty. Cannot apply shadow.")
        else:
            for name, param in self.model.named_parameters():
                if name in self.shadow:
                    self.backup[name] = param.data
                    param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        

def estimate_covmat(field_tensor,nsamp=None):
    """ Estimate covariance matrix from some tensor of fields. Can either be
        flattened to batched 1D tensors, or batched 2D fields 
        use nsamp to estimate covmat from a subsample of the data """

    ## If nsamp is not provided, use every sample in field_tensor
    if nsamp==None:
        nsamp=len(field_tensor)

    ## If the field tensor isn't flattened, flatten
    if len(field_tensor.shape)>2:
        field_tensor=field_tensor.reshape((len(field_tensor),64*64))

    ## Initialise covariance matrix
    cov=torch.zeros((64**2,64**2))

    for aa in tqdm(range(nsamp)):
        cov+=torch.outer(field_tensor[aa],field_tensor[aa])
    cov/=(nsamp-1)
    return cov


