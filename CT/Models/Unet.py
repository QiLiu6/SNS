from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import os
import math
import pickle
import sys
import CT.Models.misc as misc

class Residual_block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        first_embedding_dim = None,
        second_embedding_dim = None,
        third_embedding_dim = None,
        separate_activation = True,
    ):
        
        super().__init__()
        self.activation: nn.Module = misc.ACTIVATION_REGISTRY.get(activation, None)
        self.separate_activation = separate_activation
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode="circular")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode="circular")

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding_mode="circular")
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if first_embedding_dim is not None:
            self.first_embedding_dim = first_embedding_dim
            self.first_dense1 = nn.Linear(first_embedding_dim, out_channels)
            self.first_dense2 = nn.Linear(first_embedding_dim, out_channels)
        else:
            self.first_embedding_dim = None
            
        if second_embedding_dim is not None:
            self.second_embedding_dim = second_embedding_dim
            self.second_dense1 = nn.Linear(second_embedding_dim, out_channels)
            self.second_dense2 = nn.Linear(second_embedding_dim, out_channels)
        else:
            self.second_embedding_dim = None

        if third_embedding_dim is not None:
            self.third_embedding_dim = third_embedding_dim
            self.third_dense1 = nn.Linear(third_embedding_dim, out_channels)
            self.third_dense2 = nn.Linear(third_embedding_dim, out_channels)
        else:
            self.third_embedding_dim = None


        
    def forward(self, x: torch.Tensor, first = None, second = None, third = None):
        if self.first_embedding_dim is not None:
            if first is None:
                raise ValueError("`first` must be provided when first_embedding_dim is set.")
        if self.second_embedding_dim is not None:
            if second is None:
                raise ValueError("`second` must be provided when second_embedding_dim is set.")
        if self.third_embedding_dim is not None:
            if third is None:
                raise ValueError("`third` must be provided when third_embedding_dim is set.")
        
        h = self.conv1(self.activation(self.norm1(x)))
        if self.separate_activation:
            if self.first_embedding_dim is not None:
                h += self.first_dense1(first)[:, :, None, None]
                h = self.activation(h)
            if self.second_embedding_dim is not None:
                h += self.second_dense1(second)[:, :, None, None]
                h = self.activation(h)
            if self.third_embedding_dim is not None:
                h += self.third_dense1(third)[:, :, None, None]
                h = self.activation(h)
            # Second convolution layer
            h = self.conv2(self.activation(self.norm2(h)))
            if self.first_embedding_dim is not None:
                h += self.first_dense2(first)[:, :, None, None]
                h = self.activation(h)
            if self.second_embedding_dim is not None:
                h += self.second_dense2(second)[:, :, None, None]
                h = self.activation(h)
            if self.third_embedding_dim is not None:
                h += self.third_dense2(third)[:, :, None, None]
                h = self.activation(h)
        else:
            if self.first_embedding_dim is not None:
                h += self.first_dense1(first)[:, :, None, None]
            if self.second_embedding_dim is not None:
                h += self.second_dense1(second)[:, :, None, None]
            if self.third_embedding_dim is not None:
                h += self.third_dense1(third)[:, :, None, None]

            h = self.conv2(self.activation(self.norm2(h)))
            if self.first_embedding_dim is not None:
                h += self.first_dense2(first)[:, :, None, None]
            if self.second_embedding_dim is not None:
                h += self.second_dense2(second)[:, :, None, None]
            if self.third_embedding_dim is not None:
                h += self.third_dense2(third)[:, :, None, None]
            h = self.activation(h)
                
        # Add the shortcut connection and return
        return h + self.shortcut(x)

class Attention_block(nn.Module):

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: Optional[int] = None, n_groups: int = 1):
        super().__init__()
        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k**-0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = attn.softmax(dim=1)
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x

        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class Down_block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        first_embedding_dim = None,
        second_embedding_dim = None,
        third_embedding_dim = None,
        separate_activation = True,
    ):
        super().__init__()
        self.res = Residual_block(in_channels, out_channels, activation = activation, norm = norm, first_embedding_dim = first_embedding_dim, second_embedding_dim = second_embedding_dim, third_embedding_dim = third_embedding_dim, separate_activation = separate_activation)
        if has_attn:
            self.attn = Attention_block(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, first = None, second = None, third = None):
        x = self.res(x,first,second,third)
        x = self.attn(x)
        return x

class Middle_block(nn.Module):
    def __init__(self, n_channels: int, has_attn: bool = False, activation: str = "gelu", norm: bool = False, first_embedding_dim = None, second_embedding_dim = None, third_embedding_dim = None, separate_activation = True,):
        super().__init__()
        self.res1 = Residual_block(n_channels, n_channels, activation=activation, norm=norm, first_embedding_dim = first_embedding_dim, second_embedding_dim = second_embedding_dim, third_embedding_dim = third_embedding_dim, separate_activation = separate_activation)
        
        self.attn = Attention_block(n_channels) if has_attn else nn.Identity()
        
        self.res2 = Residual_block(n_channels, n_channels, activation=activation, norm=norm, first_embedding_dim = first_embedding_dim, second_embedding_dim = second_embedding_dim, third_embedding_dim = third_embedding_dim, separate_activation = separate_activation)

    def forward(self, x: torch.Tensor, first = None, second = None, third = None):
        x = self.res1(x,first,second,third)
        x = self.attn(x)
        x = self.res2(x,first,second,third)
        return x

class Up_block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        first_embedding_dim = None,
        second_embedding_dim = None,
        third_embedding_dim = None,
        separate_activation = True,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = Residual_block(in_channels + out_channels, out_channels, activation=activation, norm=norm, first_embedding_dim = first_embedding_dim, second_embedding_dim = second_embedding_dim, third_embedding_dim = third_embedding_dim, separate_activation = separate_activation)
        if has_attn:
            self.attn = Attention_block(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, first = None, second = None, third = None):
        x = self.res(x,first,second,third)
        x = self.attn(x)
        return x

class Upsample(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, first = None, second = None, third = None):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1), padding_mode="circular")

    def forward(self, x: torch.Tensor, first = None, second = None, third = None):
        return self.conv(x)

class Regressor_block(nn.Module):
    def __init__(self, mid_channels: int, mid_width: int, mid_height: int, mlp_dim: int=256, out_dim: int=1,
                         mlp_max: int=4096, activation: str = "gelu", norm: bool = False):
        super().__init__()

        self.intermediate_filters=round(mlp_max/(mid_width*mid_height))
        self.vector_size=(mid_width*mid_height)*self.intermediate_filters
        
        self.activation: nn.Module = misc.ACTIVATION_REGISTRY.get(activation, None)
        self.conv1=nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode="circular")
        self.conv2=nn.Conv2d(mid_channels, self.intermediate_filters, kernel_size=(1, 1), padding_mode="circular")
        self.linear1=nn.Linear(self.vector_size, out_dim)
        self.linear2=nn.Linear(out_dim, out_dim)

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, mid_channels)
            self.norm2 = nn.GroupNorm(n_groups, self.intermediate_filters)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self,x):
        x = self.conv1(self.activation(self.norm1(x)))
        x = self.conv2(self.activation(self.norm2(x)))
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x

class ModernUnet(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config
        self.n_input_channels = self.config["input_channels"]
        self.n_output_channels = self.config["output_channels"]
        self.hidden_channels = self.config["hidden_channels"]
        self.activation: nn.Module = misc.ACTIVATION_REGISTRY.get(self.config["activation"], None)
        self.n_resolutions = len(self.config["dim_mults"])
        self.n_channels = self.config["hidden_channels"]
        
        # Project image into feature map
        self.image_proj = nn.Conv2d(self.n_input_channels, self.n_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode="circular")
        self.normBool=self.config["norm"]
        self.mid_attn=self.config["mid_attn"]
        self.is_attn = self.config["is_attn"]
        
        self.first_embedding_dim = self.config.get("first_embedding_dim")
        if self.first_embedding_dim is not None:
            self.first_mlp1=nn.Linear(self.first_embedding_dim,self.first_embedding_dim)
            self.first_mlp2=nn.Linear(self.first_embedding_dim,self.first_embedding_dim)

        self.second_embedding_dim = self.config.get("second_embedding_dim")
        if self.second_embedding_dim is not None:
            self.second_mlp1=nn.Linear(self.second_embedding_dim,self.second_embedding_dim)
            self.second_mlp2=nn.Linear(self.second_embedding_dim,self.second_embedding_dim)

        self.third_embedding_dim = self.config.get("third_embedding_dim")
        if self.third_embedding_dim is not None:
            self.third_mlp1=nn.Linear(self.third_embedding_dim,self.third_embedding_dim)
            self.third_mlp2=nn.Linear(self.third_embedding_dim,self.third_embedding_dim)
      
        down = []
        out_channels = in_channels = self.n_channels

        for i in range(self.n_resolutions):
            out_channels = in_channels * self.config["dim_mults"][i]
            for _ in range(self.config["n_blocks"]):
                down.append(
                    Down_block(
                        in_channels,
                        out_channels,
                        has_attn=self.is_attn[i],
                        activation=self.config["activation"],
                        norm=self.normBool,
                        first_embedding_dim=self.first_embedding_dim,
                        second_embedding_dim=self.second_embedding_dim,
                        third_embedding_dim=self.third_embedding_dim,
                    )
                )
                in_channels = out_channels
            if i < self.n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)

        self.middle = Middle_block(out_channels, has_attn=self.mid_attn, activation=self.config["activation"],
                                  norm=self.normBool, first_embedding_dim=self.first_embedding_dim,second_embedding_dim=self.second_embedding_dim,third_embedding_dim=self.third_embedding_dim)

        up = []
        in_channels = out_channels
        for i in reversed(range(self.n_resolutions)):
            out_channels = in_channels
            for _ in range(self.config["n_blocks"]):
                up.append(
                    Up_block(
                        in_channels,
                        out_channels,
                        has_attn=self.is_attn[i],
                        activation=self.config["activation"],
                        norm=self.normBool,
                        first_embedding_dim=self.first_embedding_dim,
                        second_embedding_dim=self.second_embedding_dim,
                        third_embedding_dim=self.third_embedding_dim
                    )
                )
            out_channels = in_channels // self.config["dim_mults"][i]
            up.append(Up_block(in_channels, out_channels, has_attn=self.is_attn[i], activation=self.config["activation"],
                              norm=self.normBool, first_embedding_dim=self.first_embedding_dim,second_embedding_dim=self.second_embedding_dim,third_embedding_dim=self.third_embedding_dim))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)

        if self.normBool:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = self.n_output_channels
        #
        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), 
                            padding=(1, 1), padding_mode="circular")

    def forward(self, x: torch.Tensor, first = None, second = None, third = None):
        x = self.image_proj(x)
        if self.first_embedding_dim is not None:
            first = misc.get_timestep_embedding(first, self.first_embedding_dim)
            first = self.activation(self.first_mlp1(first))
            first = self.activation(self.first_mlp2(first))
        if self.second_embedding_dim is not None:
            second = misc.get_timestep_embedding(second, self.second_embedding_dim)
            second = self.activation(self.second_mlp1(second))
            second = self.activation(self.second_mlp2(second))
        if self.third_embedding_dim is not None:
            third = misc.get_timestep_embedding(third, self.third_embedding_dim)
            third = self.activation(self.third_mlp1(third))
            third = self.activation(self.third_mlp2(third))

        h = [x]
        for m in self.down:
            x = m(x,first,second,third)
            h.append(x)

        x = self.middle(x,first,second,third)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x,first,second,third)
            else:
                skip = h.pop()
                x = torch.cat((x, skip), dim=1)
                x = m(x,first,second,third)

        x = self.final(self.activation(self.norm(x)))
        return x
        
    def save_model(self):
        if self.config["save_path"] is None:
            print("No save path provided, not saving")
            return
        save_dict={}
        save_dict["state_dict"]=self.state_dict() ## Dict containing optimised weights and biases
        save_dict["config"]=self.config           ## Dict containing config for the dataset and model
        save_string=os.path.join(self.config["save_path"],self.config["save_name"])
        torch.save(save_dict,save_string)
        print("Model saved as %s" % save_string)
        return

class ModernUnetRegressor(ModernUnet):
    def __init__(self,config):
        super().__init__(config)
        self.mid_pix=int(self.config["image_width"]/(2*(len(self.config["dim_mults"])-1)))
        self.mid_channels=self.config["hidden_channels"]*math.prod(self.config["dim_mults"])
        self.out_features=self.config["regression_output_dim"]
        self.regressor_block=Regressor_block(self.mid_channels,self.mid_pix,self.mid_pix,out_dim=self.out_features)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, first = None, second = None, third = None, regression_output: bool=True):
        x = self.image_proj(x)
        if self.first_embedding_dim is not None:
            first = misc.get_timestep_embedding(first, self.first_embedding_dim)
            first = self.activation(self.first_mlp1(first))
            first = self.activation(self.first_mlp2(first))
        if self.second_embedding_dim is not None:
            second = misc.get_timestep_embedding(second, self.second_embedding_dim)
            second = self.activation(self.second_mlp1(second))
            second = self.activation(self.second_mlp2(second))
        if self.third_embedding_dim is not None:
            third = misc.get_timestep_embedding(third, self.third_embedding_dim)
            third = self.activation(self.third_mlp1(third))
            third = self.activation(self.third_mlp2(third))
            
        h = [x]
        for m in self.down:
            x = m(x,first,second,third)
            h.append(x)

        x = self.middle(x,first,second,third)
        if regression_output:
            s_prob_vector = self.regressor_block(x)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x,first,second,third)
            else:
                skip = h.pop()
                x = torch.cat((x, skip), dim=1)
                x = m(x,first,second,third)

        x = self.final(self.activation(self.norm(x)))
        if regression_output:
            return x, s_prob_vector
        else:
            return x

    @torch.no_grad()
    def ratio_class_distribution(self, x: torch.Tensor, first = None, second = None, third = None):
        x = self.image_proj(x)
        if self.first_embedding_dim is not None:
            first = misc.get_timestep_embedding(first, self.first_embedding_dim)
            first = self.activation(self.first_mlp1(first))
            first = self.activation(self.first_mlp2(first))
        if self.second_embedding_dim is not None:
            second = misc.get_timestep_embedding(second, self.second_embedding_dim)
            second = self.activation(self.second_mlp1(second))
            second = self.activation(self.second_mlp2(second))
        if self.third_embedding_dim is not None:
            third = misc.get_timestep_embedding(third, self.third_embedding_dim)
            third = self.activation(self.third_mlp1(third))
            third = self.activation(self.third_mlp2(third))
        
        for m in self.down:
            x = m(x,first,second,third)
        x = self.middle(x,first,second,third)
        x = self.softmax(self.regressor_block(x))
        return x

    @torch.no_grad()
    def ratio_class(self, x: torch.Tensor, first = None, second = None, third = None):

        x = self.ratio_class_distribution(x,first,second,third)
        return x.argmax(1)

class ModernUnetDoubleRegressor(ModernUnet):
    def __init__(self, config):
        # 1) Initialize base first
        super().__init__(config)

        # 4) Recompute bottleneck geometry for regressors (your existing logic, slightly cleaned)
        n_down = len(self.config["dim_mults"]) - 1
        self.mid_pix = self.config["image_width"] // (2 ** n_down)
        self.mid_channels = self.config["hidden_channels"] * math.prod(self.config["dim_mults"])

        self.out_features_1 = int(self.config.get("regression_output_dim_1", self.config["regression_output_dim"]))
        self.out_features_2 = int(self.config.get("regression_output_dim_2", self.config["regression_output_dim"]))

        self.regressor_block_1 = Regressor_block(self.mid_channels, self.mid_pix, self.mid_pix, out_dim=self.out_features_1)
        self.regressor_block_2 = Regressor_block(self.mid_channels, self.mid_pix, self.mid_pix, out_dim=self.out_features_2)

        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=1)
        
        down = []
        out_channels = in_channels = self.n_channels

        for i in range(self.n_resolutions):
            out_channels = in_channels * self.config["dim_mults"][i]
            for _ in range(self.config["n_blocks"]):
                down.append(
                    Down_block(
                        in_channels,
                        out_channels,
                        has_attn=self.is_attn[i],
                        activation=self.config["activation"],
                        norm=self.normBool,
                        first_embedding_dim=None,
                        second_embedding_dim=None,
                        third_embedding_dim=self.third_embedding_dim,
                    )
                )
                in_channels = out_channels
            if i < self.n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)

        self.middle = Middle_block(out_channels, has_attn=self.mid_attn, activation=self.config["activation"],
                                  norm=self.normBool, first_embedding_dim=None,second_embedding_dim=None,third_embedding_dim=self.third_embedding_dim)
        
        up = []
        in_channels = self.hidden_channels * math.prod(self.config["dim_mults"])
        for i in reversed(range(self.n_resolutions)):
            out_channels = in_channels
            for _ in range(self.config["n_blocks"]):
                up.append(
                    Up_block(
                        in_channels,
                        out_channels,
                        has_attn=self.is_attn[i],
                        activation=self.config["activation"],
                        norm=self.normBool,
                        first_embedding_dim=self.first_embedding_dim,
                        second_embedding_dim=self.second_embedding_dim,
                        third_embedding_dim=self.third_embedding_dim
                    )
                )
            out_channels = in_channels // self.config["dim_mults"][i]
            up.append(Up_block(in_channels, out_channels, has_attn=self.is_attn[i], activation=self.config["activation"],
                              norm=self.normBool, first_embedding_dim=self.first_embedding_dim,second_embedding_dim=self.second_embedding_dim,third_embedding_dim=self.third_embedding_dim))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)


    def forward(self, x, first=None, second=None, third=None, regression_output: bool = True):
        x = self.image_proj(x)
        
        if self.third_embedding_dim is not None:
            third = misc.get_timestep_embedding(third, self.third_embedding_dim)
            third = self.activation(self.third_mlp1(third))
            third = self.activation(self.third_mlp2(third))
            
        h = [x]
        for m in self.down:
            x = m(x, first, second, third)
            h.append(x)
    
        x = self.middle(x, first, second, third)
    
        # logits: (B, 1000)
        logits_short = self.regressor_block_1(x)
        logits_t     = self.regressor_block_2(x)

        prob_short = self.softmax_1(logits_short)  # (B, 1000)
        prob_t     = self.softmax_2(logits_t)
    
        # embeddings: (B, 1000)
        emb_short = self.activation(self.first_mlp1(prob_short))
        emb_short = self.activation(self.first_mlp2(emb_short))

        emb_t = self.activation(self.second_mlp1(prob_t))
        emb_t = self.activation(self.second_mlp2(emb_t))
    
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, emb_short, emb_t, third)
            else:
                skip = h.pop()
                x = torch.cat((x, skip), dim=1)
                x = m(x, emb_short, emb_t, third)
    
        x = self.final(self.activation(self.norm(x)))
    
        if regression_output:
            # Return both logits (for CE) and/or indices (for monitoring), plus denoised output
            return x, logits_short, logits_t
        return x


    @torch.no_grad()
    def ratio_class_distribution(
        self,
        x: torch.Tensor,
        head: int = 1,
        first=None,
        second=None,
        third=None,
        apply_softmax: bool = True,
    ):
        x = self.image_proj(x)
        if self.third_embedding_dim is not None:
            third = misc.get_timestep_embedding(third, self.third_embedding_dim)
            third = self.activation(self.third_mlp1(third))
            third = self.activation(self.third_mlp2(third))

        for m in self.down:
            x = m(x, first, second, third)

        x = self.middle(x, first, second, third)

        logits = self.regressor_block_1(x) if head == 1 else self.regressor_block_2(x)
        if apply_softmax:
            return self.softmax_1(logits) if head == 1 else self.softmax_2(logits)
        return logits

    @torch.no_grad()
    def ratio_class(
        self,
        x: torch.Tensor,
        head: int = 1,
        first=None,
        second=None,
        third=None,
    ):
        probs = self.ratio_class_distribution(x, head=head, first=first, second=second, third=third, apply_softmax=True)
        return probs.argmax(1)
        
    @torch.no_grad()
    def ratio_class_both(self, x, first=None, second=None, third=None):
        p1 = self.ratio_class_distribution(x, head=1, first=first, second=second, third=third)
        p2 = self.ratio_class_distribution(x, head=2, first=first, second=second, third=third)
    
        c1 = p1.argmax(1)  # (B,)
        c2 = p2.argmax(1)  # (B,)
    
        return torch.stack([c1, c2], dim=1)  # (B, 2)

class ModernUnetDoubleRegressor_plus(ModernUnet):
    def __init__(self, config):
        # 1) Initialize base first
        super().__init__(config)

        # 4) Recompute bottleneck geometry for regressors (your existing logic, slightly cleaned)
        n_down = len(self.config["dim_mults"]) - 1
        self.mid_pix = self.config["image_width"] // (2 ** n_down)
        self.mid_channels = self.config["hidden_channels"] * math.prod(self.config["dim_mults"])

        self.out_features_1 = int(self.config.get("regression_output_dim_1", self.config["regression_output_dim"]))
        self.out_features_2 = int(self.config.get("regression_output_dim_2", self.config["regression_output_dim"]))

        self.regressor_block_1 = Regressor_block(self.mid_channels, self.mid_pix, self.mid_pix, out_dim=self.out_features_1)
        self.regressor_block_2 = Regressor_block(self.mid_channels, self.mid_pix, self.mid_pix, out_dim=self.out_features_2)

        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=1)
        
        down = []
        out_channels = in_channels = self.n_channels

        for i in range(self.n_resolutions):
            out_channels = in_channels * self.config["dim_mults"][i]
            for _ in range(self.config["n_blocks"]):
                down.append(
                    Down_block(
                        in_channels,
                        out_channels,
                        has_attn=self.is_attn[i],
                        activation=self.config["activation"],
                        norm=self.normBool,
                        first_embedding_dim=None,
                        second_embedding_dim=None,
                        third_embedding_dim=None,
                    )
                )
                in_channels = out_channels
            if i < self.n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)

        self.middle = Middle_block(out_channels, has_attn=self.mid_attn, activation=self.config["activation"],
                                  norm=self.normBool, first_embedding_dim=None,second_embedding_dim=None,third_embedding_dim=None)
        
        up = []
        in_channels = self.hidden_channels * math.prod(self.config["dim_mults"])
        for i in reversed(range(self.n_resolutions)):
            out_channels = in_channels
            for _ in range(self.config["n_blocks"]):
                up.append(
                    Up_block(
                        in_channels,
                        out_channels,
                        has_attn=self.is_attn[i],
                        activation=self.config["activation"],
                        norm=self.normBool,
                        first_embedding_dim=self.first_embedding_dim,
                        second_embedding_dim=self.second_embedding_dim,
                        third_embedding_dim=self.third_embedding_dim
                    )
                )
            out_channels = in_channels // self.config["dim_mults"][i]
            up.append(Up_block(in_channels, out_channels, has_attn=self.is_attn[i], activation=self.config["activation"],
                              norm=self.normBool, first_embedding_dim=self.first_embedding_dim,second_embedding_dim=self.second_embedding_dim,third_embedding_dim=self.third_embedding_dim))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels))

        self.up = nn.ModuleList(up)


    def forward(self, x, first=None, second=None, third=None, regression_output: bool = True):
        x = self.image_proj(x)
    
        h = [x]
        for m in self.down:
            x = m(x)
            h.append(x)
    
        x = self.middle(x)
    
        # logits: (B, 1000)
        logits_short = self.regressor_block_1(x)
        logits_t     = self.regressor_block_2(x)

        prob_short = self.softmax_1(logits_short)  # (B, 1000)
        prob_t     = self.softmax_2(logits_t)
    
        # embeddings: (B, 1000)
        emb_short = self.activation(self.first_mlp1(prob_short))
        emb_short = self.activation(self.first_mlp2(emb_short))

        emb_t = self.activation(self.second_mlp1(prob_t))
        emb_t = self.activation(self.second_mlp2(emb_t))
    
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, emb_short, emb_t, third)
            else:
                skip = h.pop()
                x = torch.cat((x, skip), dim=1)
                x = m(x, emb_short, emb_t, third)
    
        x = self.final(self.activation(self.norm(x)))
    
        if regression_output:
            # Return both logits (for CE) and/or indices (for monitoring), plus denoised output
            return x, logits_short, logits_t
        return x


    @torch.no_grad()
    def ratio_class_distribution(
        self,
        x: torch.Tensor,
        head: int = 1,
        first=None,
        second=None,
        third=None,
        apply_softmax: bool = True,
    ):
        x = self.image_proj(x)

        for m in self.down:
            x = m(x)

        x = self.middle(x)

        logits = self.regressor_block_1(x) if head == 1 else self.regressor_block_2(x)
        if apply_softmax:
            return self.softmax_1(logits) if head == 1 else self.softmax_2(logits)
        return logits

    @torch.no_grad()
    def ratio_class(
        self,
        x: torch.Tensor,
        head: int = 1,
        first=None,
        second=None,
        third=None,
    ):
        probs = self.ratio_class_distribution(x, head=head, first=first, second=second, third=third, apply_softmax=True)
        return probs.argmax(1)
        
    @torch.no_grad()
    def ratio_class_both(self, x, first=None, second=None, third=None):
        p1 = self.ratio_class_distribution(x, head=1, first=first, second=second, third=third)
        p2 = self.ratio_class_distribution(x, head=2, first=first, second=second, third=third)
    
        c1 = p1.argmax(1)  # (B,)
        c2 = p2.argmax(1)  # (B,)
    
        return torch.stack([c1, c2], dim=1)  # (B, 2)
    
