from typing import Optional
import torch
from torch import nn
import os
import math
import pickle
import CT.Models.misc as misc


class Residual_block(nn.Module):
    """Wide Residual Blocks used in modern Unet architectures."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        first_embedding=None,
        second_embedding=None,
        third_embedding=None,
        separate_activation=True,
    ):
        super().__init__()
        self.activation: nn.Module = misc.ACTIVATION_REGISTRY.get(activation, None)
        self.separate_activation = separate_activation
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode="circular"
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode="circular"
        )

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

        if first_embedding is not None:
            self.first_embedding = first_embedding
            self.first_dense1 = nn.Linear(first_embedding, out_channels)
            self.first_dense2 = nn.Linear(first_embedding, out_channels)
        else:
            self.first_embedding = None

        if second_embedding is not None:
            self.second_embedding = second_embedding
            self.second_dense1 = nn.Linear(second_embedding, out_channels)
            self.second_dense2 = nn.Linear(second_embedding, out_channels)
        else:
            self.second_embedding = None

        if third_embedding is not None:
            self.third_embedding = third_embedding
            self.third_dense1 = nn.Linear(third_embedding, out_channels)
            self.third_dense2 = nn.Linear(third_embedding, out_channels)
        else:
            self.third_embedding = None

    def forward(self, x: torch.Tensor, first=None, second=None, third=None):
        if self.first_embedding is not None and first is None:
            raise ValueError("`first` must be provided when first_embedding is set.")
        if self.second_embedding is not None and second is None:
            raise ValueError("`second` must be provided when second_embedding is set.")
        if self.third_embedding is not None and third is None:
            raise ValueError("`third` must be provided when third_embedding is set.")

        h = self.conv1(self.activation(self.norm1(x)))

        if self.separate_activation:
            if self.first_embedding is not None:
                h += self.first_dense1(first)[:, :, None, None]
                h = self.activation(h)
            if self.second_embedding is not None:
                h += self.second_dense1(second)[:, :, None, None]
                h = self.activation(h)
            if self.third_embedding is not None:
                h += self.third_dense1(third)[:, :, None, None]
                h = self.activation(h)

            h = self.conv2(self.activation(self.norm2(h)))

            if self.first_embedding is not None:
                h += self.first_dense2(first)[:, :, None, None]
                h = self.activation(h)
            if self.second_embedding is not None:
                h += self.second_dense2(second)[:, :, None, None]
                h = self.activation(h)
            if self.third_embedding is not None:
                h += self.third_dense2(third)[:, :, None, None]
                h = self.activation(h)
        else:
            if self.first_embedding is not None:
                h += self.first_dense1(first)[:, :, None, None]
            if self.second_embedding is not None:
                h += self.second_dense1(second)[:, :, None, None]
            if self.third_embedding is not None:
                h += self.third_dense1(third)[:, :, None, None]

            h = self.conv2(self.activation(self.norm2(h)))

            if self.first_embedding is not None:
                h += self.first_dense2(first)[:, :, None, None]
            if self.second_embedding is not None:
                h += self.second_dense2(second)[:, :, None, None]
            if self.third_embedding is not None:
                h += self.third_dense2(third)[:, :, None, None]

            h = self.activation(h)

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
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)  # [B, HW, C]
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
        first_embedding=None,
        second_embedding=None,
        third_embedding=None,
        separate_activation=True,
    ):
        super().__init__()
        self.res = Residual_block(
            in_channels,
            out_channels,
            activation=activation,
            norm=norm,
            first_embedding=first_embedding,
            second_embedding=second_embedding,
            third_embedding=third_embedding,
            separate_activation=separate_activation,
        )
        self.attn = Attention_block(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, first=None, second=None, third=None):
        x = self.res(x, first, second, third)
        x = self.attn(x)
        return x


class Middle_block(nn.Module):
    def __init__(
        self,
        n_channels: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
        first_embedding=None,
        second_embedding=None,
        third_embedding=None,
        separate_activation=True,
    ):
        super().__init__()
        self.res1 = Residual_block(
            n_channels,
            n_channels,
            activation=activation,
            norm=norm,
            first_embedding=first_embedding,
            second_embedding=second_embedding,
            third_embedding=third_embedding,
            separate_activation=separate_activation,
        )
        self.attn = Attention_block(n_channels) if has_attn else nn.Identity()
        self.res2 = Residual_block(
            n_channels,
            n_channels,
            activation=activation,
            norm=norm,
            first_embedding=first_embedding,
            second_embedding=second_embedding,
            third_embedding=third_embedding,
            separate_activation=separate_activation,
        )

    def forward(self, x: torch.Tensor, first=None, second=None, third=None):
        x = self.res1(x, first, second, third)
        x = self.attn(x)
        x = self.res2(x, first, second, third)
        return x


class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1), padding_mode="circular")

    def forward(self, x: torch.Tensor, first=None, second=None, third=None):
        return self.conv(x)


class Regressor_block(nn.Module):
    def __init__(
        self,
        mid_channels: int,
        mid_width: int,
        mid_height: int,
        mlp_dim: int = 256,
        out_dim: int = 1,
        mlp_max: int = 4096,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
    ):
        super().__init__()
        self.intermediate_filters = round(mlp_max / (mid_width * mid_height))
        self.vector_size = (mid_width * mid_height) * self.intermediate_filters

        self.activation: nn.Module = misc.ACTIVATION_REGISTRY.get(activation, None)
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode="circular")
        self.conv2 = nn.Conv2d(mid_channels, self.intermediate_filters, kernel_size=(1, 1), padding_mode="circular")
        self.linear1 = nn.Linear(self.vector_size, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, mid_channels)
            self.norm2 = nn.GroupNorm(n_groups, self.intermediate_filters)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x):
        x = self.conv1(self.activation(self.norm1(x)))
        x = self.conv2(self.activation(self.norm2(x)))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x


class Classifier(nn.Module):
    """
    Your original Classifier, modified to output TWO noise-level predictions
    (two independent regressor heads).
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.n_input_channels = self.config["input_channels"]
        self.hidden_channels = self.config["hidden_channels"]
        self.activation: nn.Module = misc.ACTIVATION_REGISTRY.get(self.config["activation"], None)

        self.n_resolutions = len(self.config["dim_mults"])
        self.n_channels = self.config["hidden_channels"]

        # geometry at bottleneck (keep your original logic)
        self.mid_width = int(self.config["image_width"] / (2 * (len(self.config["dim_mults"]) - 1)))
        self.mid_height = int(self.config["image_height"] / (2 * (len(self.config["dim_mults"]) - 1)))
        self.mid_channels = self.config["hidden_channels"] * math.prod(self.config["dim_mults"])

        # Two output dims (fallback to regression_output_dim)
        base_out = int(self.config["regression_output_dim"])
        self.out_features_1 = int(self.config.get("regression_output_dim_1", base_out))
        self.out_features_2 = int(self.config.get("regression_output_dim_2", base_out))

        self.regressor_block_1 = Regressor_block(
            self.mid_channels, self.mid_width, self.mid_height, out_dim=self.out_features_1,
            activation=self.config["activation"], norm=self.config["norm"]
        )
        self.regressor_block_2 = Regressor_block(
            self.mid_channels, self.mid_width, self.mid_height, out_dim=self.out_features_2,
            activation=self.config["activation"], norm=self.config["norm"]
        )

        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=1)

        # Project image into feature map
        self.image_proj = nn.Conv2d(
            self.n_input_channels, self.n_channels, kernel_size=(3, 3), padding=(1, 1), padding_mode="circular"
        )
        self.normBool = self.config["norm"]
        self.mid_attn = self.config["mid_attn"]
        self.is_attn = self.config["is_attn"]

        # Embeddings (conditioning into residual blocks)
        self.first_embedding = self.config.get("first_embedding")
        if self.first_embedding is not None:
            self.first_mlp1 = nn.Linear(self.first_embedding, self.first_embedding)
            self.first_mlp2 = nn.Linear(self.first_embedding, self.first_embedding)

        self.second_embedding = self.config.get("second_embedding")
        if self.second_embedding is not None:
            self.second_mlp1 = nn.Linear(self.second_embedding, self.second_embedding)
            self.second_mlp2 = nn.Linear(self.second_embedding, self.second_embedding)

        self.third_embedding = self.config.get("third_embedding")
        if self.third_embedding is not None:
            self.third_mlp1 = nn.Linear(self.third_embedding, self.third_embedding)
            self.third_mlp2 = nn.Linear(self.third_embedding, self.third_embedding)

        # Down path
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
                        first_embedding=self.first_embedding,
                        second_embedding=self.second_embedding,
                        third_embedding=self.third_embedding,
                    )
                )
                in_channels = out_channels
            if i < self.n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)
        self.middle = Middle_block(
            out_channels,
            has_attn=self.mid_attn,
            activation=self.config["activation"],
            norm=self.normBool,
            first_embedding=self.first_embedding,
            second_embedding=self.second_embedding,
            third_embedding=self.third_embedding,
        )

    def _embed_conditioning(self, first=None, second=None, third=None):
        if self.first_embedding is not None:
            first = misc.get_timestep_embedding(first, self.first_embedding)
            first = self.activation(self.first_mlp1(first))
            first = self.activation(self.first_mlp2(first))
        if self.second_embedding is not None:
            second = misc.get_timestep_embedding(second, self.second_embedding)
            second = self.activation(self.second_mlp1(second))
            second = self.activation(self.second_mlp2(second))
        if self.third_embedding is not None:
            third = misc.get_timestep_embedding(third, self.third_embedding)
            third = self.activation(self.third_mlp1(third))
            third = self.activation(self.third_mlp2(third))
        return first, second, third

    def forward(
        self,
        x: torch.Tensor,
        first=None,
        second=None,
        third=None,
        return_probs: bool = False,
    ):
        """
        Returns:
          - if return_probs=False: (logits_1, logits_2)
          - if return_probs=True : (prob_1, prob_2)
        """
        x = self.image_proj(x)
        first, second, third = self._embed_conditioning(first, second, third)

        for m in self.down:
            x = m(x, first, second, third)

        x = self.middle(x, first, second, third)

        logits_1 = self.regressor_block_1(x)
        logits_2 = self.regressor_block_2(x)

        if return_probs:
            prob_1 = self.softmax_1(logits_1)
            prob_2 = self.softmax_2(logits_2)
            return prob_1, prob_2

        return logits_1, logits_2

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
        """
        head=1 uses regressor_block_1, head=2 uses regressor_block_2
        """
        x = self.image_proj(x)
        first, second, third = self._embed_conditioning(first, second, third)

        for m in self.down:
            x = m(x, first, second, third)

        x = self.middle(x, first, second, third)

        if head == 1:
            logits = self.regressor_block_1(x)
            return self.softmax_1(logits) if apply_softmax else logits
        elif head == 2:
            logits = self.regressor_block_2(x)
            return self.softmax_2(logits) if apply_softmax else logits
        else:
            raise ValueError(f"head must be 1 or 2, got {head}")

    @torch.no_grad()
    def ratio_class(self, x: torch.Tensor, head: int = 1, first=None, second=None, third=None):
        probs = self.ratio_class_distribution(x, head=head, first=first, second=second, third=third, apply_softmax=True)
        return probs.argmax(1)

    @torch.no_grad()
    def ratio_class_both(self, x: torch.Tensor, first=None, second=None, third=None):
        p1 = self.ratio_class_distribution(x, head=1, first=first, second=second, third=third, apply_softmax=True)
        p2 = self.ratio_class_distribution(x, head=2, first=first, second=second, third=third, apply_softmax=True)
        c1 = p1.argmax(1)
        c2 = p2.argmax(1)
        return torch.stack([c1, c2], dim=1)

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
