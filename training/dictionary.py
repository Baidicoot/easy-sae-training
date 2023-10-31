import torch
import torch.nn as nn

from torchtyping import TensorType

import copy
from typing import Literal, Callable

from torch.func import stack_module_state, vmap, functional_call

import torchopt

import training.ensemble as ens

SOFTPLUS_BETA: int = 10_000_000

class SparseLinearAutoencoder(nn.Module):
    def __init__(
        self,
        activation_size,
        n_dict_components,
        l1_penalty,
        activation: Literal["relu", "softplus"] = "relu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.center = nn.Parameter(torch.zeros((activation_size,), device=device, dtype=dtype))

        self.encoder = nn.Parameter(torch.empty((n_dict_components, activation_size), device=device, dtype=dtype))
        nn.init.xavier_uniform_(self.encoder)

        self.encoder_bias = nn.Parameter(torch.empty((n_dict_components,), device=device, dtype=dtype))
        nn.init.zeros_(self.encoder_bias)

        self.act: nn.Module
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "softplus":
            self.act = nn.Softplus(beta=SOFTPLUS_BETA)

        self.register_buffer("l1_penalty", torch.tensor(l1_penalty, device=device, dtype=dtype), persistent=True)
    
    def forward(self, batch):
        batch_ = batch - self.center[None, :]

        decoder_norms = torch.norm(self.encoder, 2, dim=-1)
        learned_dict = self.encoder / torch.clamp(decoder_norms, 1e-8)[:, None]

        c = torch.einsum("nd,bd->bn", learned_dict, batch_)
        c = c + self.encoder_bias[None, :]
        c = self.act(c)

        x_hat_ = torch.einsum("nd,bn->bd", learned_dict, c)

        x_hat = x_hat_ + self.center[None, :]

        l_reconstruction = (x_hat - batch).pow(2).mean()
        l_l1 = self.l1_penalty * torch.norm(c, 1, dim=-1).mean()

        bias_norm = torch.norm(self.encoder_bias, 2)
        center_norm = torch.norm(self.center, 2)
        
        return l_reconstruction + l_l1, l_reconstruction, c, x_hat, bias_norm, center_norm

def make_ensemble(input_dim, hidden_dim, l1_range, adam_settings, activation="relu", device="cuda"):
    # create a list of models
    models = []
    for l1_penalty in l1_range:
        models.append(SparseLinearAutoencoder(input_dim, hidden_dim, l1_penalty, device=device, dtype=torch.float32))

    ensemble = ens.Ensemble(
        models,
        optimizer_func=torchopt.adam,
        optimizer_kwargs=adam_settings,
        model_hyperparams={"activation_size": input_dim, "n_dict_components": hidden_dim, "l1_penalty": 0, "activation": activation},
    )

    return ensemble