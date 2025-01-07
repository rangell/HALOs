"""
Contains the code necessary to augment a model with unsafe steering vector. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import rank0_print, on_rank0, delete_dict

from IPython import embed


class UnsafeSteeringVectorLayer(nn.Module):
    def __init__(self, unsafe_direction_init: torch.Tensor):
        super(UnsafeSteeringVectorLayer, self).__init__()
        self.unsafe_direction = unsafe_direction_init

    def forward(self, x):
        _normed_unsafe_direction = F.normalize(self.unsafe_direction, p=2, dim=0)
        x = x - torch.einsum("bl,d->bld", F.gelu(torch.matmul(x, _normed_unsafe_direction)), _normed_unsafe_direction)
        return x


class TiedUnsafeSteeringVectorLayer(nn.Module):
    def __init__(self, tied_module: nn.Module):
        super(TiedUnsafeSteeringVectorLayer, self).__init__()
        self.tied_module = tied_module

    def forward(self, x):
        _normed_unsafe_direction = F.normalize(self.tied_module.unsafe_direction, p=2, dim=0)
        x = x - torch.einsum("bl,d->bld", F.gelu(torch.matmul(x, _normed_unsafe_direction)), _normed_unsafe_direction)
        return x


def get_steering_model(model):
    # we only want to train the steering vector to find the unsafe direction
    for param in model.parameters():
        param.requires_grad = False

    # for some reason we need to do this? not sure why
    model.enable_input_require_grads()

    unsafe_direction_init = nn.Parameter(
        torch.empty(model.config.hidden_size, dtype=model.config.torch_dtype).uniform_(-0.1, 0.1),
        requires_grad=True)

    # add steering layers
    steering_layers = []
    for i in range(model.config.num_hidden_layers // 2, model.config.num_hidden_layers):
        if len(steering_layers) == 0:
            steering_layers.append(UnsafeSteeringVectorLayer(unsafe_direction_init=unsafe_direction_init))
        else:
            steering_layers.append(TiedUnsafeSteeringVectorLayer(steering_layers[0]))
        #steering_layers.append(UnsafeSteeringVectorLayer(unsafe_direction_init=unsafe_direction_init))
        model.model.layers[i].mlp = nn.Sequential(model.model.layers[i].mlp, steering_layers[-1])
        
    return model