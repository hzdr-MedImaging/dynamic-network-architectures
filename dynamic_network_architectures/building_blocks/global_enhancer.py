from typing import Tuple, List, Union, Type
import torch.nn
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.building_blocks.regularization import DropPath
import numpy as np


class GlobalEnhancer(nn.Module):
    """
    This is inspired by self-attention mechanism as introduced in
    https://link.springer.com/article/10.1007/s00259-023-06197-1

    Often, in experiments, the attention matrix degraded to a constant, suggesting that it just need a global flow of
    information across the feature maps. This block implements just that: skips the attention matrix computation and
    just spreads "values" uniformly
    """
    def __init__(
            self, channels, conv_op, depth = 1, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=None, gate_layer=None):
        super(GlobalEnhancer, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = conv_op(channels + add_maxpool*channels, channels, kernel_size=1, bias=True)
        self.max_norm = nn.BatchNorm3d(channels)
        self.depth = depth
        self.extra_layers = nn.Sequential()
        for i in range(self.depth - 1):
            self.extra_layers.append(norm_layer(channels) if norm_layer else nn.Identity())
            self.extra_layers.append(act_layer(inplace=True))
            self.extra_layers.append(conv_op(channels, channels, kernel_size=1, bias=True))
        self.gate = gate_layer() if gate_layer else nn.Identity()

    def forward(self, x: torch.Tensor):
        x_signature = x.mean([*range(2,x.ndim)], keepdim=True) # collapse spatial dimensions
        if self.add_maxpool:
            # experimental codepath
            # x_max = nn.Sigmoid()(x.amax([*range(2,x.ndim)], keepdim=True))
            x_norm = self.max_norm(x)
            x_max = x_norm.amax([*range(2, x.ndim)], keepdim=True)
            x_signature = torch.concat([x_max, x_max], dim=1)
            # alternatively: first project, then max. Order does not matter for mean operation.
        x_ge = self.fc(x_signature)
        # if self.add_maxpool:
        #     x_signature = self.fc(x)
        #     x_ge = x_signature.amax([*range(2, x_signature.ndim)], keepdim=True)
        # else:
        #     x_signature = x.mean([*range(2, x.ndim)], keepdim=True)
        #     x_ge = self.fc(x_signature)
        x_ge = self.extra_layers(x_ge)
        return x + self.gate(x_ge)