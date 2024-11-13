# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi Layer Perceptron
"""

from typing import Union

from torch import nn

from examples.mlp.external import TCNN_EXISTS, tcnn


def activation_to_tcnn_string(activation: Union[nn.Module, None]) -> str:
    """Converts a torch.nn activation function to a string that can be used to
    initialize a TCNN activation function.

    Args:
        activation: torch.nn activation function
    Returns:
        str: TCNN activation function string
    """

    if isinstance(activation, nn.ReLU):
        return "ReLU"
    if isinstance(activation, nn.LeakyReLU):
        return "LeakyReLU"
    if isinstance(activation, nn.Sigmoid):
        return "Sigmoid"
    if isinstance(activation, nn.Softplus):
        return "Softplus"
    if isinstance(activation, nn.Tanh):
        return "Tanh"
    if isinstance(activation, type(None)):
        return "None"
    tcnn_documentation_url = "https://github.com/NVlabs/tiny-cuda-nn/blob/master/DOCUMENTATION.md#activation-functions"
    raise ValueError(
        f"TCNN activation {activation} not supported for now.\nSee {tcnn_documentation_url} for TCNN documentation."
    )


def get_tcnn_network_config(
    activation, out_activation, layer_width, num_layers
) -> dict:
    """Get the network configuration for tcnn if implemented"""
    activation_str = activation_to_tcnn_string(activation)
    output_activation_str = activation_to_tcnn_string(out_activation)
    assert layer_width in [16, 32, 64, 128]
    network_config = {
        "otype": "FullyFusedMLP",
        "activation": activation_str,
        "output_activation": output_activation_str,
        "n_neurons": layer_width,
        "n_hidden_layers": num_layers - 1,
    }
    return network_config


def create_mlp(
    in_dim: int,
    num_layers: int,
    layer_width: int,
    out_dim: int,
):
    if TCNN_EXISTS:
        return _create_mlp_tcnn(in_dim, num_layers, layer_width, out_dim)
    else:
        return _create_mlp_torch(in_dim, num_layers, layer_width, out_dim)


def _create_mlp_tcnn(
    in_dim: int,
    num_layers: int,
    layer_width: int,
    out_dim: int,
):
    """Create a fully-connected neural network with tiny-cuda-nn."""
    network_config = get_tcnn_network_config(
        activation=nn.LeakyReLU(),
        out_activation=None,
        layer_width=layer_width,
        num_layers=num_layers,
    )
    tcnn_encoding = tcnn.Network(
        n_input_dims=in_dim,
        n_output_dims=out_dim,
        network_config=network_config,
    )
    return tcnn_encoding


def _create_mlp_torch(
    in_dim: int,
    num_layers: int,
    layer_width: int,
    out_dim: int,
):
    """Create a fully-connected neural network with PyTorch."""
    layers = []
    layer_in = in_dim
    for i in range(num_layers):
        layer_out = layer_width if i != num_layers - 1 else out_dim
        layers.append(nn.Linear(layer_in, layer_out, bias=False))
        if i != num_layers - 1:
            layers.append(nn.LeakyReLU())
        layer_in = layer_width
    return nn.Sequential(*layers)
