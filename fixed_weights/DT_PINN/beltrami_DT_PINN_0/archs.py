import torch
import deepxde as dde
import numpy as np
import prm
from deepxde.nn.pytorch.nn import NN
from deepxde.nn import activations
from deepxde.nn import initializers
from deepxde import config


# Fourier Features
class FourierEmbs(NN):

    def __init__(self, input_dim, embed_dim, embed_scale):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embed_scale = embed_scale

        # Normal distribution: N~(0, embed_scale**2)
        B = torch.normal(mean=0.0, std=embed_scale, size=(input_dim, embed_dim // 2))

        self.register_buffer("B", B)

    def forward(self, x):

        # pass B to GPU (without this, it returns an error)
        B = self.B.to(x.device)

        # Element-wise product of x with B
        x_proj = x @ B

        # Concatenate over the last dimension (in this case, columns)
        gamma = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)

        return gamma


# Modified MLP
class ModifiedMLP(NN):

    def __init__(self, layer_sizes, activation, kernel_initializer, regularization=None, dropout_rate=0):
        super().__init__()

        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)

        if isinstance(dropout_rate, list):
            if len(layer_sizes) - 1 != len(dropout_rate):
                raise ValueError(
                    f"Number of dropout rates must be equal to {len(layer_sizes) - 1}"
                )
            self.dropout_rate = dropout_rate
        else:
            self.dropout_rate = [dropout_rate] * (len(layer_sizes) - 1)

        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros")
        self.regularizer = regularization

        ##### U and V layers #####
        self.u_layer = torch.nn.Linear(layer_sizes[0], layer_sizes[1], dtype=config.real(torch))
        self.v_layer = torch.nn.Linear(layer_sizes[0], layer_sizes[1], dtype=config.real(torch))

        ##### First hidden layer #####
        self.first_layer = torch.nn.Linear(layer_sizes[0], layer_sizes[1], dtype=config.real(torch))

        ##### Subsequent hidden layers #####
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(2, len(layer_sizes) - 1):
            self.hidden_layers.append(torch.nn.Linear(layer_sizes[i-1], layer_sizes[i], dtype=config.real(torch)))

        ##### Output layer #####
        self.output_layer = torch.nn.Linear(layer_sizes[-2], layer_sizes[-1], dtype=config.real(torch))

        ##### Weight initialization #####
        standalone_layers = [self.u_layer, self.v_layer, self.first_layer, self.output_layer]
        for layer in standalone_layers:
            initializer(layer.weight)
            initializer_zero(layer.bias)

        for layer in self.hidden_layers:
            initializer(layer.weight)
            initializer_zero(layer.bias)

    def forward(self, inputs):

        x = inputs

        if self._input_transform is not None:
            x = self._input_transform(x)

        # Choose between list of activation functions and single activation function for all layers
        get_act = lambda j: self.activation[j] if isinstance(self.activation, list) else self.activation

        # U and V should work with the same (and first, if list) activation function
        U = get_act(0)(self.u_layer(x))
        V = get_act(0)(self.v_layer(x))

        # First hidden layer
        H = get_act(0)(self.first_layer(x))
        if self.dropout_rate[0] > 0:
            H = torch.nn.functional.dropout(H, p=self.dropout_rate[0], training=self.training)

        # Hidden layers
        for j, hlayer in enumerate(self.hidden_layers):
            Z = get_act(j + 1)(hlayer(H))
            H = (1.0 - Z) * U + Z * V
            if self.dropout_rate[j + 1] > 0:
                H = torch.nn.functional.dropout(H, p=self.dropout_rate[j + 1], training=self.training)

        # Output layer
        y = self.output_layer(H)

        if self._output_transform is not None:
            y = self._output_transform(inputs, y)

        return y

