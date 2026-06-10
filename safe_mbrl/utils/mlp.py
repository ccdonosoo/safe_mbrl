from typing import Sequence, Callable

import jax
import jax.numpy as jp
from flax import nnx
from flax.nnx import rnglib


class MLP(nnx.Module):

    def __init__(self, input_dim: int, features: Sequence[int], output_dim: int,
                  activation: Callable, rngs: rnglib.Rngs):

        self.input_dim = input_dim
        self.features = features
        self.output_dim = output_dim
        self.activation = activation

        layers_sz = (input_dim, *features, output_dim)
        layers = []

        for i in range(len(layers_sz)-1):
            layers.append(nnx.Linear(layers_sz[i], layers_sz[i+1], rngs=rngs))

        self.layers = nnx.List(layers)

    def __call__(self, x: jax.Array):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x