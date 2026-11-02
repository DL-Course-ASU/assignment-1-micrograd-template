"""
nn.py -- a small neural-network API on top of the scalar autograd engine.

Neuron -> Layer -> MLP, same layout as PyTorch's nn.Module idea but with
plain Python floats hiding inside `Value` objects.
"""

import random

from src.engine import Value


class Module:
    """Tiny base class so every building block can list its parameters and
    reset its gradients."""

    def zero_grad(self):
        # Gradients accumulate with +=, so they must be cleared before every
        # new backward pass or they'd pile up across training steps.
        for p in self.parameters():
            p.reset_grad()

    def parameters(self):
        return []


class Neuron(Module):
    """One neuron: activation(w . x + b)."""

    def __init__(self, nin, nonlin=True, activation="tanh"):
        # weights and bias both start as random numbers in [-1, 1]
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin
        self.activation = activation

    def __call__(self, x):
        # w . x + b  (start the sum from the bias so we don't add a stray 0)
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi

        if not self.nonlin:
            return act                      # linear neuron (used for the output layer)
        if self.activation == "tanh":
            return act.tanh()
        if self.activation == "relu":
            return act.relu()
        if self.activation == "sigmoid":
            return act.sigmoid()
        raise ValueError(f"unknown activation: {self.activation}")

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        kind = self.activation if self.nonlin else "Linear"
        return f"{kind}Neuron({len(self.w)})"


class Layer(Module):
    """A bunch of neurons that all look at the same input."""

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """Multi-layer perceptron. `nouts` lists the size of every layer, e.g.
    MLP(3, [4, 4, 1]) is 3 inputs -> 4 -> 4 -> 1 output.
    Hidden layers use the chosen activation; the last layer is linear."""

    def __init__(self, nin, nouts, activation="tanh"):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1],
                  nonlin=(i != len(nouts) - 1),
                  activation=activation)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
