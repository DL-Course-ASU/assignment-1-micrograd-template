"""Fully connected tanh networks built entirely from scalar Values."""
import random
from src.engine import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.reset_grad()


def _positive_size(size):
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError('Network dimensions must be positive integers')


class Neuron(Module):
    def __init__(self, nin):
        _positive_size(nin)
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        x = tuple(x)
        if len(x) != len(self.w):
            raise ValueError(f'Expected {len(self.w)} inputs, got {len(x)}')
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b).tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout):
        _positive_size(nin)
        _positive_size(nout)
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        x = tuple(x)
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts):
        _positive_size(nin)
        nouts = list(nouts)
        if not nouts:
            raise ValueError('MLP requires at least one output layer')
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            # Preserve Layer's scalar return convention at width one.
            x = layer([x] if isinstance(x, Value) else x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

