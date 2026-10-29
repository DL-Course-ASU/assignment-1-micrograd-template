"""Tiny neural network API built entirely on src.engine.Value."""

import random
from src.engine import Value


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        if isinstance(x, (Value, int, float)):
            x = [x]
        if len(x) != len(self.w):
            raise ValueError(f'Expected {len(self.w)} inputs, received {len(x)}')
        activation = sum((weight * feature for weight, feature in zip(self.w, x)), self.b)
        return activation.tanh()

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.reset_grad()


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + list(nouts)
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()