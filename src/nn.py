"""
nn.py -- a minimal PyTorch-like neural-network API built on top of ``Value``.

    Neuron  : tanh(w . x + b)
    Layer   : a list of independent Neurons that see the same input
    MLP     : a stack of Layers, each feeding the next
"""
import random
from src.engine import Value


class Neuron:
    def __init__(self, nin):
        # weights and bias ~ Uniform(-1, 1), each wrapped in a Value
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):  # Neuron(nin)(x)
        # w . x + b  (start the sum at the bias so no extra "0 +" node is built)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + list(nouts)
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            if i > 0 and isinstance(x, Value):  # a 1-neuron hidden layer returns a bare Value
                x = [x]
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()
