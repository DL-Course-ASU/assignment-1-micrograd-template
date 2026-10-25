import random
from src.engine import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)] # nin: number of i/p's
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):  # Neuron(nin)(x)
        if len(x) != len(self.w):
            raise ValueError(f"Neuron expects {len(self.w)} inputs, got {len(x)}")
        # w*x + b: start the sum at b, then add each wi*xi
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
        # every neuron sees the same input x
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()

class MLP:
    def __init__(self, nin, nouts):
        # ex// nin=3, nouts=[4, 4, 1] -> sizes [3, 4, 4, 1] -> layers 3->4, 4->4, 4->1
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        # the output of each layer is the input of the next
        for layer in self.layers:
            if isinstance(x, Value):  # a 1-neuron layer returned a single Value
                x = [x]
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()