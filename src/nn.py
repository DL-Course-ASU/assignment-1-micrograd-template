import random
from src.engine import Value

class Neuron:
    def __init__(self, nin):
        # Weights initialized uniformly in [-1, 1]
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # Bias initialized uniformly in [-1, 1]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):  # Neuron(nin)(x)
        # Compute activation: sum(w_i * x_i) + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        # Return all learnable parameters in this neuron
        return self.w + [self.b]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()


class Layer:
    def __init__(self, nin, nout):
        # Create nout individual neurons, each taking nin inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        # Return scalar Value if single-output, otherwise return list
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # Flatten parameters across all neurons
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()


class MLP:
    def __init__(self, nin, nouts):
        # Construct list of layer input/output dimensions: [nin] + nouts
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        # Feed-forward through all layers sequentially
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # Flatten parameters across all layers
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()
