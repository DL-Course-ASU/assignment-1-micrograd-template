
import random
from src.engine import Value


class Neuron:
    def __init__(self, nin):
        # nin = number of inputs
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Weighted sum + bias
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # Activation
        return out.tanh()

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]

        # A single neuron returns a single Value
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons
                for p in neuron.parameters()]

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + list(nouts)

        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) if isinstance(x, list) else layer([x])
        return x

    def parameters(self):
        return [p for layer in self.layers
                for p in layer.parameters()]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
