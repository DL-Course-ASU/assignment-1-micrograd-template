import random
from src.engine import Value

class Neuron:
    """
    A single artificial neuron with weights, bias, and a tanh activation function.
    """
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(random.uniform(-1.0, 1.0))

    def __call__(self, x):  # Neuron(nin)(x)
        # Ensure single scalar Value or float can be handled gracefully
        if isinstance(x, (Value, int, float)):
            x = [x]
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()

    def __repr__(self):
        return f"Neuron({len(self.w)})"


class Layer:
    """
    A layer of neurons.
    """
    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()

    def __repr__(self):
        return f"Layer([{', '.join(str(n) for n in self.neurons)}])"


class MLP:
    """
    Multi-Layer Perceptron consisting of stacked Layers.
    """
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()

    def __repr__(self):
        return f"MLP([{', '.join(str(layer) for layer in self.layers)}])"
