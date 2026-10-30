import random
from src.engine import Value

class Neuron:
    """
    A single artificial neuron computing: tanh(sum(w_i * x_i) + b)
    """
    def __init__(self, nin):
        # Initialize weights and bias drawn uniformly from [-1, 1]
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(random.uniform(-1.0, 1.0))

    def __call__(self, x):  # Neuron(nin)(x)
        # Compute activation: act = sum(w_i * x_i) + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        # Return all trainable parameters in the neuron
        return self.w + [self.b]

    def zero_grad(self):
        # Reset gradients of all parameters to 0.0
        for p in self.parameters():
            p.reset_grad()

    def __repr__(self):
        return f"Neuron({len(self.w)})"


class Layer:
    """
    A layer consisting of nout independent neurons.
    """
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        # Return single scalar Value if nout == 1, otherwise return list of Values
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
    Multi-Layer Perceptron: a cascade of connected Layers.
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
