import random
from src.engine import Value


class Neuron:
    def __init__(self, nin):
        # small symmetric random init: if every weight started at the same
        # value all neurons in a layer would stay identical forever
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):  # Neuron(nin)(x)
        # sum(w_i * x_i) + b, with b used as the starting value of the sum so
        # we never add a bare python 0 into the graph
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()

    def __repr__(self):
        return f"TanhNeuron({len(self.w)})"


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        # a layer with one neuron returns the Value itself, not a list of one,
        # so the final output of an MLP can be used directly in a loss
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self):
        for n in self.neurons:
            n.zero_grad()

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + list(nouts)
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def __repr__(self):
        return f"MLP of [{', '.join(str(l) for l in self.layers)}]"
