import random
from src.engine import Value


class Neuron:
    """A single neuron: out = tanh(sum_i w_i * x_i + b)."""

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1), label=f'w{i}') for i in range(nin)]
        self.b = Value(random.uniform(-1, 1), label='b')
        self.nonlin = nonlin  # False -> linear output (useful for regression)

    def __call__(self, x):  # Neuron(nin)(x)
        assert len(x) == len(self.w), \
            f"expected {len(self.w)} inputs, got {len(x)}"
        # sum(..., self.b) starts the running sum at the bias
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()

    def __repr__(self):
        return f"{'Tanh' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer:
    """A fully connected layer of `nout` independent neurons."""

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP:
    """Multi-layer perceptron, e.g. MLP(3, [4, 4, 1]) = 3 -> 4 -> 4 -> 1."""

    def __init__(self, nin, nouts):
        sz = [nin] + list(nouts)
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            if isinstance(x, Value):  # a 1-neuron hidden layer returns a scalar
                x = [x] if layer is not self.layers[-1] else x
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()

    def __repr__(self):
        return f"MLP of [{', '.join(str(l) for l in self.layers)}]"
