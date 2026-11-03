"""A small PyTorch-like neural network API built on top of the Value engine."""
import random

from src.engine import Value


class Module:
    """Base class: anything with parameters can have its gradients reset."""

    def parameters(self):
        return []  # subclasses override this with their own parameters

    def zero_grad(self):
        # Gradients accumulate with +=, so they must be cleared before each new
        # backward pass or the previous step's gradients would be added in.
        for p in self.parameters():
            p.reset_grad()


class Neuron(Module):
    """Computes sum(w_i * x_i) + b, followed by tanh when nonlin is True."""

    def __init__(self, nin, nonlin=True):
        # Random weights break symmetry: identical neurons would receive identical
        # gradients and stay identical forever.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):  # Neuron(nin)(x)
        # A layer with one neuron outputs a single Value rather than a list, so a
        # lone number is treated here as a one-element input.
        if isinstance(x, (Value, int, float)):
            x = [x]
        # zip() would silently drop inputs if the lengths differed.
        if len(x) != len(self.w):
            raise ValueError(f"expected {len(self.w)} inputs, got {len(x)}")

        activation = self.b
        for wi, xi in zip(self.w, x):
            activation = activation + wi * xi
        return activation.tanh() if self.nonlin else activation

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    """A list of independent neurons that all receive the same input."""

    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    """A stack of layers, for example MLP(2, [4, 4, 1])."""

    def __init__(self, nin, nouts):
        sizes = [nin] + nouts  # [2, 4, 4, 1] gives the layers 2->4, 4->4 and 4->1
        # Every layer uses tanh except the last, which stays linear so that the
        # output is an unbounded score rather than being squashed into (-1, 1).
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)  # each layer's output is the next layer's input
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
