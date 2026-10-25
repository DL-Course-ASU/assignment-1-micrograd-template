
import random
from src.engine import Value

class Module:
    """Base class for neural network modules."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    """A single neuron with weights, bias, and optional activation."""

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        activation = "Tanh" if self.nonlin else "Linear"
        return f"{activation}Neuron({len(self.w)})"


class Layer(Module):
    """A layer containing multiple neurons."""

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [
            Neuron(nin, **kwargs) for _ in range(nout)
        ]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [
            p
            for neuron in self.neurons
            for p in neuron.parameters()
        ]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """A multilayer perceptron."""

    def __init__(self, nin, nouts):
        if nin < 1 or not nouts or any(n < 1 for n in nouts):
            raise ValueError("Layer sizes must be positive integers.")

        sizes = [nin] + list(nouts)

        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                nonlin=(i != len(nouts) - 1)
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [
            p
            for layer in self.layers
            for p in layer.parameters()
        ]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
