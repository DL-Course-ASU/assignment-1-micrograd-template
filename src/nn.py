import random

from .engine import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0.0


class Neuron(Module):
    def __init__(self, nin):
        self.w = [
            Value(random.uniform(-1, 1))
            for _ in range(nin)
        ]
        self.b = Value(0.0)

    def __call__(self, x):
        if len(x) != len(self.w):
            raise ValueError(
                f"Expected {len(self.w)} inputs, got {len(x)}."
            )

        activation = sum(
            (weight * value for weight, value in zip(self.w, x)),
            self.b,
        )

        return activation.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [
            Neuron(nin)
            for _ in range(nout)
        ]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]


class MLP(Module):
    def __init__(self, nin, nouts):
        sizes = [nin] + list(nouts)

        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x):
        for layer in self.layers:
            if isinstance(x, Value):
                x = [x]

            x = layer(x)

        return x

    def parameters(self):
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]