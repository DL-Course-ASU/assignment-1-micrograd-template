"""Minimal neural network components built on the custom Value engine."""
from __future__ import annotations
import random
from src.engine import Value


class Neuron:
    def __init__(self, nin: int, *, activation: bool = True, rng: random.Random | None = None) -> None:
        if nin < 1:
            raise ValueError("A neuron needs at least one input")
        rng = rng or random
        self.w = [Value(rng.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(rng.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x: list[Value | float]) -> Value:
        if len(x) != len(self.w):
            raise ValueError(f"expected {len(self.w)} inputs, got {len(x)}")
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return out.tanh() if self.activation else out

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.reset_grad()


class Layer:
    def __init__(self, nin: int, nout: int, *, activation: bool = True, rng: random.Random | None = None) -> None:
        if nout < 1:
            raise ValueError("A layer needs at least one neuron")
        self.neurons = [Neuron(nin, activation=activation, rng=rng) for _ in range(nout)]

    def __call__(self, x: list[Value | float]) -> Value | list[Value]:
        output = [neuron(x) for neuron in self.neurons]
        return output[0] if len(output) == 1 else output

    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.reset_grad()


class MLP:
    def __init__(self, nin: int, nouts: list[int], *, rng: random.Random | None = None) -> None:
        if nin < 1 or not nouts or any(n < 1 for n in nouts):
            raise ValueError("MLP dimensions must be positive and include at least one layer")
        sizes = [nin, *nouts]
        self.layers = [Layer(sizes[i], sizes[i + 1], activation=i < len(nouts) - 1, rng=rng)
                       for i in range(len(nouts))]

    def __call__(self, x: list[Value | float]) -> Value | list[Value]:
        if len(x) != len(self.layers[0].neurons[0].w):
            raise ValueError("input width does not match MLP architecture")
        out = x
        for index, layer in enumerate(self.layers):
            out = layer(out)
            if index < len(self.layers) - 1 and not isinstance(out, list):
                out = [out]
        return out

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.reset_grad()
