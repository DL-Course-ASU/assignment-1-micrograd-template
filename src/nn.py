"""A minimal neural-network API built entirely from scalar ``Value`` nodes."""

from __future__ import annotations

import random
from collections.abc import Sequence

from src.engine import Value


class Neuron:
    """A fully connected neuron with a tanh activation."""

    def __init__(self, nin: int):
        if nin <= 0:
            raise ValueError("A neuron must have at least one input")
        self.w = [Value(random.uniform(-1.0, 1.0), label=f"w{i}") for i in range(nin)]
        self.b = Value(random.uniform(-1.0, 1.0), label="b")

    def __call__(self, x: Sequence[float | Value]) -> Value:
        if len(x) != len(self.w):
            raise ValueError(f"Expected {len(self.w)} inputs, got {len(x)}")
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return activation.tanh()

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.reset_grad()


class Layer:
    """A collection of independently parameterized neurons."""

    def __init__(self, nin: int, nout: int):
        if nout <= 0:
            raise ValueError("A layer must contain at least one neuron")
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: Sequence[float | Value]) -> Value | list[Value]:
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self) -> list[Value]:
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.reset_grad()


class MLP:
    """A multi-layer perceptron with tanh activations at every layer."""

    def __init__(self, nin: int, nouts: Sequence[int]):
        if nin <= 0:
            raise ValueError("An MLP must have at least one input")
        if not nouts:
            raise ValueError("An MLP must contain at least one layer")

        sizes = [nin, *nouts]
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x: Sequence[float | Value]) -> Value | list[Value]:
        for layer in self.layers:
            result = layer(x)
            x = result if isinstance(result, list) else [result]
        return result

    def parameters(self) -> list[Value]:
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.reset_grad()
