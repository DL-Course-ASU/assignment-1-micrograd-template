"""A small scalar reverse-mode automatic differentiation engine."""
from __future__ import annotations

import math
from typing import Callable, Iterable


class Value:
    """A scalar value and its place in a differentiable computation graph."""

    def __init__(self, data: float, _children: Iterable[Value] = (), _op: str = "", label: str = "") -> None:
        if not isinstance(data, (int, float)):
            raise TypeError("Value data must be a real scalar")
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward: Callable[[], None] = lambda: None

    def __add__(self, other: float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def backward() -> None:
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    __radd__ = __add__

    def __mul__(self, other: float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    __rmul__ = __mul__

    def __pow__(self, exponent: int | float) -> Value:
        assert isinstance(exponent, (int, float)), "Exponent must be a real scalar"
        result = self.data ** exponent
        out = Value(result, (self,), f"**{exponent}")
        derivative = exponent * self.data ** (exponent - 1) if exponent != 0 else 0.0
        def backward() -> None:
            self.grad += derivative * out.grad
        out._backward = backward
        return out

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: float | Value) -> Value:
        return self + (-other if isinstance(other, Value) else -other)

    def __rsub__(self, other: float | Value) -> Value:
        return (-self) + other

    def __truediv__(self, other: float | Value) -> Value:
        if isinstance(other, Value):
            return self * (other ** -1)
        if other == 0:
            raise ZeroDivisionError("division by zero")
        return self * (1.0 / other)

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def backward() -> None:
            self.grad += (1.0 - t * t) * out.grad
        out._backward = backward
        return out

    def exp(self) -> Value:
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        def backward() -> None:
            self.grad += e * out.grad
        out._backward = backward
        return out

    def backward(self) -> None:
        """Recompute gradients for this graph, seeding this output with one.

        Gradients are cleared on reachable nodes for each call. Contributions
        from multiple paths within this pass are accumulated by local rules.
        """
        topo = self._get_topo()
        for node in topo:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def reset_grad(self) -> None:
        """Reset this value's gradient, as expected by the template API."""
        self.grad = 0.0

    def _get_topo(self) -> list[Value]:
        """Return this value's ancestors in dependency-first topological order."""
        topo: list[Value] = []
        visited: set[Value] = set()
        def build(node: Value) -> None:
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build(parent)
                topo.append(node)
        build(self)
        return topo

    def __int__(self) -> int:
        return int(self.data)

    def __float__(self) -> float:
        return float(self.data)

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
