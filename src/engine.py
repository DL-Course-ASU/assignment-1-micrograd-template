"""A small reverse-mode automatic differentiation engine for scalar values."""

from __future__ import annotations

import math
from numbers import Real
from typing import Callable, Iterable


class Value:
    """A scalar in a computation graph, together with its reverse-mode gradient.

    Operations create new ``Value`` objects and retain their immediate parents.
    Calling :meth:`backward` on an output walks that graph in reverse topological
    order and accumulates derivatives into each node's ``grad`` field.
    """

    def __init__(self, data: Real, _children: Iterable[Value] = (), _op: str = "", label: str = ""):
        if not isinstance(data, Real):
            raise TypeError(f"Value data must be a real number, got {type(data).__name__}")
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    @staticmethod
    def _coerce(other: Real | Value) -> Value:
        if isinstance(other, Value):
            return other
        if isinstance(other, Real):
            return Value(other)
        raise TypeError(f"Expected a real number or Value, got {type(other).__name__}")

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: Real | Value) -> Value:
        other = self._coerce(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Real | Value) -> Value:
        return self + other

    def __mul__(self, other: Real | Value) -> Value:
        other = self._coerce(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Real | Value) -> Value:
        return self * other

    def __neg__(self) -> Value:
        return self * -1.0

    def __sub__(self, other: Real | Value) -> Value:
        return self + (-self._coerce(other))

    def __rsub__(self, other: Real | Value) -> Value:
        return self._coerce(other) - self

    def __truediv__(self, other: Real | Value) -> Value:
        other = self._coerce(other)
        return self * (other ** -1)

    def __rtruediv__(self, other: Real | Value) -> Value:
        return self._coerce(other) / self

    def __pow__(self, other: int | float) -> Value:
        assert isinstance(other, (int, float)), "Only integer and floating-point exponents are supported"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            if other == 0:
                local_derivative = 0.0
            else:
                local_derivative = other * (self.data ** (other - 1))
            self.grad += local_derivative * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        result = math.tanh(self.data)
        out = Value(result, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - result**2) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        result = math.exp(self.data)
        out = Value(result, (self,), "exp")

        def _backward() -> None:
            self.grad += result * out.grad

        out._backward = _backward
        return out

    def reset_grad(self) -> None:
        """Clear this node's accumulated derivative."""
        self.grad = 0.0

    def _get_topo(self) -> list[Value]:
        """Return every reachable node once, with parents before consumers."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(node: Value) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        return topo

    def backward(self) -> None:
        """Backpropagate a unit derivative from this output through its graph."""
        topo = self._get_topo()
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __int__(self) -> int:
        return int(self.data)

    def __float__(self) -> float:
        return float(self.data)
