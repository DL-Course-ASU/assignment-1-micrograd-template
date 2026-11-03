"""A small, scalar reverse-mode automatic differentiation engine.

Every arithmetic operation creates a new Value and records how its inputs
contribute to the output. A reverse topological pass applies the chain rule.
No machine-learning framework is used.
"""

import math
from numbers import Real


class Value:
    """A scalar value, its derivative, and its computational-graph parents."""

    def __init__(self, data, _children=(), _op='', label=''):
        if not isinstance(data, Real):
            raise TypeError('Value data must be a real number')
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    @staticmethod
    def _as_value(other):
        if isinstance(other, Value):
            return other
        if isinstance(other, Real):
            return Value(other)
        raise TypeError(f'Unsupported operand type: {type(other).__name__}')

    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self, other):
        other = self._as_value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Each edge contributes: the same Value may occur twice (a + a).
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._as_value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._as_value(other))

    def __rsub__(self, other):
        return self._as_value(other) - self

    def __truediv__(self, other):
        other = self._as_value(other)
        if other.data == 0:
            raise ZeroDivisionError('division by zero')
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return self._as_value(other) / self

    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'Only numeric constant powers are supported'
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        activation = math.tanh(self.data)
        out = Value(activation, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - activation ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        result = math.exp(self.data)
        out = Value(result, (self,), 'exp')

        def _backward():
            self.grad += result * out.grad

        out._backward = _backward
        return out

    def reset_grad(self):
        """Set the gradient of this Value to zero."""
        self.grad = 0.0

    def _get_topo(self):
        """DFS postorder: parents appear before the nodes that consume them."""
        topo, visited = [], set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)
                topo.append(node)  # Append AFTER exploring the parents.

        build_topo(self)
        return topo

    def backward(self):
        """Compute all gradients of this scalar output via reverse-mode AD."""
        topo = self._get_topo()
        self.grad = 1.0  # d(output)/d(output)
        for node in reversed(topo):
            node._backward()

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)
