"""A minimal scalar reverse-mode automatic differentiation engine.

Every operation creates a Value node with a closure for its local derivative.
Calling backward() walks the computational DAG in reverse topological order.
"""

import math


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        if not isinstance(data, (int, float)):
            raise TypeError(f'Value data must be int or float, got {type(data).__name__}')
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f'Value(data={self.data})'

    @staticmethod
    def _as_value(other):
        if isinstance(other, Value):
            return other
        if isinstance(other, (int, float)):
            return Value(other)
        return NotImplemented

    def __add__(self, other):
        other = self._as_value(other)
        if other is NotImplemented:
            return NotImplemented
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._as_value(other)
        if other is NotImplemented:
            return NotImplemented
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
        other = self._as_value(other)
        if other is NotImplemented:
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other):
        return other + -self

    def __truediv__(self, other):
        other = self._as_value(other)
        if other is NotImplemented:
            return NotImplemented
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = self._as_value(other)
        if other is NotImplemented:
            return NotImplemented
        return other / self

    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'Only numeric constant powers are supported'
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # The zero exponent is constant, including at x = 0.
            local = 0.0 if other == 0 else other * (self.data ** (other - 1))
            self.grad += local * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)  # Post-order: children before parent.

        build_topo(self)
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0  # d(output)/d(output)
        for node in reversed(topo):
            node._backward()

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)