"""Scalar reverse-mode autodiff; depends only on Python's standard library."""
import math


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        if not isinstance(data, (int, float)):
            raise TypeError('Value data must be an int or float')
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f'Value(data={self.data})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
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
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'Exponent must be int or float'
        result = self.data**other
        if isinstance(result, complex):
            raise ValueError('Power result must be real')
        out = Value(result, (self,), f'**{other}')
        def _backward():
            # x**0 is constant, including Python's convention 0**0 == 1.
            self.grad += (0.0 if other == 0 else other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        # Stable even when exp(2*x) would overflow.
        out = Value(math.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        """Optional activation, with subgradient zero at the origin."""
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        topo, visited = [], set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)  # Post-order: after ALL dependencies.
        build_topo(self)
        return topo

    def backward(self):
        """Accumulate gradients; use a fresh graph for each training step.

        For a second pass on this same graph, reset ALL graph nodes first.
        Do not mutate data between constructing a graph and differentiating it.
        """
        topo = self._get_topo()
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

