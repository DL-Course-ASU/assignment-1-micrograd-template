"""Scalar reverse-mode differentiation, implemented without a tensor framework."""
import math
from numbers import Real


class Value:
    """A scalar, its accumulated derivative, and its local computation graph."""

    def __init__(self, data, _children=(), _op="", label=""):
        if not isinstance(data, Real):
            raise TypeError("Value requires a real scalar")
        self.data = float(data)
        self.grad = 0.0
        # Keep operand order deterministic. DFS still visits shared nodes once.
        self._prev = tuple(dict.fromkeys(_children))
        self._op, self.label = _op, label
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    @staticmethod
    def _wrap(item):
        if isinstance(item, Value):
            return item
        if isinstance(item, Real):
            return Value(item)
        raise TypeError("operands must be Values or real scalars")

    def __add__(self, other):
        other = self._wrap(other)
        result = Value(self.data + other.data, (self, other), "+")

        def propagate():
            # Add both edges, even when both operands are the same object.
            self.grad += result.grad
            other.grad += result.grad

        result._backward = propagate
        return result

    def __mul__(self, other):
        other = self._wrap(other)
        left, right = self.data, other.data
        result = Value(left * right, (self, other), "*")

        def propagate():
            self.grad += right * result.grad
            other.grad += left * result.grad

        result._backward = propagate
        return result

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "exponent must be constant"
        x = self.data
        data = x ** exponent
        if isinstance(data, complex):
            raise ValueError("power is outside the real domain")
        result = Value(data, (self,), f"**{exponent}")

        def propagate():
            # x**0 is constant, including at x=0; do not evaluate 0**-1.
            local = 0.0 if exponent == 0 else exponent * x ** (exponent - 1)
            self.grad += local * result.grad

        result._backward = propagate
        return result

    def tanh(self):
        t = math.tanh(self.data)
        result = Value(t, (self,), "tanh")

        def propagate():
            self.grad += (1.0 - t * t) * result.grad

        result._backward = propagate
        return result

    def exp(self):
        e = math.exp(self.data)
        result = Value(e, (self,), "exp")

        def propagate():
            self.grad += e * result.grad

        result._backward = propagate
        return result

    def relu(self):
        positive = self.data > 0
        result = Value(self.data if positive else 0.0, (self,), "relu")

        def propagate():
            self.grad += float(positive) * result.grad

        result._backward = propagate
        return result

    def log(self):
        if self.data <= 0:
            raise ValueError("log requires positive input")
        x = self.data
        result = Value(math.log(x), (self,), "log")

        def propagate():
            self.grad += result.grad / x

        result._backward = propagate
        return result

    def sigmoid(self):
        # Equivalent formulas avoid overflow on very negative inputs.
        x = self.data
        s = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
        result = Value(s, (self,), "sigmoid")

        def propagate():
            self.grad += s * (1 - s) * result.grad

        result._backward = propagate
        return result

    __radd__ = __add__
    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __truediv__(self, other):
        return self * self._wrap(other) ** -1

    def __rtruediv__(self, other):
        return self._wrap(other) * self ** -1

    def __float__(self):
        return float(self.data)

    def __int__(self):
        return int(self.data)

    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        """Post-order DFS: operands precede results, shared nodes appear once."""
        order, visited = [], set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for operand in node._prev:
                visit(operand)
            order.append(node)

        visit(self)
        return order

    def backward(self):
        """Accumulate a fresh output derivative; training builds a new graph."""
        order = self._get_topo()
        # Clear intermediate adjoints when reusing a graph, retaining leaf grads.
        for node in order:
            if node._prev:
                node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()
