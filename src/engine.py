import math


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    # Addition
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            '+'
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    # Multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            '*'
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # Negation
    def __neg__(self):
        return self * -1

    # Subtraction
    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -other)

    # Division
    def __truediv__(self, other):
        return self * (other ** -1)

    # Power
    def __pow__(self, other):
        assert isinstance(other, (int, float)), \
            "Only int and float powers are supported"

        out = Value(
            self.data ** other,
            (self,),
            f'**{other}'
        )

        def _backward():
            self.grad += (
                other * (self.data ** (other - 1)) * out.grad
            )

        out._backward = _backward
        return out

    # Hyperbolic tangent
    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    # Exponential
    def exp(self):
        x = math.exp(self.data)

        out = Value(x, (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    # Reset gradient
    def reset_grad(self):
        self.grad = 0.0

    # Topological sorting
    def _get_topo(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for child in node._prev:
                    build_topo(child)

                topo.append(node)

        build_topo(self)

        return topo

    # Backpropagation
    def backward(self):
        topo = self._get_topo()

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    # Type conversion
    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)