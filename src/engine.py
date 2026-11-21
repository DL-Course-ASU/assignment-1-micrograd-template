
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

    # Addition: self + other
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        elif not isinstance(other, Value):
            raise TypeError("Addition requires a Value or a number")

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    # Reverse addition: other + self
    def __radd__(self, other):
        return self + other

    # Multiplication: self * other
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        elif not isinstance(other, Value):
            raise TypeError("Multiplication requires a Value or a number")

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # Reverse multiplication: other * self
    def __rmul__(self, other):
        return self * other

    # Negation: -self
    def __neg__(self):
        return self * -1

    # Subtraction: self - other
    def __sub__(self, other):
        return self + (-other)

    # Division: self / other
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Value(other)
        elif not isinstance(other, Value):
            raise TypeError("Division requires a Value or a number")

        return self * (other ** -1)

    # Power: self ** other
    def __pow__(self, other):
        assert isinstance(other, (int, float)), \
            "Power must be an int or float"

        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    # Hyperbolic tangent activation
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    # Exponential function
    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    # Reset this Value's gradient
    def reset_grad(self):
        self.grad = 0.0

    # Topological sorting using Depth-First Search
    def _get_topo(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                for child in node._prev:
                    build_topo(child)

                # Append AFTER visiting all children
                topo.append(node)

        build_topo(self)
        return topo

    # Backpropagation
    def backward(self):
        topo = self._get_topo()

        # Seed the output gradient
        self.grad = 1.0

        # Traverse in reverse topological order
        for node in reversed(topo):
            node._backward()

    # Convert Value to integer
    def __int__(self):
        return int(self.data)

    # Convert Value to float
    def __float__(self):
        return float(self.data)
