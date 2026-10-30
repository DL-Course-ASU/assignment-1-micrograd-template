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

    def __add__(self, other):
        if not isinstance(other, Value):
            if not isinstance(other, (int, float)):
                raise TypeError("Can only add a Value or a number")
            other = Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            if not isinstance(other, (int, float)):
                raise TypeError("Can only multiply a Value by a number")
            other = Value(other)
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
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __truediv__(self, other):
        if not isinstance(other, Value):
            if not isinstance(other, (int, float)):
                raise TypeError("Can only divide a Value by a number")
            other = Value(other)
        return self * other ** -1

    def __rtruediv__(self, other):
        if not isinstance(other, (Value, int, float)):
            raise TypeError("Can only divide a Value into a number")
        return other * self ** -1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Exponent must be int or float"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            if other != 0:
                self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

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
                for parent in node._prev:
                    build_topo(parent)
                topo.append(node)

        build_topo(self)
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)