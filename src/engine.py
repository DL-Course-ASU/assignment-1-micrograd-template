import math


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):
      return f'Value(data={self.data})'

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    __rmul__ = __mul__

    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'Exponent must be numeric'
        out = Value(self.data ** other, (self,), f'**{other}')
        def backward():
            self.grad += (0.0 if other == 0 else other * self.data ** (other - 1)) * out.grad
        out._backward = backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')
        def backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def backward():
            self.grad += out.data * out.grad
        out._backward = backward
        return out

    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        topo, visited = [], set()
        def visit(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    visit(parent)
                topo.append(node)
        visit(self)
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __float__(self):
        return float(self.data)

    def __int__(self):
        return int(self.data)
