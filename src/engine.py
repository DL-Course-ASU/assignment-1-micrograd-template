import math


def toValue(item):
    if isinstance(item, Value):
        return item
    if isinstance(item, (int, float)):
        return Value(item)
    raise TypeError(f"unsupported operand type for Value: '{type(item).__name__}'")


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
        other = toValue(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = toValue(other)
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
        return self + (-toValue(other))

    def __rsub__(self, other):
        return toValue(other) + (-self)

    def __truediv__(self, other):
        return self * toValue(other) ** -1

    def __rtruediv__(self, other):
        return toValue(other) * self ** -1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int or float powers are supported"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        tanhValue = math.tanh(self.data)
        out = Value(tanhValue, (self,), 'tanh')

        def _backward():
            self.grad += (1 - tanhValue ** 2) * out.grad

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

        def buildTopo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    buildTopo(child)
                topo.append(node)

        buildTopo(self)
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
