import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # means no effect by default
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):  # self + other
        if isinstance(other, (int, float)):
            other = Value(other)
        if not isinstance(other, Value):
            raise TypeError(f"unsupported operand type(s) for +: 'Value' and '{type(other).__name__}'")
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        if isinstance(other, (int, float)):
            other = Value(other)
        if not isinstance(other, Value):
            raise TypeError(f"unsupported operand type(s) for *: 'Value' and '{type(other).__name__}'")
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other if isinstance(other, Value) else -other)

    def __rsub__(self, other):  # other - self
        return (-self) + other

    def __truediv__(self, other):  # self / other
        return self * (other ** -1 if isinstance(other, Value) else Value(other) ** -1)

    def __rtruediv__(self, other):  # other / self
        return other * (self ** -1)

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), "power must be an int or float"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):  # tanh(self)
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def exp(self):  # e**self
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

        def build(value):
            if value not in visited:
                visited.add(value)
                for child in value._prev:
                    build(child)
                topo.append(value)

        build(self)
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)
