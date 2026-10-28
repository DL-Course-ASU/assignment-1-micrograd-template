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

    @staticmethod
    def _wrap(other):
        if isinstance(other, Value):
            return other
        if isinstance(other, (int, float)):
            return Value(other)
        raise TypeError(f"unsupported operand type for Value: {type(other).__name__}")

    def __add__(self, other):  # self + other
        other = Value._wrap(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad * 1
            other.grad += out.grad * 1
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):  # self * other
        other = Value._wrap(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):  # -self
        out = Value(-self.data, (self,))
        def _backward():
            self.grad += out.grad* -1
        out._backward = _backward
        return out

    def __sub__(self, other):  # self - other
        return self + (-Value._wrap(other))

    def __truediv__(self, other):  # self / other
        return self * Value._wrap(other) ** -1

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), "only int/float powers are supported"
        out = Value(self.data ** other, (self,))
        def _backward():
            self.grad += out.grad * (other * self.data ** (other - 1))
        out._backward = _backward
        return out

    def tanh(self):
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t, (self,))

        def _backward():
            self.grad += out.grad * (1 - t**2)

        out._backward = _backward
        return out
    
    def exp(self):  # e**self
        out = Value(math.exp(self.data), (self,))
        def _backward():
            self.grad += out.grad * out.data
        out._backward = _backward
        return out

    def reset_grad(self):
        self.grad = 0

    def _get_topo(self):
        topo = []
        visited = set()

        def _build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    _build_topo(child)

                topo.append(v)

        _build_topo(self)
        return topo


    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)