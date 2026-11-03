import math


class Value:
    """Stores a single scalar and its gradient, and remembers how it was made."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # means no effect by default
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    # ---------- helpers ----------
    @staticmethod
    def _wrap(other):
        """Turn plain numbers into Values, reject everything else."""
        if isinstance(other, Value):
            return other
        if isinstance(other, (int, float)):
            return Value(other)
        raise TypeError(f"unsupported operand type: {type(other).__name__}")

    # ---------- arithmetic ----------
    def __add__(self, other):  # self + other
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            # use += so a node used twice (a + a) accumulates both paths
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # product rule
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-self._wrap(other))

    def __rsub__(self, other):  # other - self
        return self._wrap(other) + (-self)

    def __truediv__(self, other):  # self / other
        return self * self._wrap(other) ** -1

    def __rtruediv__(self, other):  # other / self
        return self._wrap(other) * self ** -1

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    # ---------- activations ----------
    def tanh(self):  # tanh(self)
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d tanh(x)/dx = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):  # e**self
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            # d e^x/dx = e^x
            self.grad += e * out.grad
        out._backward = _backward
        return out

    # ---------- backprop ----------
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
                # post-order: only add a node after all its children
                topo.append(node)

        build_topo(self)
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0  # seed the output derivative
        for node in reversed(topo):
            node._backward()

    # ---------- casting ----------
    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)