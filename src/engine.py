import math


class Value:
    """A scalar value that records the operations that produced it,
    so gradients can be computed with reverse-mode autodiff."""

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
        """Accept Value or a real number; reject anything else."""
        if isinstance(other, Value):
            return other
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return Value(other)
        raise TypeError(f"unsupported operand type for Value: {type(other).__name__}")

    # ---------- arithmetic ----------
    def __add__(self, other):  # self + other
        try:
            other = Value._wrap(other)
        except TypeError:
            return NotImplemented
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        try:
            other = Value._wrap(other)
        except TypeError:
            return NotImplemented
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # product rule: d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), "only int/float powers are supported"
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
            # d e^x / dx = e^x
            self.grad += e * out.grad
        out._backward = _backward
        return out

    # ---------- backprop ----------
    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        topo, visited = [], set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)  # post-order: after all children
        build_topo(self)
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0  # seed dL/dL = 1
        for node in reversed(topo):
            node._backward()

    # ---------- casting ----------
    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)
