import math


def _as_value(other):
    """Wrap a number in a Value; reject anything that is not numeric.

    The explicit check matters: without it, Value(2) * [1, 2, 3] would silently
    evaluate 2 * [1, 2, 3] and produce a repeated list instead of an error.
    """
    if isinstance(other, Value):
        return other
    if isinstance(other, (int, float)):
        return Value(other)
    raise TypeError(f"unsupported operand for Value: {type(other).__name__}")


class Value:
    """A scalar node in a computational graph, with reverse-mode autodiff."""

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
        other = _as_value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a + b)/da = 1, d(a + b)/db = 1
            # += (not =) so that a + a, where self is other, contributes twice.
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        other = _as_value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a * b)/da = b, d(a * b)/db = a
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
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), "only int/float exponents are supported"
        if self.data < 0 and not float(other).is_integer():
            # Python would return a complex number here, which the engine cannot represent.
            raise ValueError(f"{self.data} ** {other} is not a real number")
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # d(x^n)/dx = n * x^(n-1). The n == 0 case is skipped because the
            # derivative is 0 and evaluating 0.0 ** -1 would raise.
            if other != 0:
                self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):  # tanh(self)
        # math.tanh, not (e^2x - 1) / (e^2x + 1), which overflows for large x.
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh x)/dx = 1 - tanh^2 x
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):  # e**self
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            # d(e^x)/dx = e^x
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        """Return every node reachable from self, children before parents."""
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                # Post-order: append only after all children have been appended.
                topo.append(node)

        build_topo(self)
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0  # seed: d(self)/d(self) = 1
        for node in reversed(topo):
            node._backward()

    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)
