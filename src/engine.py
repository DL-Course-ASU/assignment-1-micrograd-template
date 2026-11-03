import math
import numbers


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

    @staticmethod
    def _as_value(other):
        """Wrap a real number in a Value; reject anything else with TypeError."""
        if isinstance(other, Value):
            return other
        if isinstance(other, numbers.Real):
            return Value(other)
        raise TypeError(f"unsupported operand type for Value: {type(other).__name__}")

    def __add__(self, other):  # self + other
        other = self._as_value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a + b)/da = 1 and d(a + b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        other = self._as_value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a * b)/da = b and d(a * b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-self._as_value(other))

    def __rsub__(self, other):  # other - self
        return self._as_value(other) + (-self)

    def __truediv__(self, other):  # self / other
        return self * self._as_value(other) ** -1

    def __rtruediv__(self, other):  # other / self
        return self._as_value(other) * self ** -1

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), "only int/float exponents are supported"
        result = self.data ** other
        if isinstance(result, complex):
            raise ValueError(f"{self.data} ** {other} is not a real number")
        out = Value(result, (self,), f'**{other}')

        def _backward():
            # d(a ** n)/da = n * a ** (n - 1). A constant (n == 0) has zero slope,
            # and skipping it avoids evaluating 0 ** -1 when a == 0.
            if other != 0:
                self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):  # tanh(self)
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh(a))/da = 1 - tanh(a) ** 2
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):  # e**self
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            # d(e ** a)/da = e ** a
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
                topo.append(node)  # post-order: only after all children are placed

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
