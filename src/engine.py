import math


class Value:
    """A scalar node in a computational graph that stores data and its gradient."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # means no effect by default
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _wrap(other):
        """Promote int/float to Value; reject anything else with TypeError."""
        if isinstance(other, Value):
            return other
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return Value(float(other))
        raise TypeError(f"unsupported operand type for Value: '{type(other).__name__}'")

    def __repr__(self):
        return f"Value(data={self.data})"

    # --------------------------------------------------------------- operators
    def __add__(self, other):  # self + other
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1  -> gradient flows through unchanged
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        other = self._wrap(other)
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
        return self + (-self._wrap(other))

    def __rsub__(self, other):  # other - self
        return self._wrap(other) + (-self)

    def __truediv__(self, other):  # self / other
        return self * self._wrap(other) ** -1

    def __rtruediv__(self, other):  # other / self
        return self._wrap(other) * self ** -1

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), "only int/float powers are supported"
        if self.data == 0 and other < 0:
            raise ZeroDivisionError("0 cannot be raised to a negative power")
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    # ------------------------------------------------------------- activations
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

    # ------------------------------------------------------------ backprop
    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        """Return all nodes reachable from self in topological order (DFS post-order)."""
        topo, visited = [], set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)  # post-order: only after all children
        build_topo(self)
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0  # seed: d(self)/d(self) = 1
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------ casting
    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)
