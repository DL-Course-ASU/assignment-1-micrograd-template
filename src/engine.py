"""engine.py - a tiny scalar automatic-differentiation engine (reverse-mode autodiff)."""
import math


class Value:
    """Stores a scalar, its gradient, and how it was produced (the computational graph)."""

    def __init__(self, data, _children=(), _op="", label=""):
        if not isinstance(data, (int, float)):
            raise TypeError("Value data must be an int or float")
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None      # local chain-rule step, set by each operator
        self._prev = set(_children)        # parents in the graph (the operands)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    # ---------------- forward ops (each defines its local backward) ----------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1.  "+=" accumulates (diamond graphs / reused nodes)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # d(a^n)/da = n * a^(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d tanh(x)/dx = 1 - tanh(x)^2
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            # d exp(x)/dx = exp(x)
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "ReLU")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data)) if self.data >= 0 else \
            math.exp(self.data) / (1.0 + math.exp(self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    # ---------------- derived ops (built from the primitives above) ----------------
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self ** -1

    # ---------------- backprop ----------------
    def _get_topo(self):
        """Post-order DFS: every node appears AFTER all of its children (operands)."""
        topo, visited = [], set()

        def build_topo(node):
            if node not in visited:          # 'visited' prevents re-visits in diamond graphs
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)            # appended AFTER the children (post-order)
        build_topo(self)
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0                      # seed: d(self)/d(self) = 1
        for node in reversed(topo):          # outputs first, inputs last
            node._backward()
