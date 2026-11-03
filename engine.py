"""engine.py - a tiny scalar automatic differentiation engine (MicroGrad)."""
import math


class Value:
    """Stores a scalar `data`, its gradient `grad`, and the graph that produced it."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None      # local chain-rule step, set by each operator
        self._prev = set(_children)        # parents in the computational graph
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    # ------------------------------------------------------------------ forward ops
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1.  "+=" accumulates over multiple uses.
            self.grad += out.grad
            other.grad += out.grad

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
        assert isinstance(other, (int, float)), "only int/float powers are supported"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # d(a^n)/da = n * a^(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------ activations
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

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------- derived ops (built from the above)
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    # --------------------------------------------------------------- backprop
    def _get_topo(self):
        """Post-order DFS. Every node appears AFTER all of its children (_prev).

        Written iteratively (explicit stack) so deep graphs cannot hit Python's
        recursion limit; the ordering is identical to the recursive version.
        """
        topo, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                topo.append(node)          # append AFTER visiting children
                continue
            if node in visited:            # diamond graphs: never expand twice
                continue
            visited.add(node)
            stack.append((node, True))
            for child in node._prev:
                stack.append((child, False))
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0                    # seed: d(self)/d(self) = 1
        for node in reversed(topo):
            node._backward()
