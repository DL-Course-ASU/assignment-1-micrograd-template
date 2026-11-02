"""
engine.py -- a tiny scalar autograd engine (CSE485/488 Deep Learning, Assignment 1)

Every number in a computation is wrapped in a `Value`. While we do the normal
forward math, each Value quietly remembers (a) which Values it was made from and
(b) how to push gradient back to them. Calling `.backward()` on the final output
then walks the graph in reverse and applies the chain rule node by node.

Heavily inspired by Andrej Karpathy's micrograd, but written out by hand so I
actually understand every line.
"""

import math


class Value:
    """A single scalar plus the gradient of the final output w.r.t. this scalar."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        # Everything below is only used to rebuild the graph during backprop.
        self._backward = lambda: None      # leaf nodes have nothing to propagate
        self._prev = set(_children)        # the nodes that produced this one
        self._op = _op                     # just for debugging / drawing
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    # ------------------------------------------------------------------
    # Basic arithmetic. Each op builds the forward result and attaches a
    # small closure `_backward` that holds the *local* derivative.
    #
    # Important: we use `+=` on the gradients, not `=`. A Value can feed into
    # several downstream nodes (e.g. c = a + a) and each path contributes its
    # own piece of the gradient. Using `=` would silently overwrite them.
    # ------------------------------------------------------------------
    def __add__(self, other):
        if not isinstance(other, (Value, int, float)):
            return NotImplemented            # Python then raises TypeError for us
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1 and d(a+b)/db = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        if not isinstance(other, (Value, int, float)):
            return NotImplemented
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b and d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        # Only plain int/float exponents. Differentiating w.r.t. the exponent
        # would need a log and we don't need it for this assignment.
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # d(a^n)/da = n * a^(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Activation / elementary functions
    # ------------------------------------------------------------------
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d tanh(x)/dx = 1 - tanh(x)^2   (reuse the forward value t)
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            # d e^x / dx = e^x, which we already computed in the forward pass
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "ReLU")

        def _backward():
            # gradient passes through where the output is positive, else 0
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def log(self):
        assert self.data > 0, "log is only defined for positive values"
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Everything else is built from the primitives above, so it gets correct
    # gradients for free (no new _backward needed).
    # ------------------------------------------------------------------
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    def __radd__(self, other):      # 3 + Value
        return self + other

    def __rmul__(self, other):      # 3 * Value
        return self * other

    def __rsub__(self, other):      # 3 - Value
        return other + (-self)

    def __rtruediv__(self, other):  # 3 / Value
        return other * self ** -1

    def reset_grad(self):
        self.grad = 0.0

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------
    def _get_topo(self):
        """Return all nodes reachable from self, ordered so that every node
        appears AFTER all of the nodes it was computed from (post-order DFS)."""
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)               # mark first -> handles diamond graphs
                for child in node._prev:
                    build_topo(child)
                topo.append(node)               # append AFTER the children (post-order)

        build_topo(self)
        return topo

    def backward(self):
        topo = self._get_topo()

        # Seed: d(out)/d(out) = 1
        self.grad = 1.0

        # Walk from the output back to the inputs. In reversed topo order a node
        # is only visited once everything that depends on it has already pushed
        # its contribution into node.grad.
        for node in reversed(topo):
            node._backward()
