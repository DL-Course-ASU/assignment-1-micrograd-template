"""A tiny scalar reverse-mode automatic differentiation engine.

Every arithmetic operation on a Value creates a new Value that remembers its
inputs (_prev) and how to send gradient back to them (_backward). Together
those Values form a computational graph, and backward() walks that graph from
the output to the inputs, applying the chain rule one node at a time.
"""
import math


class Value:
    """A single number plus the gradient of the final output with respect to it."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # means no effect by default
        self._backward = lambda: None  # a leaf has no inputs to send gradient to
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    # ------------------------------------------------------------------
    # Core operations. Each one computes the result (forward pass) and
    # defines a closure holding its local derivatives (backward pass).
    #
    # Every closure uses += and never = because a node may feed several
    # later operations: its total gradient is the SUM over all those paths.
    # ------------------------------------------------------------------

    def __add__(self, other):  # self + other
        # Wrap plain numbers. Returning NotImplemented for anything else makes
        # Python raise TypeError, so "abc" or a list can never slip into a graph.
        if isinstance(other, (int, float)):
            other = Value(other)
        elif not isinstance(other, Value):
            return NotImplemented

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a + b)/da = 1 and d(a + b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):  # self * other
        if isinstance(other, (int, float)):
            other = Value(other)
        elif not isinstance(other, Value):
            return NotImplemented

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a * b)/da = b and d(a * b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):  # self ** other
        # Only constant exponents are supported, so the power rule is enough.
        assert isinstance(other, (int, float)), "only int/float exponents are supported"

        # A negative base with a fractional exponent gives a complex number in
        # Python, which this engine does not support.
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # d(a ** n)/da = n * a ** (n - 1)
            # a ** 0 is the constant 1, so its derivative is 0. Skipping it avoids
            # computing 0.0 ** -1 (a ZeroDivisionError) when a is also 0.
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

    def exp(self):  # e ** self
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            # d(e ** a)/da = e ** a, which is out.data
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Derived operations. They are built from the core operations above, so
    # they need no _backward of their own: the graph records the core steps.
    # ------------------------------------------------------------------

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    # The reversed versions run when the left operand is a plain number, as in
    # 2 * x. Python tries int.__mul__ first, gets NotImplemented, then calls
    # x.__rmul__(2).

    def __radd__(self, other):  # other + self
        return self + other

    def __rmul__(self, other):  # other * self
        return self * other

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        """Return every node reachable from self, inputs first and self last."""
        topo = []
        visited = set()  # guarantees each node appears once, however many paths reach it

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                # Post-order: a node is added only after everything it depends on.
                topo.append(node)

        build(self)
        return topo

    def backward(self):
        """Fill in .grad for every node: d(self)/d(node)."""
        topo = self._get_topo()
        self.grad = 1.0  # d(self)/d(self) = 1, the starting point of the chain rule
        # Reversed post-order visits a node only after all of its consumers, so
        # its gradient is complete by the time it passes it on to its inputs.
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------
    # Type casting
    # ------------------------------------------------------------------

    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)
