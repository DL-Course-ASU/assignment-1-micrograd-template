import math


class Value:
    """A scalar node in a computational graph that supports reverse-mode autodiff.

    Every arithmetic operation on Value objects creates a new Value that remembers
    the nodes it was computed from (``_prev``), the operation that produced it
    (``_op``) and a closure (``_backward``) that applies the chain rule locally:
    it takes ``out.grad`` (dL/d out) and adds the contribution
    ``dL/d out * d out/d input`` to each input's ``grad``.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # means no effect by default
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _wrap(other):
        """Convert an int/float to a constant Value; return None for unsupported types.

        Returning None lets the operators return ``NotImplemented``, so Python
        raises a proper ``TypeError`` for things like ``Value(2.0) + "string"``.
        """
        if isinstance(other, Value):
            return other
        if isinstance(other, (int, float)):
            return Value(other)
        return None

    # ---------------------------------------------------------------- operators
    def __add__(self, other):  # self + other
        other = Value._wrap(other)
        if other is None:
            return NotImplemented
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a + b)/da = 1 and d(a + b)/db = 1: the gradient passes through unchanged.
            # "+=" (not "=") so a node used several times accumulates every contribution.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        other = Value._wrap(other)
        if other is None:
            return NotImplemented
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Product rule: d(a * b)/da = b and d(a * b)/db = a.
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
        # a / b = a * b**-1. A Value divisor is raised with __pow__; a plain number
        # is raised directly. In both cases 0 ** -1 raises ZeroDivisionError.
        if not isinstance(other, (Value, int, float)):
            return NotImplemented
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # Power rule: d(a^n)/da = n * a^(n-1).
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    # -------------------------------------------------------------- activations
    def tanh(self):  # tanh(self)
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d tanh(x)/dx = 1 - tanh(x)^2, reusing the forward result t.
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):  # e**self
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            # d e^x/dx = e^x, which is exactly the forward output.
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    # ----------------------------------------------------------------- autodiff
    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        """Return every node reachable from self, ordered so each node comes after its inputs.

        Depth-first search with post-order insertion: a node is appended only after all
        of its children (inputs) have been appended. The ``visited`` set ensures a node
        shared by several paths (diamond graph, e.g. ``a + a``) appears exactly once.
        """
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)  # post-order: after all children

        build_topo(self)
        return topo

    def backward(self):
        """Backpropagate from this node: fill ``grad`` of every node in its graph.

        Gradients accumulate (``+=``), as in PyTorch; call ``reset_grad``/``zero_grad``
        before a new backward pass on the same graph.
        """
        topo = self._get_topo()
        self.grad = 1.0  # seed: dself/dself = 1
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------- conversions
    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)
