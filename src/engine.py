import math
import numbers


class Value:
    """A scalar node in a computational graph with reverse-mode autodiff.

    Every operation on Values creates a new Value that remembers its operands
    (``_prev``), the operation that produced it (``_op``) and a closure
    (``_backward``) that pushes the output's gradient back to its operands
    using that operation's local derivative (the chain rule).
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

    # ------------------------------------------------------------------ #
    # Helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _wrap(other):
        """Promote a Python number to a Value; reject anything else."""
        if isinstance(other, Value):
            return other
        if isinstance(other, numbers.Real):   # int, float, numpy scalars...
            return Value(float(other))
        raise TypeError(
            f"unsupported operand type for Value: '{type(other).__name__}'"
        )

    # ------------------------------------------------------------------ #
    # Arithmetic operators (each builds one node of the forward graph)
    # ------------------------------------------------------------------ #
    def __add__(self, other):  # self + other
        other = Value._wrap(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1 and d(a+b)/db = 1 -> gradient is routed unchanged.
            # "+=" (not "=") so contributions from every path accumulate.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        other = Value._wrap(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Product rule: d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-Value._wrap(other))

    def __rsub__(self, other):  # other - self
        return Value._wrap(other) + (-self)

    def __truediv__(self, other):  # self / other  ==  self * other**-1
        return self * Value._wrap(other) ** -1

    def __rtruediv__(self, other):  # other / self
        return Value._wrap(other) * self ** -1

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), \
            "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # Power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Activation / elementary functions
    # ------------------------------------------------------------------ #
    def tanh(self):  # tanh(self)
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh x)/dx = 1 - tanh(x)^2  (reuses the forward result t)
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):  # e**self
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            # d(e^x)/dx = e^x  (reuses the forward result e)
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def relu(self):  # max(0, self) -- extra activation, used in the notebook
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Backpropagation
    # ------------------------------------------------------------------ #
    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        """All nodes reachable from self, ordered so that every node comes
        AFTER all of its children (inputs). Built with post-order DFS."""
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)          # visit each node once (diamond graphs)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)          # post-order: after its children

        build_topo(self)
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0                    # seed: d(out)/d(out) = 1
        for node in reversed(topo):        # output -> inputs
            node._backward()

    # ------------------------------------------------------------------ #
    # Casting
    # ------------------------------------------------------------------ #
    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)
