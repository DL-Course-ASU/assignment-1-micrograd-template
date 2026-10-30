"""
engine.py -- a tiny scalar automatic-differentiation engine (MicroGrad).

Every ``Value`` wraps one Python float (``data``) together with the derivative
of the *final output* with respect to it (``grad``).  Each arithmetic operation
creates a new ``Value`` and remembers (a) which Values it was computed from
(``_prev``) and (b) a closure ``_backward`` that knows the *local* derivative of
that single operation.  Calling ``backward()`` on the output walks the graph in
reverse topological order and applies the chain rule node by node.
"""
import math
import numbers


class Value:
    """A scalar that records the computational graph that produced it."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # means no effect by default
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _wrap(other):
        """Promote a real number to a Value; reject every other type.

        ``bool`` is an ``int`` subclass and is accepted (True == 1).  Strings,
        lists, None, ... raise ``TypeError`` so mistakes surface immediately
        instead of failing obscurely later inside the graph.
        """
        if isinstance(other, Value):
            return other
        if isinstance(other, numbers.Real):          # int, float, bool, numpy scalars...
            if not isinstance(other, (int, float)):
                other = float(other)                  # e.g. np.int64 -> float
            return Value(other)
        raise TypeError(
            f"unsupported operand type for Value arithmetic: {type(other).__name__!r}"
        )

    def __repr__(self):
        return f"Value(data={self.data})"

    # ------------------------------------------------------------------ #
    # forward ops + local derivatives
    # ------------------------------------------------------------------ #
    def __add__(self, other):  # self + other
        other = Value._wrap(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1.  "+=" (not "=") so that a node
            # used several times (diamond graph) accumulates every path.
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
        return self + (-Value._wrap(other))

    def __rsub__(self, other):  # other - self
        return Value._wrap(other) + (-self)

    def __truediv__(self, other):  # self / other
        # a / b = a * b**-1  (raises ZeroDivisionError if b == 0)
        return self * Value._wrap(other) ** -1

    def __rtruediv__(self, other):  # other / self
        return Value._wrap(other) * self ** -1

    def __pow__(self, other):  # self ** other
        assert isinstance(other, numbers.Real), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # power rule: d(a^n)/da = n * a^(n-1)   (n is a constant)
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # activation / elementary functions
    # ------------------------------------------------------------------ #
    def tanh(self):  # tanh(self)
        t = math.tanh(self.data)  # math.tanh is overflow-safe for large |x|
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d tanh(x)/dx = 1 - tanh(x)^2
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):  # e**self
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            # d e^x/dx = e^x  (the output itself)
            self.grad += e * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # backpropagation
    # ------------------------------------------------------------------ #
    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        """Return every node reachable from ``self`` in topological order
        (children/inputs first, ``self`` last) using a depth-first search.

        This is the exact algorithm from the assignment hint (visited set +
        *post-order* append), written with an explicit stack instead of Python
        recursion so that very deep graphs (e.g. a loss summed over thousands
        of samples) never hit ``RecursionError``.  See ``_get_topo_recursive``
        for the literal recursive version; the two always agree.
        """
        topo, visited = [], set()
        stack = [(self, iter(self._prev))]
        visited.add(self)
        while stack:
            node, children = stack[-1]
            for child in children:
                if child not in visited:
                    visited.add(child)
                    stack.append((child, iter(child._prev)))
                    break
            else:  # all children finished -> post-order append
                topo.append(node)
                stack.pop()
        return topo

    def _get_topo_recursive(self):
        """Literal recursive version of the hint (kept for reference/tests)."""
        topo, visited = [], set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)                 # 1. mark visited
                for child in node._prev:          # 2. visit every child
                    build_topo(child)
                topo.append(node)                 # 3. append AFTER children (post-order)

        build_topo(self)
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0  # seed: d(self)/d(self) = 1
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------ #
    # casting
    # ------------------------------------------------------------------ #
    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)
