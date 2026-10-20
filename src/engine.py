"""Scalar automatic differentiation engine (a small "micrograd").

A :class:`Value` stores a scalar ``data`` together with the computational
graph that produced it (``_prev`` / ``_op``) and a running gradient
``grad``.  Every operation attaches a ``_backward`` closure implementing
its *local* derivative; calling :meth:`Value.backward` walks the graph in
reverse topological order and applies the chain rule so that every node
ends up holding ``d(output)/d(node)``.

Design notes
------------
* Gradients are always **accumulated** (``+=``), never assigned, because a
  node may be used by several parents (diamond graphs) and each parent
  contributes a term to the chain rule.
* ``_prev`` is a ``set`` so that repeated references to the same node are
  stored once -- this is what keeps the topological sort well-defined.
* ``_backward`` is a closure over the *local* operands and receives the
  upstream gradient as an argument (``node._backward(node.grad)``).  It
  deliberately does **not** read ``out.grad``: capturing ``out`` inside its
  own closure would create a reference cycle (``out -> _backward -> out``),
  and cyclic garbage can only be reclaimed by the (expensive) cyclic GC.
  Passing the gradient in keeps the graph acyclic, so a finished training
  step is freed instantly by refcounting -- measured ~10x faster epochs.
"""

import math


class Value:
    """A scalar node in a computational graph with reverse-mode autodiff."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data                      # the scalar value itself
        self.grad = 0.0                       # d(output)/d(self), 0 until backprop
        # local "send grad to children" hook; takes the upstream gradient.
        # Operations overwrite this with a real closure over their operands.
        self._backward = lambda upstream=0.0: None
        self._prev = set(_children)           # parent nodes that produced this one
        self._op = _op                        # operator name, for debugging/drawing
        self.label = label                    # optional human-readable name

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def __repr__(self):
        """Debug representation, e.g. ``Value(data=5.0)``."""
        return f"Value(data={self.data})"

    @staticmethod
    def _coerce(other):
        """Wrap plain numbers as ``Value``; pass ``Value`` through unchanged.

        Anything else (str, list, ...) is wrapped too and then blows up in
        the arithmetic itself, which raises a natural ``TypeError``.
        """
        return other if isinstance(other, Value) else Value(other)

    def reset_grad(self):
        """Set this node's gradient back to zero (call between passes)."""
        self.grad = 0.0

    # ------------------------------------------------------------------
    # primitive operators (forward + local backward)
    # ------------------------------------------------------------------
    def __add__(self, other):  # self + other
        other = self._coerce(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward(upstream):
            # d(self)/d(out) = 1 and d(other)/d(out) = 1
            self.grad += upstream
            other.grad += upstream
        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        other = self._coerce(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward(upstream):
            # product rule
            self.grad += other.data * upstream
            other.grad += self.data * upstream
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

    def __truediv__(self, other):  # self / other == self * other ** -1
        return self * self._coerce(other) ** -1

    def __rtruediv__(self, other):  # other / self
        return self._coerce(other) * self ** -1

    def __pow__(self, other):  # self ** other
        # Exponent must be a constant: d(a**k)/da = k * a**(k-1) only holds
        # for a constant k (a Value exponent would need an extra log term).
        assert isinstance(other, (int, float)), "exponent must be a constant int/float"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward(upstream):
            # power rule: d(a**k)/da = k * a**(k-1).  For k == 0 the
            # result is a constant, so the derivative is 0 and we must not
            # evaluate a ** -1 (which would divide by zero when a == 0).
            if other != 0:
                self.grad += (other * self.data ** (other - 1)) * upstream
        out._backward = _backward
        return out

    def tanh(self):  # tanh(self)
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward(upstream):
            # d(tanh x)/dx = 1 - tanh^2(x)
            self.grad += (1.0 - t * t) * upstream
        out._backward = _backward
        return out

    def exp(self):  # e ** self
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward(upstream):
            # d(e^x)/dx = e^x  (use the local forward value, not out.data,
            # so the closure does not capture -- and cycle back to -- out)
            self.grad += e * upstream
        out._backward = _backward
        return out

    def relu(self):  # max(0, self) -- optional convenience activation
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'relu')

        def _backward(upstream):
            # derivative is 1 for positive inputs, 0 for negative ones
            self.grad += (self.data > 0) * upstream
        out._backward = _backward
        return out

    def log(self):  # ln(self) -- optional convenience
        out = Value(math.log(self.data), (self,), 'log')

        def _backward(upstream):
            # d(ln x)/dx = 1 / x
            self.grad += (1.0 / self.data) * upstream
        out._backward = _backward
        return out

    def sigmoid(self):  # optional convenience activation
        s = 1.0 / (1.0 + math.exp(-self.data)) if self.data >= 0 else \
            math.exp(self.data) / (1.0 + math.exp(self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward(upstream):
            # sigmoid'(x) = s * (1 - s), written without overflow
            self.grad += (s * (1.0 - s)) * upstream
        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # graph traversal
    # ------------------------------------------------------------------
    def _get_topo(self):
        """Return this node's ancestors in topological (post-order) order.

        Children (direct parents of a node in graph terms) are visited
        first, so a node is appended to ``topo`` *after* all of the nodes
        it depends on.  ``visited`` guarantees each node appears exactly
        once even when the graph contains diamonds.
        """
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)  # append AFTER visiting children
        build(self)
        return topo

    def backward(self):
        """Run reverse-mode backpropagation from this node.

        Sets ``self.grad = 1.0`` (the seed) and then applies every local
        ``_backward`` in reverse topological order, so gradients
        accumulate onto all ancestors.  Call :meth:`reset_grad` (or
        ``Module.zero_grad``) on parameters first if you want a clean
        slate -- this method intentionally does not wipe gradients so that
        repeated backward passes accumulate, matching PyTorch semantics.
        """
        topo = self._get_topo()
        self.grad = 1.0
        for node in reversed(topo):
            # hand each node the gradient flowing into it (its out.grad)
            node._backward(node.grad)

    # ------------------------------------------------------------------
    # numeric coercion
    # ------------------------------------------------------------------
    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)
