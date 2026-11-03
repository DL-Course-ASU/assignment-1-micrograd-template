import math
import numbers


def _wrap(other):
    """Turn a plain number into a Value so the operators don't have to care.

    numbers.Real instead of (int, float) so that numpy scalars coming out of a
    dataset (np.float64, np.int64) are accepted too - anything that isn't a
    real number is a genuine mistake and should blow up here.
    """
    if isinstance(other, Value):
        return other
    if isinstance(other, numbers.Real):
        return Value(float(other))
    raise TypeError(f"unsupported operand type for Value: {type(other).__name__}")


class Value:
    """A single scalar plus the little piece of graph that produced it."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # means no effect by default
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):  # self + other
        other = _wrap(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = d(a+b)/db = 1, so the incoming gradient just flows
            # through unchanged. += and not =, otherwise a node that is used
            # twice loses one of its two contributions
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        other = _wrap(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # product rule: d(a*b)/da = b and d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-_wrap(other))

    def __rsub__(self, other):  # other - self
        return (-self) + other

    def __truediv__(self, other):  # self / other
        # a / b is just a * b**-1, so division reuses __pow__ and we don't
        # have to write a second derivative for it
        return self * _wrap(other) ** -1

    def __rtruediv__(self, other):  # other / self
        return self ** -1 * other

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), "only int/float exponents are supported"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # power rule: d(a**n)/da = n * a**(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):  # tanh(self)
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)**2, and tanh(x) is already sitting
            # in t so there is no reason to recompute it
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):  # e**self
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            # d(e^x)/dx = e^x, which is exactly the forward result
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                # append after visiting the children, never before: that is
                # what guarantees every node appears later in the list than
                # the nodes it was built from
                topo.append(node)

        build_topo(self)
        return topo

    def backward(self):
        topo = self._get_topo()

        # seed the recursion, d(self)/d(self) = 1
        self.grad = 1.0

        # walking backwards means a node is only processed once everything
        # downstream of it has already poured its gradient in
        for node in reversed(topo):
            node._backward()

    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)
