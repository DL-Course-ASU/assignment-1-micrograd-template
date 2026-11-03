import math


def _wrap(other):
    """Turn a plain int/float into a Value; reject everything else."""
    if isinstance(other, Value):
        return other
    if isinstance(other, (int, float)):
        return Value(other)
    raise TypeError(f"unsupported operand type: {type(other).__name__}")


class Value:
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
            # d(a+b)/da = 1, d(a+b)/db = 1  (+= so shared nodes accumulate)
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
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __truediv__(self, other):  # self / other
        # other ** -1 raises ZeroDivisionError when other == 0
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # power rule: d(a^n)/da = n * a^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):  # tanh(self)
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh x)/dx = 1 - tanh^2(x)
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):  # e**self
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            # d(e^x)/dx = e^x
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        """Post-order DFS: every node is appended AFTER all of its children.

        Written iteratively (explicit stack) so very deep graphs do not hit
        Python's recursion limit; the visiting order is identical to the
        recursive build_topo from the hint. The `visited` set prevents
        double-visiting nodes shared by several paths (diamond graphs).
        """
        topo, visited = [], {self}
        stack = [(self, iter(self._prev))]
        while stack:
            node, children = stack[-1]
            for child in children:
                if child not in visited:
                    visited.add(child)
                    stack.append((child, iter(child._prev)))
                    break
            else:
                topo.append(node)  # post-order: after the children loop
                stack.pop()
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0  # seed d(out)/d(out) = 1
        for node in reversed(topo):
            node._backward()

    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)