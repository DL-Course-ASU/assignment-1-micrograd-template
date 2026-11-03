
import math
from numbers import Real


class Value:
    """A scalar value that supports automatic differentiation."""

    def __init__(self, data, _children=(), _op='', label=''):
        if isinstance(data, bool) or not isinstance(data, Real):
            raise TypeError("Value data must be a real numeric scalar")

        self.data = data
        self.grad = 0.0

        # Local gradient function, assigned by each operation.
        self._backward = lambda: None

        # Parent nodes in the computational graph.
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    @staticmethod
    def _as_value(other):
        """Convert valid numeric operands into Value objects."""
        if isinstance(other, Value):
            return other

        if isinstance(other, bool) or not isinstance(other, Real):
            raise TypeError(
                f"Unsupported operand type: {type(other).__name__}"
            )

        return Value(other)

    def __add__(self, other):
        other = self._as_value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._as_value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Product rule: gradients depend on opposite operands.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = self._as_value(other)
        return self + (-other)

    def __rsub__(self, other):
        other = self._as_value(other)
        return other - self

    def __truediv__(self, other):
        other = self._as_value(other)

        if other.data == 0:
            raise ZeroDivisionError("Division by zero")

        # a / b = a * b^(-1)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = self._as_value(other)
        return other / self

    def __pow__(self, other):
        # Preserve the assignment's requirement that exponents
        # must be numeric constants, not Value objects.
        assert isinstance(other, (int, float)) and \
            not isinstance(other, bool), \
            "Exponent must be an int or float"

        if not math.isfinite(other):
            raise ValueError("Exponent must be finite")

        if self.data == 0 and other < 0:
            raise ZeroDivisionError(
                "Zero cannot be raised to a negative power"
            )

        if self.data < 0 and not float(other).is_integer():
            raise ValueError(
                "Fractional powers of negative values "
                "are outside the real-number domain"
            )

        result = self.data ** other
        out = Value(result, (self,), f'**{other}')

        def _backward():
            # d(x^n)/dx = n*x^(n-1)
            if other == 0:
                local_grad = 0.0
            elif self.data == 0 and other < 1:
                raise ValueError(
                    "Derivative undefined at zero"
                )
            else:
                local_grad = other * self.data ** (other - 1)

            self.grad += local_grad * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
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
        """Return nodes in dependency-first topological order."""
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)

                # Visit parent nodes before appending this node.
                for parent in node._prev:
                    build_topo(parent)

                topo.append(node)

        build_topo(self)
        return topo

    def backward(self):
        """Propagate gradients from the output to its ancestors."""
        topo = self._get_topo()

        # Clear stale gradients throughout the reachable graph.
        for node in topo:
            node.reset_grad()

        # The derivative of the output with respect to itself.
        self.grad = 1.0

        # Reverse topological order applies the chain rule.
        for node in reversed(topo):
            node._backward()

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)
