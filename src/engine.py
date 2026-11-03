import math
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        if not isinstance(data, (int, float)):
            raise TypeError("Value data must be an int or float")

        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    @staticmethod
    def _to_value(other):
        # Law el input raqam, bn7awelo le Value 3ashan n3raf n3mel 3aleh operations.
        if isinstance(other, Value):
            return other
        if isinstance(other, (int, float)):
            return Value(other)
        raise TypeError(f"Unsupported operand type: {type(other).__name__}")

    def __add__(self, other):
        other = self._to_value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Derivative of addition is 1 for both inputs.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._to_value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Hena bnesta5dem el product rule 3ashan n7seb gradient kol input.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._to_value(other))

    def __rsub__(self, other):
        return self._to_value(other) - self

    def __truediv__(self, other):
        other = self._to_value(other)
        if other.data == 0:
            raise ZeroDivisionError("division by zero")
        # a / b = a * b^(-1), fa bn3melha bel operations elly 3andena.
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return self._to_value(other) / self

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Exponent must be an int or float"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # Derivative of x^n is n * x^(n-1).
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # Derivative of tanh(x) is 1 - tanh(x)^2.
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            # El derivative beta3 exp(x) howa nafs el output.
            self.grad += e * out.grad

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

                # Lazem nزور el children awel, ba3den nضيف el node lel list.
                for child in node._prev:
                    build_topo(child)

                topo.append(node)

        build_topo(self)
        return topo

    def backward(self):
        topo = self._get_topo()

        # El output gradient byebda2 b 1, w ba3den byetwazza3 bel chain rule.
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)
