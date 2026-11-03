import math

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
        if isinstance(other, Value):
            other_value = other
        elif isinstance(other, (int, float)):
            other_value = Value(other)
        else:
            raise TypeError(f"unsupported operand type for +: '{type(other).__name__}'")

        out = Value(self.data + other_value.data, (self, other_value), '+')

        def _backward():
            self.grad += out.grad
            other_value.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        if isinstance(other, Value):
            other_value = other
        elif isinstance(other, (int, float)):
            other_value = Value(other)
        else:
            raise TypeError(f"unsupported operand type for *: '{type(other).__name__}'")

        out = Value(self.data * other_value.data, (self, other_value), '*')

        def _backward():
            self.grad += other_value.data * out.grad
            other_value.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        if not isinstance(other, (int, float, Value)):
            raise TypeError(f"unsupported operand type for -: '{type(other).__name__}'")
        return (-self) + other

    def __truediv__(self, other):  # self / other
        if isinstance(other, Value):
            if other.data == 0:
                raise ZeroDivisionError("division by zero")
        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("division by zero")
        else:
            raise TypeError(f"unsupported operand type for /: '{type(other).__name__}'")
        return self * (other ** -1)

    def __rtruediv__(self, other):  # other / self
        if self.data == 0:
            raise ZeroDivisionError("division by zero")
        if not isinstance(other, (int, float, Value)):
            raise TypeError(f"unsupported operand type for /: '{type(other).__name__}'")
        return other * (self ** -1)

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), "exponent must be an int or float"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            if other != 0:
                self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):  # tanh(self)
        value = math.tanh(self.data)
        out = Value(value, (self,), 'tanh')

        def _backward():
            self.grad += (1 - value ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):  # e**self
        value = math.exp(self.data)
        out = Value(value, (self,), 'exp')

        def _backward():
            self.grad += value * out.grad

        out._backward = _backward
        return out

    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self):
        topo = []
        visited = set()

        def visit(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    visit(parent)
                topo.append(node)

        visit(self)
        return topo

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)