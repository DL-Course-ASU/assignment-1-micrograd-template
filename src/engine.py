import math

class Value:
    """
    Stores a single scalar value and its gradient for reverse-mode automatic differentiation.
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data) if isinstance(data, (int, float)) and not isinstance(data, bool) else data
        self.grad = 0.0  # means no effect by default
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):  # self + other
        if not isinstance(other, (Value, int, float)) or isinstance(other, bool):
            raise TypeError(f"unsupported operand type(s) for +: 'Value' and '{type(other).__name__}'")
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):  # other + self
        if not isinstance(other, (Value, int, float)) or isinstance(other, bool):
            raise TypeError(f"unsupported operand type(s) for +: '{type(other).__name__}' and 'Value'")
        return self + other

    def __mul__(self, other):  # self * other
        if not isinstance(other, (Value, int, float)) or isinstance(other, bool):
            raise TypeError(f"unsupported operand type(s) for *: 'Value' and '{type(other).__name__}'")
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        if not isinstance(other, (Value, int, float)) or isinstance(other, bool):
            raise TypeError(f"unsupported operand type(s) for *: '{type(other).__name__}' and 'Value'")
        return self * other

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)) and not isinstance(other, bool), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):  # -self
        return self * -1.0

    def __sub__(self, other):  # self - other
        if not isinstance(other, (Value, int, float)) or isinstance(other, bool):
            raise TypeError(f"unsupported operand type(s) for -: 'Value' and '{type(other).__name__}'")
        return self + (-other)

    def __rsub__(self, other):  # other - self
        if not isinstance(other, (Value, int, float)) or isinstance(other, bool):
            raise TypeError(f"unsupported operand type(s) for -: '{type(other).__name__}' and 'Value'")
        other = other if isinstance(other, Value) else Value(other)
        return other + (-self)

    def __truediv__(self, other):  # self / other
        if not isinstance(other, (Value, int, float)) or isinstance(other, bool):
            raise TypeError(f"unsupported operand type(s) for /: 'Value' and '{type(other).__name__}'")
        other = other if isinstance(other, Value) else Value(other)
        if other.data == 0.0:
            raise ZeroDivisionError("division by zero")
        return self * (other ** -1.0)

    def __rtruediv__(self, other):  # other / self
        if not isinstance(other, (Value, int, float)) or isinstance(other, bool):
            raise TypeError(f"unsupported operand type(s) for /: '{type(other).__name__}' and 'Value'")
        other = other if isinstance(other, Value) else Value(other)
        if self.data == 0.0:
            raise ZeroDivisionError("division by zero")
        return other * (self ** -1.0)

    def tanh(self):  # tanh(self)
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward

        return out

    def exp(self):  # e**self
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
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
                topo.append(node)

        build_topo(self)
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
