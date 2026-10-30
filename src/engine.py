import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        if not isinstance(data, (int, float)):
            raise TypeError(f"Value only accepts numeric data (int, float), got {type(data).__name__}")
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        if not isinstance(other, (Value, int, float)):
            raise TypeError(f"unsupported operand type(s) for +: 'Value' and '{type(other).__name__}'")
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, (Value, int, float)):
            raise TypeError(f"unsupported operand type(s) for *: 'Value' and '{type(other).__name__}'")
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        if not isinstance(other, (Value, int, float)):
            raise TypeError(f"unsupported operand type(s) for -: 'Value' and '{type(other).__name__}'")
        return self + (-other)

    def __rsub__(self, other):
        if not isinstance(other, (Value, int, float)):
            raise TypeError(f"unsupported operand type(s) for -: '{type(other).__name__}' and 'Value'")
        return Value(other) + (-self)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), f"only int/float powers currently supported, got {type(other).__name__}"
        if self.data == 0.0 and other < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")

        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        if not isinstance(other, (Value, int, float)):
            raise TypeError(f"unsupported operand type(s) for /: 'Value' and '{type(other).__name__}'")
        other_val = other.data if isinstance(other, Value) else other
        if other_val == 0.0:
            raise ZeroDivisionError("division by zero")
        return self * (other ** -1)

    def __rtruediv__(self, other):
        if not isinstance(other, (Value, int, float)):
            raise TypeError(f"unsupported operand type(s) for /: '{type(other).__name__}' and 'Value'")
        if self.data == 0.0:
            raise ZeroDivisionError("division by zero")
        return Value(other) * (self ** -1)

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def _get_topo(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        return topo

    def reset_grad(self):
        for node in self._get_topo():
            node.grad = 0.0

    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __int__(self):
        return int(self.data)

    def __float__(self):
        return float(self.data)
