import math
import numbers

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data) if isinstance(data, (int, float, numbers.Number)) else data
        self.grad = 0.0  # means no effect by default
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):  # self + other
        if not isinstance(other, (Value, int, float, numbers.Number)):
            raise TypeError(f"Unsupported operand type for +: 'Value' and '{type(other).__name__}'")
        other = other if isinstance(other, Value) else Value(float(other))
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        if not isinstance(other, (Value, int, float, numbers.Number)):
            raise TypeError(f"Unsupported operand type for *: 'Value' and '{type(other).__name__}'")
        other = other if isinstance(other, Value) else Value(float(other))
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        if not isinstance(other, (Value, int, float, numbers.Number)):
            raise TypeError(f"Unsupported operand type for -: 'Value' and '{type(other).__name__}'")
        other = other if isinstance(other, Value) else Value(float(other))
        return self + (-other)

    def __rsub__(self, other):  # other - self
        if not isinstance(other, (Value, int, float, numbers.Number)):
            raise TypeError(f"Unsupported operand type for -: '{type(other).__name__}' and 'Value'")
        other = other if isinstance(other, Value) else Value(float(other))
        return other + (-self)

    def __truediv__(self, other):  # self / other
        if not isinstance(other, (Value, int, float, numbers.Number)):
            raise TypeError(f"Unsupported operand type for /: 'Value' and '{type(other).__name__}'")
        if isinstance(other, Value):
            if other.data == 0:
                raise ZeroDivisionError("division by zero")
            return self * (other ** -1)
        else:
            if other == 0:
                raise ZeroDivisionError("division by zero")
            return self * (float(other) ** -1)

    def __rtruediv__(self, other):  # other / self
        if not isinstance(other, (Value, int, float, numbers.Number)):
            raise TypeError(f"Unsupported operand type for /: '{type(other).__name__}' and 'Value'")
        if self.data == 0:
            raise ZeroDivisionError("division by zero")
        other = other if isinstance(other, Value) else Value(float(other))
        return other * (self ** -1)

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float, numbers.Number)) and not isinstance(other, Value), "only supporting int/float powers for now"
        exp_val = int(other) if isinstance(other, (int, numbers.Integral)) and not isinstance(other, float) else float(other)
        out = Value(self.data ** exp_val, (self,), f'**{exp_val}')

        def _backward():
            self.grad += (exp_val * (self.data ** (exp_val - 1))) * out.grad
        out._backward = _backward

        return out

    def tanh(self):  # tanh(self)
        t = math.tanh(self.data)
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