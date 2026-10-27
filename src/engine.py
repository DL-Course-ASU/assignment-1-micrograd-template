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
        # TODO: Return string representation, e.g., "Value(data=...)"
        return f"Value(data={self.data})"

    def __add__(self, other):  # self + other
        # TODO: Handle both Value and scalar inputs
        # TODO: Return new Value and implement _backward
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        # TODO: Implement reverse addition
        return self + other

    def __mul__(self, other):  # self * other
        if not isinstance(other, (Value, int, float)):
            raise TypeError("Can only multiply Value by a number or Value")

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        # TODO: Implement reverse multiplication
        return self * other

    def __neg__(self):  # -self
        # TODO: Implement negation
        return self * -1

    def __sub__(self, other):  # self - other
        # TODO: Implement subtraction
        return self + (-other)

    def __truediv__(self, other):  # self / other
        # TODO: Implement division using self * other**-1
        if not isinstance(other, (Value, int, float)):
            raise TypeError("Can only divide Value by a number or Value")

        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1

    def __pow__(self, other):  # self ** other
        # TODO: Assert other is int or float
        # TODO: Return new Value and implement _backward using power rule
        assert isinstance(other, (int, float))

        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):  # tanh(self)
        # TODO: Implement hyperbolic tangent activation
        # TODO: Implement _backward for tanh
        x = self.data
        t = math.tanh(x)

        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):  # e**self
        # TODO: Implement exponential function (math.exp)
        # TODO: Implement _backward for exp
        x = self.data
        e = math.exp(x)

        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def reset_grad(self):
        # TODO: Reset self.grad to 0.0
        self.grad = 0.0

    def _get_topo(self):
        # TODO: Return a list of all nodes in topological order
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

    def backward(self):
        # TODO: Retrieve topological order, set root grad to 1.0, and apply _backward in reverse
        topo = self._get_topo()

        self.grad = 1.0

        for v in reversed(topo):
            v._backward()

    def __int__(self):  # int(self)
        # TODO: Return data as int
        return int(self.data)

    def __float__(self):  # float(self)
        # TODO: Return data as float
        return float(self.data)