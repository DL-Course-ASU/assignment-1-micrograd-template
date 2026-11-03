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
        return f'Value(data={self.data})'

    def __add__(self, other):  # self + other
        # TODO: Handle both Value and scalar inputs
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        # TODO: Return new Value and implement _backward
        out._backward = lambda: (setattr(self, 'grad', self.grad + out.grad),setattr(other, 'grad', other.grad + out.grad))
        return out

    def __radd__(self, other):  # other + self
        # TODO: Implement reverse addition
        return self + other

    def __mul__(self, other):  # self * other
        # TODO: Handle both Value and scalar inputs
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        # TODO: Return new Value and implement _backward using product rule
        out._backward = lambda: (setattr(self, 'grad', self.grad + out.grad * other.data),setattr(other, 'grad', other.grad + out.grad * self.data))
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
        return self * other**-1

    def __pow__(self, other):  # self ** other
        # TODO: Assert other is int or float
        assert isinstance(other, (int, float)), "Only supports int or float powers."
        out = Value(self.data ** other, (self,), f'**{other}')
        # TODO: Return new Value and implement _backward using power rule
        out._backward = lambda: setattr(self, 'grad', self.grad + out.grad * other * (self.data ** (other - 1)))
        return out

    def tanh(self):  # tanh(self)
        # TODO: Implement hyperbolic tangent activation
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t, (self,), 'tanh')
        # TODO: Implement _backward for tanh
        out._backward = lambda: setattr(self, 'grad', self.grad + out.grad * (1 - t ** 2))
        return out

    def exp(self):  # e**self
        # TODO: Implement exponential function (math.exp)
        out = Value(math.exp(self.data), (self,), 'exp')
        # TODO: Implement _backward for exp
        out._backward = lambda: setattr(self, 'grad', self.grad + out.grad * out.data)
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