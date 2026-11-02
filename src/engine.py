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
        if not isinstance(other, (Value, int, float)):
            raise TypeError(f"unsupported operand type(s) for +: 'Value' and '{type(other).__name__}'")
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1  (+= so shared nodes accumulate)
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):  # other + self
        # TODO: Implement reverse addition
        return self + other

    def __mul__(self, other):  # self * other
        # TODO: Handle both Value and scalar inputs
        # TODO: Return new Value and implement _backward using product rule
        if not isinstance(other, (Value, int, float)):
            raise TypeError(f"unsupported operand type(s) for *: 'Value' and '{type(other).__name__}'")
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # product rule: d(ab)/da = b, d(ab)/db = a
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
        return self * other**-1

    def __pow__(self, other):  # self ** other
        # TODO: Assert other is int or float
        # TODO: Return new Value and implement _backward using power rule
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # power rule: d(a^n)/da = n * a^(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):  # tanh(self)
        # TODO: Implement hyperbolic tangent activation
        # TODO: Implement _backward for tanh
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d tanh(x)/dx = 1 - tanh(x)^2
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):  # e**self
        # TODO: Implement exponential function (math.exp)
        # TODO: Implement _backward for exp
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            # d e^x/dx = e^x
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

        def build_topo(node):
            if node not in visited:
                visited.add(node)               # visited set avoids the diamond-graph trap
                for child in node._prev:
                    build_topo(child)
                topo.append(node)               # post-order: append AFTER visiting children
        build_topo(self)
        return topo

    def backward(self):
        # TODO: Retrieve topological order, set root grad to 1.0, and apply _backward in reverse
        topo = self._get_topo()
        self.grad = 1.0                         # seed d(out)/d(out) = 1
        for node in reversed(topo):
            node._backward()

    def __int__(self):  # int(self)
        # TODO: Return data as int
        return int(self.data)

    def __float__(self):  # float(self)
        # TODO: Return data as float
        return float(self.data)