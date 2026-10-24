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

    def __add__(self, other):
        if not isinstance(other, (Value, int, float)):
            raise TypeError("Unsupported type for addition")

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

    def __mul__(self, other):
        if not isinstance(other, (Value, int, float)):
            raise TypeError("Unsupported type for multiplication")

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
        return self * (other ** -1)

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def reset_grad(self):
        # TODO: Reset self.grad to 0.0
        self.grad = 0.0

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