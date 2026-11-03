import math


class Value:
    """A scalar node in a computational graph that supports reverse-mode autodiff."""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0  # msh hyasar
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    
    @staticmethod
    def _wrap(other):
        """Convert a Python scalar into a Value; reject anything else."""
        if isinstance(other, Value):
            return other
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return Value(other)
        raise TypeError(
            f"unsupported operand type for Value: '{type(other).__name__}'"
        )

    def __repr__(self):
        return f"Value(data={self.data})"

   
    def __add__(self, other):  # self + other
        other = Value._wrap(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        other = Value._wrap(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # product rule: d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-Value._wrap(other))

    def __rsub__(self, other):  # other - self
        return Value._wrap(other) + (-self)

    def __truediv__(self, other):  # self / other
        return self * Value._wrap(other) ** -1

    def __rtruediv__(self, other):  # other / self
        return Value._wrap(other) * self ** -1

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

  
    def tanh(self):  # tanh(self)
        t = math.tanh(self.data)  # numerically stable for large |x|
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh x)/dx = 1 - tanh^2(x)
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward

        return out

    def exp(self):  # e**self
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

  
    def __int__(self):  
        return int(self.data)

    def __float__(self):  
        return float(self.data)