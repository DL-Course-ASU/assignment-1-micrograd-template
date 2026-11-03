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
        other = other if isinstance(other, Value) else Value(other)
        ans = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += ans.grad
            other.grad += ans.grad
        ans._backward = _backward
        return ans
        # TODO: Return new Value and implement _backward
        

    def __radd__(self, other):  # other + self
        # TODO: Implement reverse addition
        return self + other

    def __mul__(self, other):  # self * other
        # TODO: Handle both Value and scalar inputs
        other = other if isinstance(other, Value) else Value(other)
        ans = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * ans.grad
            other.grad += self.grad * ans.grad
        ans._backward = _backward
        return ans
        # TODO: Return new Value and implement _backward using product rule
        pass

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
        assert isinstance(other, (int, float))
        # TODO: Return new Value and implement _backward using power rule
        ans = Value(self.data**other, (self, ), 'pow')
        
        def _backward():
            self.grad += other * (self.data**(other-1)) * ans.grad
        ans._backward = _backward
        return ans

    def tanh(self):  # tanh(self)
        # TODO: Implement hyperbolic tangent activation
        # TODO: Implement _backward for tanh
        num = self.data
        temp = (math.exp(2*num)-1) / (math.exp(2*num)+1)
        ans = Value(temp, (self, ), 'tanh')
        
        def _backward():
            self.grad += (1 - temp**2) * ans.grad
        ans._backward = _backward
        return ans

    def exp(self):  # e**self
        # TODO: Implement exponential function (math.exp)
        num = self.data
        ans = Value(math.exp(num), (self, ), 'exp')
        
        def _backward():
            self.grad += ans.data * ans.grad
        ans._backward = _backward
        # TODO: Implement _backward for exp
        return ans

    def reset_grad(self):
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
        for node in reversed(topo):
            node._backward()

    def __int__(self):  # int(self)
        # TODO: Return data as int
        return int(self.data)

    def __float__(self):  # float(self)
        # TODO: Return data as float
        return float(self.data)