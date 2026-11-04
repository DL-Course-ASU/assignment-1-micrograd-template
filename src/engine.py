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
       other = other if isinstance(other, Value) else Value(other)
       out = Value(
           self.data + other.data,
           (self,other),
           '+'
       )

       def _backward():
           self.grad += 1.0 * out.grad
           other.grad +=1.0 * out.grad
       out._backward = _backward
       return out  

    def __radd__(self, other):  # other + self
        return self + other

    def __mul__(self, other):  # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(
            self.data * other.data,
            (self,other),
            '*'
        )
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __neg__(self):  # -self
        return self *-1
    
    def __sub__(self, other):  # self - other
        return self + (-other if isinstance(other, Value) else -Value(other))
    
    def __truediv__(self, other):  # self / other
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)

    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float))

        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):  # tanh(self)
        t = math.tanh(self.data)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
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