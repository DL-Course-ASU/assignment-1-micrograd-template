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
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad +=out.grad
        out._backward = _backward
        return out 
    
    def __radd__(self, other):  # other + self
        return self + other
    

    def __mul__(self, other):  # self * other
        other = other if isinstance(other, Value) else Value(other)
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
        return self + (-other)
    
    def __truediv__(self, other):  # self / other
        return self * (other ** -1)
    
    def __pow__(self, other):  # self ** other
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self, ), f'**{other}')
        def _backward(): # 3ashan dah el derivative
            if other != 0:
                self.grad += other * (self.data **(other-1)) * out.grad # nazel el n we na2as wa7ed 
        out._backward = _backward
        return out


    
    def tanh(self):  # tanh(self)
        t = math.tanh(self.data)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            
            self.grad += (1.0 -t** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):  # e**self
        e = math.exp(self.data)
        out = Value(e, (self, ), 'exp')

        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out 

    def reset_grad(self):
        self.grad = 0.0

    def _get_topo(self): #DFS zakertaha ahoh
        topo, visited = [], set()

        def build(node):
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                build(child)
            topo.append(node)
        build(self)
        return topo


    def backward(self):
        topo = self._get_topo()
        self.grad = 1.0
        for n in reversed(topo):
            n._backward()

        

    def __int__(self):  # int(self)
        return int(self.data)

    def __float__(self):  # float(self)
        return float(self.data)