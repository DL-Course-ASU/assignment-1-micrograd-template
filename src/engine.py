import math


class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # ==========================================
    # 1. STRING REPRESENTATION
    # ==========================================

    def __repr__(self):
        return f"Value(data={self.data})"

    # ==========================================
    # 2. ADDITION
    # ==========================================

    def __add__(self, other):

        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            '+'
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    # ==========================================
    # 3. REVERSE ADDITION
    # ==========================================

    def __radd__(self, other):
        return self + other

    # ==========================================
    # 4. MULTIPLICATION
    # ==========================================

    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            '*'
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    # ==========================================
    # 5. REVERSE MULTIPLICATION
    # ==========================================

    def __rmul__(self, other):
        return self * other

    # ==========================================
    # 6. NEGATION
    # ==========================================

    def __neg__(self):
        return self * -1

    # ==========================================
    # 7. SUBTRACTION
    # ==========================================

    def __sub__(self, other):
        return self + (-other)

    # ==========================================
    # 8. DIVISION
    # ==========================================

    def __truediv__(self, other):

        other = other if isinstance(other, Value) else Value(other)

        return self * other**-1

    # ==========================================
    # 9. POWER
    # ==========================================

    def __pow__(self, other):

        assert isinstance(other, (int, float))

        out = Value(
            self.data ** other,
            (self,),
            f'**{other}'
        )

        def _backward():
            self.grad += (
                other * self.data ** (other - 1)
            ) * out.grad

        out._backward = _backward

        return out

    # ==========================================
    # 10. TANH ACTIVATION
    # ==========================================

    def tanh(self):

        t = math.tanh(self.data)

        out = Value(
            t,
            (self,),
            'tanh'
        )

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    # ==========================================
    # 11. EXPONENTIAL FUNCTION
    # ==========================================

    def exp(self):

        x = math.exp(self.data)

        out = Value(
            x,
            (self,),
            'exp'
        )

        def _backward():
            self.grad += x * out.grad

        out._backward = _backward

        return out

    # ==========================================
    # 12. RESET GRADIENT
    # ==========================================

    def reset_grad(self):
        self.grad = 0.0

    # ==========================================
    # 13. TOPOLOGICAL SORT
    # ==========================================

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

    # ==========================================
    # 14. BACKPROPAGATION
    # ==========================================

    def backward(self):

        topo = self._get_topo()

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    # ==========================================
    # 15. INTEGER CONVERSION
    # ==========================================

    def __int__(self):
        return int(self.data)

    # ==========================================
    # 16. FLOAT CONVERSION
    # ==========================================

    def __float__(self):
        return float(self.data)