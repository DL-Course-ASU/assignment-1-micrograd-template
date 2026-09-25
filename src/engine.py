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
        pass

    def __add__(self, other):  # self + other
        # TODO: Handle both Value and scalar inputs
        # TODO: Return new Value and implement _backward
        pass

    def __radd__(self, other):  # other + self
        # TODO: Implement reverse addition
        pass

    def __mul__(self, other):  # self * other
        # TODO: Handle both Value and scalar inputs
        # TODO: Return new Value and implement _backward using product rule
        pass

    def __rmul__(self, other):  # other * self
        # TODO: Implement reverse multiplication
        pass

    def __neg__(self):  # -self
        # TODO: Implement negation
        pass

    def __sub__(self, other):  # self - other
        # TODO: Implement subtraction
        pass

    def __truediv__(self, other):  # self / other
        # TODO: Implement division using self * other**-1
        pass

    def __pow__(self, other):  # self ** other
        # TODO: Assert other is int or float
        # TODO: Return new Value and implement _backward using power rule
        pass

    def tanh(self):  # tanh(self)
        # TODO: Implement hyperbolic tangent activation
        # TODO: Implement _backward for tanh
        pass

    def exp(self):  # e**self
        # TODO: Implement exponential function (math.exp)
        # TODO: Implement _backward for exp
        pass

    def reset_grad(self):
        # TODO: Reset self.grad to 0.0
        pass

    def _get_topo(self):
        # TODO: Return a list of all nodes in topological order
        pass

    def backward(self):
        # TODO: Retrieve topological order, set root grad to 1.0, and apply _backward in reverse
        pass

    def __int__(self):  # int(self)
        # TODO: Return data as int
        pass

    def __float__(self):  # float(self)
        # TODO: Return data as float
        pass