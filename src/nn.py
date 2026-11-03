import random

from src.engine import Value


class Neuron:
    def __init__(self, nin):
        if not isinstance(nin, int) or nin < 1:
            raise ValueError('The number of inputs must be a positive integer')
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        if len(x) != len(self.w):
            raise ValueError('The input size does not match the neuron weights')
        total = sum((weight * value for weight, value in zip(self.w, x)), self.b)
        return total.tanh()

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.reset_grad()


class Layer:
    def __init__(self, nin, nout):
        if not isinstance(nout, int) or nout < 1:
            raise ValueError('The number of outputs must be a positive integer')
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.reset_grad()


class MLP:
    def __init__(self, nin, nouts):
        if not nouts:
            raise ValueError('The network must contain at least one layer')
        sizes = [nin] + list(nouts)
        self.layers = [Layer(sizes[i], sizes[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            if isinstance(x, Value):
                x = [x]
            x = layer(x)
        return x

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.reset_grad()
