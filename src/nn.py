import random
from src.engine import Value


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        total = sum((weight * inputValue for weight, inputValue in zip(self.w, x)), self.b)
        return total.tanh()

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for param in self.parameters():
            param.reset_grad()


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def zero_grad(self):
        for param in self.parameters():
            param.reset_grad()


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[index], sizes[index + 1]) for index in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def zero_grad(self):
        for param in self.parameters():
            param.reset_grad()
