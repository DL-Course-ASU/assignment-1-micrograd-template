import random
from src.engine import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):  # Neuron(nin)(x)
        activation = sum((weight * value for weight, value in zip(self.w, x)), self.b)
        return activation.tanh()

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.reset_grad()

class Layer:
    def __init__(self, nin, nout):
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
        sizes = [nin] + list(nouts)
        self.layers = [Layer(sizes[index], sizes[index + 1]) for index in range(len(nouts))]

    def __call__(self, x):
        for index, layer in enumerate(self.layers):
            if not isinstance(x, (list, tuple)):
                x = [x]
            x = layer(x)
            if index < len(self.layers) - 1 and not isinstance(x, list):
                x = [x]
        return x

    def parameters(self):
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.reset_grad()