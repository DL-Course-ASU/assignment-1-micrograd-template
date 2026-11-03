import random
from src.engine import Value


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):  # Neuron(nin)(x)
        if len(x) != len(self.w):
            raise ValueError("Input count must match the neuron's weight count")

        total = self.b
        for weight, input_value in zip(self.w, x):
            total = total + weight * input_value
        return total.tanh()

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
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def parameters(self):
        parameters = []
        for neuron in self.neurons:
            parameters.extend(neuron.parameters())
        return parameters

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.reset_grad()


class MLP:
    def __init__(self, nin, nouts):
        self.layers = []
        for nout in nouts:
            self.layers.append(Layer(nin, nout))
            nin = nout

    def __call__(self, x):
        for layer in self.layers:
            # A layer with one neuron returns a single Value.
            if isinstance(x, Value):
                x = [x]
            x = layer(x)
        return x

    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.reset_grad()
