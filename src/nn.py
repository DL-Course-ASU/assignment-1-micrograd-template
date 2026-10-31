import random
from src.engine import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        if isinstance(x, (Value, int, float)):
            x = [x]

        total = self.b
        for i in range(len(self.w)):
            total = total + self.w[i] * x[i]
        return total.tanh()

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        parameters = []
        for neuron in self.neurons:
            parameters.extend(neuron.parameters())
        return parameters

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()

class MLP:
    def __init__(self, nin, nouts):
        self.layers = []
        for i, nout in enumerate(nouts):
            layer_nin = nin if i == 0 else nouts[i - 1]
            self.layers.append(Layer(layer_nin, nout))

    def __call__(self, x):
        inputs = x
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.parameters())
        return parameters

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
