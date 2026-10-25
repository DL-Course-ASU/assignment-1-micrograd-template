import random
from src.engine import Value

class Neuron:
    def __init__(self, nin):
       self.w = [Value(random.uniform(-1 , 1)) for _ in range(nin)]
       self.b = Value(random.uniform(-1 , 1))

    def __call__(self, x):  # Neuron(nin)(x)
        weighted_sum = sum(value * weight for value , weight in zip(x , self.w))
        result = weighted_sum + self.b
        return result.tanh()
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
        params = [p for neuron in self.neurons for p in neuron.parameters()]
        return params

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()

class MLP:
    def __init__(self, nin, nouts):
        size = [nin] + nouts
        self.layers = [Layer(size[i] , size[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
       params = [p for layer in self.layers for p in layer.parameters()]
       return params

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()