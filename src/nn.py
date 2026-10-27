import random
from src.engine import Value

class Neuron:
    def __init__(self, nin):
        # TODO: Initialize weights (w) as a list of Value objects (random uniform -1 to 1)
        # TODO: Initialize bias (b) as a Value object (random uniform -1 to 1)
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):  # Neuron(nin)(x)
        # TODO: Calculate w*x + b
        # TODO: Pass the result through a tanh activation and return it
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        # TODO: Return a list of all parameters (weights + bias)
        return self.w + [self.b]

    def zero_grad(self):
        # TODO: Call reset_grad() on all parameters
        for p in self.parameters():
            p.reset_grad()

class Layer:
    def __init__(self, nin, nout):
        # TODO: Initialize a list of `nout` Neurons, each with `nin` inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # TODO: Pass input x through all neurons
        # TODO: Return a single output if nout==1, else return a list of outputs
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # TODO: Return a flattened list of all parameters in all neurons
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def zero_grad(self):
        # TODO: Zero the gradients of all parameters in this layer
        for neuron in self.neurons:
            neuron.zero_grad()

class MLP:
    def __init__(self, nin, nouts):
        # TODO: Initialize a sequence of Layers based on the sizes in `nin` and `nouts` list
        sz = [nin] + nouts
        self.layers = [
        Layer(sz[i], sz[i + 1])
        for i in range(len(nouts))
    ]

    def __call__(self, x):
        # TODO: Pass input x sequentially through all layers and return the final output
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        # TODO: Return a flattened list of all parameters in all layers
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        # TODO: Zero the gradients of all parameters in the MLP
        for layer in self.layers:
            layer.zero_grad()