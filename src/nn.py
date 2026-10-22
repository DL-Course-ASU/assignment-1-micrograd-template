import random
from src.engine import Value

class Neuron:
    def __init__(self, nin):
        # TODO: Initialize weights (w) as a list of Value objects (random uniform -1 to 1)
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # TODO: Initialize bias (b) as a Value object (random uniform -1 to 1)
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):  # Neuron(nin)(x)
        # TODO: Calculate w*x + b
        sum_product = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # TODO: Pass the result through a tanh activation and return it
        return sum_product.tanh()

    def parameters(self):
        # TODO: Return a list of all parameters (weights + bias)
        return self.w + [self.b]

    def zero_grad(self):
        # TODO: Call reset_grad() on all parameters
        for param in self.parameters():
            param.reset_grad()

class Layer:
    def __init__(self, nin, nout):
        # TODO: Initialize a list of `nout` Neurons, each with `nin` inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # TODO: Pass input x through all neurons
        outputs = [neuron(x) for neuron in self.neurons]

        # TODO: Return a single output if nout==1, else return a list of outputs
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        # TODO: Return a flattened list of all parameters in all neurons
        return [param for neuron in self.neurons for param in neuron.parameters()]

    def zero_grad(self):
        # TODO: Zero the gradients of all parameters in this layer
        for neuron in self.neurons:
            neuron.zero_grad()

class MLP:
    def __init__(self, nin, nouts):
        # TODO: Initialize a sequence of Layers based on the sizes in `nin` and `nouts` list
        self.layers = []
        layer_sizes = [nin] + nouts
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def __call__(self, x):
        # TODO: Pass input x sequentially through all layers and return the final output
        for layer in self.layers:
            if isinstance(x, Value):
                x = [x]
            x = layer(x)
        return x

    def parameters(self):
        # TODO: Return a flattened list of all parameters in all layers
        return [param for layer in self.layers for param in layer.parameters()]

    def zero_grad(self):
        # TODO: Zero the gradients of all parameters in the MLP
        for layer in self.layers:
            layer.zero_grad()