
import random
from src.engine import Value


class Neuron:
    def __init__(self, nin):
        # Initialize weights and bias randomly between -1 and 1
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Calculate weighted sum: w*x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # Apply tanh activation
        out = act.tanh()

        return out

    def parameters(self):
        # Return all weights and bias
        return self.w + [self.b]

    def zero_grad(self):
        # Reset gradients of all parameters
        for p in self.parameters():
            p.reset_grad()


class Layer:
    def __init__(self, nin, nout):
        # Create nout neurons, each accepting nin inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # Pass inputs through every neuron
        outs = [neuron(x) for neuron in self.neurons]

        # Return a scalar Value for a single-neuron layer
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        # Flatten parameters from all neurons
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def zero_grad(self):
        # Reset all gradients in this layer
        for neuron in self.neurons:
            neuron.zero_grad()


class MLP:
    def __init__(self, nin, nouts):
        # Build layers using consecutive input/output sizes
        sz = [nin] + list(nouts)
        self.layers = [
            Layer(sz[i], sz[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        # Forward propagation through all layers
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        # Flatten parameters from all layers
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        # Reset all model parameter gradients
        for layer in self.layers:
            layer.zero_grad()
