
import random
from src.engine import Value


class Neuron:
    def __init__(self, nin):
        # Initialize random weights and bias
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Weighted sum of inputs plus bias
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # Apply tanh activation function
        return activation.tanh()

    def parameters(self):
        # Return all weights and bias
        return self.w + [self.b]

    def zero_grad(self):
        # Reset gradients of all parameters
        for p in self.parameters():
            p.reset_grad()


class Layer:
    def __init__(self, nin, nout):
        # Create nout neurons, each receiving nin inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # Forward pass through all neurons
        outputs = [neuron(x) for neuron in self.neurons]

        # Return one Value if layer has one neuron
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        # Collect parameters from all neurons
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def zero_grad(self):
        # Reset gradients in all neurons
        for neuron in self.neurons:
            neuron.zero_grad()


class MLP:
    def __init__(self, nin, nouts):
        # Build layer sizes: input size followed by output sizes
        sizes = [nin] + nouts

        # Create all layers
        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        # Forward pass through each layer
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        # Collect all network parameters
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        # Reset gradients in every layer
        for layer in self.layers:
            layer.zero_grad()
