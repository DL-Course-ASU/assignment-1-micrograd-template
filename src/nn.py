
import random
from src.engine import Value


class Neuron:
    def __init__(self, nin):
        # Initialize weights and bias randomly between -1 and 1
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Calculate weighted sum + bias
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # Apply tanh activation function
        out = activation.tanh()

        return out

    def parameters(self):
        # Return weights and bias
        return self.w + [self.b]

    def zero_grad(self):
        # Reset gradients
        for p in self.parameters():
            p.reset_grad()


class Layer:
    def __init__(self, nin, nout):
        # Create nout neurons, each receiving nin inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        # Pass input through every neuron
        outputs = [neuron(x) for neuron in self.neurons]

        # Return a single Value if there is only one neuron
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        # Flatten parameters from all neurons
        return [
            p
            for neuron in self.neurons
            for p in neuron.parameters()
        ]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()


class MLP:
    def __init__(self, nin, nouts):
        # Combine input and output sizes
        sz = [nin] + nouts

        # Build consecutive layers
        self.layers = [
            Layer(sz[i], sz[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        # Pass data through all layers sequentially
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        # Flatten parameters from all layers
        return [
            p
            for layer in self.layers
            for p in layer.parameters()
        ]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()
