import random
from src.engine import Value


# ==========================================
# 1. NEURON CLASS
# ==========================================

class Neuron:

    def __init__(self, nin):

        # Initialize random weights between -1 and 1
        self.w = [
            Value(random.uniform(-1, 1))
            for _ in range(nin)
        ]

        # Initialize random bias between -1 and 1
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):

        # Calculate weighted sum + bias
        activation = sum(
            (wi * xi for wi, xi in zip(self.w, x)),
            self.b
        )

        # Apply tanh activation
        output = activation.tanh()

        return output

    def parameters(self):

        # Return all weights and bias
        return self.w + [self.b]

    def zero_grad(self):

        # Reset gradients of all parameters
        for p in self.parameters():
            p.reset_grad()


# ==========================================
# 2. LAYER CLASS
# ==========================================

class Layer:

    def __init__(self, nin, nout):

        # Create nout neurons, each with nin inputs
        self.neurons = [
            Neuron(nin)
            for _ in range(nout)
        ]

    def __call__(self, x):

        # Pass input through all neurons
        outputs = [
            neuron(x)
            for neuron in self.neurons
        ]

        # Return single Value if only one neuron
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):

        # Collect all parameters from all neurons
        return [
            p
            for neuron in self.neurons
            for p in neuron.parameters()
        ]

    def zero_grad(self):

        # Reset gradients of all neurons
        for neuron in self.neurons:
            neuron.zero_grad()


# ==========================================
# 3. MLP CLASS
# ==========================================

class MLP:

    def __init__(self, nin, nouts):

        # Example:
        # nin = 3
        # nouts = [4, 4, 1]
        # sizes = [3, 4, 4, 1]

        sizes = [nin] + nouts

        # Create consecutive layers
        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):

        # Pass input through each layer sequentially
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):

        # Collect parameters from every layer
        return [
            p
            for layer in self.layers
            for p in layer.parameters()
        ]

    def zero_grad(self):

        # Reset all parameter gradients
        for layer in self.layers:
            layer.zero_grad()