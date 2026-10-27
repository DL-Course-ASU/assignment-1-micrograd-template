import random

from src.engine import Value


class Neuron:
    def __init__(self, nin):
        # One weight for each input.
        self.w = [
            Value(random.uniform(-1, 1))
            for _ in range(nin)
        ]

        # One bias for the neuron.
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):  # Neuron(nin)(x)
        if len(x) != len(self.w):
            raise ValueError(
                f"Expected {len(self.w)} inputs, but received {len(x)}"
            )

        # Calculate:
        # w1*x1 + w2*x2 + ... + wn*xn + b
        activation = sum(
            (weight * input_value for weight, input_value in zip(self.w, x)),
            self.b
        )

        # Apply the tanh activation function.
        return activation.tanh()

    def parameters(self):
        # Return all weights followed by the bias.
        return self.w + [self.b]

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.reset_grad()


class Layer:
    def __init__(self, nin, nout):
        # Create nout neurons, each accepting nin inputs.
        self.neurons = [
            Neuron(nin)
            for _ in range(nout)
        ]

    def __call__(self, x):
        outputs = [
            neuron(x)
            for neuron in self.neurons
        ]

        # Return a Value for a one-neuron layer.
        # Return a list for a multi-neuron layer.
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        # Flatten the parameters from all neurons.
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()


class MLP:
    def __init__(self, nin, nouts):
        # Example:
        # nin = 3
        # nouts = [4, 4, 1]
        # sizes = [3, 4, 4, 1]
        sizes = [nin] + nouts

        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        # Pass the output of each layer into the next layer.
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        # Flatten the parameters from all layers.
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()