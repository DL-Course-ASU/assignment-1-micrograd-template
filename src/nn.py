
import random
from src.engine import Value


class Neuron:
    """A fully connected neuron with tanh activation."""

    def __init__(self, nin):
        if not isinstance(nin, int) or isinstance(nin, bool) or nin <= 0:
            raise ValueError("nin must be a positive integer")

        self.nin = nin

        # One learnable weight for each input.
        self.w = [
            Value(random.uniform(-1, 1))
            for _ in range(nin)
        ]

        # One learnable bias.
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Convert iterable inputs into a list.
        x = list(x)

        # Prevent silent truncation by zip().
        if len(x) != self.nin:
            raise ValueError(
                f"Expected {self.nin} inputs, received {len(x)}"
            )

        # z = w1*x1 + w2*x2 + ... + wn*xn + b
        activation = sum(
            (wi * xi for wi, xi in zip(self.w, x)),
            self.b
        )

        # Nonlinear activation.
        return activation.tanh()

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()


class Layer:
    """A fully connected layer containing multiple neurons."""

    def __init__(self, nin, nout):
        if not isinstance(nout, int) or isinstance(nout, bool) or nout <= 0:
            raise ValueError("nout must be a positive integer")

        self.neurons = [
            Neuron(nin)
            for _ in range(nout)
        ]

    def __call__(self, x):
        # All neurons receive the same input vector.
        outputs = [
            neuron(x)
            for neuron in self.neurons
        ]

        # Single-output layers return a Value, not a list.
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        # Flatten parameters from every neuron.
        return [
            p
            for neuron in self.neurons
            for p in neuron.parameters()
        ]

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()


class MLP:
    """A multilayer perceptron built from Layer objects."""

    def __init__(self, nin, nouts):
        if not isinstance(nin, int) or isinstance(nin, bool) or nin <= 0:
            raise ValueError("nin must be a positive integer")

        if not isinstance(nouts, (list, tuple)) or not nouts:
            raise ValueError("nouts must be a non-empty sequence")

        if any(
            not isinstance(n, int) or isinstance(n, bool) or n <= 0
            for n in nouts
        ):
            raise ValueError("All layer sizes must be positive integers")

        # Example: nin=3, nouts=[4, 4, 1]
        # sizes = [3, 4, 4, 1]
        sizes = [nin] + list(nouts)

        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        # Forward propagation through each layer.
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [
            p
            for layer in self.layers
            for p in layer.parameters()
        ]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
