"""A small neuron/layer/MLP API built entirely on scalar Values."""
import random
from src.engine import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.reset_grad()


class Neuron(Module):
    def __init__(self, nin, nonlin=True, activation="tanh"):
        if nin < 1:
            raise ValueError("nin must be positive")
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin, self.activation = nonlin, activation

    def __call__(self, x):
        if len(x) != len(self.w):
            raise ValueError("input length does not match the neuron")
        total = self.b
        for weight, item in zip(self.w, x):
            total = total + weight * item
        if not self.nonlin:
            return total
        if self.activation not in ("tanh", "relu", "sigmoid"):
            raise ValueError("unsupported activation")
        return getattr(total, self.activation)()

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        name = self.activation if self.nonlin else "Linear"
        return f"{name}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        if nout < 1:
            raise ValueError("nout must be positive")
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(map(str, self.neurons))}]"


class MLP(Module):
    def __init__(self, nin, nouts, activation="tanh"):
        if not nouts:
            raise ValueError("at least one output layer is required")
        widths = [nin] + list(nouts)
        self.layers = [Layer(widths[i], widths[i+1],
                             nonlin=(i < len(nouts)-1), activation=activation)
                       for i in range(len(nouts))]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(x, Value) and i < len(self.layers)-1:
                x = [x]
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(map(str, self.layers))}]"
