"""A tiny PyTorch-like neural network API built on :mod:`src.engine`.

The hierarchy is deliberately simple:

``Neuron``  -- one weighted sum plus bias, passed through an activation
``Layer``   -- a list of ``nout`` neurons with ``nin`` inputs
``MLP``     -- a chain of layers; the final layer is linear (no
               activation) so the network can emit unbounded scores,
               which is what loss functions such as hinge/MSE prefer.

Every module exposes the same two methods:

``parameters()`` -- a flat list of every :class:`Value` that can be trained
``zero_grad()``  -- resets the gradient of every parameter between steps
"""

import random

from src.engine import Value


class Neuron:
    """A single perceptron: ``activation(w . x + b)``."""

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        """Return ``tanh(w . x + b)`` (or the raw sum if ``nonlin=False``)."""
        assert len(x) == len(self.w), "input size must match the number of weights"
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        return act.tanh() if self.nonlin else act

    def parameters(self):
        """All trainable parameters: the weights followed by the bias."""
        return self.w + [self.b]

    def zero_grad(self):
        """Reset the gradients of this neuron's parameters to zero."""
        for p in self.parameters():
            p.reset_grad()

    def __repr__(self):
        return f"Neuron(nin={len(self.w)}, nonlin={self.nonlin})"


class Layer:
    """A row of ``nout`` neurons, each seeing the same ``nin`` inputs."""

    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        """Apply every neuron; collapse to a single Value when ``nout == 1``."""
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """Flattened parameters of all neurons in this layer."""
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self):
        """Zero the gradients of every neuron in this layer."""
        for n in self.neurons:
            n.zero_grad()

    def __repr__(self):
        return f"Layer(nin={len(self.neurons[0].w)}, nout={len(self.neurons)})"


class MLP:
    """A multi-layer perceptron: ``MLP(3, [4, 4, 1])`` -> 3 layers.

    Sizes are read as ``nin -> nouts[0] -> nouts[1] -> ...``.  The hidden
    layers use ``tanh``; the last layer is linear so the output is not
    squashed into ``(-1, 1)``.
    """

    def __init__(self, nin, nouts):
        sizes = [nin] + list(nouts)
        self.layers = [
            Layer(sz, szz, nonlin=(i != len(nouts) - 1))
            for i, (sz, szz) in enumerate(zip(sizes[:-1], nouts))
        ]

    def __call__(self, x):
        """Push the input through every layer in order."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """Flattened parameters of all layers, in order."""
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        """Zero the gradients of every parameter in the network."""
        for layer in self.layers:
            layer.zero_grad()

    def __repr__(self):
        sizes = [len(self.layers[0].neurons[0].w)] + [len(l.neurons) for l in self.layers]
        return f"MLP({sizes})"
