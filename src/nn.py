import random
from src.engine import Value

class Neuron:
    def __init__(self, nin):
        # TODO: Initialize weights (w) as a list of Value objects (random uniform -1 to 1)
        # TODO: Initialize bias (b) as a Value object (random uniform -1 to 1)
        pass

    def __call__(self, x):  # Neuron(nin)(x)
        # TODO: Calculate w*x + b
        # TODO: Pass the result through a tanh activation and return it
        pass

    def parameters(self):
        # TODO: Return a list of all parameters (weights + bias)
        pass

    def zero_grad(self):
        # TODO: Call reset_grad() on all parameters
        pass

class Layer:
    def __init__(self, nin, nout):
        # TODO: Initialize a list of `nout` Neurons, each with `nin` inputs
        pass

    def __call__(self, x):
        # TODO: Pass input x through all neurons
        # TODO: Return a single output if nout==1, else return a list of outputs
        pass

    def parameters(self):
        # TODO: Return a flattened list of all parameters in all neurons
        pass

    def zero_grad(self):
        # TODO: Zero the gradients of all parameters in this layer
        pass

class MLP:
    def __init__(self, nin, nouts):
        # TODO: Initialize a sequence of Layers based on the sizes in `nin` and `nouts` list
        pass

    def __call__(self, x):
        # TODO: Pass input x sequentially through all layers and return the final output
        pass

    def parameters(self):
        # TODO: Return a flattened list of all parameters in all layers
        pass

    def zero_grad(self):
        # TODO: Zero the gradients of all parameters in the MLP
        pass