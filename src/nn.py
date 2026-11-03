import random
from src.engine import Value
class Neuron:
    def __init__(self, nin):
        # Kol neuron 3ando weight le kol input, w bias wa7ed.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Bngma3 el inputs ba3d ma ndrab kol wa7ed fel weight beta3o, w nzed el bias.
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return activation.tanh()

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()


class Layer:
    def __init__(self, nin, nout):
        # El layer 3ebara 3an مجموعة neurons, kol wa7ed byest2bel nafs 3adad el inputs.
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]

        # Law fe neuron wa7ed bas, bnرجع Value mesh list.
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()


class MLP:
    def __init__(self, nin, nouts):
        # Bn7awel a3dad el neurons le sizes 3ashan nebny el layers wara ba3d.
        sizes = [nin] + list(nouts)
        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        # El output beta3 kol layer byed5ol input lel layer elly ba3do.
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()
