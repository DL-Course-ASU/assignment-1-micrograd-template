import pytest
from src.engine import Value
from src.nn import Neuron, Layer, MLP

def test_neuron():
    n = Neuron(3)
    assert len(n.w) == 3
    assert isinstance(n.b, Value)
    assert len(n.parameters()) == 4
    
    x = [Value(1.0), Value(2.0), Value(3.0)]
    out = n(x)
    assert isinstance(out, Value)

    n.zero_grad()
    for p in n.parameters():
        assert p.grad == 0.0

def test_layer():
    l = Layer(nin=3, nout=4)
    assert len(l.neurons) == 4
    assert len(l.parameters()) == 4 * 4  # 4 neurons * (3 weights + 1 bias)
    
    x = [2.0, 3.0, -1.0]
    out = l(x)
    assert len(out) == 4

def test_mlp():
    m = MLP(nin=3, nouts=[4, 4, 1])
    assert len(m.layers) == 3
    # params: (3*4+4) + (4*4+4) + (4*1+1) = 16 + 20 + 5 = 41
    assert len(m.parameters()) == 41
    
    x = [2.0, 3.0, -1.0]
    out = m(x)
    assert isinstance(out, Value)

    # Test backprop through the whole network
    out.backward()
    for p in m.parameters():
        assert p.grad != 0.0  # Assumes random init didn't hit dead zones entirely