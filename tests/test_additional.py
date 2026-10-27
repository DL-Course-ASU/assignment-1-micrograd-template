import math
import random

import pytest

from src.engine import Value
from src.nn import Layer, MLP, Neuron


def test_zero_power_at_zero():
    x = Value(0.0)
    y = x ** 0
    y.backward()
    assert y.data == 1.0
    assert x.grad == 0.0


def test_mixed_operations_match_numerical_derivatives():
    def expression(a, b):
        return ((a * b + a / b - 2 / a + 3 - b) ** 2).tanh() + (a * b).exp()

    a, b = Value(1.2), Value(-0.7)
    expression(a, b).backward()
    h = 1e-6
    da = (expression(Value(a.data + h), Value(b.data)).data
          - expression(Value(a.data - h), Value(b.data)).data) / (2 * h)
    db = (expression(Value(a.data), Value(b.data + h)).data
          - expression(Value(a.data), Value(b.data - h)).data) / (2 * h)
    assert a.grad == pytest.approx(da, rel=1e-5, abs=1e-7)
    assert b.grad == pytest.approx(db, rel=1e-5, abs=1e-7)


def test_topological_order_visits_shared_nodes_once():
    a = Value(3.0)
    b = a + a
    out = b * a
    topo = out._get_topo()
    assert len(topo) == len(set(topo))
    for node in topo:
        assert all(topo.index(child) < topo.index(node) for child in node._prev)
    out.backward()
    assert a.grad == 12.0


def test_neuron_forward_with_known_parameters():
    n = Neuron(2)
    n.w[0].data, n.w[1].data, n.b.data = 0.5, -1.0, 0.25
    assert n([2.0, -1.0]).data == pytest.approx(math.tanh(2.25))
    assert all(isinstance(p, Value) for p in n.parameters())
    assert len(set(n.parameters())) == 3


@pytest.mark.parametrize('module', [lambda: Neuron(2), lambda: Layer(2, 3),
                                   lambda: MLP(2, [3, 1])])
def test_zero_grad_clears_nonzero_parameters(module):
    model = module()
    for p in model.parameters():
        p.grad = 7.0
    model.zero_grad()
    assert all(p.grad == 0.0 for p in model.parameters())


def test_single_neuron_hidden_layers():
    model = MLP(2, [1, 1, 2])
    out = model([0.2, -0.5])
    assert len(out) == 2
    assert all(isinstance(v, Value) for v in out)
    assert len(model.parameters()) == 9


def test_neuron_rejects_wrong_input_length():
    n = Neuron(2)
    with pytest.raises(ValueError):
        n([1.0])
    with pytest.raises(ValueError):
        n([1.0, 2.0, 3.0])


def test_all_network_gradients_match_finite_differences():
    random.seed(11)
    model = MLP(2, [2, 1])
    x = [0.3, -0.8]
    model(x).backward()
    h = 1e-6
    for p in model.parameters():
        original = p.data
        p.data = original + h
        plus = model(x).data
        p.data = original - h
        minus = model(x).data
        p.data = original
        assert p.grad == pytest.approx((plus - minus) / (2 * h), abs=1e-7)


def test_gradient_descent_reduces_loss():
    random.seed(7)
    model = MLP(2, [3, 1])
    xs = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    ys = [-1.0, 1.0, 1.0, -1.0]
    def loss():
        return sum((model(x) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)
    initial = loss().data
    for _ in range(100):
        value = loss()
        model.zero_grad()
        value.backward()
        for p in model.parameters():
            p.data -= 0.1 * p.grad
    assert loss().data < 0.2 * initial
