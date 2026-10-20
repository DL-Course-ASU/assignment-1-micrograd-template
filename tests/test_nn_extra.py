"""Extra local tests for the neural-network API (plan.md, section 5).

Covers parameter counts, ``zero_grad()``, return types, the linear-vs-tanh
layer behaviour, hand-checked forward math, and a full finite-difference
gradient check of an ``MLP``.

Standard library + pytest only, so it also runs under the course CI.
"""

import random

import pytest

from src.engine import Value
from src.nn import MLP, Layer, Neuron

H = 1e-6      # step for central finite differences
h = H         # short alias used in the gradient checks below


# ----------------------------------------------------------------------
# structure and parameter counting
# ----------------------------------------------------------------------
def test_neuron_parameter_count_and_shape():
    random.seed(0)
    n = Neuron(5)
    assert len(n.w) == 5
    assert len(n.parameters()) == 6            # 5 weights + 1 bias
    for w in n.w:
        assert -1.0 <= w.data <= 1.0           # uniform init range
    assert isinstance(n.b, Value)
    assert -1.0 <= n.b.data <= 1.0


def test_neuron_forward_hand_checked():
    # nonlin=False so we can check the raw affine transform by hand
    n = Neuron(2, nonlin=False)
    n.w[0].data, n.w[1].data, n.b.data = 0.5, -1.5, 0.25
    out = n([Value(2.0), Value(4.0)])
    # 0.5*2 + (-1.5)*4 + 0.25 = 1 - 6 + 0.25 = -4.75
    assert out.data == pytest.approx(-4.75)


def test_neuron_tanh_output_range():
    random.seed(1)
    n = Neuron(3)
    # moderate inputs stay strictly inside (-1, 1)
    out = n([Value(0.5), Value(-0.5), Value(0.25)])
    assert -1.0 < out.data < 1.0
    # saturated inputs clamp to [-1, 1] without overflowing
    out = n([Value(1e3), Value(-1e3), Value(1e3)])
    assert -1.0 <= out.data <= 1.0


def test_neuron_wrong_input_size_raises():
    n = Neuron(3)
    with pytest.raises(AssertionError):
        n([Value(1.0), Value(2.0)])


def test_layer_shapes_and_parameter_count():
    random.seed(2)
    l = Layer(nin=3, nout=4)
    assert len(l.neurons) == 4
    assert len(l.parameters()) == 16           # 4 * (3 + 1)
    out = l([2.0, 3.0, -1.0])
    assert isinstance(out, list) and len(out) == 4

    # a single-neuron layer collapses to a scalar Value
    l1 = Layer(nin=3, nout=1)
    out1 = l1([2.0, 3.0, -1.0])
    assert isinstance(out1, Value)


def test_layer_nonlin_flag():
    l = Layer(nin=2, nout=1, nonlin=False)
    l.neurons[0].w[0].data = 1.0
    l.neurons[0].w[1].data = 1.0
    l.neurons[0].b.data = 0.0
    out = l([3.0, 4.0])
    assert out.data == pytest.approx(7.0)      # no tanh squash


def test_mlp_parameter_count_and_depth():
    random.seed(3)
    m = MLP(nin=3, nouts=[4, 4, 1])
    assert len(m.layers) == 3
    # (3*4+4) + (4*4+4) + (4*1+1) = 16 + 20 + 5 = 41
    assert len(m.parameters()) == 41
    assert isinstance(m([2.0, 3.0, -1.0]), Value)

    # a different shape: (2*8+8) + (8*1+1) = 24 + 9 = 33
    m2 = MLP(nin=2, nouts=[8, 1])
    assert len(m2.parameters()) == 33


def test_mlp_multi_output_returns_list():
    random.seed(4)
    m = MLP(nin=2, nouts=[4, 3])
    out = m([1.0, 2.0])
    assert isinstance(out, list) and len(out) == 3


def test_mlp_last_layer_is_linear():
    # the final layer must not squash its output (micrograd convention)
    random.seed(5)
    m = MLP(nin=2, nouts=[4, 1])
    assert m.layers[0].neurons[0].nonlin is True
    assert m.layers[-1].neurons[0].nonlin is False
    # hidden layer tanh, output linear
    assert m.layers[0].neurons[0].w[0] is not m.layers[-1].neurons[0].w[0]


def test_zero_grad_across_hierarchy():
    random.seed(6)
    m = MLP(nin=3, nouts=[4, 4, 1])
    x = [Value(1.0), Value(-2.0), Value(0.5)]
    out = m(x)
    out.backward()
    assert any(p.grad != 0.0 for p in m.parameters())   # something got gradient

    m.zero_grad()
    for p in m.parameters():
        assert p.grad == 0.0

    # per-module zero_grad works too
    n = Neuron(2)
    n([Value(1.0), Value(2.0)]).backward()
    n.zero_grad()
    assert all(p.grad == 0.0 for p in n.parameters())


# ----------------------------------------------------------------------
# end-to-end: finite-difference check of every MLP parameter
# ----------------------------------------------------------------------
def _build_net():
    """Deterministically build the same MLP every time it is called."""
    random.seed(1234)
    return MLP(2, [3, 3, 1])


def _loss_data(net, xs, ys):
    """Same mean-squared-error as :func:`_loss_value`, returned as a float."""
    total = 0.0
    for x, y in zip(xs, ys):
        pred = float(net([Value(v) for v in x]))
        diff = pred - float(y)
        total += diff * diff
    return total / len(xs)


def _loss_value(net, xs, ys):
    """Mean squared error as a ``Value`` (mean, not sum, keeps SGD stable)."""
    total = Value(0.0)
    for x, y in zip(xs, ys):
        pred = net([Value(v) for v in x])
        diff = pred - y
        total = total + diff * diff
    return total * (1.0 / len(xs))


def test_mlp_gradients_match_finite_differences():
    xs = [(0.5, -1.0), (1.2, 0.3), (-0.7, 2.0)]
    ys = [Value(1.0), Value(-0.5), Value(0.25)]

    net = _build_net()
    loss = _loss_value(net, xs, ys)
    loss.backward()
    analytic = [p.grad for p in net.parameters()]

    for i, (p, g) in enumerate(zip(net.parameters(), analytic)):
        original = p.data

        p.data = original + h
        f_plus = _loss_data(net, xs, ys)

        p.data = original - h
        f_minus = _loss_data(net, xs, ys)

        p.data = original
        numeric = (f_plus - f_minus) / (2 * h)
        assert g == pytest.approx(numeric, abs=1e-6), f"param {i} mismatch"


def test_training_step_reduces_loss():
    """One plain SGD step on a tiny problem must reduce the loss."""
    random.seed(7)
    xs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
    ys = [Value(1.0), Value(-1.0), Value(-1.0), Value(1.0)]   # XOR-ish

    net = MLP(2, [8, 1])

    def step(lr):
        loss = _loss_value(net, xs, ys)
        net.zero_grad()
        loss.backward()
        for p in net.parameters():
            p.data -= lr * p.grad
        return loss.data

    before = _loss_data(net, xs, ys)
    for _ in range(50):
        step(0.1)
    after = _loss_data(net, xs, ys)
    assert after < before, f"loss did not decrease: {before} -> {after}"
