"""pytest suite: edge cases and math constraints for engine.py / nn.py."""
import math
import random

import pytest

from engine import Value
from nn import MLP, Layer, Neuron


def numeric_grad(f, xs, i, h=1e-6):
    """Central finite difference of f w.r.t. xs[i]."""
    hi = list(xs); hi[i] += h
    lo = list(xs); lo[i] -= h
    return (f(*hi) - f(*lo)) / (2 * h)


# ---------------------------------------------------------------- forward values
def test_forward_arithmetic():
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == -1.0
    assert (a * b).data == -6.0
    assert (a - b).data == 5.0
    assert (a / b).data == pytest.approx(-2 / 3)
    assert (a ** 3).data == 8.0
    assert (-a).data == -2.0
    assert (1 + a).data == 3.0 and (2 * a).data == 4.0
    assert (10 - a).data == 8.0 and (6 / a).data == 3.0


# --------------------------------------------------------------- local derivatives
def test_add_mul_grads():
    a, b = Value(2.0), Value(3.0)
    c = a * b + a
    c.backward()
    assert a.grad == pytest.approx(b.data + 1)
    assert b.grad == pytest.approx(a.data)


def test_pow_grad():
    x = Value(3.0)
    y = x ** 4
    y.backward()
    assert x.grad == pytest.approx(4 * 3.0 ** 3)


def test_pow_fractional_and_negative():
    x = Value(4.0)
    (x ** 0.5).backward()
    assert x.grad == pytest.approx(0.5 * 4.0 ** -0.5)
    x = Value(2.0)
    (x ** -1).backward()
    assert x.grad == pytest.approx(-1 / 4)


def test_tanh_grad():
    x = Value(0.5)
    y = x.tanh()
    y.backward()
    assert y.data == pytest.approx(math.tanh(0.5))
    assert x.grad == pytest.approx(1 - math.tanh(0.5) ** 2)


def test_exp_grad():
    x = Value(1.3)
    y = x.exp()
    y.backward()
    assert y.data == pytest.approx(math.exp(1.3))
    assert x.grad == pytest.approx(math.exp(1.3))


def test_relu_grad():
    p, n = Value(2.0), Value(-2.0)
    p.relu().backward(); n.relu().backward()
    assert p.grad == 1.0 and n.grad == 0.0


# ------------------------------------------------------------------- graph traps
def test_diamond_a_plus_a():
    a = Value(3.0)
    c = a + a
    c.backward()
    assert a.grad == 2.0


def test_diamond_shared_subexpression():
    a = Value(2.0)
    b = a * a          # a used twice
    c = b + b          # b used twice
    d = c * a          # a used again
    d.backward()       # d = 2a^3 -> dd/da = 6a^2 = 24
    assert a.grad == pytest.approx(24.0)


def test_topo_children_before_parents():
    a, b = Value(1.0), Value(2.0)
    c = (a * b + a).tanh() * b
    topo = c._get_topo()
    pos = {id(n): i for i, n in enumerate(topo)}
    assert len(topo) == len(set(map(id, topo)))      # no duplicates
    for node in topo:
        for child in node._prev:
            assert pos[id(child)] < pos[id(node)]    # post-order
    assert topo[-1] is c


def test_deep_chain_no_recursion_error():
    x = Value(1.0)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.backward()
    assert x.grad == 1.0


def test_leaf_backward_seeds_one():
    x = Value(5.0)
    x.backward()
    assert x.grad == 1.0


# ---------------------------------------------------- full expression vs numerics
def test_complex_expression_matches_numeric():
    def f(a, b, c):
        return ((a * b + c ** 2).tanh() + (a - c).exp() / (b ** 2 + 1)) ** 2

    xs = [0.3, -0.7, 1.1]
    vals = [Value(x) for x in xs]
    out = f(*vals)
    out.backward()
    plain = lambda a, b, c: ((math.tanh(a * b + c ** 2) + math.exp(a - c) / (b ** 2 + 1)) ** 2)
    assert out.data == pytest.approx(plain(*xs))
    for i, v in enumerate(vals):
        assert v.grad == pytest.approx(numeric_grad(plain, xs, i), rel=1e-5, abs=1e-7)


def test_torch_agreement_if_available():
    torch = pytest.importorskip("torch")
    xs = [0.3, -0.7, 1.1]
    t = [torch.tensor(x, dtype=torch.double, requires_grad=True) for x in xs]
    ((t[0] * t[1] + t[2]).tanh() * t[0].exp() / (t[1] ** 2 + 1)).backward()
    v = [Value(x) for x in xs]
    ((v[0] * v[1] + v[2]).tanh() * v[0].exp() / (v[1] ** 2 + 1)).backward()
    for tv, vv in zip(t, v):
        assert vv.grad == pytest.approx(tv.grad.item(), rel=1e-9)


# ------------------------------------------------------------------------- nn.py
def test_neuron_shapes_and_range():
    random.seed(0)
    n = Neuron(3)
    assert len(n.parameters()) == 4
    assert -1.0 <= n([1.0, 2.0, 3.0]).data <= 1.0


def test_init_in_range_and_random():
    random.seed(1)
    n = Neuron(50)
    ws = [w.data for w in n.w]
    assert all(-1 <= w <= 1 for w in ws) and len(set(ws)) > 1


def test_layer_and_mlp_output_shapes():
    random.seed(0)
    assert len(Layer(3, 4)([1.0, 2.0, 3.0])) == 4
    m = MLP(3, [4, 4, 1])
    assert isinstance(m([1.0, 2.0, 3.0]), Value)
    assert len(m.parameters()) == (3 * 4 + 4) + (4 * 4 + 4) + (4 * 1 + 1)
    assert len(MLP(2, [3, 2])([0.0, 1.0])) == 2


def test_zero_grad():
    random.seed(0)
    m = MLP(2, [3, 1])
    m([1.0, -1.0]).backward()
    assert any(p.grad != 0 for p in m.parameters())
    m.zero_grad()
    assert all(p.grad == 0 for p in m.parameters())


def test_mlp_gradient_matches_numeric():
    random.seed(3)
    m = MLP(2, [3, 1])
    x = [0.4, -0.9]
    m.zero_grad()
    m(x).backward()
    for p in m.parameters()[:5]:
        analytic, old, h = p.grad, p.data, 1e-6
        p.data = old + h; hi = m(x).data
        p.data = old - h; lo = m(x).data
        p.data = old
        assert analytic == pytest.approx((hi - lo) / (2 * h), rel=1e-4, abs=1e-7)


def test_training_reduces_loss():
    random.seed(7)
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    m = MLP(3, [4, 4, 1])
    losses = []
    for _ in range(60):
        loss = sum((m(x) - y) ** 2 for x, y in zip(xs, ys))
        m.zero_grad()
        loss.backward()
        for p in m.parameters():
            p.data -= 0.05 * p.grad
        losses.append(loss.data)
    assert losses[-1] < 0.1 * losses[0]
