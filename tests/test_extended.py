"""Extended tests (not part of the official autograder).

Analytical gradients from Value.backward() are compared against central finite
differences:  df/dx ~= (f(x + h) - f(x - h)) / (2h).
"""
import math
import random

import pytest

from src.engine import Value
from src.nn import Neuron, Layer, MLP

H = 1e-6
TOL = 1e-5


def numerical_grad(f, xs, i):
    """Central difference of f (a function of plain floats) w.r.t. xs[i]."""
    plus = list(xs); plus[i] += H
    minus = list(xs); minus[i] -= H
    return (f(*plus) - f(*minus)) / (2 * H)


def check_gradients(f, xs):
    """Build f on Value inputs, backprop, and compare with finite differences."""
    vals = [Value(x) for x in xs]
    out = f(*vals)
    out.backward()
    f_float = lambda *a: f(*[Value(v) for v in a]).data
    for i, v in enumerate(vals):
        assert v.grad == pytest.approx(numerical_grad(f_float, xs, i), abs=TOL, rel=TOL)


# ------------------------------------------------------------ single operators
@pytest.mark.parametrize("f, xs", [
    (lambda a, b: a + b, [1.5, -2.0]),
    (lambda a, b: a * b, [1.5, -2.0]),
    (lambda a, b: a - b, [1.5, -2.0]),
    (lambda a, b: a / b, [1.5, -2.0]),
    (lambda a: a ** 3, [1.7]),
    (lambda a: a ** -2, [1.7]),
    (lambda a: a ** 0.5, [1.7]),
    (lambda a: a.exp(), [0.8]),
    (lambda a: a.tanh(), [0.8]),
    (lambda a: -a, [0.8]),
])
def test_operator_gradients(f, xs):
    check_gradients(f, xs)


def test_scalar_operands_on_both_sides():
    a = Value(4.0)
    exprs = [a + 1, 1 + a, a * 2, 2 * a, a - 1, 1 - a, a / 2, 2 / a]
    assert [e.data for e in exprs] == [5.0, 5.0, 8.0, 8.0, 3.0, -3.0, 2.0, 0.5]
    total = sum(exprs)
    total.backward()
    # d/da of the sum: 1 + 1 + 2 + 2 + 1 - 1 + 0.5 - 2/a^2
    assert a.grad == pytest.approx(6.5 - 2 / 16)


# --------------------------------------------------------- composite graphs
def test_multilevel_expression_matches_finite_differences():
    f = lambda a, b, c: ((a * b + c).tanh() * (a ** 2).exp() - b / c) ** 2
    check_gradients(f, [0.3, -1.2, 2.5])


def test_diamond_graph_with_shared_intermediate():
    # x feeds two branches that rejoin: y = (x*x) + (x*x).exp(), with x*x shared.
    def f(x):
        s = x * x
        return s + s.exp()
    check_gradients(f, [0.7])


def test_repeated_variable_self_ops():
    a = Value(3.0)
    (a * a * a).backward()  # a^3 -> 3a^2 = 27
    assert a.grad == pytest.approx(27.0)

    b = Value(2.0)
    (b + b + b + b).backward()
    assert b.grad == 4.0


def test_topological_order_is_post_order_and_unique():
    a, b = Value(1.0), Value(2.0)
    c = a * b
    d = c + a
    e = d * c
    topo = e._get_topo()
    assert len(topo) == len(set(topo)) == 5
    pos = {id(n): i for i, n in enumerate(topo)}
    for node in topo:
        for child in node._prev:
            assert pos[id(child)] < pos[id(node)]
    assert topo[-1] is e


def test_backward_twice_accumulates_and_reset_clears():
    a = Value(2.0)
    b = a * 3
    b.backward()
    b.backward()
    assert a.grad == 6.0  # accumulates like PyTorch
    a.reset_grad()
    assert a.grad == 0.0


def test_leaf_node_backward():
    a = Value(5.0)
    a.backward()
    assert a.grad == 1.0


def test_tanh_saturation_and_exp_large():
    a = Value(50.0)
    t = a.tanh()
    t.backward()
    assert t.data == pytest.approx(1.0) and a.grad == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(OverflowError):
        Value(1000.0).exp()


def test_unsupported_types():
    with pytest.raises(TypeError):
        _ = "s" + Value(1.0)
    with pytest.raises(TypeError):
        _ = Value(1.0) / "s"
    with pytest.raises(AssertionError):
        _ = Value(2.0) ** "s"


def test_division_by_zero_scalar():
    with pytest.raises(ZeroDivisionError):
        _ = Value(1.0) / 0


# --------------------------------------------------------- neural network
def test_neuron_forward_matches_manual_computation():
    random.seed(0)
    n = Neuron(3)
    x = [0.5, -1.0, 2.0]
    expected = math.tanh(sum(w.data * xi for w, xi in zip(n.w, x)) + n.b.data)
    assert n(x).data == pytest.approx(expected)


def test_neuron_rejects_wrong_input_size():
    with pytest.raises(ValueError):
        Neuron(3)([1.0, 2.0])


@pytest.mark.parametrize("nin, nouts", [(1, [1]), (2, [3]), (3, [4, 4, 1]), (2, [1, 3, 2]), (5, [8, 1])])
def test_mlp_configurations(nin, nouts):
    random.seed(1)
    m = MLP(nin, nouts)
    sizes = [nin] + nouts
    assert len(m.parameters()) == sum((sizes[i] + 1) * sizes[i + 1] for i in range(len(nouts)))
    out = m([0.1 * (i + 1) for i in range(nin)])
    if nouts[-1] == 1:
        assert isinstance(out, Value)
    else:
        assert isinstance(out, list) and len(out) == nouts[-1]
        assert all(isinstance(o, Value) for o in out)


def test_mlp_gradients_match_finite_differences():
    random.seed(2)
    m = MLP(2, [3, 1])
    x, y = [0.4, -0.7], 0.5
    params = m.parameters()

    def loss_from(values):
        for p, v in zip(params, values):
            p.data = v
        return ((m(x) - y) ** 2).data

    base = [p.data for p in params]
    m.zero_grad()
    loss = (m(x) - y) ** 2
    loss.backward()
    analytic = [p.grad for p in params]
    for i in range(len(params)):
        num = numerical_grad(lambda *v: loss_from(v), base, i)
        assert analytic[i] == pytest.approx(num, abs=TOL, rel=TOL)
    loss_from(base)


def test_zero_grad_clears_all_levels():
    random.seed(3)
    m = MLP(3, [4, 1])
    m([1.0, 2.0, 3.0]).backward()
    assert any(p.grad != 0 for p in m.parameters())
    m.zero_grad()
    assert all(p.grad == 0.0 for p in m.parameters())


def test_gradient_descent_reduces_loss():
    random.seed(4)
    m = MLP(2, [4, 1])
    xs = [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
    ys = [1.0, 1.0, -1.0, -1.0]

    def total_loss():
        return sum((m(x) - y) ** 2 for x, y in zip(xs, ys))

    first = total_loss().data
    for _ in range(50):
        loss = total_loss()
        m.zero_grad()
        loss.backward()
        for p in m.parameters():
            p.data -= 0.05 * p.grad
    assert total_loss().data < first
