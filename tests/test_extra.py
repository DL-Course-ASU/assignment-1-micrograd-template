"""Extra tests written for this submission (the template's own tests are untouched).

Hand-computed gradients always run. The comparisons against PyTorch are skipped
automatically when PyTorch is not installed. PyTorch is used only in this file,
never in the engine.
"""
import math
import random

import pytest

from src.engine import Value
from src.nn import MLP, Layer, Neuron

TOLERANCE = 1e-6


# --------------------------------------------------------------------------
# Hand-computed gradients
# --------------------------------------------------------------------------

def test_a_plus_a_accumulates():
    a = Value(3.0)
    c = a + a
    c.backward()
    assert c.data == 6.0
    assert a.grad == 2.0  # would be 1.0 if _backward used = instead of +=


def test_a_times_a_times_a():
    a = Value(2.0)
    b = a * a * a
    b.backward()
    assert b.data == 8.0
    assert a.grad == pytest.approx(3 * 2.0 ** 2)


def test_mixing_floats_and_values():
    x = Value(4.0)
    y = 2 * x + 1 - x / 2 + 3 / x - (1 - x)
    y.backward()
    # y = 2x + 1 - x/2 + 3/x - 1 + x
    assert y.data == pytest.approx(8 + 1 - 2 + 0.75 - 1 + 4)
    # dy/dx = 2 - 1/2 - 3/x^2 + 1
    assert x.grad == pytest.approx(2 - 0.5 - 3 / 16 + 1)


def test_reversed_operators_with_plain_numbers():
    x = Value(2.0)
    assert (3 + x).data == 5.0
    assert (3 * x).data == 6.0
    assert (3 - x).data == 1.0
    assert (3 / x).data == 1.5


def test_tanh_gradient_away_from_zero():
    a = Value(0.7)
    b = a.tanh()
    b.backward()
    assert b.data == pytest.approx(math.tanh(0.7))
    assert a.grad == pytest.approx(1 - math.tanh(0.7) ** 2)


def test_exp_gradient_away_from_zero():
    a = Value(1.5)
    b = a.exp()
    b.backward()
    assert b.data == pytest.approx(math.exp(1.5))
    assert a.grad == pytest.approx(math.exp(1.5))


@pytest.mark.parametrize("exponent", [2, 3, -1, 0.5, -2.5])
def test_pow_gradient(exponent):
    a = Value(1.7)
    (a ** exponent).backward()
    assert a.grad == pytest.approx(exponent * 1.7 ** (exponent - 1))


def test_pow_zero_exponent_at_zero_base():
    a = Value(0.0)
    b = a ** 0
    b.backward()  # must not raise ZeroDivisionError
    assert b.data == 1.0
    assert a.grad == 0.0


def test_division_gradient():
    a = Value(6.0)
    b = Value(2.0)
    c = a / b
    c.backward()
    assert c.data == 3.0
    assert a.grad == pytest.approx(1 / 2.0)  # d(a/b)/da = 1/b
    assert b.grad == pytest.approx(-6.0 / 2.0 ** 2)  # d(a/b)/db = -a/b^2


def test_reused_interior_node():
    # x is an interior node reached along two paths: directly, and through y.
    # z = 2a(2a + 1) = 4a^2 + 2a, so dz/da = 8a + 2. Without the visited set
    # x._backward would run twice, and in a wrong topological order it would run
    # before y had added its share to x.grad.
    a = Value(3.0)
    x = a * 2
    y = x + 1
    z = x * y
    z.backward()
    assert z.data == 42.0
    assert a.grad == 26.0
    assert x.grad == 13.0  # dz/dx = y + x = 7 + 6


def test_topological_order_is_valid():
    a = Value(3.0)
    x = a * 2
    y = x + 1
    z = x * y
    topo = z._get_topo()
    assert len(topo) == len(set(topo))  # no duplicates (the visited set)
    assert topo[-1] is z  # the root comes last (post-order)
    position = {node: i for i, node in enumerate(topo)}
    for node in topo:
        for child in node._prev:
            assert position[child] < position[node]  # inputs before their consumers


def test_reset_grad():
    a = Value(2.0)
    (a * 3).backward()
    assert a.grad == 3.0
    a.reset_grad()
    assert a.grad == 0.0


@pytest.mark.parametrize("bad", ["string", [1, 2], None])
def test_operations_reject_non_numbers(bad):
    # An int-valued Value matters here: 2 * [1, 2] is valid Python list repetition.
    with pytest.raises(TypeError):
        Value(2) + bad
    with pytest.raises(TypeError):
        Value(2) * bad
    with pytest.raises(TypeError):
        bad + Value(2)
    with pytest.raises(TypeError):
        bad * Value(2)
    with pytest.raises(TypeError):
        Value(2) - bad
    with pytest.raises(TypeError):
        Value(2) / bad


# --------------------------------------------------------------------------
# Neural network modules
# --------------------------------------------------------------------------

def test_neuron_nonlin_flag():
    linear = Neuron(2, nonlin=False)
    linear.w[0].data, linear.w[1].data, linear.b.data = 2.0, 3.0, 1.0
    assert linear([1.0, 1.0]).data == 6.0  # 2 + 3 + 1, with no activation

    bounded = Neuron(2)
    bounded.w[0].data, bounded.w[1].data, bounded.b.data = 2.0, 3.0, 1.0
    assert bounded([1.0, 1.0]).data == pytest.approx(math.tanh(6.0))


def test_neuron_rejects_wrong_input_length():
    with pytest.raises(ValueError):
        Neuron(3)([1.0, 2.0])


def test_layer_returns_single_value_when_nout_is_one():
    assert isinstance(Layer(2, 1)([1.0, 2.0]), Value)
    assert isinstance(Layer(2, 3)([1.0, 2.0]), list)


def test_mlp_last_layer_is_linear_and_others_are_not():
    m = MLP(2, [4, 4, 1])
    assert all(n.nonlin for layer in m.layers[:-1] for n in layer.neurons)
    assert not any(n.nonlin for n in m.layers[-1].neurons)


def test_mlp_with_a_one_neuron_hidden_layer():
    # The hidden layer returns a single Value, which the next layer must accept.
    assert isinstance(MLP(2, [1, 3, 1])([1.0, 2.0]), Value)


def test_mlp_multiple_outputs():
    out = MLP(2, [3, 2])([1.0, 2.0])
    assert len(out) == 2


def test_mlp_zero_grad_clears_gradients():
    m = MLP(2, [3, 1])
    m([1.0, 2.0]).backward()
    assert any(p.grad != 0.0 for p in m.parameters())
    m.zero_grad()
    assert all(p.grad == 0.0 for p in m.parameters())


def test_mlp_training_reduces_loss():
    random.seed(0)
    m = MLP(2, [4, 1])
    xs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    ys = [-1.0, 1.0, 1.0, -1.0]  # XOR

    def loss_fn():
        return sum((m(x) - y) ** 2 for x, y in zip(xs, ys))

    first = loss_fn().data
    for _ in range(50):
        loss = loss_fn()
        m.zero_grad()
        loss.backward()
        for p in m.parameters():
            p.data -= 0.05 * p.grad
    assert loss_fn().data < first


# --------------------------------------------------------------------------
# Comparison against PyTorch (skipped when PyTorch is not installed)
# --------------------------------------------------------------------------

def _torch():
    return pytest.importorskip("torch")


# Each expression is written once and runs on both Value and torch.Tensor. The
# tanh/exp methods exist on both, so the same code exercises both engines.
EXPRESSIONS = {
    "a_plus_a": lambda a, b: a + a,
    "a_cubed_by_mul": lambda a, b: a * a * a,
    "mix_floats": lambda a, b: 2 * a + 1 - b / 2 + 3 / a - (1 - b),
    "tanh": lambda a, b: (a * b).tanh(),
    "exp": lambda a, b: (a + b).exp(),
    "pow": lambda a, b: a ** 3 + b ** -2 + a ** 0.5,
    "division": lambda a, b: a / b,
    "reused_node": lambda a, b: (lambda x: x * (x + 1) + x.tanh())(a * b),
    "karpathy_style": lambda a, b: ((a * b + b ** 3).tanh() * a.exp() - b / a) ** 2,
}


@pytest.mark.parametrize("name", EXPRESSIONS)
def test_expression_gradients_match_pytorch(name):
    torch = _torch()
    expression = EXPRESSIONS[name]

    a, b = Value(1.3), Value(-0.7)
    out = expression(a, b)
    out.backward()

    ta = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)
    tout = expression(ta, tb)
    tout.backward()

    assert out.data == pytest.approx(tout.item(), abs=TOLERANCE)
    assert a.grad == pytest.approx(ta.grad.item(), abs=TOLERANCE)
    # An expression may ignore b, in which case PyTorch leaves its grad as None.
    expected_b = 0.0 if tb.grad is None else tb.grad.item()
    assert b.grad == pytest.approx(expected_b, abs=TOLERANCE)


def test_mlp_gradients_match_pytorch():
    torch = _torch()
    random.seed(1)
    m = MLP(2, [3, 3, 1])
    x = [0.5, -1.2]

    loss = (m(x) - 1.0) ** 2
    loss.backward()

    # Rebuild the same network in PyTorch, copying each weight across.
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64, requires_grad=True)

    t_params = []  # (micrograd parameter, matching tensor)
    activations = [tensor(v) for v in x]
    for layer in m.layers:
        outputs = []
        for neuron in layer.neurons:
            total = tensor(neuron.b.data)
            t_params.append((neuron.b, total))
            for wi, xi in zip(neuron.w, activations):
                tw = tensor(wi.data)
                t_params.append((wi, tw))
                total = total + tw * xi
            outputs.append(total.tanh() if neuron.nonlin else total)
        activations = outputs

    t_loss = (activations[0] - 1.0) ** 2
    t_loss.backward()

    assert loss.data == pytest.approx(t_loss.item(), abs=TOLERANCE)
    assert len(t_params) == len(m.parameters())
    for p, tp in t_params:
        assert p.grad == pytest.approx(tp.grad.item(), abs=TOLERANCE)
