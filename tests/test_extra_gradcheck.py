"""
Extra (student-written) tests: numerical gradient checking, topological-sort
properties, the two "TA traps", deep graphs and a tiny end-to-end training run.
These are NOT part of the official autograder; they are for local confidence
and are referenced in the report.
"""
import math
import random

import pytest

from src.engine import Value
from src.nn import MLP, Neuron


# ---------------------------------------------------------------- helpers
def numeric_grad(f, xs, i, h=1e-6):
    """Central finite difference of f w.r.t. the i-th input."""
    hi = list(xs); hi[i] += h
    lo = list(xs); lo[i] -= h
    return (f(*hi) - f(*lo)) / (2 * h)


def check(f, xs, tol=1e-5):
    """f maps Values -> Value. Compare autograd with finite differences."""
    vals = [Value(x) for x in xs]
    f(*vals).backward()
    plain = lambda *a: f(*[Value(v) for v in a]).data
    for i, v in enumerate(vals):
        assert v.grad == pytest.approx(numeric_grad(plain, xs, i), rel=tol, abs=tol)


# ----------------------------------------------------- gradient checking
@pytest.mark.parametrize("f,xs", [
    (lambda a, b: a + b, [1.5, -2.0]),
    (lambda a, b: a - b, [1.5, -2.0]),
    (lambda a, b: a * b, [1.5, -2.0]),
    (lambda a, b: a / b, [1.5, -2.0]),
    (lambda a, b: a ** 3 + b ** -2, [1.3, 0.7]),
    (lambda a, b: a ** 0.5 * b, [2.0, 3.0]),
    (lambda a, b: (a * b).tanh(), [0.3, -0.8]),
    (lambda a, b: (a + b).exp(), [0.3, -0.8]),
    (lambda a, b: 2.0 - a * 3.0 + 1.0 / b, [0.4, 1.7]),
    (lambda a, b: ((a * b + a).tanh() * (a - b).exp()) / (b ** 2 + 1), [0.4, -1.2]),
])
def test_gradients_match_finite_differences(f, xs):
    check(f, xs)


def test_random_expression_gradients():
    rng = random.Random(0)
    for _ in range(25):
        xs = [rng.uniform(0.5, 1.5) for _ in range(3)]
        f = lambda a, b, c: ((a * b - c).tanh() + (a / c).exp()) * b ** 2 - c
        check(f, xs, tol=1e-4)


# --------------------------------------------------------- the TA traps
def test_topological_order_is_valid():
    a, b = Value(2.0), Value(-3.0)
    c = a * b
    d = c + a
    e = d.tanh()
    topo = e._get_topo()
    pos = {id(n): i for i, n in enumerate(topo)}
    assert topo[-1] is e                      # root comes last (post-order)
    assert len(topo) == len(set(map(id, topo)))  # each node exactly once
    for node in topo:                         # every child precedes its parent
        for ch in node._prev:
            assert pos[id(ch)] < pos[id(node)]


def test_iterative_and_recursive_topo_are_both_valid():
    a, b = Value(1.0), Value(2.0)
    out = ((a * b + a) * b).tanh() + a.exp()
    for topo in (out._get_topo(), out._get_topo_recursive()):
        pos = {id(n): i for i, n in enumerate(topo)}
        assert set(map(id, topo)) == set(map(id, out._get_topo()))
        assert all(pos[id(ch)] < pos[id(n)] for n in topo for ch in n._prev)


def test_diamond_graph_visited_set_no_double_counting():
    a = Value(3.0)
    c = a + a              # a appears twice as an input of the same node
    assert len(c._get_topo()) == 2   # {a, c}: 'a' must be visited only once
    c.backward()
    assert a.grad == 2.0   # dc/da = 2


def test_shared_subexpression():
    a = Value(2.0)
    b = a * a              # uses a twice
    c = b + b              # uses b twice
    c.backward()
    assert a.grad == 4 * 2.0   # c = 2a^2 -> 4a


def test_backward_twice_accumulates_unless_reset():
    a = Value(2.0)
    out = a * a
    out.backward(); first = a.grad
    out.backward()
    assert a.grad == pytest.approx(2 * first)   # accumulation semantics
    a.reset_grad(); out.grad = 0.0
    out.backward()
    assert a.grad == pytest.approx(first)


# ----------------------------------------------------------- edge cases
def test_deep_graph_does_not_hit_recursion_limit():
    x = Value(1.0)
    total = Value(0.0)
    for _ in range(5000):          # chain far deeper than Python's default limit
        total = total + x
    total.backward()
    assert x.grad == 5000.0


def test_tanh_and_exp_numerical_stability():
    big = Value(1000.0).tanh()
    assert big.data == 1.0
    with pytest.raises(OverflowError):
        Value(1000.0).exp()


def test_reverse_operators_and_scalars():
    a = Value(4.0)
    assert (10 - a).data == 6.0
    assert (8 / a).data == 2.0
    assert (a ** 0.5).data == 2.0
    assert (a - 1).data == 3.0
    assert (-a).data == -4.0


def test_pow_rejects_non_numbers():
    with pytest.raises(AssertionError):
        Value(2.0) ** "2"


def test_type_errors_on_all_binary_ops():
    a = Value(1.0)
    for bad in ("x", [1], None, {}):
        for op in (lambda: a + bad, lambda: a - bad, lambda: a * bad, lambda: a / bad,
                   lambda: bad + a, lambda: bad * a):
            with pytest.raises(TypeError):
                op()


# ------------------------------------------------------------------ nn
def test_neuron_output_is_in_tanh_range():
    n = Neuron(5)
    for _ in range(20):
        out = n([random.uniform(-10, 10) for _ in range(5)])
        assert -1.0 <= out.data <= 1.0


def test_mlp_parameter_gradients_match_finite_differences():
    random.seed(1)
    m = MLP(2, [3, 1])
    x, y = [0.5, -1.2], 0.7
    loss = lambda: (m(x) - y) ** 2
    m.zero_grad(); loss().backward()
    for p in m.parameters():
        old, h = p.data, 1e-6
        p.data = old + h; hi = loss().data
        p.data = old - h; lo = loss().data
        p.data = old
        assert p.grad == pytest.approx((hi - lo) / (2 * h), rel=1e-4, abs=1e-6)


def test_training_reduces_loss():
    random.seed(0)
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    m = MLP(3, [4, 4, 1])
    def total_loss():
        return sum((m(x) - y) ** 2 for x, y in zip(xs, ys))
    start = total_loss().data
    for _ in range(100):
        m.zero_grad()
        total_loss().backward()
        for p in m.parameters():
            p.data -= 0.05 * p.grad
    assert total_loss().data < 0.1 * start


def test_numpy_scalars_are_accepted():
    np = pytest.importorskip("numpy")
    a = Value(2.0)
    assert (a * np.int64(3)).data == 6.0
    assert (np.float64(1.5) + a).data == 3.5
    assert (a - np.int32(1)).data == 1.0
