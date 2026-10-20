"""Extra local tests for the autograd engine (plan.md, section 5).

These complement the template's ``tests/test_engine.py`` with hand-checked
gradients, scalar-on-the-left operators, numerical stability, the
topological-order property and finite-difference gradient checks.

Everything here uses only the standard library + pytest so it runs
identically under the course CI (Python 3.10, ``pip install pytest``).
"""

import math

import pytest

from src.engine import Value

H = 1e-6  # step for central finite differences


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def num_grad(f, x, h=H):
    """Central-difference estimate of df/dx at the scalar ``x``."""
    return (f(x + h) - f(x - h)) / (2 * h)


def close(a, b, tol=1e-5):
    assert a == pytest.approx(b, abs=tol), f"{a} != {b}"


# ----------------------------------------------------------------------
# forward values + hand-derived gradients for every primitive operator
# ----------------------------------------------------------------------
def test_add_forward_and_grad():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    close(c.data, 5.0)
    c.backward()
    close(a.grad, 1.0)
    close(b.grad, 1.0)


def test_mul_forward_and_grad():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    close(c.data, 6.0)
    c.backward()
    close(a.grad, 3.0)   # d(ab)/da = b
    close(b.grad, 2.0)   # d(ab)/db = a


def test_sub_and_div_grads():
    a = Value(6.0)
    b = Value(3.0)
    s = a - b
    s.backward()
    close(a.grad, 1.0)
    close(b.grad, -1.0)

    a.grad = b.grad = 0.0
    d = a / b            # 2.0
    close(d.data, 2.0)
    d.backward()
    close(a.grad, 1.0 / 3.0)          # d(a/b)/da = 1/b
    close(b.grad, -6.0 / 9.0)         # d(a/b)/db = -a/b^2


def test_pow_gradients_and_edge_cases():
    # x ** 2 at x = 3 -> 2x = 6
    x = Value(3.0)
    (x ** 2).backward()
    close(x.grad, 6.0)

    # negative exponent: d(x^-2)/dx = -2 x^-3
    x = Value(2.0)
    y = x ** -2          # 0.25
    close(y.data, 0.25)
    y.backward()
    close(x.grad, -2 / 8)

    # x ** 1 is the identity
    x = Value(4.0)
    (x ** 1).backward()
    close(x.grad, 1.0)

    # x ** 0 is a constant -> gradient 0 (even at x = 0)
    x = Value(4.0)
    y = x ** 0
    close(y.data, 1.0)
    y.backward()
    close(x.grad, 0.0)

    x = Value(0.0)
    (x ** 0).backward()
    close(x.grad, 0.0)


def test_pow_rejects_value_exponent():
    with pytest.raises(AssertionError):
        _ = Value(2.0) ** Value(0.5)


# ----------------------------------------------------------------------
# scalars on the LEFT of operators (reflected dunder methods)
# ----------------------------------------------------------------------
def test_scalar_on_the_left():
    x = Value(4.0)

    y = 2 * x
    close(y.data, 8.0)
    y.backward()
    close(x.grad, 2.0)

    x.grad = 0.0
    y = 1 - x             # __rsub__
    close(y.data, -3.0)
    y.backward()
    close(x.grad, -1.0)

    x.grad = 0.0
    y = 3 / x             # __rtruediv__
    close(y.data, 0.75)
    y.backward()
    close(x.grad, -3.0 / 16.0)   # d(3/x)/dx = -3/x^2

    x.grad = 0.0
    y = 5 + x             # __radd__
    close(y.data, 9.0)
    y.backward()
    close(x.grad, 1.0)


# ----------------------------------------------------------------------
# numerical stability of tanh / exp
# ----------------------------------------------------------------------
def test_tanh_at_zero_and_extremes():
    x = Value(0.0)
    t = x.tanh()
    close(t.data, 0.0)
    t.backward()
    close(x.grad, 1.0)              # 1 - tanh(0)^2 = 1

    for v in (50.0, -50.0, 100.0, -100.0):
        x = Value(v)
        t = x.tanh()
        expected = math.tanh(v)
        close(t.data, expected, tol=1e-12)
        t.backward()
        # saturated: derivative 1 - tanh^2 is ~0, no overflow/NaN
        assert math.isfinite(x.grad)
        close(x.grad, 1.0 - expected ** 2, tol=1e-9)


def test_exp_numerical_stability():
    x = Value(0.0)
    e = x.exp()
    close(e.data, 1.0)
    e.backward()
    close(x.grad, 1.0)

    # very negative: underflows smoothly to 0, gradient matches the output
    x = Value(-700.0)
    e = x.exp()
    assert e.data == pytest.approx(0.0, abs=1e-300)
    e.backward()
    assert x.grad == pytest.approx(e.data, rel=1e-9, abs=1e-300)

    # large but finite
    x = Value(700.0)
    e = x.exp()
    assert math.isfinite(e.data)
    e.backward()
    close(x.grad, e.data)

    # genuinely too large: Python's own overflow, not silent corruption
    x = Value(1000.0)
    with pytest.raises(OverflowError):
        _ = x.exp()


def test_sigmoid_stability_and_grad():
    for v in (0.0, 1.0, -1.0, 500.0, -500.0):
        x = Value(v)
        s = x.sigmoid()
        assert math.isfinite(s.data)
        assert 0.0 <= s.data <= 1.0
        x.grad = 0.0
        s.backward()
        close(x.grad, s.data * (1 - s.data), tol=1e-9)


def test_relu_grad():
    x = Value(2.0)
    r = x.relu()
    close(r.data, 2.0)
    r.backward()
    close(x.grad, 1.0)

    x = Value(-2.0)
    r = x.relu()
    close(r.data, 0.0)
    r.backward()
    close(x.grad, 0.0)


def test_log_grad():
    x = Value(4.0)
    l = x.log()
    close(l.data, math.log(4.0))
    l.backward()
    close(x.grad, 0.25)   # d(ln x)/dx = 1/x


# ----------------------------------------------------------------------
# graph structure: topological order + diamond accumulation
# ----------------------------------------------------------------------
def test_topo_order_children_before_parents():
    a = Value(1.0)
    b = Value(2.0)
    c = (a * b).tanh() + (a - b) ** 2
    d = c * a + b.exp()
    d.backward()

    topo = d._get_topo()
    # every node appears exactly once
    assert len(topo) == len(set(topo))
    # every node comes after all of its children (post-order property)
    position = {node: i for i, node in enumerate(topo)}
    for node in topo:
        for child in node._prev:
            assert position[child] < position[node], "child must precede its parent"
    assert topo[-1] is d


def test_diamond_graphs_accumulate():
    # c = a + a  ->  a.grad = 2
    a = Value(3.0)
    c = a + a
    c.backward()
    close(a.grad, 2.0)

    # b = a * a  ->  a.grad = 2a
    a = Value(3.0)
    b = a * a
    b.backward()
    close(a.grad, 6.0)

    # reuse in three places: f = a*a + a + a
    a = Value(3.0)
    f = a * a + a + a
    f.backward()
    close(a.grad, 2 * 3.0 + 1 + 1)


def test_shared_subexpression_used_twice():
    # e = a*b is consumed by two parents -> contribution counted twice
    a = Value(2.0)
    b = Value(5.0)
    e = a * b
    f = e * e + e
    # f = (ab)^2 + ab ; d/da = 2ab*b + b = 2*2*5*5 + 5 = 105
    f.backward()
    close(a.grad, 105.0)
    close(b.grad, 2 * 2 * 5 * 2 + 2)   # 42


def test_backward_does_not_reset_previous_grads():
    # gradients accumulate across passes unless reset_grad() is called
    a = Value(2.0)
    b = a * 3
    b.backward()
    close(a.grad, 3.0)
    b.backward()
    close(a.grad, 6.0)      # accumulated on purpose
    a.reset_grad()
    close(a.grad, 0.0)


# ----------------------------------------------------------------------
# finite-difference (numerical) gradient checks
# ----------------------------------------------------------------------
def test_numerical_grad_scalar_expression():
    def f(x):                       # x is a plain float
        a = Value(x)
        return ((a.tanh() * a) + a ** 3).data

    a = Value(0.7)
    out = ((a.tanh() * a) + a ** 3)
    out.backward()
    close(a.grad, num_grad(f, 0.7), tol=1e-6)


def test_numerical_grad_two_inputs():
    def f_factory(u, v):
        def f(x):                   # perturb u, hold v fixed
            a, b = Value(x), Value(v)
            return ((a * b) / (a + b)).data
        return f

    def g_factory(u, v):
        def f(x):                   # perturb v, hold u fixed
            a, b = Value(u), Value(x)
            return ((a * b) / (a + b)).data
        return f

    u, v = 1.3, -0.6
    a, b = Value(u), Value(v)
    out = (a * b) / (a + b)
    out.backward()
    close(a.grad, num_grad(f_factory(u, v), u), tol=1e-6)
    close(b.grad, num_grad(g_factory(u, v), v), tol=1e-6)


def test_numerical_grad_deep_chain():
    """A five-op chain checked against finite differences."""

    def f(x):
        a = Value(x)
        b = a * 2 + 1
        c = b.tanh()
        d = c * b
        e = d.exp()
        return e.data

    x0 = -0.35
    a = Value(x0)
    b = a * 2 + 1
    c = b.tanh()
    d = c * b
    e = d.exp()
    e.backward()
    close(a.grad, num_grad(f, x0), tol=1e-6)
