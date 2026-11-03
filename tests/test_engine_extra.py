
import math

import pytest

from src.engine import Value


def test_reverse_subtraction():
    x = Value(2.0)
    y = 5.0 - x

    assert y.data == 3.0

    y.backward()
    assert x.grad == pytest.approx(-1.0)


def test_reverse_division():
    x = Value(2.0)
    y = 6.0 / x

    assert y.data == 3.0

    y.backward()
    assert x.grad == pytest.approx(-1.5)


def test_invalid_operands():
    x = Value(2.0)

    with pytest.raises(TypeError):
        _ = x + "hello"

    with pytest.raises(TypeError):
        _ = x * [1, 2]

    with pytest.raises(TypeError):
        _ = x - None

    with pytest.raises(TypeError):
        _ = x / "2"


def test_invalid_initialization():
    with pytest.raises(TypeError):
        Value("5")

    with pytest.raises(TypeError):
        Value(True)


def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        _ = Value(5.0) / Value(0.0)

    with pytest.raises(ZeroDivisionError):
        _ = 5.0 / Value(0.0)


def test_zero_power_gradient():
    x = Value(0.0)
    y = x ** 0

    assert y.data == 1.0

    y.backward()
    assert x.grad == 0.0


def test_fractional_power():
    x = Value(4.0)
    y = x ** 0.5

    assert y.data == pytest.approx(2.0)

    y.backward()
    assert x.grad == pytest.approx(0.25)


def test_invalid_power_domain():
    with pytest.raises(ValueError):
        _ = Value(-2.0) ** 0.5

    with pytest.raises(ZeroDivisionError):
        _ = Value(0.0) ** -1


def test_shared_node_gradient():
    x = Value(3.0)
    y = x * x + x

    # y = x^2 + x
    # dy/dx = 2x + 1 = 7
    y.backward()

    assert x.grad == pytest.approx(7.0)


def test_repeated_backward():
    x = Value(2.0)
    y = x ** 3

    y.backward()
    first_grad = x.grad

    y.backward()
    second_grad = x.grad

    assert first_grad == pytest.approx(12.0)
    assert second_grad == pytest.approx(12.0)


def test_topological_order():
    a = Value(2.0)
    b = a * 3
    c = b + a

    topo = c._get_topo()

    assert len(topo) == len(set(topo))
    assert topo[-1] is c
    assert topo.index(a) < topo.index(b)
    assert topo.index(b) < topo.index(c)


def test_numerical_gradient():
    x = Value(1.5)
    y = (x * x + 2 * x).tanh()

    y.backward()
    analytical = x.grad

    def f(v):
        return math.tanh(v * v + 2 * v)

    h = 1e-5
    numerical = (f(1.5 + h) - f(1.5 - h)) / (2 * h)

    assert analytical == pytest.approx(
        numerical, rel=1e-4, abs=1e-7
    )


def test_exponential_gradient():
    x = Value(1.2)
    y = x.exp()

    y.backward()

    assert y.data == pytest.approx(math.exp(1.2))
    assert x.grad == pytest.approx(math.exp(1.2))
