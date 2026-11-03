import pytest
import math
from src.engine import Value
from src.nn import Neuron, Layer, MLP


def test_validation_a_basic_arithmetic():
    """Test A: Basic Arithmetic."""
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    d = a * b
    assert c.data == 5.0
    assert d.data == 6.0


def test_validation_b_power():
    """Test B: Power and its backward pass."""
    a = Value(2.0)
    b = a ** 3
    assert b.data == 8.0
    b.backward()
    assert a.grad == 12.0


def test_validation_c_diamond_graph():
    """Test C: Diamond Graph gradient accumulation."""
    a = Value(3.0)
    c = a + a
    d = c * a
    assert d.data == 18.0
    d.backward()
    assert a.grad == 12.0


def test_validation_d_chain_rule():
    """Test D: Chain Rule composite graph."""
    a = Value(-2.0)
    b = Value(3.0)
    d = a * b
    e = a + b
    f = d * e
    f.backward()

    # Analytical calculation:
    # f = (a * b) * (a + b) = a^2 * b + a * b^2
    # df/da = 2*a*b + b^2 = 2*(-2)*(3) + 9 = -12 + 9 = -3.0
    # df/db = a^2 + 2*a*b = 4 + 2*(-2)*(3) = 4 - 12 = -8.0
    assert math.isclose(a.grad, -3.0, rel_tol=1e-7, abs_tol=1e-7)
    assert math.isclose(b.grad, -8.0, rel_tol=1e-7, abs_tol=1e-7)


def test_validation_e_numerical_gradient_checking():
    """
    Test E: Numerical Gradient Checking using finite differences:
    f'(x) approx (f(x + h) - f(x - h)) / (2 * h)
    """
    h = 1e-6
    tolerance = 1e-5

    # Case 1: Non-linear expression with tanh, exp, pow, mul, add, div
    def forward(x_val, y_val):
        x = Value(x_val)
        y = Value(y_val)
        z = ((x * y + (x ** 2)) / (y + 1.0)).tanh() + (x * 0.5).exp()
        return z, x, y

    x0 = 1.5
    y0 = 2.5

    # Analytical
    z, x, y = forward(x0, y0)
    z.backward()
    analytical_dx = x.grad
    analytical_dy = y.grad

    # Numerical wrt x
    z_plus, _, _ = forward(x0 + h, y0)
    z_minus, _, _ = forward(x0 - h, y0)
    numerical_dx = (z_plus.data - z_minus.data) / (2.0 * h)

    # Numerical wrt y
    z_plus_y, _, _ = forward(x0, y0 + h)
    z_minus_y, _, _ = forward(x0, y0 - h)
    numerical_dy = (z_plus_y.data - z_minus_y.data) / (2.0 * h)

    assert math.isclose(analytical_dx, numerical_dx, rel_tol=tolerance, abs_tol=tolerance)
    assert math.isclose(analytical_dy, numerical_dy, rel_tol=tolerance, abs_tol=tolerance)


def test_validation_f_neural_network():
    """Test F: Neural Network architecture verification."""
    nin = 3
    nouts = [4, 4, 1]
    mlp = MLP(nin=nin, nouts=nouts)

    # Verify number of layers
    assert len(mlp.layers) == 3

    # Verify number of neurons per layer
    assert len(mlp.layers[0].neurons) == 4
    assert len(mlp.layers[1].neurons) == 4
    assert len(mlp.layers[2].neurons) == 1

    # Verify parameters count
    # Layer 0: 4 * (3 weights + 1 bias) = 16
    # Layer 1: 4 * (4 weights + 1 bias) = 20
    # Layer 2: 1 * (4 weights + 1 bias) = 5
    # Total = 41
    params = mlp.parameters()
    assert len(params) == 41

    # Verify forward pass
    x = [0.5, -1.0, 2.0]
    out = mlp(x)

    # Output shape and type
    assert isinstance(out, Value)
    assert -1.0 <= out.data <= 1.0  # bounded by tanh

    # Backward pass verification
    mlp.zero_grad()
    for p in params:
        assert p.grad == 0.0

    out.backward()
    for p in params:
        assert isinstance(p.grad, float)
