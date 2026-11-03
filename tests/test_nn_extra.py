
import random

import pytest

from src.engine import Value
from src.nn import Neuron, Layer, MLP


# --------------------------------------------------
# 1. Input dimension validation
# --------------------------------------------------

def test_neuron_input_dimensions():
    neuron = Neuron(3)

    with pytest.raises(ValueError):
        neuron([1.0, 2.0])

    with pytest.raises(ValueError):
        neuron([1.0, 2.0, 3.0, 4.0])

    output = neuron([1.0, 2.0, 3.0])
    assert isinstance(output, Value)


def test_layer_input_dimensions():
    layer = Layer(3, 4)

    with pytest.raises(ValueError):
        layer([1.0, 2.0])

    outputs = layer([1.0, 2.0, 3.0])
    assert len(outputs) == 4


def test_mlp_input_dimensions():
    model = MLP(3, [4, 4, 1])

    with pytest.raises(ValueError):
        model([1.0, 2.0])

    output = model([1.0, 2.0, 3.0])
    assert isinstance(output, Value)


# --------------------------------------------------
# 2. Invalid architecture validation
# --------------------------------------------------

def test_invalid_architectures():
    with pytest.raises(ValueError):
        Neuron(0)

    with pytest.raises(ValueError):
        Layer(3, 0)

    with pytest.raises(ValueError):
        MLP(3, [])

    with pytest.raises(ValueError):
        MLP(3, [4, -1, 1])


# --------------------------------------------------
# 3. Parameter count and initialization
# --------------------------------------------------

def test_parameter_count():
    model = MLP(3, [4, 4, 1])

    # (3*4+4) + (4*4+4) + (4*1+1)
    assert len(model.parameters()) == 41

    assert all(
        isinstance(p, Value)
        for p in model.parameters()
    )


def test_initialization_range():
    random.seed(42)
    model = MLP(3, [4, 1])

    for p in model.parameters():
        assert -1.0 <= p.data <= 1.0


# --------------------------------------------------
# 4. Full MLP numerical gradient checking
# --------------------------------------------------

def test_mlp_numerical_gradients():
    random.seed(42)

    # Small network for fast gradient checking.
    model = MLP(2, [3, 1])

    x = [0.5, -0.3]
    target = 0.4

    def compute_loss():
        prediction = model(x)
        return (prediction - target) ** 2

    # Compute analytical gradients.
    model.zero_grad()
    loss = compute_loss()
    loss.backward()

    params = model.parameters()
    analytical_grads = [p.grad for p in params]

    # Central finite difference.
    h = 1e-5

    for p, analytical in zip(params, analytical_grads):
        original = p.data

        try:
            p.data = original + h
            loss_plus = compute_loss().data

            p.data = original - h
            loss_minus = compute_loss().data

        finally:
            p.data = original

        numerical = (loss_plus - loss_minus) / (2 * h)

        assert analytical == pytest.approx(
            numerical,
            rel=1e-4,
            abs=1e-6
        )


# --------------------------------------------------
# 5. Gradient reset
# --------------------------------------------------

def test_mlp_zero_grad():
    random.seed(42)
    model = MLP(2, [3, 1])

    output = model([1.0, -1.0])
    output.backward()

    # At least one parameter should receive a gradient.
    assert any(
        abs(p.grad) > 1e-12
        for p in model.parameters()
    )

    model.zero_grad()

    assert all(
        p.grad == 0.0
        for p in model.parameters()
    )


# --------------------------------------------------
# 6. Training sanity test
# --------------------------------------------------

def test_training_reduces_loss():
    random.seed(42)

    # Simple supervised regression dataset.
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]

    y = [-0.5, 0.0, 0.0, 0.5]

    model = MLP(2, [4, 1])

    def compute_loss():
        predictions = [model(x) for x in X]

        # Mean squared error.
        return sum(
            (pred - target) ** 2
            for pred, target in zip(predictions, y)
        ) / len(y)

    initial_loss = compute_loss().data

    learning_rate = 0.05

    for epoch in range(300):
        loss = compute_loss()

        # Reset gradients before backpropagation.
        model.zero_grad()
        loss.backward()

        # Gradient descent parameter update.
        for p in model.parameters():
            p.data -= learning_rate * p.grad

    final_loss = compute_loss().data

    print(f"\nInitial loss: {initial_loss:.6f}")
    print(f"Final loss:   {final_loss:.6f}")

    assert final_loss < initial_loss
