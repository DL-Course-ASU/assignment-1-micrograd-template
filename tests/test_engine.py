import pytest
import math
from src.engine import Value

def test_value_initialization():
    v = Value(5.0, label='test_node')
    assert v.data == 5.0
    assert v.grad == 0.0
    assert v.label == 'test_node'
    assert repr(v) == "Value(data=5.0)"

def test_basic_operations():
    a = Value(2.0)
    b = Value(3.0)
    
    # Addition and reverse addition
    c = a + b
    d = 2.0 + a
    assert c.data == 5.0
    assert d.data == 4.0

    # Multiplication and reverse multiplication
    e = a * b
    f = 3.0 * a
    assert e.data == 6.0
    assert f.data == 6.0

    # Subtraction
    g = b - a  # 3 - 2 = 1
    assert g.data == 1.0

    # Division 
    h = Value(6.0) / Value(2.0) # 3.0
    assert h.data == 3.0

    # Negation
    i = -a
    assert i.data == -2.0

def test_type_errors():
    # Ensure students don't blindly add strings or lists to Values
    with pytest.raises(TypeError):
        _ = Value(2.0) + "string"
    with pytest.raises(TypeError):
        _ = Value(2.0) * [1, 2, 3]

def test_power():
    a = Value(2.0)
    b = a ** 3
    assert b.data == 8.0
    
    b.backward()
    # d(a^3)/da = 3 * a^2 = 3 * 4 = 12
    assert a.grad == 12.0

    # Power should explicitly assert only int/float are supported
    with pytest.raises(AssertionError):
        _ = Value(2.0) ** Value(3.0)

def test_activations_and_exp():
    # Test tanh
    a = Value(0.0)
    b = a.tanh()
    assert b.data == 0.0
    
    b.backward()
    # d(tanh(x))/dx = 1 - tanh^2(x). If x=0, grad = 1
    assert a.grad == 1.0

    # Test exp
    c = Value(0.0)
    d = c.exp()
    assert d.data == 1.0
    
    # Reset grads to test exp backward cleanly
    c.grad = 0.0
    d.backward()
    # d(e^x)/dx = e^x. If x=0, grad = 1
    assert c.grad == 1.0

def test_gradient_accumulation_diamond_graph():
    """
    Tests a diamond graph where 'a' branches out and rejoins.
    If students use self.grad = ... instead of self.grad += ..., this fails.
    """
    a = Value(3.0)
    c = a + a
    d = c * a
    d.backward()
    
    # d = (a + a) * a = 2a * a = 2a^2
    # dd/da = 4a = 4 * 3 = 12.0
    assert a.grad == 12.0

def test_chain_rule_and_backprop():
    """
    Comprehensive composite function test evaluating the chain rule
    """
    a = Value(-2.0, label='a')
    b = Value(3.0, label='b')
    d = a * b; d.label = 'd'
    e = a + b; e.label = 'e'
    f = d * e; f.label = 'f'
    
    f.backward()
    
    # f = (a * b) * (a + b) = ab(a + b) = a^2 b + ab^2
    # df/da = 2ab + b^2 = 2(-2)(3) + 9 = -12 + 9 = -3.0
    # df/db = a^2 + 2ab = 4 + 2(-2)(3) = 4 - 12 = -8.0
    assert round(a.grad, 4) == -3.0
    assert round(b.grad, 4) == -8.0

def test_edge_case_division_by_zero():
    # Ensure division by zero raises appropriate exception
    a = Value(5.0)
    b = Value(0.0)
    
    with pytest.raises(ZeroDivisionError):
        c = a / b
        _ = c.data # Force evaluation

def test_type_casting():
    # Ensure __int__ and __float__ are implemented correctly
    v = Value(5.7)
    assert int(v) == 5
    assert float(v) == 5.7