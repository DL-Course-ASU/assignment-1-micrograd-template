"""Independent analytic, numerical, topology, and network checks."""
import math
import random
import pytest
from src.engine import Value
from src.nn import Neuron, Layer, MLP


@pytest.mark.parametrize('fn,expected', [
    (lambda a: a+a, 2), (lambda a: a*a+a, 7),
    (lambda a: (a+a)*a, 12), (lambda a: (a*a)*(a*a), 108),
])
def test_shared_paths(fn, expected):
    a = Value(3.0)
    root = fn(a)
    topo = root._get_topo()
    assert len(topo) == len(set(topo))
    positions = {node: i for i, node in enumerate(topo)}
    assert topo[-1] is root
    assert all(positions[p] < positions[n] for n in topo for p in n._prev)
    root.backward()
    assert a.grad == expected


@pytest.mark.parametrize('value_fn,real_fn,x', [
    (lambda a: 5-a, lambda a: 5-a, 1.3),
    (lambda a: 5/a, lambda a: 5/a, 1.3),
    (lambda a: (a**3).tanh()+a.exp()/(a**2+2),
     lambda a: math.tanh(a**3)+math.exp(a)/(a**2+2), 0.7),
    (lambda a: a**-2, lambda a: a**-2, 1.2),
    (lambda a: a**0.5, lambda a: a**0.5, 1.2),
    (lambda a: a.relu(), lambda a: max(0, a), -1.2),
    (lambda a: a.relu(), lambda a: max(0, a), 1.2),
])
def test_finite_difference(value_fn, real_fn, x):
    a = Value(x)
    y = value_fn(a)
    y.backward()
    h = 1e-6
    numerical = (real_fn(x+h)-real_fn(x-h))/(2*h)
    assert y.data == pytest.approx(real_fn(x))
    assert a.grad == pytest.approx(numerical, rel=1e-6, abs=1e-7)


@pytest.mark.parametrize('x', [-1000.0, 1000.0])
def test_stable_tanh(x):
    a = Value(x)
    a.tanh().backward()
    assert a.grad == 0


def test_zero_power_and_relu_origin():
    a = Value(0.0)
    (a**0).backward()
    assert a.grad == 0
    a.relu().backward()
    assert a.grad == 0


def test_reset_and_fresh_graph_accumulation():
    a = Value(2.0)
    (a*a).backward()
    (3*a).backward()
    assert a.grad == 7
    a.reset_grad()
    (a*a).backward()
    assert a.grad == 4


def test_leaf_backward_and_graph_reset():
    a = Value(2.0)
    a.backward()
    assert a.grad == 1
    a.reset_grad()
    y = (a*a+a).tanh()
    y.backward()
    first = a.grad
    for node in y._get_topo():
        node.reset_grad()
    y.backward()
    assert a.grad == first


def test_multivariate_composite():
    def real(x, y):
        return math.tanh(x*y+x*x)+math.exp(x-y)/(y*y+2)
    a, b = Value(.4), Value(-.8)
    ((a*b+a**2).tanh()+(a-b).exp()/(b**2+2)).backward()
    h = 1e-6
    assert a.grad == pytest.approx((real(.4+h,-.8)-real(.4-h,-.8))/(2*h), rel=1e-6)
    assert b.grad == pytest.approx((real(.4,-.8+h)-real(.4,-.8-h))/(2*h), rel=1e-6)


def test_mlp_bottleneck_and_all_parameter_gradients():
    random.seed(17)
    m = MLP(2, [1, 3, 1])
    assert len(m.parameters()) == 13
    x = [.3, -.2]
    m(x).backward()
    for p in m.parameters():
        analytic, original, h = p.grad, p.data, 1e-6
        p.data = original+h
        plus = m(x).data
        p.data = original-h
        minus = m(x).data
        p.data = original
        assert analytic == pytest.approx((plus-minus)/(2*h), rel=1e-5, abs=1e-7)
    m.zero_grad()
    assert all(p.grad == 0 for p in m.parameters())


@pytest.mark.parametrize('factory', [
    lambda: Neuron(0), lambda: Layer(2,0), lambda: MLP(2,[]),
    lambda: MLP(2,[1.5]),
])
def test_invalid_network_dimensions(factory):
    with pytest.raises(ValueError):
        factory()


def test_wrong_input_width():
    with pytest.raises(ValueError):
        Neuron(2)([1])


def test_nonreal_power():
    with pytest.raises(ValueError):
        Value(-1)**.5


def test_descent_decreases_loss():
    random.seed(23)
    m = MLP(2, [3,1])
    samples = [([0.,0.],-1.), ([0.,1.],1.), ([1.,0.],1.)]
    def loss():
        return sum((m(x)-y)**2 for x,y in samples)/len(samples)
    initial = loss().data
    for _ in range(30):
        current = loss()
        m.zero_grad()
        current.backward()
        for p in m.parameters():
            p.data -= .1*p.grad
    assert loss().data < .35*initial

