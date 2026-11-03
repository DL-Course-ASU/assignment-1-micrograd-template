"""Supplemental finite-difference verification; run as python -m scripts.verify."""
import json
import math
import random
from pathlib import Path
from src.engine import Value


def run_checks():
    for expression, expected in [(lambda a: a+a, 2), (lambda a: a*a+a, 7),
                                 (lambda a: (a+a)*a, 12)]:
        a = Value(3.0)
        out = expression(a)
        topo = out._get_topo()
        assert len(topo) == len(set(topo))
        out.backward()
        assert a.grad == expected
    def real(x, y):
        return math.tanh(x*y+x**3)+math.exp(x-y)/(y*y+2)+2/(x*x+1)
    rng = random.Random(2026)
    errors = []
    h = 1e-6
    for _ in range(30):
        x, y = rng.uniform(-1, 1), rng.uniform(-1, 1)
        a, b = Value(x), Value(y)
        out = (a*b+a**3).tanh()+(a-b).exp()/(b**2+2)+2/(a*a+1)
        out.backward()
        numerical = [(real(x+h,y)-real(x-h,y))/(2*h),
                     (real(x,y+h)-real(x,y-h))/(2*h)]
        for autodiff, estimate in zip([a.grad,b.grad], numerical):
            error = abs(autodiff-estimate)
            errors.append(error)
            assert error < 1e-7 + 1e-6*abs(estimate)
    return {'diamond_checks': 3, 'composite_points': 30,
            'partial_derivatives_checked': len(errors), 'step': h,
            'max_absolute_error': max(errors), 'status': 'PASS'}


if __name__ == '__main__':
    result = run_checks()
    root = Path(__file__).resolve().parents[1]
    (root/'artifacts').mkdir(exist_ok=True)
    (root/'artifacts/verification.json').write_text(json.dumps(result, indent=2))
    print('Shared-node graphs: 3/3 PASS')
    print('Composite pow/tanh/exp/division: 60/60 partial derivatives PASS')
    print(f"Central difference h={result['step']:.0e}")
    print(f"Maximum absolute error: {result['max_absolute_error']:.3e}")
    print('SUPPLEMENTAL VERIFICATION: PASS')
