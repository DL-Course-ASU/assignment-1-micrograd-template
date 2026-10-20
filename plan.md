# Plan: Assignment 1 — MicroGrad (Scalar Autograd Engine)

**Course:** CSE485-CSE488 Deep Learning, Ain Shams University
**Goal:** Build a scalar autograd engine and a small neural-network API from scratch (no PyTorch), then submit via GitHub CI (Part A) and an LMS zip (Part B).

---

## 0. Implementation status

Everything below was implemented in this repository (the course template was
cloned in-place, so `src/`, `tests/` and `.github/workflows/` are present).

| Phase | Deliverable | Status |
| ----- | ----------- | ------ |
| 0 | Template cloned, `.venv` created (`--system-site-packages`, shares numpy/matplotlib/pytest) | done |
| 1 | `src/engine.py` — full `Value`: `add/mul/pow/tanh/exp` + `relu/log/sigmoid`, convenience ops, topological sort, `backward()`, `reset_grad()`, `__int__/__float__` | done |
| 2 | `src/nn.py` — `Neuron/Layer/MLP`, `nonlin` flag, linear final layer, `parameters()`/`zero_grad()` | done |
| 3 | Tests — template **12/12** + extra **30/30** = **42 passed** (`python -m pytest`, no `PYTHONPATH` needed); finite-difference check of all 41 MLP parameters | done |
| 4 | `demo.ipynb` — seeded, executed top-to-bottom: 98.5% train accuracy, gradient check max rel. error 5.8e-10, three figures saved to `report/figures/` | done |
| 5 | `report/main.pdf` — 11 pages, 0 LaTeX warnings, compiled with `tectonic` (binary in `.tools/`; no system TeX / sudo password available, so not `pdflatex`) | done |
| 6 | `submission.zip` built at the repo root (ready for the LMS) | done |

**Intentional deviations from the original plan**

* **Layout follows the template, not §2's sketch:** code lives in
  `src/engine.py` / `src/nn.py` because the CI imports
  `from src.engine import Value` (no top-level `engine.py`).
* **`_backward(upstream)` takes the gradient as an argument**
  (`node._backward(node.grad)`) instead of reading `out.grad`. The classic
  closure creates a reference cycle `out -> out._backward -> out`, which only
  the cyclic GC can reclaim — that alone cost 1532 → 669 ms/epoch when fixed
  (same math; justification in report §5).
* **Bias is random uniform `[-1, 1]`** (per the template's TODO), not `Value(0)`.
* **Last MLP layer is linear** (micrograd convention) so hinge margins can grow.

**Remaining manual steps (need your accounts)**

1. Push to GitHub as branch `name-with-hyphens_StudentID`
   (`git branch -m <your-name>_<your-id>` — the repo is still on `main`),
   open the PR into `main` → Actions grades it.
2. Upload `submission.zip` to the LMS before the deadline.
3. Check the LMS/template for published conceptual questions and append them to
   report §9 if they differ from the ones answered there.

---


## 1. Deliverables at a glance

| Part | What                                                                               | Where                                           |
| ---- | ---------------------------------------------------------------------------------- | ----------------------------------------------- |
| A    | `engine.py` + `nn.py` passing the autograder (pytest via GitHub Actions)           | GitHub template repo, merge request into `main` |
| B    | `.zip` with cleaned code, demo notebook (`.ipynb`), LaTeX-compiled report (`.pdf`) | LMS                                             |

**Branch name (strict):** `Name-With-Hyphens_StudentID` (the CI parses it to upsert your score into the instructor's sheet, so a wrong name = no grade).

---

## 2. Setup (Day 0)

- [ ] Fork/clone `https://github.com/DL-Course-ASU/assignment-1-micrograd-template`
- [ ] Create branch `Your-Name-Here_StudentID` and push it immediately
- [ ] Read the template's tests (`tests/`) and workflow file (`.github/workflows/`) to see exactly what the autograder checks
- [ ] Create a venv, install `pytest` (and `numpy`, `matplotlib`, `jupyter` for the notebook)
- [ ] Confirm `pytest` runs locally (it will fail, which is fine)

Suggested layout:

```
.
├── engine.py          # Value class + autograd
├── nn.py              # Neuron, Layer, MLP
├── tests/             # from template (+ your own extra tests)
├── demo.ipynb         # Part B notebook
├── report/            # main.tex, figures/
└── plan.md
```

---

## 3. Phase 1 — `engine.py` (the computational graph)

### 3.1 `Value` skeleton

- [ ] `__init__(self, data, _children=(), _op='')` storing `data`, `grad = 0.0`, `_backward = lambda: None`, `_prev = set(_children)`, `_op`
- [ ] `__repr__` for debugging

### 3.2 Operators (forward + local backward)

Every `_backward` must **accumulate with `+=`**, never assign with `=`.

| Op                        | Forward       | Local backward (`out.grad` = upstream)                       |
| ------------------------- | ------------- | ------------------------------------------------------------ |
| `__add__`                 | `a + b`       | `a.grad += out.grad`; `b.grad += out.grad`                   |
| `__mul__`                 | `a * b`       | `a.grad += b.data * out.grad`; `b.grad += a.data * out.grad` |
| `__pow__` (int/float `k`) | `a ** k`      | `a.grad += k * a.data**(k-1) * out.grad`                     |
| `tanh`                    | `t = tanh(a)` | `a.grad += (1 - t**2) * out.grad`                            |
| `exp`                     | `e = exp(a)`  | `a.grad += e * out.grad`                                     |

- [ ] Wrap plain numbers: `other = other if isinstance(other, Value) else Value(other)`
- [ ] Assert `__pow__` exponent is `int`/`float` (not `Value`)

### 3.3 Convenience operators (built from the primitives above)

- [ ] `__neg__` → `self * -1`
- [ ] `__sub__` → `self + (-other)`
- [ ] `__radd__`, `__rmul__`, `__rsub__`
- [ ] `__truediv__` → `self * other**-1`
- [ ] `__rtruediv__`
- [ ] Optional extras (cheap and often tested): `relu`, `log`, `sigmoid`

### 3.4 Topological sort and `backward()` (from the hint)

- [ ] `_get_topo`: DFS with `topo` list + `visited` set; recurse on `node._prev`; **append node after visiting children** (post-order)
- [ ] `backward`: `topo = self._get_topo()`, set `self.grad = 1.0`, then call `node._backward()` for each node in `reversed(topo)`

**Traps to test explicitly**

- **Post-order trap:** append _after_ the children loop.
- **Diamond trap:** keep the `visited` set. Test `c = a + a` (expect `a.grad == 2`) and `b = a * a` (expect `a.grad == 2 * a.data`).
- Reusing one `Value` in several places must accumulate gradients (this is why `+=` matters).
- Consider whether `backward()` should reset grads first, or document that the caller must zero them between passes (see Section 5).

### 3.4 Verification

- [ ] Hand-check small expressions (e.g. `x*y + z`, `tanh(x*w + b)`)
- [ ] Numerical gradient check: `(f(x+h) - f(x-h)) / 2h` with `h = 1e-6`, compare to autograd grads
- [ ] Optional: compare against PyTorch **only in a throwaway test script**, not in submitted code

---

## 4. Phase 2 — `nn.py` (neural network API)

- [ ] Base `Module` with `zero_grad()` and `parameters()`
- [ ] `Neuron(nin, nonlin=True)`
  - weights `w`: `nin` Values, random uniform in `[-1, 1]`; bias `b`: `Value(0)` (or random, check the tests)
  - `__call__(x)`: `act = sum(wi*xi) + b`, then `tanh` (or identity if `nonlin=False`)
  - `parameters()` returns `w + [b]`
- [ ] `Layer(nin, nout)`: list of `nout` neurons; `__call__` returns a single Value if `nout == 1`, else a list
- [ ] `MLP(nin, nouts)`: e.g. `MLP(3, [4, 4, 1])`; chain layers; make the last layer linear if the tests/assignment expect it
- [ ] `parameters()` flattens all params through the hierarchy
- [ ] Check what the template tests expect for constructor signatures, return types, and parameter counts (e.g. `MLP(3,[4,4,1])` should have `(3+1)*4 + (4+1)*4 + (4+1)*1 = 41` parameters)

---

## 5. Phase 3 — Tests (local, before pushing)

- [ ] Run the template's tests: `pytest -v`
- [ ] Add your own tests for:
  - each operator's forward value and gradient
  - diamond graphs (`a + a`, `a * a`, shared sub-expressions)
  - chained ops with scalars on the left (`2 * x`, `1 - x`, `3 / x`)
  - `tanh` and `exp` at 0 and at large |x| (numerical stability)
  - topological order property: every node appears after all of its children
  - parameter counts and `zero_grad()`
  - edge cases: `x ** 0`, `x ** 1`, negative exponents, `exp` of a large value
- [ ] Gradient-check the whole MLP against finite differences

---

## 6. Phase 4 — Demo notebook (`demo.ipynb`)

Must be reproducible: load data, train, visualize.

- [ ] Fix random seeds at the top
- [ ] Sanity section: the small `Value` expression with printed grads
- [ ] Dataset: a toy binary classification set (e.g. `make_moons`, or a small generated dataset). Load and plot it.
- [ ] Build `MLP(2, [16, 16, 1])`
- [ ] Loss: MSE or hinge/max-margin (plus optional L2 regularization)
- [ ] Training loop: forward → `zero_grad()` → `loss.backward()` → SGD update (`p.data -= lr * p.grad`), with learning-rate decay if desired
- [ ] Plots: loss curve, accuracy curve, decision boundary
- [ ] Run **Restart & Run All** once at the end to confirm it works top to bottom

---

## 7. Phase 5 — LaTeX report (`report.pdf`)

Required: mathematical derivations, design justifications, answers to conceptual questions.

- [ ] **Chain rule and reverse-mode autodiff:** explain why the gradient flows from output to inputs
- [ ] **Local derivatives**, derived by hand:
  - add, mul, pow, tanh (`1 - tanh²`), exp
- [ ] **Topological sort:** why it is needed, why post-order, a worked example on a small graph
- [ ] **Diamond graph:** why the `visited` set and `+=` are both necessary, with `c = a + a` worked through
- [ ] **Design justifications:** why a closure-based `_backward`, `_prev` as a set, the random initialization range, why `tanh`
- [ ] **Neural network section:** the Neuron/Layer/MLP math with equations
- [ ] **Results:** figures from the notebook, loss curves, decision boundary, gradient-check table
- [ ] **Answers to the conceptual questions** (watch for questions in the template repo or on the LMS, since the PDF itself doesn't list them)
- [ ] Compile with `pdflatex`/`latexmk`, check that references and figures render

---

## 8. Git workflow and submission

1. Commit small, meaningful steps on your named branch.
2. When `pytest` is green locally, push.
3. Open a **merge/pull request into `main`** (this triggers the GitHub Actions autograder).
4. Check the Actions run. Fix failures and push again (it re-runs).
5. Confirm the score appears in the instructor's sheet.
6. Clean the code (remove debug prints, add docstrings), then build the zip:

```
submission.zip
├── engine.py
├── nn.py
├── tests/            (optional)
├── demo.ipynb
└── report.pdf        (plus report/main.tex as a bonus)
```

7. Upload the zip to the LMS.

---

## 9. Suggested schedule

| Block | Focus                                                   | Exit criteria                                    |
| ----- | ------------------------------------------------------- | ------------------------------------------------ |
| 1     | Setup, read template tests, `Value` + add/mul/pow       | Basic gradient checks pass                       |
| 2     | `tanh`, `exp`, convenience ops, topo sort, `backward()` | Diamond tests pass, numerical grad check matches |
| 3     | `nn.py` + local pytest green                            | All template tests pass                          |
| 4     | Open merge request, fix CI failures                     | CI green, score recorded                         |
| 5     | Notebook (train + plots)                                | Restart & Run All works                          |
| 6     | Report in LaTeX                                         | Compiles, all required sections present          |
| 7     | Final cleanup, zip, LMS upload                          | Submitted before the deadline                    |

---

## 10. Common pitfalls (checklist)

- [ ] `grad` accumulates with `+=` in every `_backward`
- [ ] Topo append happens **after** the children loop
- [ ] `visited` set is present
- [ ] Python numbers on the left of operators work (`__radd__`, `__rmul__`, `__rsub__`, `__rtruediv__`)
- [ ] `__pow__` only takes int/float exponents
- [ ] Gradients are zeroed between training steps
- [ ] Branch name matches `Name-With-Hyphens_StudentID` exactly
- [ ] Notebook runs from scratch with a fixed seed
- [ ] Report is LaTeX-compiled (not exported from Word/Markdown)
- [ ] No PyTorch imports in `engine.py`/`nn.py`

---

## 11. Open questions to resolve early

- What are the exact function signatures and return types expected by the template's tests?
- Are there extra operations the tests check beyond `add/mul/pow/tanh/exp` (e.g. `relu`, `log`, `sub`, `div`)?
- Where are the "conceptual assignment questions" for the report listed?
- What is the submission deadline and what is the penalty for late merges?
- Is `Name-With-Hyphens_StudentID` using the full name or first and last only?
