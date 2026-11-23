# Assignment 1: MicroGrad (Scalar Autograd Engine)

Welcome to Assignment 1 of the **Deep Learning & Computer Vision** course. In this assignment, you will build a tiny scalar automatic differentiation engine and a basic neural network API from scratch, inspired by Andrej Karpathy's `micrograd`.

By building this engine, you will understand the mechanics of computational graphs, the chain rule, topological sorting, and how deep learning frameworks like PyTorch handle backpropagation under the hood.

---

## Repository Structure

```text
.
├── conftest.py         # Pytest configuration to fix import paths (DO NOT MODIFY)
├── src/
│   ├── engine.py       # Core scalar autograd engine (YOU FILL THIS IN)
│   └── nn.py           # Neural network modules (YOU FILL THIS IN)
├── tests/
│   ├── test_engine.py  # Test suite for the autograd engine
│   └── test_nn.py      # Test suite for the neural network modules
└── README.md           # This instructions file

```

---

## Getting Started

1. **Clone the repository** locally or open it in your development environment.
2. **Open `src/engine.py**` and implement the missing logic inside the `Value` class:
* Arithmetic operations (`__add__`, `__mul__`, `__pow__`, etc.) and their corresponding backward pass logic (`_backward`).
* Activation functions (`tanh`, `exp`).
* The global backward pass (`backward`) utilizing **Topological Sort**.


3. **Open `src/nn.py`** and implement the PyTorch-like API:
* Forward passes and parameter collection for `Neuron`, `Layer`, and `MLP`.


4. **Run tests locally** using `pytest`. Run this exact command from the root of the repository:

```bash
# Run all tests
python -m pytest

# Run all tests with detailed verbose output
python -m pytest -v

# Run only the engine tests
python -m pytest tests/test_engine.py -v

```

---

## Submission & Branch Naming Guidelines

To submit your assignment, you must open a **Pull Request (PR)** to the main repository.

### Strict Branch Naming Convention

Your working branch **must** follow this exact format, combining your full name and university student ID separated by an underscore:

```text
name-with-hyphens_StudentID

```

**Example:**

If your name is *Seif Yasser Ahmed* and your ID is *21P0102*, your branch name must be:

```text
seif-yasser-ahmed_21P0102

```

> The automated grading system parses this branch name to log your grades directly into the course tracking sheet.

---

## Automated Grading & Feedback

When you open or push updates to your Pull Request:

1. GitHub Actions will automatically run `pytest` against your code.
2. An automated bot will comment on your PR with a detailed breakdown of your test results.
3. Your score (formatted as `Passed/Total`, e.g., `12/12 PASSED`) will automatically sync with the course gradebook.
4. **Re-submissions:** You can push updates to your branch as many times as you like before the deadline. Each push will **override** your previous score with your latest attempt.
