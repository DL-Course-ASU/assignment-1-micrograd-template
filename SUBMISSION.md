# Assignment 1: MicroGrad

Student: **Yousef Mohamed Mahmoud Helmy Ismail**<br>
ID: **20P7954**<br>
Branch: `Yousef-Mohamed-Mahmoud-Helmy-Ismail_20P7954`

[Open the completed assignment branch on GitHub](https://github.com/glacialhex/assignment-1-micrograd-template/tree/Yousef-Mohamed-Mahmoud-Helmy-Ismail_20P7954).

This submission implements the supplied course template at commit
`2a9d379546bdc61b8e3ff846e551051ac0602bc6`.
The original README, tests, workflow, and package initializers are preserved.
Only `src/engine.py` and `src/nn.py` are used by the upstream autograder.
No external autograd library is needed or imported.

## Reproduce

Use Python 3.12 for the pinned notebook environment. The scalar core is also
compatible with the grader's Python 3.10 runtime.

```text
python -m venv .venv
```

Activate on Windows PowerShell with `.\.venv\Scripts\Activate.ps1`, or on
macOS/Linux with `source .venv/bin/activate`, then run:

```text
python -m pip install -r requirements.txt
python -m pytest
python -m scripts.verify
python -m nbconvert --to notebook --execute --inplace notebooks/demo.ipynb
python -m nbconvert --to html notebooks/demo.ipynb
python -m scripts.build_evidence
```

The notebook runs offline after installation, generates its own data, and embeds
all outputs. It exports figures at 300 DPI and as vector PDFs under
`report/figures/`. Data, split indices, metrics, learned parameters, and per-update
history are saved under `artifacts/`. No checkpoint is loaded before training.
The exact installed dependency inventory is in `environment-lock.txt`.

To refresh CLI evidence, run `python -m pytest -q --junitxml=artifacts/pytest.xml`
and `python -m scripts.verify`, saving their stdout as `artifacts/pytest.txt`
and `artifacts/verification.txt`, then rerun `scripts.build_evidence`.

Compile from the `report/` directory with either:

```text
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Documentation_Report report.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=Documentation_Report report.tex
```

or `tectonic report.tex` and rename the resulting `report.pdf` to
`Documentation_Report.pdf`. This copy used Tectonic 0.17.0; the source is standard
LaTeX. Screenshots already included in the archive are required when compiling.

## Files to review

- `src/engine.py`, `src/nn.py`: scalar differentiation and network API.
- `tests/`: 12 preserved template tests and 25 supplemental cases.
- `scripts/verify.py`: three shared-graph analytic checks and 60 independent
  finite-difference comparisons at 30 seeded points.
- `notebooks/demo.ipynb`: executed end-to-end demonstration.
- `notebooks/demo.html`: browser view of the executed notebook.
- `report/Documentation_Report.pdf`: compiled mathematical and experimental report.
- `report/report.tex`, `report/results.tex`: editable LaTeX sources.
- `report/figures/`: graph, optimization curves, decision boundary, and screenshots.
- `artifacts/`: raw logs, JUnit XML, data, and structured experiment evidence.
- `SUBMISSION_MANIFEST.json`: SHA-256 inventory inside the ZIP.

## Behavior and limitations

Every neuron uses tanh, including the output, matching the template. Single-neuron
layers return a Value; MLP handles these intermediate bottlenecks explicitly.
Inputs and network dimensions are checked instead of silently truncating them.

Backward seeds the root with 1 and accumulates with `+=`. For each training step,
construct a fresh graph and call `model.zero_grad()` before `loss.backward()`.
If reusing the same graph, clear every node in `loss._get_topo()` before another
backward call; clearing only the leaves is insufficient. Do not mutate node data
between forward and backward. Recursive DFS has Python's usual recursion limit.

The evidence screenshots show an executed notebook and a browser rendering of
verbatim CLI logs. They are explicitly labeled as browser screenshots; native
terminal automation was unavailable under the computer-use rules. The actual CLI
logs and JUnit results are included separately. Local passing tests do not claim
that hosted GitHub Actions ran. The prepared branch is committed and published
on the linked GitHub fork. Opening a course PR and uploading to the LMS remain
submission actions.

Prepared with AI coding assistance. The report records this provenance.
