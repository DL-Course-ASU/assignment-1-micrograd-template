"""Keep the repository root importable when pytest runs from another directory."""
from pathlib import Path
import sys

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
