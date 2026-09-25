# conftest.py — repo root
# Ensures the repo root is on sys.path so `src.entity_resolution.*` imports
# work in pytest without needing `pip install -e .` or PYTHONPATH.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
