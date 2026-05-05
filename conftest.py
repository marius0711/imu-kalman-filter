"""conftest.py — Add src/ to sys.path so pytest finds modules without PYTHONPATH."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
