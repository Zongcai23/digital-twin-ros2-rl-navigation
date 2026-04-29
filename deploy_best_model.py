"""Convenience wrapper.

Equivalent to:
    python rl_navigation/deploy_best_model.py
"""

import runpy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "rl_navigation"))
runpy.run_path(str(ROOT / "rl_navigation" / "deploy_best_model.py"), run_name="__main__")
