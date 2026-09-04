"""
Root Entrypoint for Gym Exercise Classification CLI.
Usage:
    py run.py --help
    py run.py preprocess --help
    py run.py train --help
    py run.py evaluate --help
    py run.py ensemble --help
    py run.py reproduce --help
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.cli import main

if __name__ == "__main__":
    main()
