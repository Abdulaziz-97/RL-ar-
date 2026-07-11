"""Entry point for python -m rlvr_pipeline."""
from rlvr_pipeline.cli import main as _main
import sys

if __name__ == "__main__":
    sys.exit(_main())
