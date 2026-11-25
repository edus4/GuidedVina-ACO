#!/usr/bin/env python3
"""The stub engine advertising no bias option, to test capability refusal."""
import runpy
import sys
from pathlib import Path

sys.argv = [sys.argv[0], *sys.argv[1:], "--pretend-no-bias"]
runpy.run_path(str(Path(__file__).with_name("stub_engine.py")), run_name="__main__")
