import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

STUB = ROOT / "tests" / "stub_engine.py"
EXAMPLES = ROOT / "examples"


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    if not (EXAMPLES / "sample_receptor.pdbqt").exists():
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_examples.py")], check=True)
    return EXAMPLES


@pytest.fixture(scope="session")
def stub_path() -> Path:
    return STUB
