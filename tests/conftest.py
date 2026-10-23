import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
