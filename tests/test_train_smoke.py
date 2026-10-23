import json
from pathlib import Path

import pytest

from training.train import main


@pytest.mark.timeout(240)
def test_end_to_end(tmp_path: Path):
    rc = main([
        "--n-samples", "8", "--n-val", "4", "--epochs", "1",
        "--batch-size", "2", "--device", "cpu", "--no-amp",
        "--duration", "0.25", "--out-dir", str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / "last.pt").exists()
    report = json.loads((tmp_path / "train_report.json").read_text())
    assert report["environment"]["torch"]
    assert len(report["history"]) == 1
    assert report["history"][0]["train"]["curve"] >= 0.0


@pytest.mark.timeout(120)
def test_dry_run(tmp_path: Path):
    assert main(["--dry-run", "--n-samples", "4", "--batch-size", "2",
                 "--device", "cpu", "--duration", "0.25",
                 "--out-dir", str(tmp_path)]) == 0
