import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")


@pytest.fixture(scope="session")
def fake_root(tmp_path_factory) -> Path:
    from make_fake_dataset import main

    root = tmp_path_factory.mktemp("fake-pet")
    main(root, n=36, seed=1)
    return root
