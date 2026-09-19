import tomllib
from pathlib import Path
import pcc_poker


def test_package_versions_are_synchronized():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == pcc_poker.__version__ == "0.8.0"
