from gtsrb.utils.config import ROOT, load_config
from gtsrb.utils.seed import set_seed


def test_import_and_root():
    assert (ROOT / "pyproject.toml").exists()


def test_load_base_config():
    cfg = load_config("configs/base.yaml")
    assert cfg["data"]["img_size"] == 64
    assert cfg["model"]["name"] == "resnet18"


def test_override_merges_nested(tmp_path):
    (tmp_path / "base.yaml").write_text("aug: {geo: true, blur: true}\nseed: 1\n")
    (tmp_path / "e.yaml").write_text("_base_: base.yaml\naug: {blur: false}\n")
    cfg = load_config(tmp_path / "e.yaml")
    assert cfg["aug"] == {"geo": True, "blur": False}  # chỉ blur bị ghi đè
    assert cfg["seed"] == 1


def test_set_seed_runs():
    set_seed(42)
