"""Đặc tả chạy được cho B và C. Test tự bỏ qua nếu module chưa có, sẽ chạy khi bạn merge code của mình."""
import pytest

torch = pytest.importorskip("torch")

PARAM_RANGES = {
    "small_cnn": (9.5e4, 1.05e5),
    "resnet18": (1.11e7, 1.13e7),
    "plain18": (1.09e7, 1.11e7),
}


@pytest.mark.parametrize("name", list(PARAM_RANGES))
def test_build_model_contract(name):
    models = pytest.importorskip("gtsrb.models")
    if not hasattr(models, "build_model"):
        pytest.skip("build_model chưa có")
    m = models.build_model(name, mean=[0.5] * 3, std=[0.2] * 3).eval()
    lo, hi = PARAM_RANGES[name]
    assert lo <= sum(p.numel() for p in m.parameters()) <= hi
    with torch.no_grad():
        assert m(torch.rand(2, 3, 64, 64)).shape == (2, 43)


def test_normalization_is_inside_model():
    models = pytest.importorskip("gtsrb.models")
    if not hasattr(models, "build_model"):
        pytest.skip("build_model chưa có")
    x = torch.rand(2, 3, 64, 64)
    a = models.build_model("resnet18", mean=[0.0] * 3, std=[1.0] * 3).eval()
    b = models.build_model("resnet18", mean=[0.5] * 3, std=[0.2] * 3).eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        assert not torch.allclose(a(x), b(x))   # cùng trọng số, khác mean/std -> output khác


def test_corruptions_contract():
    c = pytest.importorskip("gtsrb.data.corruptions")
    assert c.CONDITIONS["dark"] == [0.70, 0.55, 0.35]
    assert c.CONDITIONS["blur"] == [1.0, 1.5, 2.5]
    x = torch.rand(4, 3, 64, 64)
    for f in c.CONDITIONS["dark"]:
        y = c.darken(x, f)
        assert y.shape == x.shape and 0 <= y.min() and y.max() <= 1
        assert torch.allclose(y, x * f, atol=1e-5)
    for s in c.CONDITIONS["blur"]:
        y = c.blur(x, s)
        assert y.shape == x.shape and 0 <= y.min() and y.max() <= 1
        assert y.var() < x.var()             # làm mờ giảm phương sai
    assert torch.equal(c.apply_condition(x, "orig", "-"), x)
