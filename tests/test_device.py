import torch

from turboml.device import get_device


def test_device_auto_and_override(monkeypatch):
    monkeypatch.delenv("TURBOML_DEVICE", raising=False)
    assert get_device().type == ("cuda" if torch.cuda.is_available() else "cpu")
    monkeypatch.setenv("TURBOML_DEVICE", "cpu")
    assert get_device().type == "cpu"
    assert get_device("cpu").type == "cpu"
