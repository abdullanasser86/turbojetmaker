"""Device selection for all training code (device-agnostic: CUDA workstation or CPU container)."""

from __future__ import annotations

import os


def get_device(prefer: str | None = None):
    """Return a ``torch.device``.

    Order: explicit ``prefer`` argument, then the ``TURBOML_DEVICE`` environment variable,
    then CUDA if available, else CPU. Use this everywhere instead of hard-coding "cuda".
    """
    import torch

    choice = prefer or os.environ.get("TURBOML_DEVICE")
    if choice:
        return torch.device(choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
