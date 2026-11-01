"""Hardware backend abstraction and FakeBelem loader."""

from .backend_info import BackendInfo
from .fake_belem import load_fake_belem_info

__all__ = ["BackendInfo", "load_fake_belem_info"]

