"""Test shims for optional heavy dependencies.

These stubs let unit tests import app/bootstrap/UI modules without requiring
full ML runtime packages in local and CI test environments.
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace


def _ensure_torch_stub() -> None:
    if importlib.util.find_spec("torch") is not None:
        import torch  # noqa: F401

        return
    if "torch" in sys.modules:
        return
    torch_stub = SimpleNamespace(
        __version__="0.0-test",
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    sys.modules["torch"] = torch_stub


def _ensure_kokoro_stub() -> None:
    if "kokoro" in sys.modules:
        return

    class _KModel:
        def __init__(self, repo_id=None, **_kwargs):
            self.repo_id = repo_id
            self.device = None

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    class _Lexicon:
        def __init__(self):
            self.golds = {}

    class _G2P:
        def __init__(self):
            self.lexicon = _Lexicon()

    class _KPipeline:
        def __init__(self, lang_code="a", repo_id=None, model=None, **_kwargs):
            _ = (lang_code, repo_id, model)
            self.g2p = _G2P()

        def load_voice(self, voice):
            return {"voice": voice}

    sys.modules["kokoro"] = SimpleNamespace(
        __version__="0.0-test",
        KModel=_KModel,
        KPipeline=_KPipeline,
    )


def _ensure_misaki_stub() -> None:
    if "misaki" in sys.modules:
        return
    sys.modules["misaki"] = SimpleNamespace(__version__="0.0-test")


_ensure_torch_stub()
_ensure_kokoro_stub()
_ensure_misaki_stub()
