"""Shared test setup.

The unit tests cover the parts of YaYan that are plain Python: metrics, routing, text
clean-up and configuration. They must run on a machine without a GPU, model weights or
the heavy ML stack, so CI only installs requirements-test.txt.
"""
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# yayan/llm/__init__.py imports the LLM client, which imports torch. The tests only need
# the text helpers in yayan/llm/postprocess.py, so register yayan.llm as a bare package:
# Python then finds the submodule in that directory without running __init__.py.
if "yayan.llm" not in sys.modules:
    import yayan  # noqa: F401  (parent package must exist first)

    _pkg = types.ModuleType("yayan.llm")
    _pkg.__path__ = [str(ROOT / "yayan" / "llm")]
    sys.modules["yayan.llm"] = _pkg
