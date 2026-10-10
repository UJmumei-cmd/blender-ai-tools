"""Blender の中で、tests/ の下のテストを全部動かす（unit と blender の両方）。

使い方:
    blender --background --factory-startup --python-exit-code 1 --python tests/run_in_blender.py

テストが1つでも失敗すると、終了コード 1 で終わる。
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

suite = unittest.defaultTestLoader.discover(str(REPO_ROOT / "tests"), top_level_dir=str(REPO_ROOT))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
