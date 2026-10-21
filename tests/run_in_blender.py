"""Blender の中でテストを動かす。

使い方:
    # tests/ の下を全部（unit と blender の両方）
    blender --background --factory-startup --python-exit-code 1 --python tests/run_in_blender.py
    # 選んだものだけ（モジュール名か、ファイルの場所）
    blender --background --factory-startup --python-exit-code 1 --python tests/run_in_blender.py -- \
        tests.unit.test_bones tests/blender/test_extension.py

テストが1つでも失敗すると、終了コード 1 で終わる。
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def module_name(target: str) -> str:
    """tests/unit/test_x.py のようなファイルの場所を、tests.unit.test_x のモジュール名にする"""
    if target.endswith(".py"):
        path = Path(target).resolve().relative_to(REPO_ROOT)
        return ".".join(path.with_suffix("").parts)
    return target


targets = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
loader = unittest.defaultTestLoader
if targets:
    suite = loader.loadTestsFromNames([module_name(t) for t in targets])
else:
    suite = loader.discover(str(REPO_ROOT / "tests"), top_level_dir=str(REPO_ROOT))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
