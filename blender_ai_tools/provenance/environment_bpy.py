"""A15：Blender の中でだけ分かる環境（Blender の版と、mmd_tools の版）。

記録には、python_environment()（bpy なし）と合わせて入れる：

    environment = {**python_environment(), **blender_environment()}

mmd_tools は、古い形のアドオン（モジュール名 "mmd_tools"）でも、拡張機能（"bl_ext.<リポジトリ>.mmd_tools"）でも見つける。
入っていないときは、版を None にする（測れなかった物として扱う）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import addon_utils
import bpy

# mmd_tools のモジュールの名前。拡張機能のときは、最後の部分がこの名前になる
MMD_TOOLS_MODULE = "mmd_tools"


def blender_environment() -> dict[str, Any]:
    """Blender と mmd_tools の版。

    blender_version: Blender の版の文字（例 "5.2.2 LTS"）
    blender_build_hash: Blender のビルドのハッシュ（例 "d13f752e3b9c"）
    mmd_tools_module: 見つけた mmd_tools のモジュールの名前（例 "bl_ext.blender_org.mmd_tools"）。無いときは None
    mmd_tools_version: mmd_tools の版（例 "4.3.2"）。入っていない・版が分からないときは None
    mmd_tools_enabled: mmd_tools が有効になっていて、読み込まれているか
    """
    module_name, version, enabled = _find_mmd_tools(addon_utils.modules(), addon_utils.module_bl_info, _is_enabled)
    return {
        "blender_version": bpy.app.version_string,
        "blender_build_hash": _text(bpy.app.build_hash),
        "mmd_tools_module": module_name,
        "mmd_tools_version": version,
        "mmd_tools_enabled": enabled,
    }


# MARK: - mmd_tools を探す


def _is_mmd_tools_name(module_name: str) -> bool:
    """mmd_tools のモジュールの名前か（"mmd_tools" か、"….mmd_tools" で終わる名前）"""
    return module_name == MMD_TOOLS_MODULE or module_name.endswith("." + MMD_TOOLS_MODULE)


def _is_enabled(module_name: str) -> bool:
    """アドオンが有効で、いま読み込まれているか"""
    _, loaded = addon_utils.check(module_name)
    return loaded is True


def _find_mmd_tools(modules: Iterable[Any], bl_info_of: Callable[[Any], dict[str, Any]],
                    is_enabled: Callable[[str], bool]) -> tuple[str | None, str | None, bool]:
    """アドオンの一覧から mmd_tools を探し、（モジュールの名前、版、有効か）を返す。

    いくつも見つかったときは、有効な物を先にし、同じなら名前の順で最初の物を使う。
    """
    candidates = [module for module in modules if _is_mmd_tools_name(str(getattr(module, "__name__", "")))]
    if not candidates:
        return None, None, False
    chosen = min(candidates, key=lambda module: (not is_enabled(module.__name__), module.__name__))
    info = bl_info_of(chosen)
    version = _version_text(info.get("version") if isinstance(info, dict) else None)
    return chosen.__name__, version, bool(is_enabled(chosen.__name__))


def _version_text(version: Any) -> str | None:
    """bl_info の版（(4, 3, 2) など）を "4.3.2" の形にする。版が無いときは None"""
    if isinstance(version, str):
        return version.strip() or None
    if isinstance(version, (tuple, list)) and version:
        return ".".join(str(part) for part in version)
    return None


def _text(value: Any) -> str:
    """bytes（bpy.app.build_hash など）を文字にする"""
    if isinstance(value, bytes):
        return value.decode("ascii", errors="replace")
    return str(value)
