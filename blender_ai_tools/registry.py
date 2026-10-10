"""この拡張機能の道具を、道具の目録（A8）に載せる。

道具を足すときは、ここに1つずつ書いて register() する。文字列から関数を探して動かすようなことはしない
（AI が選べるのは、ここに載せた道具だけ。CLAUDE.md 2章1）。
"""

from __future__ import annotations

from typing import Any

from . import bones, thresholds
from .catalog import Catalog, Effect, ToolSpec

# MARK: - A3 骨の名前の対応

MAP_BONE_NAMES = ToolSpec(
    id="bones.map_names",
    version="1",
    feature="A3",
    category="bones",
    title="骨の名前の対応",
    summary="骨の名前を MMD 準標準の名前に対応させ、分からない骨・足りない骨・決められない骨を一覧にする",
    effect=Effect.READ,
    heavy=False,
    input_schema={
        "type": "object",
        "properties": {
            "names": {
                "type": "array",
                "description": "モデルの骨の名前",
                "items": {"type": "string", "maxLength": thresholds.MAX_BONE_NAME_LENGTH},
                "maxItems": thresholds.MAX_BONE_NAMES,
            },
        },
        "required": ["names"],
        "additionalProperties": False,
    },
    measures_after=("分からない骨・決められない骨は、ユーザーに聞く",),
)


def map_bone_names(params: dict[str, Any]) -> dict[str, Any]:
    """bones.map_names の中身。結果は JSON にできる形で返す"""
    result = bones.map_bones(params["names"])
    return {
        "mapping": dict(result.mapping),
        "unknown": list(result.unknown),
        "known_extras": list(result.known_extras),
        "conflicts": {canonical: list(names) for canonical, names in result.conflicts.items()},
        "ambiguous": {name: list(candidates) for name, candidates in result.ambiguous.items()},
        "shadowed": dict(result.shadowed),
        "missing_standard": list(result.missing_standard),
        "missing_semi_standard": list(result.missing_semi_standard),
        "semi_standard_compliant": result.is_semi_standard_compliant,
    }


# MARK: - 目録を作る


def build_catalog() -> Catalog:
    """この拡張機能の道具をすべて載せた目録"""
    catalog = Catalog()
    catalog.register(MAP_BONE_NAMES, map_bone_names)
    return catalog
