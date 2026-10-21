"""この拡張機能の道具を、道具の目録（A8）に載せる。

道具を足すときは、ここに1つずつ書いて register() する。文字列から関数を探して動かすようなことはしない
（AI が選べるのは、ここに載せた道具だけ。CLAUDE.md 2章1）。

- 道具の関数は、値の dict を1つ受け取り、JSON にそのまま書ける物を返す（裏の係 A10 が result.json に書く）。
- bpy が要る部品（*_bpy）は、関数の中で読み込む（この部品は bpy なしでも読み込めるようにしておく）。
- 重い道具（heavy=True）は、画面からは裏の Blender（A10）で動かす。進み具合は jobs.progress.report で書く
  （画面の中で直接呼んだときは、何もしない）。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from . import bones, texts, thresholds
from .catalog import Catalog, Effect, ToolSpec
from .export.roundtrip import DEVIATION_NAME
from .jobs import progress
from .measure.results import Measurement
from .motion.course import CHECKS, Event, report_lines, summarize_course
from .motion.poses import POSES
from .versions import history
from .versions.diff import ALL_CATEGORIES, CATEGORIES, AllowedChanges, Change, change_lines

# 場所（ファイル・フォルダ）を受け取る項目のスキーマ
_PATH_SCHEMA = {"type": "string", "maxLength": thresholds.MAX_PATH_LENGTH}

# MARK: - A3 骨の名前の対応

MAP_BONE_NAMES = ToolSpec(
    id="bones.map_names",
    version="2",  # 版 2：結果に sway_extras（揺れ物の骨）を足した
    feature="A3",
    category="bones",
    title="骨の名前の対応",
    summary="骨の名前を MMD 準標準の名前に対応させ、分からない骨・足りない骨・決められない骨・揺れ物の骨を一覧にする",
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
        # known_extras のうち、揺れ物・飾りの骨（髪・スカート など）。報告の「揺れ物の骨 N 本」に使う
        "sway_extras": list(result.sway_extras),
        "conflicts": {canonical: list(names) for canonical, names in result.conflicts.items()},
        "ambiguous": {name: list(candidates) for name, candidates in result.ambiguous.items()},
        "shadowed": dict(result.shadowed),
        "missing_standard": list(result.missing_standard),
        "missing_semi_standard": list(result.missing_semi_standard),
        "semi_standard_compliant": result.is_semi_standard_compliant,
    }


# MARK: - E1・E2 書き出して、読み戻して確かめる

CONVERT_AND_CHECK = ToolSpec(
    id="export.convert_and_check",
    version="1",
    feature="E1",
    category="export",
    title="書き出して確かめる",
    summary="紅魔館の convert_blend.py で版のファイルを作業フォルダに書き出し（E1）、USDZ を読み戻して、"
            "いま開いているモデルと形・名前・ウェイトを比べる（E2）。アプリのフォルダには書かない",
    effect=Effect.READ,
    heavy=True,
    input_schema={
        "type": "object",
        "properties": {
            "input_blend": {**_PATH_SCHEMA, "description": "書き出す .blend（この道具が保存した版のファイル）"},
            "output_dir": {**_PATH_SCHEMA, "description": "出力フォルダ（まだ無いか、空のフォルダ。アプリの Model フォルダは断る）"},
            "koumakan_repo": {**_PATH_SCHEMA, "description": "紅魔館のリポジトリの場所"},
            "blender": {**_PATH_SCHEMA, "description": "書き出しに使う Blender の場所。空なら、いま動いている Blender"},
        },
        "required": ["input_blend", "output_dir", "koumakan_repo"],
        "additionalProperties": False,
    },
    checks_before=("出力フォルダが、アプリの Model フォルダと convert_blend.py のフォルダではないこと",
                   "入力が、この道具が保存した版のファイルであること"),
    measures_after=("読み戻した形のずれ（0.5mm 未満）", "1点に付く骨が4本まで", "物理の部品が入っていないこと",
                    "名前が originalNames を通して合うこと"),
)


def _digest_data(digest: Any) -> dict[str, Any] | None:
    return None if digest is None else asdict(digest)


def convert_and_check(params: dict[str, Any]) -> dict[str, Any]:
    """export.convert_and_check の中身。書き出しに失敗しても例外にせず、convert.ok と problems で返す"""
    from .export import convert
    from .export import roundtrip_bpy
    from .jobs import launcher_bpy

    blender = launcher_bpy.blender_binary(params.get("blender") or None)
    progress.report(0, 2, "export.step_convert")
    result = convert.run_convert(params["input_blend"], params["output_dir"], blender=blender,
                                 repo=params["koumakan_repo"])
    convert_data = {
        "ok": result.ok,
        "signal": result.signal.name.lower(),
        "problems": list(result.problems),
        "warnings": list(result.warnings),
        "summary": result.summary,
        "output_dir": str(result.output_dir),
        "usdz": _digest_data(result.usdz),
        "rig_path": None if result.rig_path is None else str(result.rig_path),
        "input_unchanged": result.input_unchanged,
        "timed_out": result.timed_out,
        "returncode": result.returncode,
        "record": json.loads(result.record.to_json()),
    }
    if result.ok and result.usdz is not None and result.rig_path is not None:
        progress.report(1, 2, "export.step_roundtrip")
        measurements = roundtrip_bpy.run_roundtrip(result.usdz.path, result.rig_path)
    else:
        measurements = [Measurement.unmeasured(DEVIATION_NAME, "m", texts.text("export.convert_failed"))]
    progress.report(2, 2, "export.step_done")
    return {"convert": convert_data, "roundtrip": [m.to_data() for m in measurements]}


# MARK: - A6 版の管理と変更点の一覧

SAVE_VERSION = ToolSpec(
    id="versions.save",
    version="1",
    feature="A6",
    category="versions",
    title="新しい版で保存",
    summary="いまの中身を新しい版（_v001、_v002 …）として保存し、形・骨・ウェイトの記録を残す。元のファイルは上書きしない",
    effect=Effect.READ,
    heavy=False,
    input_schema={
        "type": "object",
        "properties": {
            "base_path": {**_PATH_SCHEMA, "description": "元のファイル（版の名前のもと）。空なら、いま開いているファイルから決める"},
        },
        "additionalProperties": False,
    },
    checks_before=("いま開いているファイルが保存ずみであること（まだ保存していないファイルは断る）",),
)


def save_version(params: dict[str, Any]) -> dict[str, Any]:
    """versions.save の中身"""
    from .versions import save_bpy

    target, folder = save_bpy.save_version(base_path=params.get("base_path") or None)
    return {"version": str(target), "name": target.name, "snapshot_dir": str(folder)}


COMPARE_WITH_PREVIOUS = ToolSpec(
    id="versions.compare_with_previous",
    version="1",
    feature="A6",
    category="versions",
    title="前の版と比べる",
    summary="いまの中身を、前の版の記録と比べて、変わった所を一覧にする。「変えてよい所」の外の変化が1件でもあれば赤",
    effect=Effect.READ,
    heavy=False,
    input_schema={
        "type": "object",
        "properties": {
            "base_path": {**_PATH_SCHEMA, "description": "元のファイル。空なら、いま開いているファイルから決める"},
            "version_path": {**_PATH_SCHEMA, "description": "比べる版のファイル。空なら、いちばん新しい版"},
            "allowed": {
                "type": "array",
                "description": "変えてよい所。オブジェクトごとに、変えてよい種類（categories）か、どの種類も（all）",
                "items": {
                    "type": "object",
                    "properties": {
                        "object": {"type": "string", "maxLength": thresholds.MAX_OBJECT_NAME_LENGTH},
                        "categories": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(CATEGORIES)},
                            "minItems": 1,
                            "maxItems": len(CATEGORIES),
                            "uniqueItems": True,
                        },
                        "all": {"type": "boolean"},
                    },
                    "required": ["object"],
                    "additionalProperties": False,
                },
                "maxItems": thresholds.MAX_ALLOWED_OBJECTS,
            },
        },
        "additionalProperties": False,
    },
    measures_after=("変えてよい所の外の変化の件数（1件でもあれば赤）",),
)


def allowed_from_params(items: list[dict[str, Any]] | None) -> AllowedChanges:
    """道具の allowed（[{object, categories, all}]）を AllowedChanges にする。

    all が true なら、どの種類も変えてよい（categories は見ない）。種類も all も無い項目は ValueError
    （何も書かないのは、まちがいのことが多いため）。同じオブジェクトが2回あるときも ValueError。
    """
    objects: dict[str, Any] = {}
    for item in items or []:
        name = item["object"]
        if name in objects:
            raise ValueError(f"変えてよい所に、同じオブジェクト {name!r} が2回あります")
        if item.get("all"):
            objects[name] = ALL_CATEGORIES
        elif item.get("categories"):
            objects[name] = frozenset(item["categories"])
        else:
            raise ValueError(f"変えてよい所の {name!r} に、種類（categories）か all を書いてください")
    return AllowedChanges(objects)


def _change_data(change: Change) -> dict[str, Any]:
    return {"category": change.category, "object": change.object, "count": change.count,
            "outside_allowed": change.outside_allowed, "text": change.text()}


def compare_with_previous(params: dict[str, Any]) -> dict[str, Any]:
    """versions.compare_with_previous の中身。変わった所の中身（details）は changes に出すので、まとめには入れない"""
    from .versions import save_bpy

    allowed = allowed_from_params(params.get("allowed"))
    changes, measurement = save_bpy.compare_with_previous(base_path=params.get("base_path") or None,
                                                          version_path=params.get("version_path") or None,
                                                          allowed=allowed)
    return {"measurement": measurement.to_data(include_details=False),
            "changes": [_change_data(change) for change in changes],
            "lines": change_lines(changes)}


LIST_VERSIONS = ToolSpec(
    id="versions.list",
    version="1",
    feature="A6",
    category="versions",
    title="版の一覧",
    summary="元のファイルから作った版（開ける物）を、番号の小さい順に一覧にする",
    effect=Effect.READ,
    heavy=False,
    input_schema={
        "type": "object",
        "properties": {
            "base_path": {**_PATH_SCHEMA, "description": "元のファイル。空なら、いま開いているファイルから決める"},
        },
        "additionalProperties": False,
    },
)


def list_versions(params: dict[str, Any]) -> dict[str, Any]:
    """versions.list の中身。まだ保存していないファイルで base_path も無いときは ValueError"""
    from .versions import save_bpy

    base = save_bpy.current_base(params.get("base_path") or None)
    if base is None:
        raise ValueError("まだ保存していないファイルです。先にファイルを保存するか、元のファイル（base_path）を渡してください")
    versions = history.saved_versions(base)
    return {"base": str(base),
            "versions": [{"number": number, "path": str(path), "name": path.name} for number, path in versions],
            "lines": history.version_lines(base)}


# MARK: - A5 確かめの動き

POSE_NAMES = tuple(pose.name for pose in POSES)

RUN_COURSE = ToolSpec(
    id="motion.run_course",
    version="1",
    feature="A5",
    category="motion",
    title="確かめの動き",
    summary="決まったポーズを1つずつ付けて、ひざの逆曲がり・足の床への沈み・服からの体のはみ出しを確かめ、"
            "出来事の一覧にする。アプリの動きは、まだ計算できないので灰",
    effect=Effect.READ,
    heavy=True,
    input_schema={
        "type": "object",
        "properties": {
            "poses": {
                "type": "array",
                "description": "確かめるポーズの名前。空なら全部",
                "items": {"type": "string", "enum": list(POSE_NAMES)},
                "maxItems": len(POSE_NAMES),
                "uniqueItems": True,
            },
            "checks": {
                "type": "array",
                "description": "する確かめ。空なら全部",
                "items": {"type": "string", "enum": list(CHECKS)},
                "maxItems": len(CHECKS),
                "uniqueItems": True,
            },
            "visibility": {
                "type": "boolean",
                "description": "服からの体のはみ出しを、アプリのカメラ（まわり全体）から見える物だけ数える（重い）",
            },
            "app_motions": {
                "type": "array",
                "description": "アプリの動きの名前（motions.json から読んだ物）。無ければ、まとめて1つの灰の出来事",
                "items": {"type": "string", "maxLength": thresholds.MAX_BONE_NAME_LENGTH},
                "maxItems": thresholds.MAX_BONE_NAMES,
                "uniqueItems": True,
            },
        },
        "additionalProperties": False,
    },
    checks_before=("モデル（アーマチュアと、それに付いたメッシュ）がシーンにあること",),
    measures_after=("ポーズごとの、ひざの逆曲がり（度）・足の床への沈み（mm）・服からの体のはみ出し（mm）",),
)


def _event_data(event: Event) -> dict[str, Any]:
    return {"pose": event.pose, "check": event.check, "measurement": event.measurement.to_data()}


def run_course(params: dict[str, Any]) -> dict[str, Any]:
    """motion.run_course の中身。ポーズは POSES の順、確かめは CHECKS の順で行う（渡した順ではない）"""
    from .motion import course_bpy

    wanted = set(params.get("poses") or POSE_NAMES)
    poses = [pose for pose in POSES if pose.name in wanted]
    checks = [check for check in CHECKS if check in set(params.get("checks") or CHECKS)]
    events = course_bpy.run_course(poses=poses, checks=checks, visibility=bool(params.get("visibility", False)),
                                   app_motions=params.get("app_motions"))
    return {"events": [_event_data(event) for event in events],
            "summary": summarize_course(events).to_data(),
            "lines": report_lines(events)}


# MARK: - 目録を作る


def build_catalog() -> Catalog:
    """この拡張機能の道具をすべて載せた目録"""
    catalog = Catalog()
    catalog.register(MAP_BONE_NAMES, map_bone_names)
    catalog.register(CONVERT_AND_CHECK, convert_and_check)
    catalog.register(SAVE_VERSION, save_version)
    catalog.register(COMPARE_WITH_PREVIOUS, compare_with_previous)
    catalog.register(LIST_VERSIONS, list_versions)
    catalog.register(RUN_COURSE, run_course)
    return catalog
