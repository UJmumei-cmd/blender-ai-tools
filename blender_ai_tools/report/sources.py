"""目録の道具の結果（JSON の形）から、報告（Report）を作る（bpy なし）。

報告を作る道具と、点検の種類（Report.kind）：
- export.convert_and_check → "export"：書き出し（E1）の1件 "export/convert" と、往復の確認（E2）の測った結果ごとに1件。
  読み戻した形のずれ（roundtrip_deviation）はポーズごとに出るので、id は "export/roundtrip_deviation/<ポーズ>"。
- motion.run_course → "pose_course"：確かめの出来事ごとに1件（report.model.items_from_course_events）。
- versions.compare_with_previous → "versions"：変わった所ごとに1件（変えてよい所の外なら赤、中なら緑）。
  変わった所が無いときは、まとめの結果（灰のこともある）を1件。

項目の id は版が変わっても同じになるように作る（decisions.json の答えを次の版でも使うため）。
id には ASCII の字しか使えないので、オブジェクトの名前（日本語のこともある）は sha1 の頭 12 字にする。
数字はプログラムが決まった文の形（texts の report.*）に入れる（CLAUDE.md 2章3）。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from .. import texts
from ..export.roundtrip import DEVIATION_NAME, REST_POSE, TEST_POSES
from ..measure.results import Measurement, Signal
from .model import LANGUAGES, Item, Report, item_from_measurement, items_from_course_events, new_report

# 報告を作る道具の id → 点検の種類（Report.kind）
REPORT_KINDS: Mapping[str, str] = {
    "export.convert_and_check": "export",
    "motion.run_course": "pose_course",
    "versions.compare_with_previous": "versions",
}

# 往復の確認のポーズの名前（休みの姿勢と、確かめのポーズ）。文は report.export_pose_<名前>
EXPORT_POSES = (REST_POSE, *TEST_POSES)

# 書き出し（E1）の項目の、測った結果の名前（id は export/convert、題は report.item_convert）
CONVERT_NAME = "convert"

# 名前を id にそのまま入れてよい形（ASCII の英数字と _ -）
_SAFE_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_SIGNALS = {signal.name.lower(): signal for signal in Signal}


def makes_report(tool_id: str) -> bool:
    """その道具の結果から報告を作れるか"""
    return tool_id in REPORT_KINDS


def kind_label(kind: str, *, language: str | None = None) -> str:
    """点検の種類の名前（知らない種類は、そのままの名前）"""
    key = f"report.kind_label_{kind}"
    return texts.text(key, language=language) if key in texts.TEXTS else kind


def short_hash(value: str) -> str:
    """名前（日本語のこともある）を、id に入れられる ASCII の字にする（sha1 の頭 12 字）"""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _safe_part(value: Any) -> str:
    """id の一部：ASCII の名前ならそのまま、ほかは short_hash"""
    text = str(value)
    return text if _SAFE_NAME.fullmatch(text) else short_hash(text)


def _both(key: str, **values: Any) -> dict[str, str]:
    return {language: texts.text(key, language=language, **values) for language in LANGUAGES}


def _require(data: Any, name: str, kind: type, what: str) -> Any:
    if not isinstance(data, Mapping) or name not in data or not isinstance(data[name], kind):
        raise ValueError(f"{what} の結果の形が違います（{name} がありません）")
    return data[name]


# MARK: - E1・E2 書き出して確かめる


def _convert_item(convert: Mapping[str, Any]) -> Item:
    ok = convert.get("ok") is True
    signal = _SIGNALS.get(convert.get("signal"))
    if signal is None:
        raise ValueError(f"書き出しの結果の信号の形が違います（いまは {convert.get('signal')!r}）")
    if not ok and signal == Signal.GREEN:
        signal = Signal.RED  # 書き出せなかったのに緑にはしない
    problems = [str(line) for line in convert.get("problems") or []]
    warnings = [str(line) for line in convert.get("warnings") or []]
    if problems or warnings:
        reason = "；".join(problems + warnings)
    elif ok:
        reason = texts.text("report.convert_ok", language="ja")
    else:
        reason = texts.text("report.convert_no_detail", language="ja")
    details = {"problems": problems, "warnings": warnings, "output_dir": convert.get("output_dir")}
    measurement = Measurement(CONVERT_NAME, float(len(problems)), "count", signal, reason, details)
    return item_from_measurement("export", measurement, item_id="export/convert")


def _pose_label(pose: str, language: str) -> str:
    key = f"report.export_pose_{pose}"
    return texts.text(key, language=language) if key in texts.TEXTS else pose


def _roundtrip_item(measurement: Measurement) -> Item:
    if measurement.name != DEVIATION_NAME:
        return item_from_measurement("export", measurement, item_id=f"export/{_safe_part(measurement.name)}")
    pose = measurement.details.get("pose") if isinstance(measurement.details, Mapping) else None
    if not isinstance(pose, str) or not pose:
        # 読み戻せなかったときの灰の1件（ポーズの名前が無い）
        return item_from_measurement("export", measurement, item_id=f"export/{DEVIATION_NAME}")
    titles = {language: texts.text("report.roundtrip_pose_item", language=language, pose=_pose_label(pose, language))
              for language in LANGUAGES}
    return item_from_measurement("export", measurement, item_id=f"export/{DEVIATION_NAME}/{_safe_part(pose)}",
                                 titles=titles)


def _export_items(result: Mapping[str, Any]) -> list[Item]:
    convert = _require(result, "convert", Mapping, "書き出して確かめる")
    roundtrip = _require(result, "roundtrip", list, "書き出して確かめる")
    items = [_convert_item(convert)]
    items.extend(_roundtrip_item(Measurement.from_data(data)) for data in roundtrip)
    return items


# MARK: - A5 確かめの動き


def _course_items(result: Mapping[str, Any]) -> list[Item]:
    from ..motion.course import Event  # 遅く読む（report は motion に頼らずに使えるようにしておく）

    events = []
    for data in _require(result, "events", list, "確かめの動き"):
        if not isinstance(data, Mapping) or not {"pose", "check", "measurement"} <= set(data):
            raise ValueError("確かめの動きの出来事の形が違います（pose・check・measurement が要ります）")
        events.append(Event(pose=data["pose"], check=data["check"], measurement=Measurement.from_data(data["measurement"])))
    return items_from_course_events(events)


# MARK: - A6 前の版との比べ


def _change_titles(change: Mapping[str, Any]) -> dict[str, str]:
    titles = change.get("texts")
    if isinstance(titles, Mapping) and all(isinstance(titles.get(language), str) and titles.get(language)
                                           for language in LANGUAGES):
        return {language: titles[language] for language in LANGUAGES}
    raise ValueError("前の版との比べの結果の形が違います（変わった所に、日本語と英語の文 texts がありません。"
                     "道具 versions.compare_with_previous の版 2 の結果を渡してください）")


def _change_item(change: Any) -> Item:
    if not isinstance(change, Mapping):
        raise ValueError("前の版との比べの、変わった所の形が違います")
    category, name, key = change.get("category"), change.get("object"), change.get("key")
    if not isinstance(category, str) or not isinstance(name, str) or not isinstance(key, str) \
            or not key.startswith("versions."):
        raise ValueError("前の版との比べの、変わった所の形が違います（category・object・key が要ります）")
    count = change.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"前の版との比べの、変わった数の形が違います（いまは {count!r}）")
    outside = change.get("outside_allowed") is True
    suffix = key[len("versions."):]
    measurement = Measurement(suffix, float(count), "count", Signal.RED if outside else Signal.GREEN,
                              texts.text("report.versions_outside" if outside else "report.versions_allowed",
                                         language="ja"),
                              {"object": name, "category": category, "key": key})
    item_id = f"versions/{_safe_part(category)}/{short_hash(name)}/{_safe_part(suffix)}"
    return item_from_measurement("versions", measurement, item_id=item_id, titles=_change_titles(change))


def _versions_items(result: Mapping[str, Any]) -> list[Item]:
    changes = _require(result, "changes", list, "前の版との比べ")
    if not changes:
        summary = Measurement.from_data(_require(result, "measurement", Mapping, "前の版との比べ"))
        return [item_from_measurement("versions", summary)]
    items: list[Item] = []
    seen: dict[str, int] = {}
    for change in changes:
        item = _change_item(change)
        # 同じオブジェクトの同じ種類の変化が2つ以上あるとき（シェイプキーごとの形の変化など）は、出た順の番号を付ける
        seen[item.item_id] = seen.get(item.item_id, 0) + 1
        if seen[item.item_id] > 1:
            item = Item(item_id=f"{item.item_id}/{seen[item.item_id]}", group=item.group, titles=item.titles,
                        measurement=item.measurement)
        items.append(item)
    return items


# MARK: - 報告を作る

_BUILDERS = {"export": _export_items, "pose_course": _course_items, "versions": _versions_items}


def report_from_tool_result(tool_id: str, result: Any, *, source: Mapping[str, Any] | None = None) -> Report:
    """道具 tool_id の結果（JSON の形。裏の係の result.json の "result" と同じ）から報告を作る。

    source：出所（仕事の id・入力の版の名前など。JSON の値だけ）。道具の id は、ここで入れる。
    報告を作らない道具・結果の形が違うときは ValueError。
    """
    kind = REPORT_KINDS.get(tool_id)
    if kind is None:
        raise ValueError(f"この道具の結果からは報告を作れません：{tool_id!r}")
    if not isinstance(result, Mapping):
        raise ValueError(f"道具の結果は辞書にしてください（いまは {type(result).__name__}）")
    items = _BUILDERS[kind](result)
    return new_report(kind, items, source={**dict(source or {}), "tool_id": tool_id})
