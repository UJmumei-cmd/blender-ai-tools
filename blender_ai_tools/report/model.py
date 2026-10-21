"""報告の形（Item・Report）と、項目を作る・数える・一言を作る・前の版と比べる。

- Item：1件の結果。安定した id（同じモデル・同じ点検なら、版が変わっても同じ）、まとまり、題（日本語と英語）、
  測った結果（Measurement.to_data() の形）、ユーザーの答え。
- Report：1回の点検の全部の項目。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .. import provenance, texts
from ..jobs.spec import check_json_value, is_time, strict_json_dumps, strict_json_loads
from ..measure.results import Measurement, Signal

SCHEMA_VERSION = 1
DECISIONS = ("fix", "accept", "later")
LANGUAGES = texts.LANGUAGES

_ITEM_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:+/-]{0,255}")
_GROUP = re.compile(r"[a-z][a-z0-9_]*")
_KIND = re.compile(r"[a-z][a-z0-9_]*")
_SIGNAL_KEYS = {Signal.GREEN: "report.signal_green", Signal.YELLOW: "report.signal_yellow",
                Signal.GRAY: "report.signal_gray", Signal.RED: "report.signal_red"}


# MARK: - 項目


@dataclass(frozen=True)
class Item:
    """報告の1件。

    item_id: 安定した id（例 "pose_course/arms_up/cloth_penetration"、"roundtrip/roundtrip_deviation"）
    group: まとまり（道具や段階の名前。例 "pose_course"）
    titles: 言語 → 題（"ja" と "en"）
    measurement: Measurement.to_data() の形
    decision: ユーザーの答え（"fix"・"accept"・"later"）。まだなら None
    decision_note・decided_at: 答えのメモと時刻
    """

    item_id: str
    group: str
    titles: dict[str, str]
    measurement: dict[str, Any]
    decision: str | None = None
    decision_note: str = ""
    decided_at: str | None = None

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not isinstance(self.item_id, str) or not _ITEM_ID.fullmatch(self.item_id):
            problems.append(f"item_id の形が違います（いまは {self.item_id!r}）")
        if not isinstance(self.group, str) or not _GROUP.fullmatch(self.group):
            problems.append(f"group は小文字の英数字と _ にしてください（いまは {self.group!r}）")
        if not isinstance(self.titles, dict) or set(self.titles) != set(LANGUAGES) or not all(
                isinstance(value, str) and value for value in self.titles.values()):
            problems.append(f"titles は {LANGUAGES} の両方の、空でない文にしてください")
        try:
            Measurement.from_data(self.measurement)
        except ValueError as error:
            problems.append(f"measurement：{error}")
        if self.decision is not None and self.decision not in DECISIONS:
            problems.append(f"decision は {DECISIONS} のどれかか None にしてください（いまは {self.decision!r}）")
        if not isinstance(self.decision_note, str):
            problems.append("decision_note は文字にしてください")
        if self.decided_at is not None and not is_time(self.decided_at):
            problems.append(f"decided_at は時刻の形か None にしてください（いまは {self.decided_at!r}）")
        if problems:
            raise ValueError("報告の項目の形が違います：" + "；".join(problems))
        object.__setattr__(self, "titles", dict(self.titles))

    @property
    def signal(self) -> Signal:
        return Measurement.from_data(self.measurement).signal

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"

    @property
    def counted_signal(self) -> Signal | None:
        """数える信号。「このままでよい」とした物は数えない（None）"""
        return None if self.accepted else self.signal

    def to_data(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "group": self.group, "titles": self.titles, "measurement": self.measurement,
                "decision": self.decision, "decision_note": self.decision_note, "decided_at": self.decided_at}

    @classmethod
    def from_data(cls, data: Any) -> Item:
        if not isinstance(data, dict):
            raise ValueError(f"報告の項目は辞書にしてください（いまは {type(data).__name__}）")
        unknown = set(data) - {"item_id", "group", "titles", "measurement", "decision", "decision_note", "decided_at"}
        if unknown:
            raise ValueError(f"報告の項目に知らない名前があります: {sorted(unknown)}")
        missing = {"item_id", "group", "titles", "measurement"} - set(data)
        if missing:
            raise ValueError(f"報告の項目に足りない名前があります: {sorted(missing)}")
        return cls(item_id=data["item_id"], group=data["group"], titles=data["titles"], measurement=data["measurement"],
                   decision=data.get("decision"), decision_note=data.get("decision_note", ""),
                   decided_at=data.get("decided_at"))


def _titles_for_measurement(name: str) -> dict[str, str]:
    key = f"report.item_{name}"
    if key in texts.TEXTS:
        return {language: texts.text(key, language=language) for language in LANGUAGES}
    return {language: name for language in LANGUAGES}


def item_from_measurement(group: str, measurement: Measurement, *, item_id: str | None = None,
                          titles: Mapping[str, str] | None = None, include_details: bool = True) -> Item:
    """測った結果1つから項目を作る。id は <group>/<結果の名前>、題は texts の report.item_<名前>（無ければ名前のまま）"""
    if not isinstance(measurement, Measurement):
        raise ValueError(f"measurement は Measurement にしてください（いまは {type(measurement).__name__}）")
    return Item(item_id=item_id or f"{group}/{measurement.name}", group=group,
                titles=dict(titles) if titles is not None else _titles_for_measurement(measurement.name),
                measurement=measurement.to_data(include_details=include_details))


def items_from_measurements(group: str, measurements: Iterable[Measurement], *, include_details: bool = True) -> list[Item]:
    items = [item_from_measurement(group, m, include_details=include_details) for m in measurements]
    ids = [item.item_id for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"同じ id の項目があります: {sorted(i for i in ids if ids.count(i) > 1)}")
    return items


def items_from_course_events(events: Iterable[Any], *, include_details: bool = False) -> list[Item]:
    """確かめの動き（A5）の出来事から項目を作る。id は pose_course/<ポーズ>/<確かめ>。
    events は motion.course.Event（pose・check・measurement を持つ物）"""
    from ..motion import course  # 遅く読む（report は motion に頼らずに使えるようにしておく）

    items = []
    for event in events:
        titles = {language: texts.text("report.course_item", language=language,
                                       pose=course.pose_label(event.pose, language=language),
                                       check=course.check_label(event.check, language=language))
                  for language in LANGUAGES}
        items.append(Item(item_id=f"pose_course/{event.pose}/{event.check}", group="pose_course", titles=titles,
                          measurement=event.measurement.to_data(include_details=include_details)))
    return items


# MARK: - 報告


@dataclass(frozen=True)
class Report:
    """1回の点検の報告。

    kind: 点検の種類（例 "pose_course"・"export"・"card"）。前の版との比べは、同じ kind どうしで行う
    source: 出所（道具の id・仕事の id・版の名前・入力の指紋など。JSON の値だけ）
    """

    report_id: str
    kind: str
    created_at: str
    items: tuple[Item, ...]
    source: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not isinstance(self.report_id, str) or not _ITEM_ID.fullmatch(self.report_id):
            problems.append(f"report_id の形が違います（いまは {self.report_id!r}）")
        if not isinstance(self.kind, str) or not _KIND.fullmatch(self.kind):
            problems.append(f"kind は小文字の英数字と _ にしてください（いまは {self.kind!r}）")
        if not is_time(self.created_at):
            problems.append(f"created_at は時刻の形にしてください（いまは {self.created_at!r}）")
        if isinstance(self.items, (list, tuple)) and all(isinstance(item, Item) for item in self.items):
            object.__setattr__(self, "items", tuple(self.items))
            ids = [item.item_id for item in self.items]
            if len(set(ids)) != len(ids):
                problems.append("同じ id の項目があります")
        else:
            problems.append("items は Item の並びにしてください")
        if isinstance(self.source, dict):
            check_json_value(self.source, "source", problems)
        else:
            problems.append("source は辞書にしてください")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            problems.append(f"schema_version は {SCHEMA_VERSION} だけを扱えます")
        if problems:
            raise ValueError("報告の形が違います：" + "；".join(problems))

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "report_id": self.report_id, "kind": self.kind,
                "created_at": self.created_at, "source": self.source, "items": [item.to_data() for item in self.items]}

    def to_json(self) -> str:
        return strict_json_dumps(self.to_data())

    @classmethod
    def from_data(cls, data: Any) -> Report:
        if not isinstance(data, dict):
            raise ValueError(f"報告は辞書にしてください（いまは {type(data).__name__}）")
        unknown = set(data) - {"schema_version", "report_id", "kind", "created_at", "source", "items"}
        if unknown:
            raise ValueError(f"報告に知らない名前があります: {sorted(unknown)}")
        if not isinstance(data.get("items"), list):
            raise ValueError("報告の items は並びにしてください")
        return cls(report_id=data.get("report_id"), kind=data.get("kind"), created_at=data.get("created_at"),
                   items=tuple(Item.from_data(item) for item in data["items"]), source=data.get("source", {}),
                   schema_version=data.get("schema_version", SCHEMA_VERSION))

    @classmethod
    def from_json(cls, text: str) -> Report:
        return cls.from_data(strict_json_loads(text, "報告の JSON"))

    def with_decisions(self, decisions: Mapping[str, Mapping[str, Any]]) -> Report:
        """記録してある答え（item_id → {decision, note, decided_at}）を項目に写す"""
        items = []
        for item in self.items:
            entry = decisions.get(item.item_id)
            if entry is None:
                items.append(item)
            else:
                items.append(replace(item, decision=entry.get("decision"), decision_note=entry.get("note", ""),
                                     decided_at=entry.get("decided_at")))
        return replace(self, items=tuple(items))

    def item(self, item_id: str) -> Item:
        for item in self.items:
            if item.item_id == item_id:
                return item
        raise KeyError(item_id)


def new_report(kind: str, items: Iterable[Item], *, source: Mapping[str, Any] | None = None,
               report_id: str | None = None) -> Report:
    created = provenance.now_iso()
    return Report(report_id=report_id or re.sub(r"[^0-9TZ]", "", created) + "_" + kind, kind=kind, created_at=created,
                  items=tuple(items), source=dict(source or {}))


# MARK: - 数える・一言


def count_signals(items: Iterable[Item]) -> dict[str, int]:
    """信号ごとの数（"red"・"yellow"・"gray"・"green"）と、「このままでよい」とした数（"accepted"）"""
    counts = {"red": 0, "yellow": 0, "gray": 0, "green": 0, "accepted": 0}
    for item in items:
        signal = item.counted_signal
        if signal is None:
            counts["accepted"] += 1
        else:
            counts[signal.name.lower()] += 1
    return counts


def sort_items(items: Iterable[Item]) -> list[Item]:
    """重い順（赤 → 灰 → 黄 → 緑。「このままでよい」は最後）。同じなら出た順"""
    indexed = list(enumerate(items))
    indexed.sort(key=lambda pair: (0 if pair[1].counted_signal is not None else 1,
                                   -(pair[1].counted_signal or Signal.GREEN), pair[0]))
    return [item for _, item in indexed]


def headline(items: Iterable[Item], *, language: str | None = None) -> str:
    """いちばん上の一言。例「大丈夫です（黄 3件）」。数字はプログラムが入れる"""
    values = list(items)
    if not values:
        return texts.text("report.headline_empty", language=language)
    counts = count_signals(values)
    numbers = {key: str(counts[key]) for key in ("red", "yellow", "gray")}
    if counts["red"]:
        line = texts.text("report.headline_red", language=language, **numbers)
    elif counts["gray"]:
        line = texts.text("report.headline_gray", language=language, gray=numbers["gray"], yellow=numbers["yellow"])
    elif counts["yellow"]:
        line = texts.text("report.headline_yellow", language=language, yellow=numbers["yellow"])
    else:
        line = texts.text("report.headline_ok", language=language)
    if counts["accepted"]:
        line += texts.text("motion.list_separator", language=language) if "motion.list_separator" in texts.TEXTS else "、"
        line += texts.text("report.headline_accepted", language=language, accepted=str(counts["accepted"]))
    return line


def signal_label(signal: Signal, *, language: str | None = None) -> str:
    return texts.text(_SIGNAL_KEYS[signal], language=language)


def decision_label(decision: str | None, *, language: str | None = None) -> str:
    if decision is not None and decision not in DECISIONS:
        raise ValueError(f"知らない答えです：{decision!r}")
    return texts.text(f"report.decision_{decision or 'none'}", language=language)


def value_text(measurement: Mapping[str, Any], *, language: str | None = None) -> str:
    """値の文。長さは mm（小数 2 桁）、角度は度（小数 1 桁）、数は整数。値が無ければ「値なし」"""
    value, unit = measurement.get("value"), measurement.get("unit", "")
    if value is None:
        return texts.text("report.value_none", language=language)
    if unit == "m":
        return texts.text("report.value_mm", language=language, value=f"{value * 1000.0:.2f}")
    if unit == "deg":
        return texts.text("report.value_deg", language=language, value=f"{value:.1f}")
    if unit == "count" and float(value).is_integer():
        return texts.text("report.value_count", language=language, value=str(int(value)))
    return texts.text("report.value_other", language=language, value=f"{value:.3g}", unit=unit)


# MARK: - 前の版との比べ


def compare_red_count(items: Iterable[Item], previous_red: int | None) -> Measurement:
    """赤の数を前の版の記録と比べる（ロードマップ：数は記録し、次の版で増えたら赤）。

    記録が無いとき（最初の版）は灰（比べられない）。増えていれば赤、同じか減っていれば緑。
    「このままでよい」とした物は数えない。
    """
    current = count_signals(items)["red"]
    if previous_red is None:
        return Measurement.unmeasured("red_count_change", "count",
                                      texts.text("report.baseline_first", language="ja", current=str(current)),
                                      {"current": current, "previous": None})
    if isinstance(previous_red, bool) or not isinstance(previous_red, int) or previous_red < 0:
        raise ValueError(f"前の版の赤の数は 0 以上の整数にしてください（いまは {previous_red!r}）")
    key = "report.baseline_increase" if current > previous_red else "report.baseline_ok"
    return Measurement("red_count_change", float(current), "count", Signal.RED if current > previous_red else Signal.GREEN,
                       texts.text(key, language="ja", current=str(current), previous=str(previous_red)),
                       {"current": current, "previous": previous_red})
