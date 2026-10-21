"""ユーザーの答え（直す・このままでよい・あとで）と、赤の数の基準の記録（decisions.json）。

モデルの作業フォルダに1つ置く。項目の id は版が変わっても同じなので、「このままでよい」とした物は次の版でも
そのまま（数には入れない）。赤の数の基準は、点検の種類（kind）ごとに記録する。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .. import provenance
from ..jobs.spec import SCHEMA_VERSION, check_json_value, is_time, strict_json_dumps, strict_json_loads, write_text_atomic
from .model import DECISIONS, Item, _ITEM_ID, _KIND, count_signals

DECISIONS_FILE = "decisions.json"


class DecisionStore:
    """decisions.json の読み書き。読み込みは厳しく、書き込みは一時ファイルから置き換える"""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.decisions: dict[str, dict[str, Any]] = {}
        self.baselines: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            self._load()

    # MARK: - 読み書き

    def _load(self) -> None:
        data = strict_json_loads(self.path.read_text(encoding="utf-8"), "答えの記録の JSON")
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"答えの記録の形が違います：{self.path}")
        unknown = set(data) - {"schema_version", "decisions", "baselines"}
        if unknown:
            raise ValueError(f"答えの記録に知らない名前があります: {sorted(unknown)}")
        decisions, baselines = data.get("decisions", {}), data.get("baselines", {})
        if not isinstance(decisions, dict) or not isinstance(baselines, dict):
            raise ValueError("答えの記録の decisions と baselines は辞書にしてください")
        for item_id, entry in decisions.items():
            self._check_decision(item_id, entry)
        for kind, entry in baselines.items():
            self._check_baseline(kind, entry)
        self.decisions, self.baselines = decisions, baselines

    @staticmethod
    def _check_decision(item_id: Any, entry: Any) -> None:
        if not isinstance(item_id, str) or not _ITEM_ID.fullmatch(item_id):
            raise ValueError(f"答えの記録の項目の id の形が違います（{item_id!r}）")
        if not isinstance(entry, dict) or set(entry) - {"decision", "note", "decided_at", "version"}:
            raise ValueError(f"答えの記録の形が違います（{item_id}）")
        if entry.get("decision") not in DECISIONS:
            raise ValueError(f"答えは {DECISIONS} のどれかにしてください（{item_id}：{entry.get('decision')!r}）")
        if not isinstance(entry.get("note", ""), str) or not is_time(entry.get("decided_at")):
            raise ValueError(f"答えの記録のメモか時刻の形が違います（{item_id}）")
        if entry.get("version") is not None and not isinstance(entry["version"], str):
            raise ValueError(f"答えの記録の版は文字か None にしてください（{item_id}）")

    @staticmethod
    def _check_baseline(kind: Any, entry: Any) -> None:
        if not isinstance(kind, str) or not _KIND.fullmatch(kind):
            raise ValueError(f"基準の記録の種類の名前の形が違います（{kind!r}）")
        if not isinstance(entry, dict) or set(entry) - {"red", "updated_at", "version"}:
            raise ValueError(f"基準の記録の形が違います（{kind}）")
        red = entry.get("red")
        if isinstance(red, bool) or not isinstance(red, int) or red < 0 or not is_time(entry.get("updated_at")):
            raise ValueError(f"基準の記録の赤の数か時刻の形が違います（{kind}）")

    def save(self) -> Path:
        problems: list[str] = []
        check_json_value({"decisions": self.decisions, "baselines": self.baselines}, "記録", problems)
        if problems:
            raise ValueError("答えの記録に JSON にできない値があります：" + "；".join(problems))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.path, strict_json_dumps({"schema_version": SCHEMA_VERSION, "decisions": self.decisions,
                                                        "baselines": self.baselines}))
        return self.path

    # MARK: - 答え

    def decide(self, item_id: str, decision: str, *, note: str = "", version: str | None = None) -> dict[str, Any]:
        """答えを記録する（上書き）。decision は fix・accept・later"""
        entry = {"decision": decision, "note": note, "decided_at": provenance.now_iso(), "version": version}
        self._check_decision(item_id, entry)
        self.decisions[item_id] = entry
        return entry

    def clear(self, item_id: str) -> None:
        self.decisions.pop(item_id, None)

    def decision_of(self, item_id: str) -> dict[str, Any] | None:
        return self.decisions.get(item_id)

    # MARK: - 赤の数の基準

    def baseline_red(self, kind: str) -> int | None:
        entry = self.baselines.get(kind)
        return None if entry is None else int(entry["red"])

    def record_baseline(self, kind: str, items: list[Item] | tuple[Item, ...], *, version: str | None = None) -> int:
        """今回の赤の数（「このままでよい」は除く）を、次の版の基準として記録する"""
        red = count_signals(items)["red"]
        entry = {"red": red, "updated_at": provenance.now_iso(), "version": version}
        self._check_baseline(kind, entry)
        self.baselines[kind] = entry
        return red
