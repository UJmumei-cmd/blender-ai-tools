"""測った結果の共通の形：信号（緑・黄・赤・灰）と、数字・単位・理由。

測れなかった物は緑にしない（CLAUDE.md 2章5）。値が無い・NaN のときは、必ず灰にする。
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


class Signal(enum.IntEnum):
    """信号。まとめるときは数の大きいほうを取る（赤 > 灰 > 黄 > 緑）。
    灰（測れなかった）は、赤を隠しているかもしれないので、黄より重く扱う"""

    GREEN = 0
    YELLOW = 1
    GRAY = 2
    RED = 3

    @property
    def label(self) -> str:
        """日本語の名前（緑・黄・灰・赤）"""
        return _LABELS[self]


_LABELS = {Signal.GREEN: "緑", Signal.YELLOW: "黄", Signal.GRAY: "灰", Signal.RED: "赤"}


def worst(signals: Iterable[Signal]) -> Signal:
    """いちばん重い信号。1つも無いときは灰（何も測っていない）"""
    result: Signal | None = None
    for signal in signals:
        result = signal if result is None else max(result, signal)
    return Signal.GRAY if result is None else result


def is_missing(value: float | None) -> bool:
    """値が無い（None・NaN）か"""
    return value is None or (isinstance(value, float) and math.isnan(value))


def classify(value: float | None, *, red: float, yellow: float | None = None, higher_is_worse: bool = True) -> Signal:
    """値を信号にする。値が無い（None・NaN）ときは灰。

    higher_is_worse=True のとき：value >= red で赤、value >= yellow で黄、それ以外は緑。
    higher_is_worse=False のとき：value <= red で赤、value <= yellow で黄、それ以外は緑。
    yellow が None のときは、黄を使わない。
    """
    if is_missing(value):
        return Signal.GRAY
    value = float(value)
    if higher_is_worse:
        if value >= red:
            return Signal.RED
        if yellow is not None and value >= yellow:
            return Signal.YELLOW
    else:
        if value <= red:
            return Signal.RED
        if yellow is not None and value <= yellow:
            return Signal.YELLOW
    return Signal.GREEN


@dataclass(frozen=True)
class Measurement:
    """1つの測った結果。

    name: 測った物の名前（英語の識別子。例 "penetration_depth"）
    value: 代表の値（いちばん悪い値など）。測れなかったときは None
    unit: 単位（"m"・"m2"・"m3"・"deg"・"ratio"・"count"・"px"・"s" など）
    signal: 信号
    reason: 理由（灰のときは必ず書く。日本語）
    details: 要素ごとの値など（numpy の配列を入れてよい）
    """

    name: str
    value: float | None
    unit: str
    signal: Signal
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if is_missing(self.value) and self.signal != Signal.GRAY:
            raise ValueError(f"{self.name}: 値が無いのに信号が {self.signal.label} になっています（測れなかった物は灰にする）")
        if self.signal == Signal.GRAY and not self.reason:
            raise ValueError(f"{self.name}: 灰のときは理由を書いてください")

    @classmethod
    def unmeasured(cls, name: str, unit: str, reason: str, details: dict[str, Any] | None = None) -> Measurement:
        """測れなかった結果（灰）"""
        return cls(name=name, value=None, unit=unit, signal=Signal.GRAY, reason=reason, details=details or {})

    @classmethod
    def from_value(cls, name: str, value: float | None, unit: str, *, red: float, yellow: float | None = None,
                   higher_is_worse: bool = True, reason: str = "",
                   details: dict[str, Any] | None = None) -> Measurement:
        """値と基準から信号を決めて作る。値が無いときは灰にし、理由が無ければ「値がありません」とする"""
        signal = classify(value, red=red, yellow=yellow, higher_is_worse=higher_is_worse)
        if signal == Signal.GRAY:
            return cls.unmeasured(name, unit, reason or "値がありません", details)
        return cls(name=name, value=float(value), unit=unit, signal=signal, reason=reason, details=details or {})
