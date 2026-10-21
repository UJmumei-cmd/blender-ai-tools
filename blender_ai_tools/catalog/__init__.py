"""A8 道具の目録。AI もボタンも、ここに載った道具だけを使う。

- 道具は、説明（ToolSpec）と関数の組で Catalog に載せる。載っていない道具は動かない
  （断るかどうかは、目録にその道具が無いことで決まる。ロードマップ段階2）。
- LLM（27B・Jeff）の返事は JSON の文字列で受け取り、parse_tool_call できびしく読む。
  返事をコードとして動かすことはしない（CLAUDE.md 2章1）。値は、道具の入力のスキーマ（schema.py）で確かめてから渡す。
- 確認：
  ・見るだけ（READ）の道具は、確認なしで動く。
  ・変更（MODIFY）は、計画全体の承認か、その場の確認が要る。
  ・消す（DELETE）は、計画を承認したあとでも、毎回あらためて確認する（CLAUDE.md 2章）。
"""

from __future__ import annotations

import copy
import enum
import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..thresholds import MAX_TOOL_CALL_LENGTH
from .schema import SUPPORTED_KEYWORDS, check_schema, describe_value, quote_text, validate

__all__ = [
    "MAX_TOOL_CALL_LENGTH",
    "SUPPORTED_KEYWORDS",
    "Catalog",
    "CatalogError",
    "ConfirmationRequiredError",
    "DuplicateToolError",
    "Effect",
    "InvalidParamsError",
    "ToolCall",
    "ToolCallError",
    "ToolSpec",
    "UnknownToolError",
    "check_schema",
    "parse_tool_call",
    "validate",
]

# MARK: - 決まり

# LLM の返事（道具の呼び出しの JSON）の長さの上限（文字）は thresholds.MAX_TOOL_CALL_LENGTH（仮）。上で読み込む

# 道具の id：小文字の名前を . でつなぐ（例 "bones.map_names"）
_TOOL_ID = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")
# 道具の版：数を . でつなぐ（例 "1"・"1.2"）。0 で始まる数（"01"）は使わない
_VERSION = re.compile(r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))*")
# ロードマップの機能の番号（例 "A3"・"A15"）
_FEATURE = re.compile(r"[A-Z][0-9]+")
# 文の中に id をそのまま書いてよい形（見える ASCII の文字だけ）
_PRINTABLE_ID = re.compile(r"[!-~]{1,80}")
# エラーの文に入れる、合わない所の数
_ERRORS_IN_MESSAGE = 10


def _show_id(tool_id: Any) -> str:
    """道具の id を文の中に見せる形にする（LLM から来た変な文字列でも、短く安全に見せる）"""
    if type(tool_id) is str and _PRINTABLE_ID.fullmatch(tool_id):
        return tool_id
    return describe_value(tool_id)


def _join(problems: list[str]) -> str:
    shown = "；".join(problems[:_ERRORS_IN_MESSAGE])
    if len(problems) > _ERRORS_IN_MESSAGE:
        shown += f"（ほか {len(problems) - _ERRORS_IN_MESSAGE} 件）"
    return shown


def _require_bool(name: str, value: Any) -> None:
    """確認の印は True か False だけ（"false" のような文字列が「はい」と読まれないようにする）"""
    if type(value) is not bool:
        raise TypeError(f"{name} は True か False にしてください（いまは {describe_value(value)}）")


# MARK: - 道具の働き


class Effect(enum.Enum):
    """道具がモデルに何をするか。確認が要るかどうかを決める"""

    READ = "read"
    MODIFY = "modify"
    DELETE = "delete"

    @property
    def label(self) -> str:
        """日本語の名前（見るだけ・変更・消す）"""
        return _EFFECT_LABELS[self]


_EFFECT_LABELS = {Effect.READ: "見るだけ", Effect.MODIFY: "変更", Effect.DELETE: "消す"}


def _needs_confirmation(effect: Effect, plan_approved: bool) -> bool:
    if effect is Effect.READ:
        return False
    if effect is Effect.MODIFY:
        return not plan_approved
    return True  # 消す操作は、計画を承認したあとでも毎回確認する


# MARK: - エラー


class CatalogError(Exception):
    """目録のエラーのもと"""


class UnknownToolError(CatalogError, KeyError):
    """目録にない道具を使おうとした"""

    def __init__(self, tool_id: Any) -> None:
        self.tool_id = tool_id
        super().__init__(f"目録にない道具です: {_show_id(tool_id)}")

    def __str__(self) -> str:
        # KeyError は文を引用符で囲むので、そのままの文を返す
        return str(self.args[0])


class DuplicateToolError(CatalogError, ValueError):
    """同じ id の道具を2回載せようとした"""


class InvalidParamsError(CatalogError, ValueError):
    """道具に渡す値が、入力のスキーマに合わない。errors に合わない所がすべて入る"""

    def __init__(self, tool_id: str, errors: list[str]) -> None:
        self.tool_id = tool_id
        self.errors = list(errors)
        super().__init__(f"道具 {tool_id} の値がスキーマに合いません：{_join(self.errors)}")


class ConfirmationRequiredError(CatalogError):
    """ユーザーの確認が要るのに、確認されていない"""

    def __init__(self, tool_id: str, effect: Effect) -> None:
        self.tool_id = tool_id
        self.effect = effect
        if effect is Effect.DELETE:
            reason = "消す道具は、計画を承認したあとでも、毎回確認が要ります"
        else:
            reason = "形を変える道具は、計画の承認か、その場の確認が要ります"
        super().__init__(f"道具 {tool_id}（{effect.label}）を動かす前に確認が要ります：{reason}")


class ToolCallError(CatalogError, ValueError):
    """LLM の返事（道具の呼び出しの JSON）が決まった形でない。errors に合わない所が入る"""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        self.errors = list(errors) if errors else [message]
        super().__init__(message)


# MARK: - 道具の説明


def _check_text(name: str, value: Any, problems: list[str], *, single_line: bool = False) -> None:
    if type(value) is not str or not value.strip():
        problems.append(f"{name} は空でない文字列にしてください（いまは {describe_value(value)}）")
        return
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        problems.append(f"{name} に使えない文字（対になっていないサロゲート）が入っています")
    if single_line and ("\n" in value or "\r" in value):
        problems.append(f"{name} は1行にしてください")


def _check_texts(name: str, values: Any, problems: list[str]) -> None:
    if type(values) is not tuple and type(values) is not list:
        problems.append(f"{name} は文字列の tuple にしてください（いまは {describe_value(values)}）")
        return
    seen: set[str] = set()
    for index, value in enumerate(values):
        _check_text(f"{name}[{index}]", value, problems)
        if type(value) is str:
            if value in seen:
                problems.append(f"{name}[{index}] が2回あります")
            seen.add(value)


@dataclass(frozen=True)
class ToolSpec:
    """目録に載せる1つの道具の説明。

    id: 道具の名前（例 "bones.map_names"。小文字の名前を . でつなぐ）
    version: 道具の版（例 "1"・"1.2"）。中身を変えたら上げる（A15 の記録に残る）
    feature: ロードマップの機能の番号（例 "A3"）
    category: 分類（ASCII の識別子。例 "bones"）。27B は分類を決めてから、その分類の道具から選ぶ
    title: 日本語の名前（1行）
    summary: 日本語の説明
    effect: 何をするか（見るだけ・変更・消す）
    heavy: 重い作業か（True なら、あとで裏の Blender で動かす。A10）
    input_schema: 入力の形（JSON スキーマ。check_schema を通る物）。写しを持つ
    checks_before: 実行前に確かめること（日本語）
    measures_after: 実行後に測ること（日本語）
    """

    id: str
    version: str
    feature: str
    category: str
    title: str
    summary: str
    effect: Effect
    heavy: bool
    input_schema: dict[str, Any] = field(hash=False)
    checks_before: tuple[str, ...] = ()
    measures_after: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        problems = _spec_problems(self)
        if problems:
            raise ValueError(f"道具の説明に誤りがあります（{_show_id(self.id)}）：{_join(problems)}")
        # 渡した dict や list をあとで書き換えても、説明が変わらないように写しを持つ
        object.__setattr__(self, "input_schema", copy.deepcopy(self.input_schema))
        object.__setattr__(self, "checks_before", tuple(self.checks_before))
        object.__setattr__(self, "measures_after", tuple(self.measures_after))


def _spec_problems(spec: ToolSpec) -> list[str]:
    """道具の説明の誤りを、すべて集める（作るときと、目録に載せるときの両方で使う）"""
    problems: list[str] = []
    if type(spec.id) is not str or not _TOOL_ID.fullmatch(spec.id):
        problems.append(f"id は \"bones.map_names\" のような形にしてください（いまは {describe_value(spec.id)}）")
    if type(spec.version) is not str or not _VERSION.fullmatch(spec.version):
        problems.append(f"version は \"1\" や \"1.2\" のような形にしてください（いまは {describe_value(spec.version)}）")
    if type(spec.feature) is not str or not _FEATURE.fullmatch(spec.feature):
        problems.append(f"feature はロードマップの番号（\"A3\" など）にしてください（いまは {describe_value(spec.feature)}）")
    if type(spec.category) is not str or not (spec.category.isascii() and spec.category.isidentifier()):
        problems.append(f"category は ASCII の識別子にしてください（いまは {describe_value(spec.category)}）")
    _check_text("title", spec.title, problems, single_line=True)
    _check_text("summary", spec.summary, problems)
    if type(spec.effect) is not Effect:
        problems.append(f"effect は Effect にしてください（いまは {describe_value(spec.effect)}）")
    if type(spec.heavy) is not bool:
        problems.append(f"heavy は True か False にしてください（いまは {describe_value(spec.heavy)}）")
    schema_problems = check_schema(spec.input_schema)
    if schema_problems:
        problems.append(f"input_schema が決まりに合いません：{_join(schema_problems)}")
    _check_texts("checks_before", spec.checks_before, problems)
    _check_texts("measures_after", spec.measures_after, problems)
    return problems


# MARK: - 目録


@dataclass(frozen=True)
class _Entry:
    spec: ToolSpec
    func: Callable[[dict[str, Any]], Any]
    # 載せたときのスキーマの写し。確かめと一覧には、これだけを使う（spec の dict があとで書き換えられても変わらない）
    schema: dict[str, Any]


class Catalog:
    """道具の目録。載せた道具だけを、確かめてから動かす"""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(self, spec: ToolSpec, func: Callable[[dict[str, Any]], Any]) -> None:
        """道具を載せる。func は値の dict を1つ受け取る関数。

        説明は、ここでもう一度すべて確かめる（作ったあとで書き換えた説明や、確かめを飛ばす子クラスを載せないため）。
        """
        if not isinstance(spec, ToolSpec):
            raise TypeError(f"spec は ToolSpec にしてください（いまは {describe_value(spec)}）")
        if not callable(func):
            raise TypeError(f"道具 {_show_id(spec.id)} の func が関数ではありません（いまは {describe_value(func)}）")
        problems = _spec_problems(spec)
        if problems:
            raise ValueError(f"道具の説明に誤りがあります（{_show_id(spec.id)}）：{_join(problems)}")
        if spec.id in self._entries:
            raise DuplicateToolError(f"同じ id の道具が、もう目録に載っています: {spec.id}")
        self._entries[spec.id] = _Entry(spec=spec, func=func, schema=copy.deepcopy(spec.input_schema))

    def _entry(self, tool_id: Any) -> _Entry:
        entry = self._entries.get(tool_id) if type(tool_id) is str else None
        if entry is None:
            raise UnknownToolError(tool_id)
        return entry

    def get(self, tool_id: str) -> ToolSpec:
        """道具の説明。目録に無いときは UnknownToolError"""
        return self._entry(tool_id).spec

    def __contains__(self, tool_id: object) -> bool:
        return type(tool_id) is str and tool_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def specs(self, category: str | None = None) -> tuple[ToolSpec, ...]:
        """道具の説明を id の順に並べる。category を渡すと、その分類だけ"""
        if category is not None and type(category) is not str:
            raise TypeError(f"category は文字列か None にしてください（いまは {describe_value(category)}）")
        return tuple(self._entries[tool_id].spec for tool_id in sorted(self._entries)
                     if category is None or self._entries[tool_id].spec.category == category)

    def categories(self) -> tuple[str, ...]:
        """分類の一覧（名前の順）"""
        return tuple(sorted({entry.spec.category for entry in self._entries.values()}))

    def validate_call(self, tool_id: Any, params: Any) -> list[str]:
        """道具の呼び出しを確かめ、合わない所を返す（空なら動かせる形）。目録に無い道具も、例外でなく1つの文で返す"""
        try:
            entry = self._entry(tool_id)
        except UnknownToolError as error:
            return [str(error)]
        return validate(params, entry.schema)

    def needs_confirmation(self, tool_id: Any, *, plan_approved: bool = False) -> bool:
        """動かす前にユーザーの確認が要るか。

        見るだけ：要らない。変更：計画を承認していなければ要る。消す：いつも要る。
        """
        _require_bool("plan_approved", plan_approved)
        return _needs_confirmation(self._entry(tool_id).spec.effect, plan_approved)

    def call(self, tool_id: Any, params: Any, *, confirmed: bool = False, plan_approved: bool = False) -> Any:
        """道具を動かし、関数の戻り値を返す。

        調べる順番：目録に無い → UnknownToolError、値がスキーマに合わない → InvalidParamsError、
        確認が要るのに confirmed でない → ConfirmationRequiredError。
        関数には値の写しを渡す（関数が値を書き換えても、呼んだ側の値は変わらない）。
        """
        _require_bool("confirmed", confirmed)
        _require_bool("plan_approved", plan_approved)
        entry = self._entry(tool_id)
        errors = validate(params, entry.schema)
        if errors:
            raise InvalidParamsError(entry.spec.id, errors)
        if _needs_confirmation(entry.spec.effect, plan_approved) and not confirmed:
            raise ConfirmationRequiredError(entry.spec.id, entry.spec.effect)
        # 確かめた値は JSON の値だけなので、写しは安全に作れる。写しも確かめ直し、確かめた物だけを渡す
        safe_params = copy.deepcopy(params)
        errors = validate(safe_params, entry.schema)
        if errors:
            raise InvalidParamsError(entry.spec.id, errors)
        return entry.func(safe_params)

    def listing(self, category: str | None = None) -> list[dict[str, Any]]:
        """道具の一覧（JSON にそのまま書ける形。id の順）。LLM に見せる目録と、記録（A15）に使う"""
        return [self._listing_entry(self._entries[spec.id]) for spec in self.specs(category)]

    @staticmethod
    def _listing_entry(entry: _Entry) -> dict[str, Any]:
        spec = entry.spec
        return {
            "id": spec.id,
            "version": spec.version,
            "feature": spec.feature,
            "category": spec.category,
            "title": spec.title,
            "summary": spec.summary,
            "effect": spec.effect.value,
            "heavy": spec.heavy,
            "input_schema": copy.deepcopy(entry.schema),
            "checks_before": list(spec.checks_before),
            "measures_after": list(spec.measures_after),
        }

    def fingerprint(self) -> str:
        """目録全体の指紋（一覧を決まった形の JSON にした物の sha256、16進）。どの説明が変わっても変わる"""
        text = json.dumps(self.listing(), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# MARK: - LLM の返事を読む


@dataclass(frozen=True)
class ToolCall:
    """LLM の返事から読んだ、道具の呼び出し（まだ確かめていない）"""

    tool: str
    params: dict[str, Any] = field(hash=False)

    def __post_init__(self) -> None:
        if type(self.tool) is not str:
            raise ToolCallError(f"tool は文字列にしてください（いまは {describe_value(self.tool)}）")
        if type(self.params) is not dict:
            raise ToolCallError(f"params はオブジェクト（dict）にしてください（いまは {describe_value(self.params)}）")


class _Rejected(Exception):
    """JSON を読んでいる途中で、受け取れない物を見つけた"""


def _reject_constant(name: str) -> Any:
    raise _Rejected(f"{name} は使えません（有限の数だけを使ってください）")


def _parse_finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise _Rejected(f"数 {text[:40]} は大きすぎます（有限の数だけを使ってください）")
    return value


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise _Rejected(f"同じ名前 {quote_text(name)} が2回あります")
        result[name] = value
    return result


def parse_tool_call(text: str, *, max_length: int = MAX_TOOL_CALL_LENGTH) -> ToolCall:
    """LLM の返事（JSON の文字列）を、道具の呼び出しとしてきびしく読む。コードとしては動かさない。

    受け取る形は {"tool": "道具の id", "params": {...}} だけ。次のときは ToolCallError：
    文字列でない・max_length 文字より長い・JSON として読めない・NaN や Infinity（とても大きな数を含む）・
    同じ名前が2回ある（どの深さでも）・いちばん外がオブジェクトでない・"tool" と "params" 以外の名前がある、
    または足りない・tool が文字列でない・params がオブジェクトでない。
    道具が目録にあるか・値がスキーマに合うかは、ここでは調べない（Catalog.call が調べる）。
    """
    if type(max_length) is not int or max_length < 1:
        raise ValueError(f"max_length は 1 以上の整数にしてください（いまは {describe_value(max_length)}）")
    if type(text) is not str:
        raise ToolCallError(f"道具の呼び出しは JSON の文字列で渡してください（いまは {describe_value(text)}）")
    if len(text) > max_length:
        raise ToolCallError(f"長すぎます（{len(text)} 文字。{max_length} 文字までにしてください）")
    try:
        data = json.loads(text, parse_constant=_reject_constant, parse_float=_parse_finite_float,
                          object_pairs_hook=_object_without_duplicates)
    except _Rejected as error:
        raise ToolCallError(f"JSON に受け取れない物があります：{error}") from None
    except json.JSONDecodeError as error:
        raise ToolCallError(f"JSON として読めません（{error.lineno} 行 {error.colno} 文字目）") from None
    except RecursionError:
        raise ToolCallError("JSON の入れ子が深すぎます") from None
    except ValueError:
        # とても長い整数など（Python が数にできない物）
        raise ToolCallError("JSON として読めません（数が長すぎるなど）") from None
    if type(data) is not dict:
        raise ToolCallError('いちばん外は {"tool": ..., "params": {...}} の形のオブジェクトにしてください'
                            f"（いまは {describe_value(data)}）")
    problems: list[str] = []
    for name in ("tool", "params"):
        if name not in data:
            problems.append(f'"{name}" がありません')
    for name in data:
        if name not in ("tool", "params"):
            problems.append(f'決められていない名前 {quote_text(name)} があります（使えるのは "tool" と "params" だけ）')
    if "tool" in data and type(data["tool"]) is not str:
        problems.append(f'"tool" は文字列にしてください（いまは {describe_value(data["tool"])}）')
    if "params" in data and type(data["params"]) is not dict:
        problems.append(f'"params" はオブジェクトにしてください（いまは {describe_value(data["params"])}）')
    if problems:
        raise ToolCallError(f"道具の呼び出しの形が違います：{_join(problems)}", problems)
    return ToolCall(tool=data["tool"], params=data["params"])
