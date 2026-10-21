"""A8 道具の目録：道具の入力の形（JSON スキーマ）を決めて、値がその形に合うかを確かめる。

外のパッケージ（jsonschema）は使わず、目録に要る分だけの、きびしい小さな確かめ役を書く。

- check_schema(schema)：目録のスキーマが決まりに合うか。合わない所の一覧を返す（空なら合っている）。
- validate(instance, schema)：値がスキーマに合うか。合わない所の一覧を返す（空なら合っている）。

値の比べ方（JSON の意味に合わせる）：
- 受け取るのは、そのままの Python の型だけ（dict・list・str・int・float・bool・None）。
  numpy.float64・IntEnum・OrderedDict のような「仲間の型」は受け取らない。
  合格した値は、そのまま JSON に書けて、記録（A15）に残せるようにするため。
- true / false は数ではない（1 と True は別の値）。1 と 1.0 は、数としては同じ値（JSON Schema と同じ）。
  ただし integer の所では、1.0 は小数なので受け取らない。
- オブジェクトどうしは、名前の順番に関係なく比べる（{"a": 1, "b": 2} と {"b": 2, "a": 1} は同じ）。
- pattern は re.search で探す（文字列のどこかに合えばよい。JSON Schema と同じ）。
  JSON Schema の正規表現（ECMA-262）に近づけるため、次の2つを変える：
  ・$ は文字列の終わりだけに合わせる（Python の $ は、最後の改行の前にも合ってしまうため）。
  ・\\d・\\w・\\b は ASCII の文字だけに合わせる（re.ASCII）。全角の数字は \\d に合わない。
    なお \\s も ASCII の空白だけになる（全角の空白は \\s に合わない）。
  (?i) のような、正規表現の中のフラグは使えない。
"""

from __future__ import annotations

import functools
import json
import math
import re
from typing import Any

from ..thresholds import MAX_SCHEMA_DEPTH

__all__ = [
    "ANNOTATION_KEYWORDS",
    "MAX_SCHEMA_DEPTH",
    "SUPPORTED_KEYWORDS",
    "TYPE_NAMES",
    "VALIDATION_KEYWORDS",
    "check_schema",
    "describe_value",
    "quote_text",
    "validate",
]

# MARK: - 決まり

# スキーマの入れ子の深さの上限（いちばん外が 0 段）は thresholds.MAX_SCHEMA_DEPTH（仮）。上で読み込む

# JSON で正しくやりとりできる整数の大きさ（2**53 - 1。I-JSON（RFC 7493）と JavaScript の Number.MAX_SAFE_INTEGER）。
# スキーマに書く整数（個数・長さ・限り・enum・const）は、この中にする。これより大きい整数は LLM の側で正しく読めないことがあり、
# とても大きな整数（4300 桁より長い物）は文字にできないので、文・一覧・指紋が作れなくなるため（基準ではなく、JSON の決まり）
_MAX_SAFE_INTEGER = 2 ** 53 - 1

# type に書ける型の名前
TYPE_NAMES = ("object", "array", "string", "integer", "number", "boolean", "null")

# 値を確かめる言葉
VALIDATION_KEYWORDS = frozenset({
    "type", "properties", "required", "additionalProperties", "items", "enum", "const",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems",
})
# 説明だけの言葉（値の確かめには使わない。default は値を埋めない）
ANNOTATION_KEYWORDS = frozenset({"title", "description", "default", "examples"})
# 目録のスキーマで使える言葉。これ以外は誤り
SUPPORTED_KEYWORDS = VALIDATION_KEYWORDS | ANNOTATION_KEYWORDS

# どの型でも使える言葉
_COMMON_KEYWORDS = frozenset({"type", "enum", "const"}) | ANNOTATION_KEYWORDS
_BOUND_KEYWORDS = ("minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum")
# 型ごとに使える言葉
_TYPE_KEYWORDS = {
    "object": frozenset({"properties", "required", "additionalProperties"}),
    "array": frozenset({"items", "minItems", "maxItems", "uniqueItems"}),
    "string": frozenset({"minLength", "maxLength", "pattern"}),
    "integer": frozenset(_BOUND_KEYWORDS),
    "number": frozenset(_BOUND_KEYWORDS),
    "boolean": frozenset(),
    "null": frozenset(),
}
_TYPE_LABELS = {
    "object": "オブジェクト（dict）",
    "array": "配列（list）",
    "string": "文字列（str）",
    "integer": "整数（int）",
    "number": "数（int か float）",
    "boolean": "真偽値（true か false）",
    "null": "null（None）",
}

# 値を比べるときに、入れ子をたどる深さの上限。合格する値の深さはスキーマの深さまでなので、それより大きくしておく
_KEY_DEPTH_LIMIT = 2 * MAX_SCHEMA_DEPTH
# 文の中に値を見せるときの長さ（文字）
_SHOW_MAX_CHARS = 40
# 文の中にそのまま書く整数の大きさ（ビット）。これより大きい整数は桁を書かない（とても長い整数を文字にすると落ちるため）
_SHOW_MAX_INT_BITS = 64
# 名前を「$.name」の形で書けるか（ASCII の識別子）
_PLAIN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


# MARK: - 値を短い日本語で見せる


def _is_text(text: str) -> bool:
    """UTF-8 で書ける文字列か（対になっていないサロゲートが無いか）"""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def quote_text(text: str) -> str:
    """文字列を、文の中に見せる引用符つきの短い形にする（長い物は切り、使えない文字は \\u の形にする）"""
    cut = text[:_SHOW_MAX_CHARS]
    shown = json.dumps(cut, ensure_ascii=False)
    if not _is_text(shown):
        shown = json.dumps(cut, ensure_ascii=True)
    return shown + ("…" if len(text) > _SHOW_MAX_CHARS else "")


def _show_int(value: int) -> str:
    if value.bit_length() > _SHOW_MAX_INT_BITS:
        return "（とても大きな整数）"
    return str(value)


def _show_number(value: int | float) -> str:
    return _show_int(value) if type(value) is int else repr(value)


def _show_value(value: Any) -> str:
    """スキーマに書いた値（JSON の値）を、JSON の書き方で短く見せる"""
    if value is None:
        return "null"
    kind = type(value)
    if kind is bool:
        return "true" if value else "false"
    if kind is int or kind is float:
        return _show_number(value)
    if kind is str:
        return quote_text(value)
    if kind is list:
        return f"配列（{len(value)} 個）"
    if kind is dict:
        return f"オブジェクト（{len(value)} 個の名前）"
    return describe_value(value)


def _type_name(value: Any) -> str:
    """型の名前。変わった型でも落ちないようにする"""
    try:
        name = type(value).__qualname__
    except Exception:  # 型の名前を読むだけで失敗する、変わった型
        return "名前の分からない型"
    return name if type(name) is str else "名前の分からない型"


def describe_value(value: Any) -> str:
    """値の様子を、短い日本語で書く（どんな Python の物を渡しても落ちない。repr は呼ばない）"""
    if value is None:
        return "null"
    kind = type(value)
    if kind is bool:
        return f"真偽値 {'true' if value else 'false'}"
    if kind is int:
        return f"整数 {_show_int(value)}"
    if kind is float:
        return f"小数 {repr(value)}"
    if kind is str:
        return f"文字列 {quote_text(value)}"
    if kind is list:
        return f"配列（{len(value)} 個）"
    if kind is dict:
        return f"オブジェクト（{len(value)} 個の名前）"
    if kind is tuple:
        return "tuple"
    for base in (bool, int, float, str, list, dict):
        # isinstance は変わった物で失敗することがあるので、本当の型どうしで比べる
        if issubclass(kind, base):
            return f"{_type_name(value)} 型（{base.__name__} の仲間の型は使えません。{base.__name__} に変えてください）"
    return f"{_type_name(value)} 型"


# MARK: - 値の比べ（JSON の意味）


class _KeyMaker:
    """JSON の値を、比べられる形（ハッシュできる tuple）にする。

    - true / false と数は別にする。1 と 1.0 は同じにする（Python の 1 == 1.0 と hash(1) == hash(1.0) を使う）。
    - オブジェクトは名前の順番に関係なく比べる（frozenset にする）。
    - JSON の値でない物・深すぎる物・自分を含む物は、その物自身（id）で比べる（ほかの物とは等しくならない）。
    1回の比べの間だけ使う（id を覚えるため）。
    """

    def __init__(self) -> None:
        self._memo: dict[int, Any] = {}
        self._active: set[int] = set()

    def key(self, value: Any, depth: int = 0) -> Any:
        if value is None:
            return ("null",)
        kind = type(value)
        if kind is bool:
            return ("bool", value)
        if kind is int or kind is float:
            return ("number", value)
        if kind is str:
            return ("string", value)
        if kind is not list and kind is not dict:
            return ("other", id(value))
        ident = id(value)
        if ident in self._memo:
            return self._memo[ident]
        if depth >= _KEY_DEPTH_LIMIT or ident in self._active:
            return ("other", ident)
        self._active.add(ident)
        try:
            if kind is list:
                result = ("array", tuple(self.key(item, depth + 1) for item in value))
            elif any(type(name) is not str for name in value):
                result = ("other", ident)
            else:
                result = ("object", frozenset((name, self.key(item, depth + 1)) for name, item in value.items()))
        finally:
            self._active.discard(ident)
        self._memo[ident] = result
        return result


# MARK: - 正規表現


class _PatternError(ValueError):
    pass


def _to_python_pattern(pattern: str) -> str:
    """JSON Schema の正規表現を Python の正規表現にする：[ ] の外の $ を \\Z（文字列の終わり）に変える"""
    out: list[str] = []
    in_class = False
    class_start = False  # [ や [^ のすぐあと（ここの ] は文字として読む）
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\":
            out.append(pattern[i:i + 2])
            i += 2
            class_start = False
            continue
        if in_class:
            if char == "^" and class_start and pattern[i - 1] == "[":
                out.append(char)
                i += 1
                continue
            if char == "]" and not class_start:
                in_class = False
            class_start = False
            out.append(char)
        elif char == "[":
            in_class = True
            class_start = True
            out.append(char)
        elif char == "(" and pattern.startswith("(?", i) and i + 2 < len(pattern) and pattern[i + 2] in "aiLmsux-":
            raise _PatternError("(?i) のような、正規表現の中のフラグは使えません")
        elif char == "$":
            out.append(r"\Z")
        else:
            out.append(char)
        i += 1
    return "".join(out)


@functools.lru_cache(maxsize=256)
def _compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(_to_python_pattern(pattern), re.ASCII)


def _pattern_problem(pattern: str) -> str | None:
    """正規表現として読めないときは、その理由。読めるときは None"""
    try:
        _compile_pattern(pattern)
    except _PatternError as error:
        return str(error)
    except re.error as error:
        return f"正規表現として読めません：{error}"
    except (RecursionError, OverflowError, ValueError, TypeError):
        return "正規表現として読めません"
    return None


# MARK: - スキーマの点検


def _pointer(base: str, token: str) -> str:
    """JSON Pointer の形で1段深くする（~ と / は ~0・~1 にする）"""
    return base + "/" + token.replace("~", "~0").replace("/", "~1")


def check_schema(schema: Any) -> list[str]:
    """目録のスキーマが決まりに合うかを調べ、合わない所を日本語の文で返す（空なら合っている）。

    場所は JSON Pointer の形で書く（例 "#/properties/size/maximum"）。決まり：
    - いちばん外は type が "object"。type はどこでも1つの文字列（object・array・string・integer・number・boolean・null）。
    - object：properties（空でもよい）と、additionalProperties: false が要る。required は properties にある名前の
      重ならない list。
    - array：items と maxItems が要る。
    - string：maxLength か enum（または const）が要る。
    - integer・number：enum（または const）か、下の限り（minimum か exclusiveMinimum）と上の限り（maximum か
      exclusiveMaximum）の両方が要る。限りは有限の数で、間に入る値が無いといけない。
    - スキーマに書く整数（個数・長さ・限り・enum・const）は、JSON で正しく扱える ±(2**53 - 1) の中。
    - pattern は正規表現として読めること。知らない言葉・その型で使えない言葉は誤り。
    - default・const・enum・examples の値は、そのスキーマに合うこと。
    どんな物を渡しても例外を出さない（自分を含むスキーマ・深すぎるスキーマも誤りとして返す）。
    """
    if type(schema) is not dict:
        return [f"#: スキーマは dict にしてください（いまは {describe_value(schema)}）"]
    errors: list[str] = []
    if all(type(key) is str for key in schema):
        root_type = schema.get("type")
        if type(root_type) is str and root_type in TYPE_NAMES and root_type != "object":
            errors.append('#/type: いちばん外の type は "object" にしてください（道具の値は、名前の付いた組で受け取る）')
    _check_node(schema, "#", 0, set(), errors)
    return errors


def _check_node(node: Any, pointer: str, depth: int, active: set[int], errors: list[str]) -> None:
    """1つのスキーマ（と、その中のスキーマ）を調べる"""
    if type(node) is not dict:
        errors.append(f"{pointer}: スキーマは dict にしてください（いまは {describe_value(node)}）")
        return
    if depth > MAX_SCHEMA_DEPTH:
        errors.append(f"{pointer}: スキーマの入れ子が深すぎます（{MAX_SCHEMA_DEPTH} 段まで）")
        return
    if id(node) in active:
        errors.append(f"{pointer}: スキーマが自分自身を中に含んでいます")
        return
    start = len(errors)
    bad_keys = [key for key in node if type(key) is not str]
    if bad_keys:
        errors.append(f"{pointer}: キーは文字列にしてください（{describe_value(bad_keys[0])} のキーがあります）")
        return
    for key in node:
        if key not in SUPPORTED_KEYWORDS:
            errors.append(f"{pointer}: 使えない言葉 {quote_text(key)} があります")
    if "type" not in node:
        errors.append(f"{pointer}: type がありません")
        return
    type_name = node["type"]
    if type(type_name) is not str:
        errors.append(f"{pointer}/type: type は1つの文字列にしてください（いまは {describe_value(type_name)}）")
        return
    if type_name not in TYPE_NAMES:
        errors.append(f"{pointer}/type: 型 {quote_text(type_name)} は使えません（使えるのは {', '.join(TYPE_NAMES)}）")
        return
    allowed = _COMMON_KEYWORDS | _TYPE_KEYWORDS[type_name]
    for key in node:
        if key in SUPPORTED_KEYWORDS and key not in allowed:
            errors.append(f"{pointer}/{key}: type が \"{type_name}\" のときには使えません")
    for key in ("title", "description"):
        if key in node and not (type(node[key]) is str and _is_text(node[key])):
            errors.append(f"{pointer}/{key}: 文字列にしてください（いまは {describe_value(node[key])}）")
    if "examples" in node and type(node["examples"]) is not list:
        errors.append(f"{pointer}/examples: list にしてください（いまは {describe_value(node['examples'])}）")

    active.add(id(node))
    try:
        if type_name == "object":
            _check_object(node, pointer, depth, active, errors)
        elif type_name == "array":
            _check_array(node, pointer, depth, active, errors)
        elif type_name == "string":
            _check_string(node, pointer, errors)
        elif type_name in ("integer", "number"):
            _check_number(node, pointer, type_name, errors)
    finally:
        active.discard(id(node))
    _check_enum_list(node, pointer, errors)
    # 形に誤りが無いときだけ、値（default など）をスキーマで確かめる（形が壊れたスキーマでは確かめられないため）
    if len(errors) == start:
        _check_values(node, pointer, errors)


def _check_object(node: dict[str, Any], pointer: str, depth: int, active: set[int], errors: list[str]) -> None:
    properties = node.get("properties")
    names: set[str] = set()
    if "properties" not in node:
        errors.append(f"{pointer}: properties がありません（名前が1つも無いときも {{}} と書いてください）")
    elif type(properties) is not dict:
        errors.append(f"{pointer}/properties: dict にしてください（いまは {describe_value(properties)}）")
    else:
        for name, sub in properties.items():
            if type(name) is not str or not _is_text(name):
                errors.append(f"{pointer}/properties: 名前は文字列にしてください（{describe_value(name)} があります）")
                continue
            names.add(name)
            _check_node(sub, _pointer(pointer + "/properties", name), depth + 1, active, errors)
    if "additionalProperties" not in node:
        errors.append(f"{pointer}: additionalProperties がありません（false にしてください）")
    elif node["additionalProperties"] is not False:
        errors.append(f"{pointer}/additionalProperties: false にしてください（決められていない名前を受け取らないため）")
    if "required" in node:
        required = node["required"]
        if type(required) is not list:
            errors.append(f"{pointer}/required: 名前の list にしてください（いまは {describe_value(required)}）")
            return
        seen: set[str] = set()
        for index, name in enumerate(required):
            where = f"{pointer}/required/{index}"
            if type(name) is not str:
                errors.append(f"{where}: 文字列にしてください（いまは {describe_value(name)}）")
            elif name in seen:
                errors.append(f"{where}: {quote_text(name)} が2回あります")
            elif type(properties) is dict and name not in names:
                errors.append(f"{where}: {quote_text(name)} は properties にありません")
            if type(name) is str:
                seen.add(name)


def _check_count(node: dict[str, Any], pointer: str, key: str, errors: list[str]) -> bool:
    """数（個数・長さ）の言葉が 0 以上の整数か。書いていないか、正しいときは True"""
    if key not in node:
        return True
    value = node[key]
    if type(value) is not int or value < 0:
        errors.append(f"{pointer}/{key}: 0 以上の整数にしてください（いまは {describe_value(value)}）")
        return False
    if value > _MAX_SAFE_INTEGER:
        errors.append(f"{pointer}/{key}: 大きすぎます（JSON で正しく扱える {_MAX_SAFE_INTEGER} までの整数にしてください。"
                      f"いまは {describe_value(value)}）")
        return False
    return True


def _check_min_max(node: dict[str, Any], pointer: str, min_key: str, max_key: str, errors: list[str]) -> None:
    good_min = _check_count(node, pointer, min_key, errors)
    good_max = _check_count(node, pointer, max_key, errors)
    if good_min and good_max and min_key in node and max_key in node and node[min_key] > node[max_key]:
        errors.append(f"{pointer}: {min_key}（{_show_int(node[min_key])}）が {max_key}（{_show_int(node[max_key])}）"
                      "より大きくなっています")


def _check_array(node: dict[str, Any], pointer: str, depth: int, active: set[int], errors: list[str]) -> None:
    if "items" not in node:
        errors.append(f"{pointer}: items がありません（要素の形を決めてください）")
    else:
        _check_node(node["items"], pointer + "/items", depth + 1, active, errors)
    if "maxItems" not in node:
        errors.append(f"{pointer}: maxItems がありません（要素の数の上限を決めてください）")
    _check_min_max(node, pointer, "minItems", "maxItems", errors)
    if "uniqueItems" in node and type(node["uniqueItems"]) is not bool:
        errors.append(f"{pointer}/uniqueItems: true か false にしてください（いまは {describe_value(node['uniqueItems'])}）")


def _check_string(node: dict[str, Any], pointer: str, errors: list[str]) -> None:
    _check_min_max(node, pointer, "minLength", "maxLength", errors)
    if "pattern" in node:
        pattern = node["pattern"]
        if type(pattern) is not str or not _is_text(pattern):
            errors.append(f"{pointer}/pattern: 文字列にしてください（いまは {describe_value(pattern)}）")
        else:
            problem = _pattern_problem(pattern)
            if problem is not None:
                errors.append(f"{pointer}/pattern: {problem}")
    if "maxLength" not in node and "enum" not in node and "const" not in node:
        errors.append(f"{pointer}: 文字列には maxLength か enum を書いてください（長さの上限を決めるため）")


_SAFE_INTEGER_RULE = f"整数は JSON で正しく扱える -{_MAX_SAFE_INTEGER} 以上 {_MAX_SAFE_INTEGER} 以下にしてください"


def _is_safe_integer(value: int) -> bool:
    return -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER


def _check_number(node: dict[str, Any], pointer: str, type_name: str, errors: list[str]) -> None:
    good: dict[str, int | float] = {}
    for key in _BOUND_KEYWORDS:
        if key not in node:
            continue
        value = node[key]
        kind = type(value)
        if (kind is int and _is_safe_integer(value)) or (kind is float and math.isfinite(value)):
            good[key] = value
        elif kind is int:
            errors.append(f"{pointer}/{key}: {_SAFE_INTEGER_RULE}（いまは {describe_value(value)}）")
        else:
            errors.append(f"{pointer}/{key}: 有限の数にしてください（いまは {describe_value(value)}）")
    # 決めた値（enum・const）の整数も、JSON で正しく扱える大きさにする（default・examples は、限りか enum で止まる）
    listed = [("const", node["const"])] if "const" in node else []
    if type(node.get("enum")) is list:
        listed += [(f"enum/{index}", value) for index, value in enumerate(node["enum"])]
    for where, value in listed:
        if type(value) is int and not _is_safe_integer(value):
            errors.append(f"{pointer}/{where}: {_SAFE_INTEGER_RULE}（いまは {describe_value(value)}）")
    if "minimum" in node and "exclusiveMinimum" in node:
        errors.append(f"{pointer}: minimum と exclusiveMinimum は、どちらか1つにしてください")
    if "maximum" in node and "exclusiveMaximum" in node:
        errors.append(f"{pointer}: maximum と exclusiveMaximum は、どちらか1つにしてください")
    has_lower = "minimum" in node or "exclusiveMinimum" in node
    has_upper = "maximum" in node or "exclusiveMaximum" in node
    if not ("enum" in node or "const" in node or (has_lower and has_upper)):
        errors.append(f"{pointer}: {type_name} には enum か、下の限り（minimum か exclusiveMinimum）と"
                      "上の限り（maximum か exclusiveMaximum）の両方を書いてください")
    lowers = [key for key in ("minimum", "exclusiveMinimum") if key in good]
    uppers = [key for key in ("maximum", "exclusiveMaximum") if key in good]
    if len(lowers) == 1 and len(uppers) == 1:
        low_key, high_key = lowers[0], uppers[0]
        if _range_is_empty(good[low_key], low_key == "minimum", good[high_key], high_key == "maximum",
                           integer=type_name == "integer"):
            what = "整数" if type_name == "integer" else "数"
            errors.append(f"{pointer}: {low_key}（{_show_number(good[low_key])}）と {high_key}"
                          f"（{_show_number(good[high_key])}）の間に、入る{what}がありません")


def _range_is_empty(low: int | float, low_inclusive: bool, high: int | float, high_inclusive: bool, *,
                    integer: bool) -> bool:
    """下の限りと上の限りの間に、入る値が1つも無いか"""
    if integer:
        smallest = math.ceil(low) if low_inclusive else math.floor(low) + 1
        largest = math.floor(high) if high_inclusive else math.ceil(high) - 1
        return smallest > largest
    if low < high:
        return False
    return not (low == high and low_inclusive and high_inclusive)


def _check_enum_list(node: dict[str, Any], pointer: str, errors: list[str]) -> None:
    if "enum" not in node:
        return
    values = node["enum"]
    if type(values) is not list or not values:
        errors.append(f"{pointer}/enum: 1つ以上の値の list にしてください（いまは {describe_value(values)}）")
        return
    keys = _KeyMaker()
    seen: dict[Any, int] = {}
    for index, value in enumerate(values):
        key = keys.key(value)
        if key in seen:
            errors.append(f"{pointer}/enum/{index}: {pointer}/enum/{seen[key]} と同じ値です")
        else:
            seen[key] = index


def _check_values(node: dict[str, Any], pointer: str, errors: list[str]) -> None:
    """enum・const・default・examples の値が、そのスキーマに合うかを確かめる"""
    base = {key: value for key, value in node.items() if key not in ("default", "examples")}
    if "enum" in node:
        without_enum = {key: value for key, value in base.items() if key != "enum"}
        for index, value in enumerate(node["enum"]):
            for problem in _validate_checked(value, without_enum):
                errors.append(f"{pointer}/enum/{index}: 選べる値がスキーマに合いません（{problem}）")
    if "const" in node:
        without_const = {key: value for key, value in base.items() if key != "const"}
        for problem in _validate_checked(node["const"], without_const):
            errors.append(f"{pointer}/const: 決めた値がスキーマに合いません（{problem}）")
    if "default" in node:
        for problem in _validate_checked(node["default"], base):
            errors.append(f"{pointer}/default: 初めの値がスキーマに合いません（{problem}）")
    if "examples" in node:
        for index, value in enumerate(node["examples"]):
            for problem in _validate_checked(value, base):
                errors.append(f"{pointer}/examples/{index}: 例がスキーマに合いません（{problem}）")


# MARK: - 値の確かめ


def validate(instance: Any, schema: dict[str, Any]) -> list[str]:
    """値がスキーマに合うかを確かめ、合わない所をすべて日本語の文で返す（空なら合っている）。

    場所は "$"（いちばん外）・"$.size"・"$.items[2].name" の形で書く。
    スキーマは check_schema を通る物にする（通らないときは ValueError）。
    値はどんな Python の物でもよい（例外を出さずに、合わない所として返す）。
    default は値を埋めない（渡された値だけを確かめる）。
    """
    problems = check_schema(schema)
    if problems:
        raise ValueError("スキーマが決まりに合いません：" + "；".join(problems))
    return _validate_checked(instance, schema)


def _validate_checked(instance: Any, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _validate_node(instance, schema, "$", errors)
    return errors


def _child_path(path: str, name: str) -> str:
    if _PLAIN_NAME.fullmatch(name):
        return f"{path}.{name}"
    return f"{path}[{quote_text(name)}]"


def _type_problem(instance: Any, expected: str) -> str | None:
    """型が合わないときは、その文。合うときは None"""
    kind = type(instance)
    if expected == "object":
        ok = kind is dict
    elif expected == "array":
        ok = kind is list
    elif expected == "string":
        ok = kind is str
    elif expected == "integer":
        ok = kind is int
    elif expected == "number":
        if kind is float and not math.isfinite(instance):
            return f"有限の数にしてください（いまは {repr(instance)}）"
        ok = kind is int or kind is float
    elif expected == "boolean":
        ok = kind is bool
    else:
        ok = instance is None
    if ok:
        return None
    label = _TYPE_LABELS[expected]
    if expected in ("integer", "number") and kind is bool:
        return f"{label}にしてください（{'true' if instance else 'false'} は真偽値で、数としては使えません）"
    if expected == "integer" and kind is float:
        if math.isfinite(instance) and instance.is_integer():
            return f"{label}にしてください（{repr(instance)} は小数です。小数点を付けずに書いてください）"
        return f"{label}にしてください（{repr(instance)} は小数です）"
    if expected == "array" and kind is tuple:
        return "配列は list にしてください（tuple は使えません）"
    return f"{label}にしてください（いまは {describe_value(instance)}）"


def _validate_node(instance: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    expected = schema["type"]
    problem = _type_problem(instance, expected)
    if problem is not None:
        # 型が違うときは、ほかの言葉（範囲・enum など）は調べない（同じ誤りを何度も出さないため）
        errors.append(f"{path}: {problem}")
        return
    if expected == "object":
        _validate_object(instance, schema, path, errors)
    elif expected == "array":
        _validate_array(instance, schema, path, errors)
    elif expected == "string":
        _validate_string(instance, schema, path, errors)
    elif expected in ("integer", "number"):
        _validate_number(instance, schema, path, errors)
    if "const" in schema:
        keys = _KeyMaker()
        if keys.key(instance) != keys.key(schema["const"]):
            errors.append(f"{path}: 値は {_show_value(schema['const'])} にしてください")
    if "enum" in schema:
        keys = _KeyMaker()
        choices = {keys.key(value) for value in schema["enum"]}
        if keys.key(instance) not in choices:
            shown = ", ".join(_show_value(value) for value in schema["enum"][:10])
            more = " など" if len(schema["enum"]) > 10 else ""
            errors.append(f"{path}: 選べる値ではありません（選べるのは {shown}{more}）")


def _validate_object(instance: dict[Any, Any], schema: dict[str, Any], path: str, errors: list[str]) -> None:
    properties: dict[str, Any] = schema["properties"]
    required = set(schema.get("required", ()))
    bad_keys = [key for key in instance if type(key) is not str]
    if bad_keys:
        errors.append(f"{path}: 名前（キー）は文字列にしてください（{describe_value(bad_keys[0])} のキーがあります）")
        # 文字列でないキーを外した写しで調べる（変わったキーの == が呼ばれないようにするため）
        instance = {key: value for key, value in instance.items() if type(key) is str}
    for name, sub in properties.items():
        if name in instance:
            _validate_node(instance[name], sub, _child_path(path, name), errors)
        elif name in required:
            errors.append(f"{_child_path(path, name)}: 必要な値がありません")
    names = ", ".join(properties) if properties else "なし"
    for name in instance:
        if name not in properties:
            errors.append(f"{_child_path(path, name)}: 決められていない名前です（使える名前：{names}）")


def _validate_array(instance: list[Any], schema: dict[str, Any], path: str, errors: list[str]) -> None:
    count = len(instance)
    max_items = schema["maxItems"]
    if count > max_items:
        # 多すぎるときは、要素は調べない（調べる手間を上限の中におさめるため）
        errors.append(f"{path}: 要素が多すぎます（{count} 個。{max_items} 個までにしてください）")
        return
    min_items = schema.get("minItems", 0)
    if count < min_items:
        errors.append(f"{path}: 要素が少なすぎます（{count} 個。{min_items} 個以上にしてください）")
    item_schema = schema["items"]
    for index, item in enumerate(instance):
        _validate_node(item, item_schema, f"{path}[{index}]", errors)
    if schema.get("uniqueItems") is True:
        keys = _KeyMaker()
        seen: dict[Any, int] = {}
        for index, item in enumerate(instance):
            key = keys.key(item)
            if key in seen:
                errors.append(f"{path}[{index}]: {path}[{seen[key]}] と同じ値です（同じ値は1回だけにしてください）")
            else:
                seen[key] = index


def _validate_string(instance: str, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    if not _is_text(instance):
        errors.append(f"{path}: 使えない文字（対になっていないサロゲート）が入っています")
    length = len(instance)
    too_long = "maxLength" in schema and length > schema["maxLength"]
    if too_long:
        errors.append(f"{path}: 長すぎます（{length} 文字。{schema['maxLength']} 文字までにしてください）")
    if "minLength" in schema and length < schema["minLength"]:
        errors.append(f"{path}: 短すぎます（{length} 文字。{schema['minLength']} 文字以上にしてください）")
    # 長すぎる文字列には正規表現を使わない（手間を上限の中におさめるため）
    if "pattern" in schema and not too_long and _compile_pattern(schema["pattern"]).search(instance) is None:
        errors.append(f"{path}: 決められた形 {quote_text(schema['pattern'])} に合いません")


def _validate_number(instance: int | float, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    shown = _show_number(instance)
    if "minimum" in schema and instance < schema["minimum"]:
        errors.append(f"{path}: {shown} は小さすぎます（{_show_number(schema['minimum'])} 以上にしてください）")
    if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
        errors.append(f"{path}: {shown} は小さすぎます（{_show_number(schema['exclusiveMinimum'])} より大きくしてください）")
    if "maximum" in schema and instance > schema["maximum"]:
        errors.append(f"{path}: {shown} は大きすぎます（{_show_number(schema['maximum'])} 以下にしてください）")
    if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
        errors.append(f"{path}: {shown} は大きすぎます（{_show_number(schema['exclusiveMaximum'])} より小さくしてください）")
