"""A8 道具の目録：スキーマの点検（check_schema）と値の確かめ（validate）を確かめる"""

import copy
import decimal
import enum
import fractions
import math
import unittest
from collections import OrderedDict

import numpy as np

from blender_ai_tools.catalog.schema import (
    MAX_SCHEMA_DEPTH,
    SUPPORTED_KEYWORDS,
    check_schema,
    describe_value,
    validate,
)


# MARK: - 小さなスキーマを作る


def obj(properties=None, **extra):
    """決まりに合う object のスキーマ"""
    return {"type": "object", "properties": {} if properties is None else properties,
            "additionalProperties": False, **extra}


def root(**properties):
    return obj(properties)


def string(**extra):
    return {"type": "string", "maxLength": 10, **extra}


def integer(**extra):
    base = {"type": "integer", "minimum": 0, "maximum": 10}
    base.update(extra)
    return base


def number(**extra):
    base = {"type": "number", "minimum": 0, "maximum": 10}
    base.update(extra)
    return base


def array(items, **extra):
    return {"type": "array", "items": items, "maxItems": 5, **extra}


def without(schema, *keys):
    return {key: value for key, value in schema.items() if key not in keys}


# MARK: - 変わった Python の物（落ちないことを確かめる）


class RaisingEverything:
    """比べる・ハッシュ・長さ・文字にする、のどれでも例外を出す"""

    def __eq__(self, other):
        raise RuntimeError("eq")

    def __ne__(self, other):
        raise RuntimeError("ne")

    def __lt__(self, other):
        raise RuntimeError("lt")

    def __hash__(self):
        raise RuntimeError("hash")

    def __len__(self):
        raise RuntimeError("len")

    def __iter__(self):
        raise RuntimeError("iter")

    def __repr__(self):
        raise RuntimeError("repr")

    def __str__(self):
        raise RuntimeError("str")

    def __bool__(self):
        raise RuntimeError("bool")


class EvilMeta(type):
    """型の名前（__qualname__）を読むと例外を出す"""

    def __getattribute__(cls, name):
        if name == "__qualname__":
            raise RuntimeError("qualname")
        return super().__getattribute__(name)


class EvilType(metaclass=EvilMeta):
    pass


class EvilClassAttribute:
    """isinstance が __class__ を読むと例外になる"""

    @property
    def __class__(self):
        raise RuntimeError("__class__")


class RaisingStr(str):
    def __eq__(self, other):
        raise RuntimeError("eq")

    __hash__ = str.__hash__


class RaisingInt(int):
    def __lt__(self, other):
        raise RuntimeError("lt")

    def __gt__(self, other):
        raise RuntimeError("gt")


class RaisingDict(dict):
    def __iter__(self):
        raise RuntimeError("iter")

    def items(self):
        raise RuntimeError("items")


class RaisingList(list):
    def __len__(self):
        raise RuntimeError("len")

    def __iter__(self):
        raise RuntimeError("iter")


class Color(enum.IntEnum):
    RED = 1


def weird_values():
    cyclic_list = []
    cyclic_list.append(cyclic_list)
    cyclic_dict = {}
    cyclic_dict["size"] = cyclic_dict
    deep = []
    for _ in range(5000):
        deep = [deep]
    return [
        object(), {1, 2}, frozenset(), b"x", bytearray(b"x"), 1j, decimal.Decimal("1.5"), fractions.Fraction(1, 2),
        np.float64(1.5), np.int64(3), np.bool_(True), np.array([1.0]), (x for x in range(3)), lambda: 1, int,
        RaisingEverything(), EvilType(), EvilClassAttribute(), RaisingStr("a"), RaisingInt(3), RaisingDict(a=1),
        RaisingList([1]), Color.RED, OrderedDict(a=1), (1, 2), cyclic_list, cyclic_dict, deep, float("nan"),
        float("inf"), 10 ** 5000, "\ud800",
    ]


# 全部の型を含む、決まりに合うスキーマ
EVERYTHING = root(
    size=number(),
    count=integer(),
    name=string(pattern="^[a-z]+$"),
    flag={"type": "boolean"},
    nothing={"type": "null"},
    tags=array(string(), uniqueItems=True),
    nested=obj({"value": integer()}, required=["value"]),
    choice={"type": "string", "enum": ["a", "b"]},
    grid=array(array(number(), uniqueItems=True), uniqueItems=True),
)


# MARK: - 使える言葉


class SupportedKeywordsTest(unittest.TestCase):
    def test_exact_set(self):
        self.assertEqual(SUPPORTED_KEYWORDS, {
            "type", "properties", "required", "additionalProperties", "items", "enum", "const",
            "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "pattern",
            "minItems", "maxItems", "uniqueItems", "title", "description", "default", "examples",
        })


# MARK: - check_schema：決まりに合うスキーマ


class CheckSchemaValidTest(unittest.TestCase):
    def assertValid(self, schema):
        self.assertEqual(check_schema(schema), [])

    def test_minimal_root(self):
        self.assertValid({"type": "object", "properties": {}, "additionalProperties": False})

    def test_every_keyword(self):
        schema = {
            "type": "object", "title": "骨", "description": "説明", "additionalProperties": False,
            "required": ["size", "names"],
            "examples": [{"size": 0.5, "names": ["左腕"]}],
            "properties": {
                "size": {"type": "number", "exclusiveMinimum": 0, "maximum": 1.5, "default": 1.0, "examples": [0.5]},
                "count": {"type": "integer", "minimum": 0, "exclusiveMaximum": 10, "default": 0},
                "names": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 20,
                                                     "pattern": "^[^ ]+$"},
                          "minItems": 1, "maxItems": 100, "uniqueItems": True},
                "mode": {"type": "string", "enum": ["fast", "slow"], "default": "fast"},
                "fixed": {"type": "integer", "const": 4},
                "flag": {"type": "boolean", "default": False},
                "nothing": {"type": "null"},
            },
        }
        self.assertValid(schema)
        self.assertEqual(validate({"size": 0.5, "names": ["左腕"]}, schema), [])

    def test_bounded_by_enum_or_const_only(self):
        self.assertValid(root(a={"type": "integer", "enum": [1, 2, 3]}))
        self.assertValid(root(a={"type": "number", "enum": [0.5, 1]}))
        self.assertValid(root(a={"type": "string", "enum": ["x"]}))
        self.assertValid(root(a={"type": "string", "const": "x"}))
        self.assertValid(root(a={"type": "integer", "const": 3}))

    def test_single_value_ranges(self):
        self.assertValid(root(a={"type": "number", "minimum": 1, "maximum": 1}))
        self.assertValid(root(a={"type": "integer", "minimum": 1, "maximum": 1}))
        # 小数の限りでも、間に整数（1）があればよい
        self.assertValid(root(a={"type": "integer", "minimum": 0.5, "maximum": 1.0}))
        self.assertValid(root(a={"type": "integer", "exclusiveMinimum": 1, "exclusiveMaximum": 3}))
        # 小数の「より小さい・より大きい」の限りでも、間に整数（2）があればよい
        self.assertValid(root(a={"type": "integer", "exclusiveMinimum": 1, "exclusiveMaximum": 2.5}))
        self.assertValid(root(a={"type": "integer", "exclusiveMinimum": 1.5, "exclusiveMaximum": 2.5}))
        self.assertValid(root(a={"type": "integer", "exclusiveMinimum": -2.5, "exclusiveMaximum": -1}))

    def test_shared_subschema_is_not_a_cycle(self):
        shared = string()
        self.assertValid(root(a=shared, b=shared, c=array(shared)))

    def test_does_not_mutate(self):
        schema = copy.deepcopy(EVERYTHING)
        self.assertValid(schema)
        self.assertEqual(schema, EVERYTHING)

    def test_special_property_names(self):
        self.assertValid(root(**{"a-b": string(), "": string(), "名前": string()}))


# MARK: - check_schema：決まりごとに、合わないスキーマ


class CheckSchemaRuleTest(unittest.TestCase):
    def assertRejected(self, schema, *fragments):
        """誤りが出て、どれか1つの文に fragments がすべて入っている"""
        errors = check_schema(schema)
        self.assertTrue(errors, "誤りが出ませんでした")
        for error in errors:
            self.assertIsInstance(error, str)
        if fragments:
            self.assertTrue(any(all(f in error for f in fragments) for error in errors),
                            f"{fragments} を含む文がありません: {errors}")
        return errors

    # ---- いちばん外

    def test_root_must_be_dict(self):
        for value in ([], None, "object", ("type", "object"), 1):
            with self.subTest(value=value):
                self.assertRejected(value, "#:", "dict")

    def test_root_type_must_be_object(self):
        self.assertRejected(array(string()), "#/type", "object")
        self.assertRejected(string(), "#/type", "object")

    def test_type_required(self):
        self.assertRejected({"properties": {}, "additionalProperties": False}, "#:", "type")
        self.assertRejected(root(a={"maxLength": 3}), "#/properties/a", "type がありません")

    def test_type_must_be_single_string(self):
        self.assertRejected(root(a={"type": ["string", "null"], "maxLength": 3}), "#/properties/a/type", "1つの文字列")

    def test_unknown_type_name(self):
        for name in ("float", "Object", "int", "any"):
            with self.subTest(name=name):
                self.assertRejected(root(a={"type": name}), "#/properties/a/type", name)

    def test_unknown_keyword(self):
        for keyword in ("format", "$ref", "oneOf", "patternProperties", "maxProperties", "nullable"):
            with self.subTest(keyword=keyword):
                self.assertRejected(root(a=string(**{keyword: 1})), "#/properties/a", keyword)

    def test_non_string_key(self):
        self.assertRejected(root(a={"type": "string", "maxLength": 3, 1: "x"}), "#/properties/a", "キー")

    def test_keyword_not_for_type(self):
        cases = [
            integer(maxLength=3), string(minimum=0), obj(items=string()), array(string(), properties={}),
            string(required=["a"]), {"type": "boolean", "maxItems": 1}, {"type": "null", "pattern": "a"},
            number(uniqueItems=True),
        ]
        for schema in cases:
            with self.subTest(schema=schema):
                self.assertRejected(root(a=schema), "#/properties/a/", "使えません")

    # ---- object

    def test_object_needs_properties(self):
        self.assertRejected({"type": "object", "additionalProperties": False}, "#:", "properties")
        self.assertRejected(root(a={"type": "object", "additionalProperties": False}), "#/properties/a", "properties")

    def test_properties_must_be_dict(self):
        self.assertRejected(obj(properties=[string()]), "#/properties", "dict")

    def test_property_schema_must_be_dict(self):
        self.assertRejected(root(a="string"), "#/properties/a", "dict")
        self.assertRejected(root(a=None), "#/properties/a", "dict")

    def test_additional_properties_must_be_false(self):
        self.assertRejected({"type": "object", "properties": {}}, "#:", "additionalProperties")
        for value in (True, 0, None, {}, string(), "false"):
            with self.subTest(value=value):
                self.assertRejected({"type": "object", "properties": {}, "additionalProperties": value},
                                    "#/additionalProperties", "false")
        nested = root(a={"type": "object", "properties": {}, "additionalProperties": True})
        self.assertRejected(nested, "#/properties/a/additionalProperties")

    def test_required_rules(self):
        props = {"a": string(), "b": string()}
        self.assertRejected(obj(props, required=("a",)), "#/required", "list")
        self.assertRejected(obj(props, required="a"), "#/required", "list")
        self.assertRejected(obj(props, required=[1]), "#/required/0", "文字列")
        self.assertRejected(obj(props, required=["a", "a"]), "#/required/1", "2回")
        self.assertRejected(obj(props, required=["c"]), "#/required/0", "properties にありません")
        self.assertEqual(check_schema(obj(props, required=[])), [])
        self.assertEqual(check_schema(obj(props, required=["b", "a"])), [])

    # ---- array

    def test_array_needs_items(self):
        self.assertRejected(root(a={"type": "array", "maxItems": 3}), "#/properties/a", "items")

    def test_array_needs_max_items(self):
        self.assertRejected(root(a={"type": "array", "items": string()}), "#/properties/a", "maxItems")
        self.assertRejected(root(a={"type": "array", "items": string(), "minItems": 1}), "#/properties/a", "maxItems")

    def test_items_must_be_a_valid_schema(self):
        self.assertRejected(root(a=array("string")), "#/properties/a/items", "dict")
        self.assertRejected(root(a=array({"type": "string"})), "#/properties/a/items", "maxLength")

    def test_counts_must_be_non_negative_integers(self):
        for value in (-1, True, 3.0, "3", None):
            with self.subTest(value=value):
                self.assertRejected(root(a={"type": "array", "items": string(), "maxItems": value}),
                                    "#/properties/a/maxItems", "0 以上の整数")
                self.assertRejected(root(a=array(string(), minItems=value)), "#/properties/a/minItems", "0 以上の整数")

    def test_counts_must_be_json_safe_integers(self):
        # JSON で正しく扱える整数（2**53 - 1）まで。とても大きな整数は、文にも JSON にもできない
        safe = 2 ** 53 - 1
        self.assertEqual(check_schema(root(a={"type": "array", "items": string(), "maxItems": safe})), [])
        self.assertEqual(check_schema(root(a={"type": "string", "minLength": safe, "maxLength": safe})), [])
        for too_big in (safe + 1, 10 ** 5000):
            with self.subTest(bits=too_big.bit_length()):
                self.assertRejected(root(a={"type": "array", "items": string(), "maxItems": too_big}),
                                    "#/properties/a/maxItems", "大きすぎ")
                self.assertRejected(root(a=array(string(), minItems=too_big, maxItems=too_big)),
                                    "#/properties/a/minItems", "大きすぎ")
                self.assertRejected(root(a={"type": "string", "maxLength": too_big}), "#/properties/a/maxLength",
                                    "大きすぎ")
                self.assertRejected(root(a={"type": "string", "minLength": too_big, "maxLength": too_big}),
                                    "#/properties/a/minLength", "大きすぎ")

    def test_huge_counts_with_a_default_do_not_crash(self):
        # default を確かめるときの文で、4300 桁を超える整数を文字にしようとして落ちないこと
        huge = 10 ** 5000
        for schema in (
            root(a={"type": "array", "items": string(), "minItems": huge, "maxItems": huge, "default": []}),
            root(a={"type": "string", "minLength": huge, "maxLength": huge, "default": "x"}),
        ):
            with self.subTest(type=schema["properties"]["a"]["type"]):
                errors = check_schema(schema)
                self.assertTrue(errors)
                self.assertTrue(all(type(error) is str and len(error) < 300 for error in errors), errors)

    def test_min_items_not_above_max_items(self):
        self.assertRejected(root(a=array(string(), minItems=6)), "#/properties/a", "minItems")
        self.assertEqual(check_schema(root(a=array(string(), minItems=5))), [])

    def test_unique_items_must_be_bool(self):
        for value in (1, 0, "yes", None):
            with self.subTest(value=value):
                self.assertRejected(root(a=array(string(), uniqueItems=value)), "#/properties/a/uniqueItems")

    # ---- string

    def test_string_needs_max_length_or_enum(self):
        self.assertRejected(root(a={"type": "string"}), "#/properties/a", "maxLength")
        self.assertRejected(root(a={"type": "string", "minLength": 1, "pattern": "^a$"}), "#/properties/a", "maxLength")

    def test_length_rules(self):
        self.assertRejected(root(a=string(minLength=11)), "#/properties/a", "minLength")
        for value in (-1, 2.0, True, "3"):
            with self.subTest(value=value):
                self.assertRejected(root(a={"type": "string", "maxLength": value}), "#/properties/a/maxLength")

    def test_pattern_must_compile(self):
        for pattern in ("[a-", "(", "a{2,1}", "\\", "*a"):
            with self.subTest(pattern=pattern):
                self.assertRejected(root(a=string(pattern=pattern)), "#/properties/a/pattern")

    def test_pattern_must_be_string(self):
        self.assertRejected(root(a=string(pattern=3)), "#/properties/a/pattern", "文字列")

    def test_inline_flags_rejected(self):
        for pattern in ("(?i)abc", "(?m)^a$", "a(?s:.)b", "(?-i:a)"):
            with self.subTest(pattern=pattern):
                self.assertRejected(root(a=string(pattern=pattern)), "#/properties/a/pattern", "フラグ")
        # フラグでない (? は使える
        self.assertEqual(check_schema(root(a=string(pattern="(?:ab)+(?=c)(?!d)"))), [])
        # 文字として書いた (?i) は使える
        self.assertEqual(check_schema(root(a=string(pattern="\\(?i\\)"))), [])

    # ---- integer・number

    def test_numbers_need_both_bounds_or_enum(self):
        for type_name in ("integer", "number"):
            cases = [
                {"type": type_name},
                {"type": type_name, "minimum": 0},
                {"type": type_name, "maximum": 0},
                {"type": type_name, "exclusiveMinimum": 0},
                {"type": type_name, "exclusiveMaximum": 0},
            ]
            for schema in cases:
                with self.subTest(schema=schema):
                    self.assertRejected(root(a=schema), "#/properties/a", "enum")

    def test_bounds_must_be_finite_numbers(self):
        for value in (math.nan, math.inf, -math.inf, True, "1", None, [1], np.float64(1.0)):
            for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                with self.subTest(value=value, key=key):
                    schema = {"type": "number", "minimum": 0, "maximum": 10}
                    if key.startswith("exclusive"):
                        schema.pop("minimum" if key == "exclusiveMinimum" else "maximum")
                    schema[key] = value
                    self.assertRejected(root(a=schema), f"#/properties/a/{key}", "有限の数")

    def test_integer_bounds_must_be_json_safe(self):
        safe = 2 ** 53 - 1
        self.assertEqual(check_schema(root(a={"type": "integer", "minimum": -safe, "maximum": safe})), [])
        self.assertEqual(check_schema(root(a={"type": "number", "exclusiveMinimum": -safe, "exclusiveMaximum": safe})),
                         [])
        # 小数の限りは、有限なら大きくてもよい（小数は、いつも JSON に書ける）
        self.assertEqual(check_schema(root(a={"type": "number", "minimum": -1e308, "maximum": 1e308})), [])
        for too_big in (safe + 1, 10 ** 5000):
            for type_name in ("integer", "number"):
                cases = [
                    ("minimum", {"minimum": -too_big, "maximum": 0}),
                    ("exclusiveMinimum", {"exclusiveMinimum": -too_big, "maximum": 0}),
                    ("maximum", {"minimum": 0, "maximum": too_big}),
                    ("exclusiveMaximum", {"minimum": 0, "exclusiveMaximum": too_big}),
                ]
                for key, bounds in cases:
                    with self.subTest(bits=too_big.bit_length(), type=type_name, key=key):
                        errors = self.assertRejected(root(a={"type": type_name, **bounds}), f"#/properties/a/{key}",
                                                     "整数")
                        self.assertTrue(all(len(error) < 300 for error in errors), errors)

    def test_enum_and_const_integers_must_be_json_safe(self):
        safe = 2 ** 53 - 1
        self.assertEqual(check_schema(root(a={"type": "integer", "enum": [-safe, safe]})), [])
        self.assertEqual(check_schema(root(a={"type": "number", "const": safe})), [])
        for too_big in (safe + 1, -(10 ** 5000)):
            for type_name in ("integer", "number"):
                with self.subTest(bits=too_big.bit_length(), type=type_name):
                    self.assertRejected(root(a={"type": type_name, "enum": [1, too_big]}), "#/properties/a/enum/1",
                                        "整数")
                    self.assertRejected(root(a={"type": type_name, "const": too_big}), "#/properties/a/const", "整数")
        # 外側の const・default に入れても、葉の限りか enum で止まる
        leaf = {"type": "integer", "enum": [1, 2]}
        self.assertRejected(root(a=array(leaf, const=[10 ** 5000])), "#/properties/a/const")
        self.assertRejected(root(a=obj({"n": integer()}, default={"n": 10 ** 5000})), "#/properties/a/default")

    def test_inclusive_and_exclusive_bound_together(self):
        self.assertRejected(root(a=number(exclusiveMinimum=0)), "#/properties/a", "minimum と exclusiveMinimum")
        self.assertRejected(root(a=number(exclusiveMaximum=10)), "#/properties/a", "maximum と exclusiveMaximum")

    def test_empty_ranges(self):
        cases = [
            {"type": "number", "minimum": 2, "maximum": 1},
            {"type": "number", "exclusiveMinimum": 1, "maximum": 1},
            {"type": "number", "minimum": 1, "exclusiveMaximum": 1},
            {"type": "number", "exclusiveMinimum": 1, "exclusiveMaximum": 1},
            {"type": "integer", "minimum": 0.2, "maximum": 0.8},
            {"type": "integer", "exclusiveMinimum": 1, "exclusiveMaximum": 2},
            {"type": "integer", "exclusiveMinimum": 0.5, "maximum": 0.9},
            {"type": "integer", "minimum": 5, "maximum": 4},
        ]
        for schema in cases:
            with self.subTest(schema=schema):
                self.assertRejected(root(a=schema), "#/properties/a", "間に")

    # ---- enum・const・default・examples

    def test_enum_must_be_non_empty_list(self):
        for value in ([], "a", ("a",), None, {"a": 1}):
            with self.subTest(value=value):
                self.assertRejected(root(a={"type": "string", "maxLength": 3, "enum": value}), "#/properties/a/enum")

    def test_enum_duplicates(self):
        self.assertRejected(root(a={"type": "string", "enum": ["a", "b", "a"]}), "#/properties/a/enum/2", "enum/0")
        # JSON では 1 と 1.0 は同じ数
        self.assertRejected(root(a={"type": "number", "enum": [1, 1.0]}), "#/properties/a/enum/1", "同じ値")
        # オブジェクトは名前の順番に関係なく比べる
        schema = obj({"a": integer(), "b": integer()}, enum=[{"a": 1, "b": 2}, {"b": 2, "a": 1}])
        self.assertRejected(root(x=schema), "#/properties/x/enum/1", "同じ値")

    def test_bool_is_not_a_duplicate_of_a_number(self):
        errors = self.assertRejected(root(a={"type": "boolean", "enum": [True, 1]}), "#/properties/a/enum/1")
        self.assertFalse(any("同じ値" in error for error in errors), errors)

    def test_enum_values_must_match_schema(self):
        cases = [
            ({"type": "string", "enum": ["a", 1]}, "enum/1"),
            ({"type": "integer", "enum": [1, True]}, "enum/1"),
            ({"type": "integer", "enum": [1, 2.0]}, "enum/1"),
            ({"type": "string", "maxLength": 2, "enum": ["ab", "abc"]}, "enum/1"),
            ({"type": "string", "maxLength": 5, "pattern": "^a", "enum": ["ab", "ba"]}, "enum/1"),
            (integer(enum=[5, 11]), "enum/1"),
            ({"type": "number", "enum": [1, math.nan]}, "enum/1"),
        ]
        for schema, where in cases:
            with self.subTest(schema=schema):
                self.assertRejected(root(a=schema), f"#/properties/a/{where}", "合いません")

    def test_const_must_match_schema(self):
        self.assertRejected(root(a=integer(const="1")), "#/properties/a/const")
        self.assertRejected(root(a=integer(const=11)), "#/properties/a/const")
        self.assertRejected(root(a={"type": "string", "enum": ["a", "b"], "const": "c"}), "#/properties/a/const")

    def test_default_must_match_schema(self):
        cases = [
            number(default=11),
            number(default="1"),
            integer(default=1.0),
            string(default="x" * 11),
            string(pattern="^a", default="b"),
            obj({"a": integer()}, default={"a": 1, "b": 2}),
            obj({"a": integer()}, required=["a"], default={}),
            array(string(), default=["a"] * 6),
            array(string(), uniqueItems=True, default=["a", "a"]),
            {"type": "boolean", "default": 0},
            {"type": "null", "default": False},
            {"type": "string", "enum": ["a"], "default": "b"},
        ]
        for schema in cases:
            with self.subTest(schema=schema):
                self.assertRejected(root(a=schema), "#/properties/a/default", "初めの値")

    def test_root_default_checked(self):
        self.assertRejected(obj({"a": integer()}, default={"a": "x"}), "#/default", "$.a")

    def test_examples(self):
        self.assertRejected(root(a=integer(examples=3)), "#/properties/a/examples", "list")
        self.assertRejected(root(a=integer(examples=[1, 20])), "#/properties/a/examples/1", "例")

    def test_title_and_description_must_be_strings(self):
        self.assertRejected(root(a=integer(title=1)), "#/properties/a/title")
        self.assertRejected(root(a=integer(description=None)), "#/properties/a/description")
        self.assertRejected(obj(title=["x"]), "#/title")

    # ---- 場所の書き方・すべての誤り・落ちないこと

    def test_nested_pointer(self):
        schema = root(a=array(obj({"b": number(maximum="x")})))
        self.assertRejected(schema, "#/properties/a/items/properties/b/maximum")

    def test_pointer_escapes(self):
        self.assertRejected(root(**{"a/b~c": {"type": "string"}}), "#/properties/a~1b~0c:")

    def test_reports_all_errors(self):
        schema = root(a={"type": "string"}, b=integer(minimum="0"), c={"type": "array", "items": string()})
        errors = check_schema(schema)
        self.assertTrue(any(e.startswith("#/properties/a:") for e in errors), errors)
        self.assertTrue(any(e.startswith("#/properties/b/minimum:") for e in errors), errors)
        self.assertTrue(any(e.startswith("#/properties/c:") and "maxItems" in e for e in errors), errors)

    def test_cycle_is_an_error(self):
        schema = root()
        schema["properties"]["self"] = schema
        self.assertRejected(schema, "#/properties/self", "自分自身")
        items = array(string())
        items["items"] = items
        self.assertRejected(root(a=items), "自分自身")

    def test_too_deep(self):
        def nested(depth):
            schema = string()
            for _ in range(depth):
                schema = obj({"a": schema})
            return schema

        self.assertEqual(check_schema(nested(MAX_SCHEMA_DEPTH)), [])
        self.assertRejected(nested(MAX_SCHEMA_DEPTH + 1), "深すぎ")
        self.assertRejected(nested(5000), "深すぎ")

    def test_weird_objects_inside_schema_do_not_crash(self):
        for value in weird_values():
            for schema in (
                root(a=integer(enum=[1, value])),
                root(a=integer(default=value)),
                root(a=string(examples=[value])),
                root(a=string(title=value)),
                root(a={"type": value}),
                root(a=string(maxLength=[value])),
                root(a=number(minimum=value, maximum=-1)),
                root(a=array(value)),
                obj(required=[value]),
                obj(additionalProperties=value),
                obj(properties=value),
                root(a=value),
            ):
                with self.subTest(value=describe_value(value), schema=sorted(schema)):
                    errors = check_schema(schema)
                    self.assertTrue(errors)
                    self.assertTrue(all(type(error) is str for error in errors))
            # 数の言葉に、そのまま変わった物を書いたとき（JSON で扱える大きさの int だけが正しい）
            with self.subTest(value=describe_value(value), keyword="maxLength"):
                errors = check_schema(root(a=string(maxLength=value)))
                self.assertEqual(errors == [], type(value) is int and 0 <= value <= 2 ** 53 - 1)
                self.assertTrue(all(type(error) is str for error in errors))


# MARK: - validate


class ValidateTest(unittest.TestCase):
    def assertOneError(self, instance, schema, *fragments):
        errors = validate(instance, schema)
        self.assertEqual(len(errors), 1, errors)
        for fragment in fragments:
            self.assertIn(fragment, errors[0])
        return errors[0]

    def test_requires_a_checked_schema(self):
        with self.assertRaises(ValueError):
            validate({}, {"type": "object"})
        with self.assertRaises(ValueError):
            validate({}, None)

    def test_valid_instance(self):
        instance = {
            "size": 2.5, "count": 3, "name": "abc", "flag": False, "nothing": None, "tags": ["a", "b"],
            "nested": {"value": 1}, "choice": "b", "grid": [[1, 2], [2, 1]],
        }
        self.assertEqual(validate(instance, EVERYTHING), [])
        self.assertEqual(validate({}, EVERYTHING), [])

    def test_paths(self):
        schema = root(size=number(), items=array(obj({"name": string()})), **{"a-b": integer()})
        self.assertTrue(self.assertOneError([], schema).startswith("$: "))
        self.assertTrue(self.assertOneError({"size": "x"}, schema).startswith("$.size: "))
        items = [{"name": "a"}, {"name": "b"}, {"name": 3}]
        self.assertTrue(self.assertOneError({"items": items}, schema).startswith("$.items[2].name: "))
        self.assertTrue(self.assertOneError({"a-b": "x"}, schema).startswith('$["a-b"]: '))

    # ---- 型

    def test_integer(self):
        schema = root(n=integer(minimum=-10))
        for value in (0, 3, -5, 10):
            self.assertEqual(validate({"n": value}, schema), [], value)
        self.assertOneError({"n": True}, schema, "$.n", "真偽値")
        self.assertOneError({"n": False}, schema, "$.n", "真偽値")
        self.assertOneError({"n": 1.0}, schema, "$.n", "小数")
        self.assertOneError({"n": 1.5}, schema, "$.n", "小数")
        for value in ("3", None, [3], np.int64(3), Color.RED, math.nan):
            with self.subTest(value=value):
                self.assertOneError({"n": value}, schema, "$.n", "整数")

    def test_number(self):
        schema = root(x=number(minimum=-10))
        for value in (0, 1, 1.5, -10, 10, 10.0):
            self.assertEqual(validate({"x": value}, schema), [], value)
        self.assertOneError({"x": True}, schema, "$.x", "真偽値")
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                self.assertOneError({"x": value}, schema, "$.x", "有限の数")
        for value in ("1", None, np.float64(1.5), decimal.Decimal("1"), fractions.Fraction(1, 2), 1j):
            with self.subTest(value=value):
                self.assertOneError({"x": value}, schema, "$.x", "数")
        message = self.assertOneError({"x": np.float64(1.5)}, schema)
        self.assertIn("float64", message)

    def test_boolean_and_null(self):
        schema = root(flag={"type": "boolean"}, nothing={"type": "null"})
        self.assertEqual(validate({"flag": True, "nothing": None}, schema), [])
        for value in (1, 0, "true", None, np.bool_(True)):
            with self.subTest(value=value):
                self.assertOneError({"flag": value}, schema, "$.flag", "真偽値")
        for value in (0, "", False, "null", []):
            with self.subTest(value=value):
                self.assertOneError({"nothing": value}, schema, "$.nothing", "null")

    # ---- 数の範囲

    def test_inclusive_bounds(self):
        schema = root(x={"type": "number", "minimum": 1, "maximum": 2})
        self.assertEqual(validate({"x": 1}, schema), [])
        self.assertEqual(validate({"x": 2.0}, schema), [])
        self.assertOneError({"x": 0.999}, schema, "小さすぎ", "1 以上")
        self.assertOneError({"x": 2.0000000000000004}, schema, "大きすぎ", "2 以下")

    def test_exclusive_bounds(self):
        schema = root(x={"type": "number", "exclusiveMinimum": 1, "exclusiveMaximum": 2})
        self.assertEqual(validate({"x": 1.5}, schema), [])
        self.assertOneError({"x": 1}, schema, "小さすぎ", "1 より大きく")
        self.assertOneError({"x": 1.0}, schema, "小さすぎ")
        self.assertOneError({"x": 2}, schema, "大きすぎ", "2 より小さく")
        schema = root(n={"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 3})
        self.assertEqual(validate({"n": 1}, schema), [])
        self.assertEqual(validate({"n": 2}, schema), [])
        self.assertOneError({"n": 0}, schema)
        self.assertOneError({"n": 3}, schema)

    def test_huge_integers(self):
        schema = root(n=integer(), x={"type": "number", "minimum": -1e308, "maximum": 1e308})
        message = self.assertOneError({"n": 10 ** 5000}, schema, "$.n", "大きすぎ", "とても大きな整数")
        self.assertLess(len(message), 200)
        self.assertOneError({"x": 10 ** 400}, schema, "$.x", "大きすぎ")
        self.assertOneError({"x": -(10 ** 400)}, schema, "$.x", "小さすぎ")
        self.assertEqual(validate({"x": 10 ** 300}, schema), [])

    # ---- 文字列

    def test_string_length_counts_characters(self):
        schema = root(s={"type": "string", "minLength": 2, "maxLength": 3})
        self.assertEqual(validate({"s": "あいう"}, schema), [])
        self.assertEqual(validate({"s": "左腕"}, schema), [])
        self.assertOneError({"s": "あいうえ"}, schema, "長すぎ", "4 文字")
        self.assertOneError({"s": "あ"}, schema, "短すぎ")

    def test_pattern_is_search(self):
        schema = root(s=string(pattern="b"))
        self.assertEqual(validate({"s": "abc"}, schema), [])
        self.assertOneError({"s": "xyz"}, schema, "形")

    def test_pattern_dollar_is_end_of_string(self):
        # Python の $ は最後の改行の前にも合うが、JSON Schema では文字列の終わりだけ
        schema = root(s=string(pattern="^[a-z]+$"))
        self.assertEqual(validate({"s": "abc"}, schema), [])
        self.assertOneError({"s": "abc\n"}, schema, "$.s", "形")
        self.assertOneError({"s": "ab c"}, schema)
        # [ ] の中の $ と、\$ は文字のまま
        schema = root(s=string(pattern="^[$]\\$$"))
        self.assertEqual(validate({"s": "$$"}, schema), [])
        self.assertOneError({"s": "$$\n"}, schema)

    def test_pattern_digits_are_ascii(self):
        schema = root(s=string(pattern="^\\d+$"))
        self.assertEqual(validate({"s": "123"}, schema), [])
        self.assertOneError({"s": "１２３"}, schema, "形")

    def test_pattern_skipped_when_too_long(self):
        schema = root(s={"type": "string", "maxLength": 3, "pattern": "^a+$"})
        self.assertOneError({"s": "bbbbb"}, schema, "長すぎ")

    def test_lone_surrogate_rejected(self):
        schema = root(s=string())
        self.assertOneError({"s": "a\ud800"}, schema, "サロゲート")

    # ---- オブジェクト

    def test_object_rules(self):
        schema = root(a=integer(), b=integer())
        schema["required"] = ["a"]
        self.assertEqual(validate({"a": 1}, schema), [])
        self.assertOneError({"b": 1}, schema, "$.a", "必要")
        self.assertOneError({"a": 1, "extra": 1}, schema, "$.extra", "決められていない")
        self.assertOneError({"a": 1, 5: 1}, schema, "$:", "文字列")
        self.assertOneError(OrderedDict(a=1), schema, "$:", "OrderedDict")
        self.assertOneError({"a": 1, "あ": 1}, schema, '$["あ"]')

    def test_empty_object_schema_accepts_only_empty(self):
        schema = root()
        self.assertEqual(validate({}, schema), [])
        self.assertOneError({"x": 1}, schema, "$.x", "なし")

    # ---- 配列

    def test_array_rules(self):
        schema = root(v=array(integer(), minItems=2))
        self.assertEqual(validate({"v": [1, 2]}, schema), [])
        self.assertOneError({"v": (1, 2)}, schema, "$.v", "tuple")
        self.assertOneError({"v": [1]}, schema, "$.v", "少なすぎ")
        errors = validate({"v": ["a", 2, 3.0]}, schema)
        self.assertEqual([e.split(":")[0] for e in errors], ["$.v[0]", "$.v[2]"])

    def test_too_many_items_are_not_checked_one_by_one(self):
        schema = root(v=array(integer()))
        self.assertOneError({"v": ["x"] * 1000}, schema, "$.v", "多すぎ", "1000 個")
        self.assertEqual(len(validate({"v": ["x"] * 5}, schema)), 5)

    def test_unique_items(self):
        schema = root(v=array(number(), uniqueItems=True))
        self.assertEqual(validate({"v": [1, 2, 3]}, schema), [])
        message = self.assertOneError({"v": [1, 2, 1.0]}, schema, "$.v[2]", "$.v[0]", "同じ値")
        self.assertTrue(message.startswith("$.v[2]:"))
        self.assertEqual(validate({"v": [1, 2, 1.0]}, root(v=array(number()))), [])

    def test_unique_items_with_objects_and_arrays(self):
        schema = root(v=array(obj({"a": integer(), "b": array(integer())}), uniqueItems=True))
        self.assertEqual(validate({"v": [{"a": 1}, {"a": 2}, {"b": [1]}, {"b": [1, 1]}]}, schema), [])
        self.assertOneError({"v": [{"a": 1, "b": [2]}, {"b": [2], "a": 1}]}, schema, "$.v[1]", "同じ値")
        self.assertOneError({"v": [{"b": [1, 2]}, {"a": 1}, {"b": [1, 2]}]}, schema, "$.v[2]", "$.v[0]")
        grid = root(g=array(array(integer(), uniqueItems=False), uniqueItems=True))
        self.assertEqual(validate({"g": [[1, 2], [2, 1]]}, grid), [])
        self.assertOneError({"g": [[1, 2], [1, 2]]}, grid, "$.g[1]")

    def test_unique_items_bool_is_not_number(self):
        schema = root(v=array(array({"type": "boolean"}), uniqueItems=True))
        errors = validate({"v": [[True], [1]]}, schema)
        self.assertEqual(len(errors), 1, errors)
        self.assertTrue(errors[0].startswith("$.v[1][0]:"))
        self.assertOneError({"v": [[True], [True]]}, schema, "同じ値")

    def test_unique_items_with_unhashable_and_weird_items(self):
        schema = root(v={"type": "array", "items": integer(), "maxItems": 100, "uniqueItems": True})
        items = [{1}, {"a": [1]}, [[1]], RaisingEverything(), RaisingEverything(), 1, 1]
        errors = validate({"v": items}, schema)
        self.assertEqual(len(errors), 6, errors)
        self.assertIn("$.v[6]: $.v[5] と同じ値", errors[-1])

    # ---- enum・const

    def test_enum_is_type_strict(self):
        schema = root(n={"type": "integer", "enum": [1, 2]})
        self.assertEqual(validate({"n": 1}, schema), [])
        self.assertOneError({"n": True}, schema, "真偽値")
        self.assertOneError({"n": 1.0}, schema, "小数")
        self.assertOneError({"n": 3}, schema, "選べる値ではありません", "1, 2")
        schema = root(b={"type": "boolean", "enum": [True]})
        self.assertOneError({"b": False}, schema, "選べる値ではありません", "true")
        self.assertOneError({"b": 1}, schema, "真偽値")
        schema = root(s={"type": "string", "enum": ["a"]})
        self.assertOneError({"s": "A"}, schema, "選べる値ではありません")

    def test_enum_numbers_compare_as_json_numbers(self):
        schema = root(x={"type": "number", "enum": [1, 2.5]})
        self.assertEqual(validate({"x": 1.0}, schema), [])
        self.assertEqual(validate({"x": 2.5}, schema), [])
        self.assertOneError({"x": 2}, schema, "選べる値ではありません")

    def test_enum_of_objects(self):
        schema = root(o=obj({"a": integer(), "b": integer()}, enum=[{"a": 1, "b": 2}]))
        self.assertEqual(validate({"o": {"b": 2, "a": 1}}, schema), [])
        self.assertOneError({"o": {"a": 1}}, schema, "選べる値ではありません")

    def test_const(self):
        schema = root(n=integer(const=0))
        self.assertEqual(validate({"n": 0}, schema), [])
        self.assertOneError({"n": False}, schema, "真偽値")
        self.assertOneError({"n": 1}, schema, "0 にしてください")

    # ---- すべての誤り・落ちないこと

    def test_reports_all_errors_in_order(self):
        instance = {"size": True, "count": 11, "tags": ["a", "a"], "extra": 1, "nested": {}}
        errors = validate(instance, EVERYTHING)
        self.assertEqual([e.split(":")[0] for e in errors],
                         ["$.size", "$.count", "$.tags[1]", "$.nested.value", "$.extra"])

    def test_does_not_mutate_instance(self):
        instance = {"tags": ["b", "a"], "nested": {"value": 20}, "extra": [1]}
        before = copy.deepcopy(instance)
        validate(instance, EVERYTHING)
        self.assertEqual(instance, before)

    def test_arbitrary_python_objects_do_not_crash(self):
        armed = {"value": False}

        class CollidingKey:
            """"size" と同じハッシュで、比べると例外を出す（辞書を作ったあとで仕掛ける）"""

            def __hash__(self):
                return hash("size")

            def __eq__(self, other):
                if armed["value"]:
                    raise RuntimeError("eq")
                return self is other

        colliding = {"size": 1.0, CollidingKey(): 1}
        armed["value"] = True
        try:
            for value in [*weird_values(), colliding]:
                instances = [
                    value, {"size": value}, {"count": value}, {"name": value}, {"flag": value}, {"nothing": value},
                    {"tags": value}, {"tags": [value, value]}, {"nested": value}, {"nested": {"value": value}},
                    {"choice": value}, {"grid": [[value], [value]]}, {"grid": value},
                ]
                for instance in instances:
                    with self.subTest(value=describe_value(value)):
                        errors = validate(instance, EVERYTHING)
                        self.assertTrue(errors)
                        self.assertTrue(all(type(error) is str for error in errors))
        finally:
            armed["value"] = False


class DescribeValueTest(unittest.TestCase):
    def test_short_descriptions(self):
        self.assertEqual(describe_value(None), "null")
        self.assertEqual(describe_value(True), "真偽値 true")
        self.assertEqual(describe_value(3), "整数 3")
        self.assertEqual(describe_value(1.5), "小数 1.5")
        self.assertEqual(describe_value("abc"), '文字列 "abc"')
        self.assertEqual(describe_value([1, 2]), "配列（2 個）")
        self.assertIn("float64", describe_value(np.float64(1.0)))
        self.assertIn("…", describe_value("x" * 1000))
        self.assertLess(len(describe_value("x" * 1000)), 60)

    def test_never_raises(self):
        for value in weird_values():
            with self.subTest(value=type(value).__name__):
                self.assertIsInstance(describe_value(value), str)


if __name__ == "__main__":
    unittest.main()
