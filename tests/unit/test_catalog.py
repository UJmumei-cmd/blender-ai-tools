"""A8 道具の目録：道具の説明・目録・確認の決まり・LLM の返事の読み方を確かめる"""

import copy
import dataclasses
import hashlib
import itertools
import json
import unittest

from blender_ai_tools.catalog import (
    MAX_TOOL_CALL_LENGTH,
    Catalog,
    CatalogError,
    ConfirmationRequiredError,
    DuplicateToolError,
    Effect,
    InvalidParamsError,
    ToolCall,
    ToolCallError,
    ToolSpec,
    UnknownToolError,
    parse_tool_call,
)
from blender_ai_tools.catalog.schema import validate

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["armature"],
    "properties": {
        "armature": {"type": "string", "minLength": 1, "maxLength": 63},
        "threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
        "names": {"type": "array", "items": {"type": "string", "maxLength": 32}, "maxItems": 100,
                  "uniqueItems": True},
        "options": {"type": "object", "additionalProperties": False,
                    "properties": {"strict": {"type": "boolean"}}},
    },
}
GOOD_PARAMS = {"armature": "Armature", "threshold": 0.25, "names": ["左腕", "右腕"], "options": {"strict": True}}


def make_spec(**overrides):
    fields = {
        "id": "bones.map_names",
        "version": "1",
        "feature": "A3",
        "category": "bones",
        "title": "骨の名前の対応",
        "summary": "骨の名前を、MMD 準標準の名前に対応させます。",
        "effect": Effect.READ,
        "heavy": False,
        "input_schema": SCHEMA,
        "checks_before": ("アーマチュアがあること",),
        "measures_after": ("対応できなかった骨の数",),
    }
    fields.update(overrides)
    return ToolSpec(**fields)


class Recorder:
    """呼ばれた値を覚える道具の関数"""

    def __init__(self, result="done"):
        self.calls = []
        self.result = result

    def __call__(self, params):
        self.calls.append(params)
        return self.result


# MARK: - 道具の働き


class EffectTest(unittest.TestCase):
    def test_values_and_labels(self):
        self.assertEqual([e.value for e in Effect], ["read", "modify", "delete"])
        self.assertEqual([e.label for e in Effect], ["見るだけ", "変更", "消す"])
        self.assertIs(Effect("delete"), Effect.DELETE)


# MARK: - 道具の説明


class ToolSpecTest(unittest.TestCase):
    def test_valid(self):
        spec = make_spec()
        self.assertEqual(spec.id, "bones.map_names")
        self.assertEqual(spec.input_schema, SCHEMA)
        self.assertEqual(spec.checks_before, ("アーマチュアがあること",))
        for good_id in ("a.b", "bones.map_names", "measure.mesh.penetration", "a1.b_2"):
            self.assertEqual(make_spec(id=good_id).id, good_id)
        for version in ("1", "1.2", "0", "10.0.3"):
            self.assertEqual(make_spec(version=version).version, version)
        for feature in ("A3", "A15", "M4"):
            self.assertEqual(make_spec(feature=feature).feature, feature)

    def test_schema_is_copied(self):
        schema = copy.deepcopy(SCHEMA)
        spec = make_spec(input_schema=schema)
        schema["properties"]["threshold"]["maximum"] = 100
        schema["required"].append("names")
        self.assertEqual(spec.input_schema, SCHEMA)
        self.assertIsNot(spec.input_schema, schema)

    def test_lists_become_tuples(self):
        checks = ["アーマチュアがあること"]
        spec = make_spec(checks_before=checks, measures_after=[])
        self.assertEqual(spec.checks_before, ("アーマチュアがあること",))
        self.assertEqual(spec.measures_after, ())
        checks.append("ほか")
        self.assertEqual(spec.checks_before, ("アーマチュアがあること",))

    def test_defaults(self):
        spec = ToolSpec(id="a.b", version="1", feature="A1", category="a", title="名前", summary="説明",
                        effect=Effect.READ, heavy=False, input_schema=SCHEMA)
        self.assertEqual(spec.checks_before, ())
        self.assertEqual(spec.measures_after, ())

    def test_frozen_and_hashable(self):
        spec = make_spec()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.title = "別"
        self.assertEqual(hash(spec), hash(make_spec()))
        self.assertEqual(spec, make_spec())
        self.assertNotEqual(spec, make_spec(version="2"))

    def assertRejected(self, field_name, value):
        with self.assertRaises(ValueError) as caught:
            make_spec(**{field_name: value})
        self.assertIn(field_name, str(caught.exception))

    def test_rejects_bad_id(self):
        for value in ("bones", "Bones.map", "bones.", ".bones", "bones..map", "bones.map-names", "bones.Map",
                      "bones.map_names\n", "1bones.map", "bones.1map", "ボーン.map", "bones. map", "", None, 3):
            with self.subTest(value=value):
                self.assertRejected("id", value)

    def test_rejects_bad_version(self):
        for value in ("", "1.", ".1", "v1", "01", "1.02", "1.2\n", "1 ", "１", 1, 1.2, None):
            with self.subTest(value=value):
                self.assertRejected("version", value)

    def test_rejects_bad_feature(self):
        for value in ("a3", "A", "A3b", "AA3", "3", "A 3", "Ａ3", "A3\n", "", None):
            with self.subTest(value=value):
                self.assertRejected("feature", value)

    def test_rejects_bad_category(self):
        for value in ("", "bad-cat", "日本", "a b", "1abc", None, 3):
            with self.subTest(value=value):
                self.assertRejected("category", value)

    def test_rejects_bad_texts(self):
        for field_name in ("title", "summary"):
            for value in ("", "   ", None, 1, ["名前"], "a\ud800"):
                with self.subTest(field=field_name, value=value):
                    self.assertRejected(field_name, value)
        self.assertRejected("title", "1行目\n2行目")
        self.assertEqual(make_spec(summary="1行目\n2行目").summary, "1行目\n2行目")

    def test_rejects_bad_effect_and_heavy(self):
        for value in ("read", None, 0):
            with self.subTest(value=value):
                self.assertRejected("effect", value)
        for value in (1, 0, "yes", None):
            with self.subTest(value=value):
                self.assertRejected("heavy", value)

    def test_rejects_bad_schema(self):
        for value in (None, [], {"type": "object"}, {**SCHEMA, "additionalProperties": True}, {**SCHEMA, "x": 1}):
            with self.subTest(value=value):
                self.assertRejected("input_schema", value)

    def test_rejects_bad_check_lists(self):
        for field_name in ("checks_before", "measures_after"):
            for value in ("abc", None, ("ok", ""), ("ok", 3), ("同じ", "同じ"), {"a"}):
                with self.subTest(field=field_name, value=value):
                    self.assertRejected(field_name, value)

    def test_reports_all_problems(self):
        with self.assertRaises(ValueError) as caught:
            make_spec(version="v1", heavy=1, title="")
        message = str(caught.exception)
        for name in ("version", "heavy", "title"):
            self.assertIn(name, message)


# MARK: - 目録


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog()
        self.func = Recorder()
        self.catalog.register(make_spec(), self.func)

    def test_register_and_get(self):
        self.assertEqual(len(self.catalog), 1)
        self.assertIn("bones.map_names", self.catalog)
        self.assertNotIn("bones.other", self.catalog)
        self.assertEqual(self.catalog.get("bones.map_names"), make_spec())

    def test_duplicate_id(self):
        other = Recorder()
        with self.assertRaises(DuplicateToolError) as caught:
            self.catalog.register(make_spec(version="2"), other)
        self.assertIsInstance(caught.exception, CatalogError)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertEqual(len(self.catalog), 1)
        self.assertEqual(self.catalog.get("bones.map_names").version, "1")
        self.catalog.call("bones.map_names", GOOD_PARAMS)
        self.assertEqual(len(self.func.calls), 1)
        self.assertEqual(other.calls, [])

    def test_register_rejects_bad_arguments(self):
        with self.assertRaises(TypeError):
            self.catalog.register(make_spec(id="a.b"), "not callable")
        with self.assertRaises(TypeError):
            self.catalog.register(make_spec(id="a.b"), None)
        with self.assertRaises(TypeError):
            self.catalog.register({"id": "a.b"}, Recorder())
        self.assertNotIn("a.b", self.catalog)
        self.assertEqual(len(self.catalog), 1)

    def test_register_rechecks_a_mutated_schema(self):
        spec = make_spec(id="a.b")
        spec.input_schema["additionalProperties"] = True
        with self.assertRaises(ValueError):
            self.catalog.register(spec, Recorder())
        self.assertNotIn("a.b", self.catalog)

    def test_register_rechecks_every_field(self):
        class Unchecked(ToolSpec):
            def __post_init__(self):  # 確かめを飛ばしてしまう子クラス
                pass

        func = Recorder()
        fields = {"id": "a.b", "version": "1", "feature": "A1", "category": "a", "title": "名前", "summary": "説明",
                  "effect": Effect.READ, "heavy": False, "input_schema": SCHEMA}
        for name, value in (("effect", "read"), ("heavy", "no"), ("id", ["a.b"]), ("title", ""),
                            ("checks_before", "abc")):
            with self.subTest(field=name):
                with self.assertRaises(ValueError) as caught:
                    self.catalog.register(Unchecked(**{**fields, name: value}), func)
                self.assertIn(name, str(caught.exception))
        # 作ったあとで書き換えた説明も断る
        spec = make_spec(id="a.b")
        object.__setattr__(spec, "effect", "delete")
        with self.assertRaises(ValueError):
            self.catalog.register(spec, func)
        self.assertNotIn("a.b", self.catalog)
        self.assertEqual(len(self.catalog), 1)
        # 親の確かめを使う子クラスは載せられる
        @dataclasses.dataclass(frozen=True)
        class Tagged(ToolSpec):
            note: str = ""

        self.catalog.register(Tagged(**fields, note="印"), func)
        self.assertEqual(self.catalog.call("a.b", GOOD_PARAMS), "done")

    def test_every_registered_schema_can_be_listed(self):
        # JSON で正しく扱えない大きな整数のスキーマは、説明の時点で断る（一覧と指紋が作れなくなるため）
        safe = 2 ** 53 - 1
        for prop in ({"type": "integer", "minimum": 0, "maximum": 10 ** 5000},
                     {"type": "integer", "enum": [10 ** 5000]},
                     {"type": "string", "maxLength": 10 ** 5000},
                     {"type": "array", "items": {"type": "null"}, "maxItems": safe + 1}):
            schema = {"type": "object", "properties": {"x": prop}, "additionalProperties": False}
            with self.subTest(prop=prop["type"]):
                with self.assertRaises(ValueError):
                    make_spec(id="a.b", input_schema=schema)
        # 上限ちょうどは使えて、一覧と指紋が作れる
        schema = {"type": "object", "additionalProperties": False, "properties": {
            "n": {"type": "integer", "minimum": -safe, "maximum": safe},
            "s": {"type": "string", "maxLength": safe},
        }}
        self.catalog.register(make_spec(id="a.b", input_schema=schema), Recorder())
        self.assertRegex(self.catalog.fingerprint(), r"\A[0-9a-f]{64}\Z")
        self.assertEqual(json.loads(json.dumps(self.catalog.listing())), self.catalog.listing())

    def test_unknown_tool(self):
        with self.assertRaises(UnknownToolError) as caught:
            self.catalog.get("nope.tool")
        error = caught.exception
        self.assertIsInstance(error, KeyError)
        self.assertIsInstance(error, CatalogError)
        self.assertEqual(str(error), "目録にない道具です: nope.tool")
        self.assertEqual(error.tool_id, "nope.tool")
        for value in (None, ["bones.map_names"], 3, "bones.map_names "):
            with self.subTest(value=value):
                with self.assertRaises(UnknownToolError):
                    self.catalog.get(value)
                self.assertNotIn(value, self.catalog)

    def test_specs_and_categories_are_sorted(self):
        catalog = Catalog()
        for tool_id, category in (("mesh.fix", "mesh"), ("bones.map", "bones"), ("mesh.check", "mesh"),
                                  ("anim.bake", "anim")):
            catalog.register(make_spec(id=tool_id, category=category), Recorder())
        self.assertEqual([s.id for s in catalog.specs()], ["anim.bake", "bones.map", "mesh.check", "mesh.fix"])
        self.assertEqual([s.id for s in catalog.specs("mesh")], ["mesh.check", "mesh.fix"])
        self.assertEqual(catalog.specs("nothing"), ())
        self.assertIsInstance(catalog.specs(), tuple)
        self.assertEqual(catalog.categories(), ("anim", "bones", "mesh"))
        with self.assertRaises(TypeError):
            catalog.specs(1)

    def test_validate_call(self):
        self.assertEqual(self.catalog.validate_call("bones.map_names", GOOD_PARAMS), [])
        self.assertEqual(self.catalog.validate_call("nope.tool", GOOD_PARAMS), ["目録にない道具です: nope.tool"])
        self.assertEqual(len(self.catalog.validate_call(None, None)), 1)
        bad = {"threshold": 2, "extra": 1}
        errors = self.catalog.validate_call("bones.map_names", bad)
        self.assertEqual(errors, validate(bad, SCHEMA))
        self.assertEqual([e.split(":")[0] for e in errors], ["$.armature", "$.threshold", "$.extra"])


class ConfirmationTest(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog()
        self.funcs = {}
        for effect in Effect:
            self.funcs[effect] = Recorder(result=effect.value)
            self.catalog.register(make_spec(id=f"tools.{effect.value}", effect=effect), self.funcs[effect])

    def test_needs_confirmation_matrix(self):
        expected = {
            (Effect.READ, False): False, (Effect.READ, True): False,
            (Effect.MODIFY, False): True, (Effect.MODIFY, True): False,
            (Effect.DELETE, False): True, (Effect.DELETE, True): True,
        }
        for (effect, plan_approved), needed in expected.items():
            with self.subTest(effect=effect, plan_approved=plan_approved):
                self.assertIs(self.catalog.needs_confirmation(f"tools.{effect.value}", plan_approved=plan_approved),
                              needed)
        self.assertTrue(self.catalog.needs_confirmation("tools.modify"))
        with self.assertRaises(UnknownToolError):
            self.catalog.needs_confirmation("nope.tool")

    def test_call_matrix(self):
        for effect, plan_approved, confirmed in itertools.product(Effect, (False, True), (False, True)):
            with self.subTest(effect=effect, plan_approved=plan_approved, confirmed=confirmed):
                func = self.funcs[effect]
                func.calls.clear()
                if effect is Effect.READ:
                    allowed = True
                elif effect is Effect.MODIFY:
                    allowed = confirmed or plan_approved
                else:
                    allowed = confirmed  # 消す操作は、計画を承認していても毎回確認する
                tool_id = f"tools.{effect.value}"
                if allowed:
                    result = self.catalog.call(tool_id, GOOD_PARAMS, confirmed=confirmed, plan_approved=plan_approved)
                    self.assertEqual(result, effect.value)
                    self.assertEqual(func.calls, [GOOD_PARAMS])
                else:
                    with self.assertRaises(ConfirmationRequiredError) as caught:
                        self.catalog.call(tool_id, GOOD_PARAMS, confirmed=confirmed, plan_approved=plan_approved)
                    self.assertIsInstance(caught.exception, CatalogError)
                    self.assertEqual(caught.exception.tool_id, tool_id)
                    self.assertIs(caught.exception.effect, effect)
                    self.assertEqual(func.calls, [])

    def test_safe_defaults(self):
        # 何も渡さないときは「承認も確認もない」として扱う
        for effect in (Effect.MODIFY, Effect.DELETE):
            with self.subTest(effect=effect):
                with self.assertRaises(ConfirmationRequiredError):
                    self.catalog.call(f"tools.{effect.value}", GOOD_PARAMS)
                self.assertEqual(self.funcs[effect].calls, [])
        self.assertEqual(self.catalog.call("tools.read", GOOD_PARAMS), "read")

    def test_confirmation_does_not_skip_validation(self):
        # 確認や承認があっても、スキーマに合わない値では動かさない
        bad = {"armature": "A", "threshold": 2}
        for effect in Effect:
            with self.subTest(effect=effect):
                with self.assertRaises(InvalidParamsError) as caught:
                    self.catalog.call(f"tools.{effect.value}", bad, confirmed=True, plan_approved=True)
                self.assertEqual([e.split(":")[0] for e in caught.exception.errors], ["$.threshold"])
                self.assertEqual(self.funcs[effect].calls, [])

    def test_flags_must_be_bool(self):
        for value in ("yes", "false", 1, 0, None):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    self.catalog.call("tools.delete", GOOD_PARAMS, confirmed=value)
                with self.assertRaises(TypeError):
                    self.catalog.call("tools.modify", GOOD_PARAMS, plan_approved=value)
                with self.assertRaises(TypeError):
                    self.catalog.needs_confirmation("tools.modify", plan_approved=value)
        self.assertEqual(self.funcs[Effect.DELETE].calls, [])
        self.assertEqual(self.funcs[Effect.MODIFY].calls, [])

    def test_order_of_checks(self):
        # 目録に無い → 値 → 確認 の順に調べる
        with self.assertRaises(UnknownToolError):
            self.catalog.call("nope.tool", {"bad": 1})
        with self.assertRaises(InvalidParamsError):
            self.catalog.call("tools.delete", {"bad": 1})
        with self.assertRaises(ConfirmationRequiredError):
            self.catalog.call("tools.delete", GOOD_PARAMS, plan_approved=True)
        self.assertEqual(self.funcs[Effect.DELETE].calls, [])


class CallTest(unittest.TestCase):
    def setUp(self):
        self.catalog = Catalog()

    def test_invalid_params(self):
        func = Recorder()
        self.catalog.register(make_spec(), func)
        bad = {"armature": "", "threshold": True, "names": ["a", "a"]}
        with self.assertRaises(InvalidParamsError) as caught:
            self.catalog.call("bones.map_names", bad)
        error = caught.exception
        self.assertIsInstance(error, ValueError)
        self.assertIsInstance(error, CatalogError)
        self.assertEqual(error.errors, validate(bad, SCHEMA))
        self.assertEqual(len(error.errors), 3)
        self.assertIn(error.errors[0], str(error))
        self.assertEqual(func.calls, [])
        for params in (None, [], "x", (("armature", "a"),)):
            with self.subTest(params=params):
                with self.assertRaises(InvalidParamsError):
                    self.catalog.call("bones.map_names", params)
        self.assertEqual(func.calls, [])

    def test_func_gets_a_deep_copy(self):
        def mutate(params):
            params["names"].append("足")
            params["options"]["strict"] = False
            params["new"] = 1
            return params

        self.catalog.register(make_spec(), mutate)
        params = copy.deepcopy(GOOD_PARAMS)
        result = self.catalog.call("bones.map_names", params)
        self.assertEqual(params, GOOD_PARAMS)
        self.assertIsNot(result, params)
        self.assertIsNot(result["names"], params["names"])
        self.assertEqual(result["names"], ["左腕", "右腕", "足"])

    def test_func_errors_pass_through(self):
        def broken(params):
            raise RuntimeError("こわれた")

        self.catalog.register(make_spec(), broken)
        with self.assertRaises(RuntimeError):
            self.catalog.call("bones.map_names", GOOD_PARAMS)

    def test_mutating_the_spec_does_not_change_what_is_enforced(self):
        func = Recorder()
        spec = make_spec()
        self.catalog.register(spec, func)
        before = self.catalog.fingerprint()
        spec.input_schema["properties"]["threshold"]["maximum"] = 100
        with self.assertRaises(InvalidParamsError):
            self.catalog.call("bones.map_names", {"armature": "A", "threshold": 50})
        self.assertEqual(self.catalog.fingerprint(), before)
        self.assertEqual(self.catalog.listing()[0]["input_schema"], SCHEMA)


# MARK: - 一覧と指紋


class ListingTest(unittest.TestCase):
    TOOLS = (("mesh.fix", "mesh", Effect.MODIFY), ("bones.map", "bones", Effect.READ),
             ("mesh.remove", "mesh", Effect.DELETE))

    def build(self, order):
        catalog = Catalog()
        for index in order:
            tool_id, category, effect = self.TOOLS[index]
            catalog.register(make_spec(id=tool_id, category=category, effect=effect), Recorder())
        return catalog

    def test_listing_shape(self):
        listing = self.build((0, 1, 2)).listing()
        self.assertEqual([item["id"] for item in listing], ["bones.map", "mesh.fix", "mesh.remove"])
        first = listing[0]
        self.assertEqual(first, {
            "id": "bones.map", "version": "1", "feature": "A3", "category": "bones", "title": "骨の名前の対応",
            "summary": "骨の名前を、MMD 準標準の名前に対応させます。", "effect": "read", "heavy": False,
            "input_schema": SCHEMA, "checks_before": ["アーマチュアがあること"],
            "measures_after": ["対応できなかった骨の数"],
        })
        self.assertEqual([item["effect"] for item in listing], ["read", "modify", "delete"])
        # JSON にそのまま書けて、読み戻すと同じ
        text = json.dumps(listing, ensure_ascii=False, allow_nan=False)
        self.assertEqual(json.loads(text), listing)

    def test_listing_by_category(self):
        catalog = self.build((0, 1, 2))
        self.assertEqual([item["id"] for item in catalog.listing("mesh")], ["mesh.fix", "mesh.remove"])
        self.assertEqual(catalog.listing("none"), [])

    def test_deterministic_regardless_of_order(self):
        first = self.build((0, 1, 2))
        for order in itertools.permutations(range(3)):
            catalog = self.build(order)
            self.assertEqual(catalog.listing(), first.listing())
            self.assertEqual(catalog.fingerprint(), first.fingerprint())

    def test_listing_is_a_copy(self):
        catalog = self.build((0, 1, 2))
        before = catalog.fingerprint()
        listing = catalog.listing()
        listing[0]["input_schema"]["properties"].clear()
        listing[0]["checks_before"].append("x")
        listing[0]["title"] = "別"
        self.assertEqual(catalog.fingerprint(), before)
        self.assertEqual(catalog.listing()[0]["input_schema"], SCHEMA)

    def test_fingerprint_formula(self):
        catalog = self.build((0, 1, 2))
        text = json.dumps(catalog.listing(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(catalog.fingerprint(), hashlib.sha256(text.encode("utf-8")).hexdigest())
        self.assertEqual(Catalog().fingerprint(), hashlib.sha256(b"[]").hexdigest())
        self.assertRegex(catalog.fingerprint(), r"\A[0-9a-f]{64}\Z")

    def test_fingerprint_changes_with_every_field(self):
        changed_schema = copy.deepcopy(SCHEMA)
        changed_schema["properties"]["threshold"]["maximum"] = 2
        changes = {
            "id": "bones.map_names2", "version": "2", "feature": "A4", "category": "names", "title": "別の名前",
            "summary": "別の説明です。", "effect": Effect.MODIFY, "heavy": True, "input_schema": changed_schema,
            "checks_before": ("アーマチュアがあること", "もう1つ"), "measures_after": (),
        }
        self.assertEqual(set(changes), {f.name for f in dataclasses.fields(ToolSpec)})

        def fingerprint(**overrides):
            catalog = Catalog()
            catalog.register(make_spec(**overrides), Recorder())
            return catalog.fingerprint()

        base = fingerprint()
        seen = {base}
        for name, value in changes.items():
            with self.subTest(field=name):
                changed = fingerprint(**{name: value})
                self.assertNotEqual(changed, base)
                self.assertNotIn(changed, seen)
                seen.add(changed)
        # 関数がちがっても、説明が同じなら同じ
        self.assertEqual(fingerprint(), base)

    def test_fingerprint_changes_when_a_tool_is_added(self):
        catalog = self.build((0,))
        before = catalog.fingerprint()
        catalog.register(make_spec(id="extra.tool"), Recorder())
        self.assertNotEqual(catalog.fingerprint(), before)


# MARK: - LLM の返事を読む


class ParseToolCallTest(unittest.TestCase):
    def assertRejected(self, text, *fragments, **kwargs):
        with self.assertRaises(ToolCallError) as caught:
            parse_tool_call(text, **kwargs)
        error = caught.exception
        self.assertIsInstance(error, CatalogError)
        self.assertIsInstance(error, ValueError)
        self.assertTrue(error.errors)
        for fragment in fragments:
            self.assertIn(fragment, str(error))
        return error

    def test_round_trip(self):
        text = json.dumps({"tool": "bones.map_names", "params": GOOD_PARAMS}, ensure_ascii=False)
        call = parse_tool_call(text)
        self.assertEqual(call, ToolCall(tool="bones.map_names", params=GOOD_PARAMS))
        self.assertIs(type(call.params["threshold"]), float)
        # 読んだ呼び出しを、そのまま目録で動かせる
        catalog = Catalog()
        func = Recorder()
        catalog.register(make_spec(), func)
        self.assertEqual(catalog.call(call.tool, call.params), "done")
        self.assertEqual(func.calls, [GOOD_PARAMS])

    def test_whitespace_and_order(self):
        call = parse_tool_call('  \n{ "params" : { } , "tool" : "a.b" }\n')
        self.assertEqual(call, ToolCall("a.b", {}))

    def test_rejects_non_strings(self):
        class Text(str):
            pass

        for value in (b'{"tool":"a.b","params":{}}', None, {"tool": "a.b", "params": {}}, 1,
                      Text('{"tool":"a.b","params":{}}')):
            with self.subTest(value=value):
                self.assertRejected(value, "文字列")

    def test_length_limit(self):
        text = '{"tool":"a.b","params":{}}'
        self.assertEqual(parse_tool_call(text, max_length=len(text)).tool, "a.b")
        self.assertRejected(text, "長すぎ", max_length=len(text) - 1)
        self.assertEqual(MAX_TOOL_CALL_LENGTH, 65536)
        long_text = '{"tool":"a.b","params":{"s":"' + "x" * MAX_TOOL_CALL_LENGTH + '"}}'
        self.assertRejected(long_text, "長すぎ")

    def test_bad_max_length(self):
        for value in (0, -1, True, 1.5, "10", None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_tool_call('{"tool":"a.b","params":{}}', max_length=value)

    def test_invalid_json(self):
        for text in ("", "   ", "{", "{'tool': 'a.b', 'params': {}}", '{"tool":"a.b","params":{},}',
                     '```json\n{"tool":"a.b","params":{}}\n```', "tool(a.b)", '{"tool":"a.b","params":{}} x',
                     '﻿{"tool":"a.b","params":{}}', '{"tool":"a.b","params":{"x":01}}'):
            with self.subTest(text=text):
                self.assertRejected(text, "JSON")

    def test_rejects_python_code(self):
        namespace_before = dict(globals())
        for text in ("__import__('os').system('echo pwned')", "globals().__setitem__('pwned', 1)",
                     '{"tool": __import__("os").getcwd(), "params": {}}'):
            with self.subTest(text=text):
                self.assertRejected(text)
        self.assertNotIn("pwned", globals())
        self.assertEqual(set(globals()), set(namespace_before))

    def test_rejects_non_finite_numbers(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e400", "-1e400"):
            with self.subTest(token=token):
                self.assertRejected('{"tool":"a.b","params":{"x":' + token + "}}", "有限の数")
        self.assertEqual(parse_tool_call('{"tool":"a.b","params":{"x":1e308}}').params["x"], 1e308)

    def test_rejects_duplicate_keys(self):
        self.assertRejected('{"tool":"a.b","tool":"c.d","params":{}}', "2回", '"tool"')
        self.assertRejected('{"tool":"a.b","params":{"x":1,"x":2}}', "2回", '"x"')
        self.assertRejected('{"tool":"a.b","params":{"o":[{"y":1,"y":1}]}}', "2回")

    def test_top_level_must_be_object(self):
        for text in ("[]", '"x"', "1", "null", "true", '[{"tool":"a.b","params":{}}]'):
            with self.subTest(text=text):
                self.assertRejected(text, "いちばん外")

    def test_keys_must_be_exactly_tool_and_params(self):
        self.assertRejected('{"params":{}}', '"tool" がありません')
        self.assertRejected('{"tool":"a.b"}', '"params" がありません')
        self.assertRejected('{"tool":"a.b","params":{},"note":"x"}', '"note"')
        self.assertRejected("{}", '"tool" がありません', '"params" がありません')
        self.assertRejected('{"Tool":"a.b","params":{}}', '"Tool"')

    def test_tool_and_params_types(self):
        for value in ("1", "null", '["a.b"]', '{"a":1}', "true"):
            with self.subTest(tool=value):
                self.assertRejected('{"tool":' + value + ',"params":{}}', '"tool" は文字列')
        for value in ("[]", "null", '"x"', "1", "[{}]"):
            with self.subTest(params=value):
                self.assertRejected('{"tool":"a.b","params":' + value + "}", '"params" はオブジェクト')

    def test_reports_all_shape_problems(self):
        error = self.assertRejected('{"tool": 1, "x": 2}')
        self.assertEqual(len(error.errors), 3, error.errors)

    def test_deep_nesting_and_huge_numbers(self):
        self.assertRejected("[" * 30000 + "]" * 30000, "深すぎ")
        self.assertRejected('{"tool":"a.b","params":{"x":' + "[" * 20000 + "]" * 20000 + "}}", "深すぎ")
        self.assertRejected('{"tool":"a.b","params":{"x":' + "9" * 5000 + "}}", "JSON")

    def test_tool_call_checks_types(self):
        with self.assertRaises(ToolCallError):
            ToolCall(tool=1, params={})
        with self.assertRaises(ToolCallError):
            ToolCall(tool="a.b", params=[])
        call = ToolCall("a.b", {"x": 1})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            call.tool = "c.d"

    def test_unknown_tool_after_parsing(self):
        call = parse_tool_call('{"tool":"os.system","params":{"cmd":"rm -rf /"}}')
        catalog = Catalog()
        catalog.register(make_spec(), Recorder())
        self.assertEqual(catalog.validate_call(call.tool, call.params), ["目録にない道具です: os.system"])
        with self.assertRaises(UnknownToolError):
            catalog.call(call.tool, call.params, confirmed=True, plan_approved=True)

    def test_unknown_tool_message_is_short_and_safe(self):
        catalog = Catalog()
        message = catalog.validate_call("x" * 10000 + "\n\ud800", {})[0]
        self.assertTrue(message.startswith("目録にない道具です: "))
        self.assertLess(len(message), 120)
        message.encode("utf-8")


if __name__ == "__main__":
    unittest.main()
