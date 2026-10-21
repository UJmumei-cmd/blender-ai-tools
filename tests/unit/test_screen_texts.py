"""画面の文（texts/screen.py）がそろっているか、ui/ のコードが文を texts から作っているかを確かめる（bpy なし）。

- screen.* の文には、どれも日本語と英語があり、入れる名前がそろっている。
- 部品の種類・服の種類・信号・言語の選び方の文が、model.card などの一覧と同じだけある。
- ui/*.py に書いた "screen.…" のキーは、どれも texts にある（AST で読む）。使っていない文も無い。
- ui/*.py には、日本語の文を直接書かない（説明の文字列 docstring と、ValueError などの例外の説明は別）。
"""

import ast
import unittest
from pathlib import Path

from blender_ai_tools import texts
from blender_ai_tools.measure.results import Signal
from blender_ai_tools.model.card import CLOTH_TYPE_IDS, CLOTH_TYPE_LABELS, PART_KIND_LABELS, PART_KINDS
from blender_ai_tools.texts import screen

UI_DIR = Path(__file__).resolve().parents[2] / "blender_ai_tools" / "ui"
PREFIX = "screen."

# ui/ のコードが f"screen.<頭>{名前}" の形で作るキーの頭と、そのキーが要る名前の全部
DYNAMIC_KEYS = {
    "screen.kind_": tuple(PART_KINDS),
    "screen.cloth_": tuple(CLOTH_TYPE_IDS),
    "screen.signal_": tuple(signal.name.lower() for signal in Signal),
    "screen.pref_language_": ("auto", "ja", "en"),
    "screen.summary_": ("unknown_bones", "unconfirmed_parts", "body_measurements", "source_terms", "card_current"),
    # 裏の係に出す重い道具（labels.tool_label。道具の id の「.」を「_」にした名前）
    "screen.tool_": ("export_convert_and_check", "motion_run_course"),
}


def is_japanese(value: str) -> bool:
    """ひらがな・カタカナ・漢字・全角の記号を含むか"""
    return any("　" <= char <= "ヿ" or "一" <= char <= "鿿" or "＀" <= char <= "￯"
               for char in value)


class UiStrings(ast.NodeVisitor):
    """ui/ のコードの文字列を集める。

    keys: "screen." で始まる文字列の定数
    prefixes: f-string の、"screen." で始まる頭の部分
    japanese: 日本語を含む文字列（docstring と raise の中は除く）。(行, 文字列)
    """

    def __init__(self) -> None:
        self.keys: set[str] = set()
        self.prefixes: set[str] = set()
        self.japanese: list[tuple[int, str]] = []
        self._docstrings: set[int] = set()
        self._in_raise = 0

    def collect(self, tree: ast.AST) -> "UiStrings":
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    self._docstrings.add(id(body[0].value))
        self.visit(tree)
        return self

    def visit_Raise(self, node: ast.Raise) -> None:
        self._in_raise += 1
        self.generic_visit(node)
        self._in_raise -= 1

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        first = node.values[0] if node.values else None
        if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith(PREFIX):
            self.prefixes.add(first.value)
        for part in node.values:
            if isinstance(part, ast.Constant):
                # f-string の中の文字の部分は、キーの頭なので keys には入れない。日本語だけを見る
                if isinstance(part.value, str) and is_japanese(part.value) and not self._in_raise:
                    self.japanese.append((node.lineno, part.value))
            else:
                self.visit(part)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str):
            return
        if node.value.startswith(PREFIX):
            self.keys.add(node.value)
        if is_japanese(node.value) and id(node) not in self._docstrings and not self._in_raise:
            self.japanese.append((node.lineno, node.value))


def ui_strings() -> dict[str, UiStrings]:
    return {path.name: UiStrings().collect(ast.parse(path.read_text(encoding="utf-8")))
            for path in sorted(UI_DIR.glob("*.py"))}


class ScreenTextsTest(unittest.TestCase):
    def tearDown(self):
        texts.set_language("ja")

    def test_entries_are_complete(self):
        """screen.* の文に、日本語と英語があり、入れる名前がそろっている"""
        self.assertTrue(screen.TEXTS)
        problems = [problem for problem in texts.problems() if problem.startswith(PREFIX)]
        self.assertEqual(problems, [])
        for key, entry in screen.TEXTS.items():
            with self.subTest(key=key):
                self.assertTrue(key.startswith(PREFIX))
                self.assertEqual(set(entry), {"ja", "en"})
                self.assertTrue(entry["ja"].strip() and entry["en"].strip())
                self.assertEqual(texts.placeholders(entry["ja"]), texts.placeholders(entry["en"]))
                # 英語の文に、日本語の字がまぎれていない（言語の名前「日本語」は、どの言語でもその字で見せる）
                if key != "screen.pref_language_ja":
                    self.assertFalse(is_japanese(entry["en"]), entry["en"])

    def test_screen_texts_are_merged(self):
        for key in screen.TEXTS:
            self.assertIn(key, texts.TEXTS)

    def test_kind_and_cloth_labels_match_the_card(self):
        """部品の種類・服の種類の日本語は、登録カード（model.card）の名前と同じ"""
        for kind in PART_KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(texts.text(f"screen.kind_{kind}", language="ja"), PART_KIND_LABELS[kind])
        for cloth_type in CLOTH_TYPE_IDS:
            with self.subTest(cloth_type=cloth_type):
                self.assertEqual(texts.text(f"screen.cloth_{cloth_type}", language="ja"), CLOTH_TYPE_LABELS[cloth_type])

    def test_dynamic_keys_are_complete(self):
        for prefix, names in DYNAMIC_KEYS.items():
            for name in names:
                with self.subTest(key=prefix + name):
                    self.assertIn(prefix + name, texts.TEXTS)

    def test_signal_labels(self):
        self.assertEqual([texts.text(f"screen.signal_{s.name.lower()}", language="ja") for s in Signal],
                         [s.label for s in Signal])

    def test_values_are_filled_by_the_program(self):
        self.assertEqual(texts.text("screen.height", language="ja", value="1450.0"), "背の高さ：1450.0 mm")
        self.assertEqual(texts.text("screen.height", language="en", value="1450.0"), "Height: 1450.0 mm")
        self.assertEqual(texts.text("screen.model_found", language="ja", armature="A", meshes=2), "モデル：A（部品 2 個）")
        with self.assertRaises(ValueError):
            texts.text("screen.height", language="ja")


class UiSourceTest(unittest.TestCase):
    def setUp(self):
        self.strings = ui_strings()

    def test_ui_files_exist(self):
        self.assertTrue({"__init__.py", "preferences.py", "panel.py", "operators.py"} <= set(self.strings))

    def test_every_key_used_in_ui_exists(self):
        """ui/*.py に書いた "screen.…" のキーは、どれも texts にある"""
        for name, found in self.strings.items():
            for key in sorted(found.keys):
                with self.subTest(file=name, key=key):
                    self.assertIn(key, texts.TEXTS)

    def test_dynamic_prefixes_are_known(self):
        """f-string で作るキーの頭は、DYNAMIC_KEYS に書いた物だけ（そこで全部の名前の文があるかを確かめる）"""
        prefixes = set().union(*(found.prefixes for found in self.strings.values()))
        self.assertTrue(prefixes)
        self.assertLessEqual(prefixes, set(DYNAMIC_KEYS))

    def test_every_screen_text_is_used(self):
        """使っていない文を残さない（キーそのものか、f-string の頭で使っている）"""
        used = set().union(*(found.keys for found in self.strings.values()))
        dynamic = {prefix + name for prefix, names in DYNAMIC_KEYS.items() for name in names}
        unused = sorted(set(screen.TEXTS) - used - dynamic)
        self.assertEqual(unused, [])

    def test_no_japanese_sentences_in_ui_code(self):
        """画面の文は texts に置く。ui/*.py には日本語の文を直接書かない（docstring と例外の説明は別）"""
        for name, found in self.strings.items():
            with self.subTest(file=name):
                self.assertEqual(found.japanese, [])

    def test_scanner_finds_violations(self):
        """見張りそのものが、キーと日本語の文を見つけられるか"""
        source = ('"""説明"""\n'
                  'def f(kind):\n'
                  '    """関数の説明"""\n'
                  '    text("screen.a")\n'
                  '    text(f"screen.kind_{kind}")\n'
                  '    label(text="登録する")\n'
                  '    label(text=f"{kind} の種類")\n'
                  '    raise ValueError(f"まちがい {kind}")\n')
        found = UiStrings().collect(ast.parse(source))
        self.assertEqual(found.keys, {"screen.a"})
        self.assertEqual(found.prefixes, {"screen.kind_"})
        self.assertEqual([value for _, value in found.japanese], ["登録する", " の種類"])


if __name__ == "__main__":
    unittest.main()
