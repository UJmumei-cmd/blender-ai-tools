"""CLAUDE.md のコードの決まりを、コードを読んで確かめる。

- bpy など Blender だけのモジュールを import してよいのは、ui/ の下と、名前が _bpy.py で終わるファイルだけ。
- それ以外のファイルは、ファイルの先頭（関数の外）で ui や *_bpy を読み込まない（bpy なしで読み込めるようにする）。
- exec・eval・compile・__import__ を使わない（LLM や Jeff の返事をコードとして動かさない。CLAUDE.md 2章1）。
"""

import ast
import unittest
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "blender_ai_tools"

BLENDER_MODULES = {
    "bpy", "bmesh", "mathutils", "addon_utils", "bpy_extras", "bpy_types", "bl_math", "bl_operators", "bl_ui",
    "bl_keymap_utils", "gpu", "gpu_extras", "idprop", "aud", "bgl", "blf", "imbuf", "freestyle", "rna_info", "_bpy",
    "pxr",  # USD。Blender の中の Python にだけ入っている
}
FORBIDDEN_CALLS = {"exec", "eval", "compile", "__import__"}


def may_use_blender(path: Path) -> bool:
    relative = path.relative_to(PACKAGE_DIR)
    return "ui" in relative.parts[:-1] or path.name.endswith("_bpy.py")


def is_bpy_module_name(name: str) -> bool:
    last = name.split(".")[-1]
    return last == "ui" or last.endswith("_bpy")


class ImportVisitor(ast.NodeVisitor):
    """import を集める。関数の中かどうかも覚える"""

    def __init__(self) -> None:
        self.function_depth = 0
        self.blender_imports: list[tuple[int, str]] = []
        self.top_level_bpy_parts: list[tuple[int, str]] = []
        self.forbidden_calls: list[tuple[int, str]] = []

    def _function(self, node) -> None:
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _function

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.split(".")[0] in BLENDER_MODULES:
                self.blender_imports.append((node.lineno, alias.name))
            elif self.function_depth == 0 and is_bpy_module_name(alias.name):
                self.top_level_bpy_parts.append((node.lineno, alias.name))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level == 0 and module.split(".")[0] in BLENDER_MODULES:
            self.blender_imports.append((node.lineno, module))
        if self.function_depth == 0:
            names = [module] if module else []
            names += [f"{module}.{alias.name}" if module else alias.name for alias in node.names]
            for name in names:
                if is_bpy_module_name(name):
                    self.top_level_bpy_parts.append((node.lineno, name))

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            self.forbidden_calls.append((node.lineno, node.func.id))
        self.generic_visit(node)


def python_files() -> list[Path]:
    return sorted(PACKAGE_DIR.rglob("*.py"))


class CodeRulesTest(unittest.TestCase):
    def test_finds_files(self):
        self.assertIn(PACKAGE_DIR / "__init__.py", python_files())

    def test_blender_modules_only_in_bpy_files(self):
        for path in python_files():
            if may_use_blender(path):
                continue
            visitor = ImportVisitor()
            visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
            with self.subTest(path=str(path.relative_to(PACKAGE_DIR))):
                self.assertEqual(visitor.blender_imports, [], "bpy などは ui/ か *_bpy.py の中だけで使う")
                self.assertEqual(visitor.top_level_bpy_parts, [], "ui や *_bpy は関数の中で読み込む")

    def test_no_exec_or_eval(self):
        for path in python_files():
            visitor = ImportVisitor()
            visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
            with self.subTest(path=str(path.relative_to(PACKAGE_DIR))):
                self.assertEqual(visitor.forbidden_calls, [])

    def test_rules_catch_violations(self):
        """見張りそのものが、違反を見つけられるか"""
        visitor = ImportVisitor()
        visitor.visit(ast.parse("import bpy\nfrom mathutils import Vector\nfrom . import ui\n"
                                "from .x import y_bpy\neval('1')\n"
                                "def f():\n    from . import ui\n    import bmesh\n"))
        self.assertEqual([name for _, name in visitor.blender_imports], ["bpy", "mathutils", "bmesh"])
        self.assertEqual([name for _, name in visitor.top_level_bpy_parts], ["ui", "x.y_bpy"])
        self.assertEqual([name for _, name in visitor.forbidden_calls], ["eval"])


if __name__ == "__main__":
    unittest.main()
