"""E1 書き出し（convert_blend.py を呼ぶ）を、にせの紅魔館で確かめる（bpy なし）。

にせの紅魔館：一時フォルダに tools/model-converter/convert_blend.py（本物の print の形をまねて、ダミーの
character.usdz と character_rig.json を書く小さな Python のスクリプト）と convert_pmx.py（中身なし）を置く。
にせの Blender：受け取った引数をファイルに書き、--python の後のスクリプトを Python で動かす sh のスクリプト。
本物の Blender があるときは、本物の Blender でにせの convert_blend.py を動かす試しもする（コマンドの形の確かめ）。
"""

import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from blender_ai_tools import provenance, thresholds
from blender_ai_tools.export import convert
from blender_ai_tools.export.convert import ConvertResult, KoumakanPaths, build_command, koumakan_paths, parse_summary
from blender_ai_tools.measure.results import Signal
from blender_ai_tools.provenance import GitState, RunRecord

REAL_BLENDER = shutil.which("blender")
GIT = shutil.which("git")
SOURCE_BYTES = b"BLENDER-v405 fake model for tests"

# MARK: - にせの紅魔館

# にせの convert_blend.py。MODE で動きを変える。print の形は本物の convert_blend.py と同じ
FAKE_CONVERT_BLEND = r'''
"""テスト用の、にせの convert_blend.py（bpy を使わない）"""
import json
import sys
import time
import zipfile
from pathlib import Path

MODE = "__MODE__"
APP_MODEL = Path(__file__).resolve().parents[2] / "KoumakanApp" / "KoumakanApp" / "Model"
HUMAN = ["hips", "spine", "leftUpperArm"] + (["chest", "leftEye"] if MODE == "missing_humanoid" else [])
BONE_NAMES = {"hips": "センター", "spine": "上半身", "leftUpperArm": "左腕", "b003": "左ひじ"}


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    source, output_dir = Path(args[0]).resolve(), Path(args[1]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if MODE == "write_app":
        output_dir = APP_MODEL
        output_dir.mkdir(parents=True, exist_ok=True)
    print("骨格: TestArmature（骨 4）、メッシュ 1 個")
    print("  Face_Params がないので、表情のアクションは読み取らない")
    if MODE == "fail":
        sys.exit("アーマチュアがありません")
    if MODE == "raise":
        raise RuntimeError("にせの失敗")
    if MODE == "sleep":
        sys.stdout.flush()
        time.sleep(60)
    if MODE == "pycache":
        # 本物の Blender は convert_pmx を読み込むときに __pycache__ を作る（git では、まだ入れていない物になる）
        cache = Path(__file__).resolve().parent / "__pycache__"
        cache.mkdir(exist_ok=True)
        (cache / "convert_pmx.cpython-313.pyc").write_bytes(b"fake")
    if MODE == "modify_input":
        with source.open("ab") as stream:
            stream.write(b"changed")
    if MODE != "no_usdz":
        usdz = output_dir / "character.usdz"
        if MODE == "not_zip":
            usdz.write_bytes(b"not a zip file")
        else:
            with zipfile.ZipFile(usdz, "w", zipfile.ZIP_STORED) as archive:
                archive.writestr("character.usdc" if MODE != "no_usd_inside" else "readme.txt", b"PXR-USDC fake")
                archive.writestr("textures/t00.png", b"PNG fake")
    humanoid = {"hips": "hips", "spine": "spine", "leftUpperArm": "leftUpperArm"}
    rig = {
        "version": {"version2": 2, "version_true": True}.get(MODE, 1),
        "source": "other.blend" if MODE == "wrong_source" else source.name,
        "format": "blend",
        # json.dumps は NaN をそのまま書く（アプリの JSONDecoder は読めない）
        "restArmDownDegrees": float("nan") if MODE == "nan" else 38.7,
        "bones": dict(humanoid, head="head") if MODE == "inconsistent" else humanoid,
        "expressions": {"a": [{"shapeKey": "a", "weight": 1.0}]},
        "originalNames": {
            "bones": [] if MODE == "bad_original" else BONE_NAMES,
            "shapeKeys": {"a": "あ"},
            "materials": {"m00": "肌", "m01": "髪"},
        },
    }
    if MODE == "no_bones":
        del rig["bones"]
    text = json.dumps(rig, ensure_ascii=False, indent=2) + "\n"
    if MODE == "bad_json":
        text = "{not json"
    if MODE == "duplicate_key":
        text = '{"version": 1, "version": 1}'
    if MODE != "no_rig":
        (output_dir / "character_rig.json").write_text(text, encoding="utf-8")
    if MODE == "pydriver":
        print("Error in PyDriver: expression failed: bpy.data.objects[0]['p'] + var")
    if MODE == "modifier_notice":
        print("  Body: 骨で選ぶ MASK は焼き込めないので外す")
        print("  Body: モディファイア SOLIDIFY は書き出しに反映されない場合がある")
    if MODE in ("ok", "modifier_notice"):
        print("  動きのある表情は使わない: ['Expr_Yawn']")
    if MODE == "old_format":
        print(f"書き出し完了: {output_dir}")
        print("  ボーン: 4 本")
        return

    presets = {"a": [{"shapeKey": "a", "weight": 1.0}]}
    keep = {"あ"}
    removed, limited = 12, 3
    unmapped = ["shy", "smile"]
    material_modes = {"m01": "BLEND", "m00": "OPAQUE"}
    arm_angle = 38.7
    missing = sorted(set(HUMAN) - set(humanoid))
    limit_text = "5" if MODE == "limit5" else "4"
    print(f"書き出し完了: {output_dir}")
    print(f"  ボーン: {len(BONE_NAMES)}（ヒューマノイド {len(humanoid)}、見つからない: {missing or 'なし'}）")
    print(f"  表情プリセット: {len(presets)} 個、残したシェイプキー: {len(keep)} 個、服で隠れる頂点を {removed} 個消した、骨の影響を{limit_text}本までに絞った頂点 {limited} 個")
    print(f"  アプリの表情で対応が無いもの: {unmapped or 'なし'}")
    print(f"  透過: {dict(sorted(material_modes.items()))}")
    print(f"  レスト時の腕の下がり角: {arm_angle} 度")


if __name__ == "__main__":
    main()
'''

# にせの Blender。引数を1行ずつ記録し、"--python <スクリプト>" を Python で動かす（"--" から後ろはスクリプトに渡す）
FAKE_BLENDER = """#!/bin/sh
printf '%s\\n' "$@" > {args_file}
script=''
while [ "$#" -gt 0 ]; do
  case "$1" in
    --python) script="$2"; shift 2 ;;
    --) break ;;
    *) shift ;;
  esac
done
exec {python} "$script" "$@"
"""

# 本物の convert_blend.py の stdout（このセッションで、テスト用のモデルを書き出したときの物。場所だけ置き換えた）
REAL_STDOUT = """00:00.748  blend            | Read blend: "/work/model.blend"
骨格: TestArmature（骨 96）、メッシュ 2 個
  Face_Params がないので、表情のアクションは読み取らない
書き出し完了: /work/out
  ボーン: 96（ヒューマノイド 21、見つからない: なし）
  表情プリセット: 0 個、残したシェイプキー: 0 個、服で隠れる頂点を 0 個消した、骨の影響を4本までに絞った頂点 4 個
  アプリの表情で対応が無いもの: ['angry', 'fluster', 'happy', 'heart', 'pout', 'sad', 'shock', 'shy', 'sleep', 'sleepy', 'smile', 'smug', 'surprised', 'troubled']
  透過: {}
  レスト時の腕の下がり角: 37.5 度
Blender 5.2.2 LTS (hash d13f752e3b9c built 2026-09-15 01:34:58)
USD export of '/tmp/tmpqlyn8pwh/character.usdc' took 29.42 ms

Blender quit
"""

# にせの convert_blend.py が "ok" のときに書く報告を、読んだ値
FAKE_SUMMARY = {
    "armature_name": "TestArmature",
    "armature_bone_count": 4,
    "mesh_count": 1,
    "bone_count": 4,
    "humanoid_count": 3,
    "missing_humanoid": [],
    "preset_count": 1,
    "kept_shape_key_count": 1,
    "masked_vertices_removed": 12,
    "influence_limit": 4,
    "vertices_limited": 3,
    "unmapped_emotions": ["shy", "smile"],
    "transparency": {"m00": "OPAQUE", "m01": "BLEND"},
    "rest_arm_down_deg": 38.7,
    "notices": ["Face_Params がないので、表情のアクションは読み取らない", "動きのある表情は使わない: ['Expr_Yawn']"],
    "unread": [],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder_state(folder: Path):
    """フォルダの中身（場所・大きさ・中身の sha256）。無いときは None"""
    if not folder.exists():
        return None
    return sorted((str(path.relative_to(folder)), path.is_dir(), None if path.is_dir() else sha256(path))
                  for path in folder.rglob("*"))


def run_git(directory: Path, *args: str) -> str:
    """テスト用のリポジトリを作る git。ユーザーの設定（署名など）を読まない"""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    result = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false",
         "-c", "init.defaultBranch=main", "-C", str(directory), *args],
        capture_output=True, text=True, timeout=60, env=env, check=True)
    return result.stdout.strip()


class FakeKoumakanTestCase(unittest.TestCase):
    """にせの紅魔館・にせの Blender・入力の .blend を、一時フォルダに作る"""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.repo = self.root / "koumakan"
        self.converter_dir = self.repo / "tools" / "model-converter"
        self.converter_dir.mkdir(parents=True)
        self.convert_blend = self.converter_dir / "convert_blend.py"
        self.convert_pmx = self.converter_dir / "convert_pmx.py"
        self.convert_pmx.write_text("# にせの convert_pmx.py（読み込まれない）\n", encoding="utf-8")
        self.app_model = self.repo / "KoumakanApp" / "KoumakanApp" / "Model"
        self.app_model.mkdir(parents=True)
        (self.app_model / "character.usdz").write_bytes(b"the app's current model")
        self.set_mode("ok")
        self.args_file = self.root / "args.txt"
        self.blender = self.root / "bin" / "blender"
        self.blender.parent.mkdir()
        self.blender.write_text(FAKE_BLENDER.format(args_file=shlex.quote(str(self.args_file)),
                                                    python=shlex.quote(sys.executable)), encoding="utf-8")
        self.blender.chmod(0o755)
        self.work = self.root / "work"
        self.work.mkdir()
        self.source = self.work / "model.blend"
        self.source.write_bytes(SOURCE_BYTES)
        self.app_before = folder_state(self.app_model)

    def set_mode(self, mode: str) -> None:
        self.convert_blend.write_text(FAKE_CONVERT_BLEND.replace("__MODE__", mode), encoding="utf-8")

    def convert(self, output=None, **options) -> ConvertResult:
        options.setdefault("blender", self.blender)
        options.setdefault("repo", self.repo)
        options.setdefault("timeout", 60.0)
        return convert.run_convert(self.source, self.root / "out" if output is None else output, **options)

    def recorded_args(self) -> list[str]:
        return self.args_file.read_text(encoding="utf-8").splitlines()

    def assert_refused(self, output, message: str) -> None:
        """出力フォルダを断り、何も動かさず、アプリのフォルダも変えない"""
        with self.assertRaises(ValueError) as caught:
            self.convert(output)
        self.assertIn(message, str(caught.exception))
        self.assertFalse(self.args_file.exists(), "断ったときは Blender を動かさない")
        self.assertEqual(folder_state(self.app_model), self.app_before)


# MARK: - うまくいくとき


class SuccessTest(FakeKoumakanTestCase):
    def test_ok(self):
        result = self.convert()
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.problems, [])
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.timed_out)
        self.assertTrue(result.input_unchanged)
        self.assertEqual(result.output_dir, self.root / "out")
        self.assertEqual(folder_state(self.app_model), self.app_before)

    def test_exact_command(self):
        """決まった形のコマンドで動かす（--python-exit-code は --python より前、"--" の後ろに入力と出力）"""
        result = self.convert()
        expected = [str(self.blender), "--background", "--factory-startup", "--disable-autoexec",
                    "--python-exit-code", "1", "--python", str(self.convert_blend),
                    "--", str(self.source), str(self.root / "out")]
        self.assertEqual(list(result.command), expected)
        # にせの Blender が受け取った引数（$0 を除く）
        self.assertEqual(self.recorded_args(), expected[1:])
        self.assertEqual(result.record.settings["command"], expected)

    def test_outputs(self):
        result = self.convert()
        usdz = self.root / "out" / "character.usdz"
        self.assertEqual(result.usdz.sha256, sha256(usdz))
        self.assertEqual(result.usdz.path, str(usdz))
        self.assertEqual(result.usdz_contents, {
            "character.usdc": hashlib.sha256(b"PXR-USDC fake").hexdigest(),
            "textures/t00.png": hashlib.sha256(b"PNG fake").hexdigest(),
        })
        self.assertEqual(result.rig_path, self.root / "out" / "character_rig.json")
        self.assertEqual(result.rig_file.sha256, sha256(result.rig_path))
        self.assertEqual(result.rig, json.loads(result.rig_path.read_text(encoding="utf-8")))
        self.assertEqual(result.rig["bones"], {"hips": "hips", "spine": "spine", "leftUpperArm": "leftUpperArm"})
        self.assertEqual(result.rig["source"], "model.blend")

    def test_summary(self):
        result = self.convert()
        expected = dict(FAKE_SUMMARY, reported_output_dir=str(self.root / "out"))
        self.assertEqual(result.summary, expected)

    def test_record(self):
        result = self.convert()
        record = result.record
        self.assertEqual(record.tool_id, "export.convert")
        self.assertEqual(record.tool_version, convert.TOOL_VERSION)
        self.assertEqual(record.addon_version, provenance.addon_version())
        # 入力：.blend・convert_blend.py・convert_pmx.py の順
        self.assertEqual([item.name for item in record.inputs], ["model.blend", "convert_blend.py", "convert_pmx.py"])
        self.assertEqual(record.inputs[0].sha256, hashlib.sha256(SOURCE_BYTES).hexdigest())
        self.assertEqual(record.inputs[0].size, len(SOURCE_BYTES))
        self.assertEqual(record.inputs[1].sha256, sha256(self.convert_blend))
        self.assertEqual(record.inputs[2].sha256, sha256(self.convert_pmx))
        self.assertEqual(record.settings["output_dir"], str(self.root / "out"))
        self.assertEqual(record.settings["timeout_seconds"], 60.0)
        self.assertEqual(record.settings["blender"], str(self.blender))
        self.assertIsNone(record.seed)
        # にせの紅魔館は git のリポジトリではない
        self.assertEqual(record.convert_blend, provenance.git_state(self.repo))
        results = record.results
        self.assertIs(results["ok"], True)
        self.assertEqual(results["returncode"], 0)
        self.assertIs(results["timed_out"], False)
        self.assertIs(results["input_unchanged"], True)
        self.assertIs(results["app_model_dir_unchanged"], True)
        # 書いた場所は記録の結果に入れない（出力フォルダが違っても、同じ書き出しなら同じ結果になるように）
        self.assertEqual(results["summary"], FAKE_SUMMARY)
        self.assertEqual(results["outputs"], {
            "usdz": {"name": "character.usdz", "size": result.usdz.size, "contents": result.usdz_contents},
            "rig": {"name": "character_rig.json", "sha256": result.rig_file.sha256, "size": result.rig_file.size},
        })
        self.assertEqual(results["problems"], [])
        self.assertEqual(results["warnings"], [])
        # JSON にして読み戻せる
        self.assertEqual(RunRecord.from_json(record.to_json()), record)

    def test_record_is_reproducible_between_output_folders(self):
        """同じ入力を別のフォルダに2回書き出すと、記録の結果は同じ（USDZ の zip の時刻は入れない）"""
        first = self.convert(self.root / "first")
        self.args_file.unlink()
        time.sleep(2.1)  # zip の時刻（2秒きざみ）が変わるように待つ
        second = self.convert(self.root / "second")
        self.assertEqual(provenance.compare_results(first.record.results, second.record.results), [])

    def test_record_is_a_copy(self):
        """ConvertResult の値を変えても、記録は変わらない"""
        result = self.convert()
        result.problems.append("あとから足した")
        result.summary["bone_count"] = 999
        self.assertEqual(result.record.results["problems"], [])
        self.assertEqual(result.record.results["summary"]["bone_count"], 4)

    def test_creates_nested_output_dir(self):
        output = self.root / "new" / "deeper"
        result = self.convert(output)
        self.assertTrue(result.ok, result.problems)
        self.assertTrue((output / "character.usdz").is_file())

    def test_accepts_existing_empty_output_dir(self):
        output = self.root / "empty"
        output.mkdir()
        self.assertTrue(self.convert(output).ok)

    def test_output_dir_through_symlink_and_dots(self):
        """リンクと ".." は、たどった場所を使う（コマンドにも、たどった場所を渡す）"""
        real = self.root / "real_out"
        real.mkdir()
        link = self.root / "link_out"
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"シンボリックリンクが作れないため：{error}")
        result = self.convert(self.root / "work" / ".." / "link_out")
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.output_dir, real)
        self.assertEqual(result.command[-1], str(real))
        self.assertTrue((real / "character.usdz").is_file())

    def test_output_dir_name_ending_with_space(self):
        """フォルダの名前の終わりの空白（全角の空白も）は、場所の一部。報告の場所と比べるときに消さない"""
        for name in ("out ", "out　"):
            with self.subTest(name=name):
                output = self.root / name
                result = self.convert(output)
                self.assertTrue(result.ok, result.problems)
                self.assertEqual(result.summary["reported_output_dir"], str(output))
                self.assertEqual(result.summary["unread"], [])

    def test_allows_other_places_in_the_repo(self):
        """断るのはアプリの Model フォルダと convert_blend.py のフォルダだけ。リポジトリのほかの場所には書ける"""
        result = self.convert(self.repo / "KoumakanApp" / "export_check")
        self.assertTrue(result.ok, result.problems)

    def test_relative_paths(self):
        """入力・出力・紅魔館の場所は、いまのフォルダからの場所でもよい（絶対パスにして渡す）"""
        previous = os.getcwd()
        os.chdir(self.root)
        try:
            result = convert.run_convert("work/model.blend", "relative_out", blender=self.blender, repo="koumakan",
                                         timeout=60.0)
        finally:
            os.chdir(previous)
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.command[-2:], (str(self.source), str(self.root / "relative_out")))

    def test_blender_from_path(self):
        """区切りの無い名前は PATH から探す"""
        previous = os.environ.get("PATH", "")
        os.environ["PATH"] = str(self.blender.parent) + os.pathsep + previous
        try:
            result = self.convert(blender="blender")
        finally:
            os.environ["PATH"] = previous
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.command[0], str(self.blender))


@unittest.skipIf(GIT is None, "git が入っていないため（convert_blend.py のコミットの記録は、git がある所で試す）")
class GitRecordTest(FakeKoumakanTestCase):
    def test_records_converter_commit(self):
        run_git(self.repo, "init", "-q")
        run_git(self.repo, "add", ".")
        run_git(self.repo, "commit", "-q", "-m", "converter")
        head = run_git(self.repo, "rev-parse", "HEAD")
        result = self.convert()
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.record.convert_blend, GitState(commit=head, dirty=False))

    def test_git_state_is_taken_before_running(self):
        """書き出しの間に convert_pmx の __pycache__ ができても、記録するのは動かす前の状態（dirty にならない）"""
        run_git(self.repo, "init", "-q")
        run_git(self.repo, "add", ".")
        run_git(self.repo, "commit", "-q", "-m", "converter")
        self.set_mode("pycache")
        run_git(self.repo, "commit", "-q", "-a", "-m", "pycache mode")
        head = run_git(self.repo, "rev-parse", "HEAD")
        result = self.convert()
        self.assertTrue(result.ok, result.problems)
        self.assertTrue((self.converter_dir / "__pycache__").is_dir())
        self.assertEqual(provenance.git_state(self.repo), GitState(commit=head, dirty=True))
        self.assertEqual(result.record.convert_blend, GitState(commit=head, dirty=False))


# MARK: - 断るとき（動かす前に ValueError）


class RefuseTest(FakeKoumakanTestCase):
    def test_output_dir_is_required(self):
        """出力フォルダは必ず指定する（指定しないと convert_blend.py はアプリの Model フォルダに書く）"""
        for output in (None, ""):
            with self.subTest(output=output):
                with self.assertRaises(ValueError) as caught:
                    convert.run_convert(self.source, output, blender=self.blender, repo=self.repo)
                self.assertIn("出力フォルダを必ず指定してください", str(caught.exception))
        with self.assertRaises(TypeError):
            convert.run_convert(self.source, blender=self.blender, repo=self.repo)  # 既定の値は無い
        self.assertFalse(self.args_file.exists())
        self.assertEqual(folder_state(self.app_model), self.app_before)

    def test_refuses_app_model_dir(self):
        message = "アプリの Model フォルダ"
        self.assert_refused(self.app_model, message)
        self.assert_refused(self.app_model / "sub", message)
        self.assert_refused(self.app_model / "sub" / "deeper", message)
        self.assert_refused(self.app_model.parent / "Other" / ".." / "Model", message)
        self.assert_refused(self.converter_dir / ".." / ".." / "KoumakanApp" / "KoumakanApp" / "Model" / "x", message)
        # 大文字・小文字の違いだけの場所（Mac のディスクでは同じフォルダになる）
        self.assert_refused(self.repo / "koumakanapp" / "KOUMAKANAPP" / "model", message)
        self.assertFalse((self.app_model / "sub").exists(), "断った場所にフォルダを作らない")
        self.assertFalse((self.repo / "koumakanapp").exists())

    def test_refuses_app_model_dir_through_symlinks(self):
        message = "アプリの Model フォルダ"
        try:
            (self.root / "link_model").symlink_to(self.app_model, target_is_directory=True)
            (self.root / "link_app").symlink_to(self.app_model.parent, target_is_directory=True)
            (self.root / "link_repo").symlink_to(self.repo, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"シンボリックリンクが作れないため：{error}")
        self.assert_refused(self.root / "link_model", message)
        self.assert_refused(self.root / "link_model" / "sub", message)
        self.assert_refused(self.root / "link_app" / "Model", message)
        self.assert_refused(self.root / "link_repo" / "KoumakanApp" / "KoumakanApp" / "Model" / "sub", message)
        self.assert_refused(self.root / "work" / ".." / "link_app" / "Model", message)

    def test_refuses_app_model_dir_that_does_not_exist_yet(self):
        """アプリの Model フォルダがまだ無くても断り、作らない"""
        shutil.rmtree(self.app_model)
        self.app_before = folder_state(self.app_model)
        self.assert_refused(self.app_model, "アプリの Model フォルダ")
        self.assert_refused(self.app_model / "sub", "アプリの Model フォルダ")
        self.assertFalse(self.app_model.exists())

    def test_check_output_dir_before_running(self):
        """画面が仕事を出す前の確かめ（check_output_dir）も、同じ場所を断る。フォルダは作らない"""
        for output, message in ((self.app_model / "sub", "アプリの Model フォルダ"),
                                (self.converter_dir / "out", "convert_blend.py のあるフォルダ")):
            with self.subTest(output=output):
                with self.assertRaisesRegex(ValueError, message):
                    convert.check_output_dir(output, self.repo)
                self.assertFalse(output.exists())
        fine = self.root / "work" / "exports" / "model_v001_job"
        self.assertEqual(convert.check_output_dir(fine, self.repo), fine)
        self.assertFalse(fine.exists())
        with self.assertRaises(ValueError):
            convert.check_output_dir(fine, self.root / "no_repo")
        self.assertFalse(self.args_file.exists())

    def test_refuses_converter_dir(self):
        message = "convert_blend.py のあるフォルダ"
        self.assert_refused(self.converter_dir, message)
        self.assert_refused(self.converter_dir / "out", message)
        self.assert_refused(self.repo / "tools" / "x" / ".." / "model-converter" / "out", message)
        try:
            (self.root / "link_converter").symlink_to(self.converter_dir, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"シンボリックリンクが作れないため：{error}")
        self.assert_refused(self.root / "link_converter" / "out", message)
        self.assertFalse((self.converter_dir / "out").exists())

    def test_refuses_non_empty_output_dir(self):
        output = self.root / "used"
        output.mkdir()
        (output / "character.usdz").write_bytes(b"old export")
        self.assert_refused(output, "空ではありません")
        self.assertEqual((output / "character.usdz").read_bytes(), b"old export")
        # 入力のファイルのあるフォルダも、空ではないので断る
        self.assert_refused(self.work, "空ではありません")

    def test_refuses_file_as_output_dir(self):
        output = self.root / "a_file"
        output.write_bytes(b"x")
        self.assert_refused(output, "フォルダではない物")

    def test_input_checks(self):
        cases = {
            "ありません": self.work / "missing.blend",
            ".blend のファイルにしてください": self.work / "model.txt",
        }
        (self.work / "model.txt").write_bytes(b"x")
        (self.work / "folder.blend").mkdir()
        cases["ファイルではありません"] = self.work / "folder.blend"
        for message, source in cases.items():
            with self.subTest(source=source.name):
                with self.assertRaises(ValueError) as caught:
                    convert.run_convert(source, self.root / "out", blender=self.blender, repo=self.repo)
                self.assertIn(message, str(caught.exception))
        for source in (None, ""):
            with self.assertRaises(ValueError):
                convert.run_convert(source, self.root / "out", blender=self.blender, repo=self.repo)
        self.assertFalse((self.root / "out").exists(), "入力がまちがっているときは、出力フォルダを作らない")
        self.assertFalse(self.args_file.exists())

    def test_repo_checks(self):
        with self.assertRaises(ValueError) as caught:
            self.convert(repo=self.root / "nothing")
        self.assertIn("リポジトリのフォルダがありません", str(caught.exception))
        for value in (None, ""):
            with self.assertRaises(ValueError):
                self.convert(repo=value)
        self.convert_pmx.unlink()
        with self.assertRaises(ValueError) as caught:
            self.convert()
        self.assertIn("convert_pmx.py", str(caught.exception))
        self.assertNotIn("convert_blend.py", str(caught.exception))
        self.convert_blend.unlink()
        with self.assertRaises(ValueError) as caught:
            self.convert()
        self.assertIn("convert_blend.py・convert_pmx.py", str(caught.exception))
        self.assertFalse(self.args_file.exists())

    def test_blender_checks(self):
        cases = [self.root / "bin" / "missing", "no-such-blender-command-for-tests", None, ""]
        plain = self.root / "bin" / "not_executable"
        plain.write_text("#!/bin/sh\n", encoding="utf-8")
        plain.chmod(0o644)
        cases.append(plain)
        for blender in cases:
            with self.subTest(blender=blender):
                with self.assertRaises(ValueError):
                    self.convert(blender=blender)
        self.assertFalse((self.root / "out").exists())

    def test_unreadable_input_is_refused_before_anything(self):
        """入力が読めない（PermissionError など）ときは、出力フォルダを作る前に ValueError"""
        real_digest = provenance.file_digest

        def digest(path):
            if Path(path).name == "model.blend":
                raise PermissionError(13, "Permission denied", str(path))
            return real_digest(path)

        with mock.patch.object(convert.provenance, "file_digest", side_effect=digest):
            with self.assertRaises(ValueError) as caught:
                self.convert()
        self.assertIn("入力の .blend が読めません", str(caught.exception))
        self.assertFalse((self.root / "out").exists(), "出力フォルダを作らない")
        self.assertFalse(self.args_file.exists())

    def test_timeout_checks(self):
        for timeout in (0, -1.0, math.nan, math.inf, True, "60"):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    self.convert(timeout=timeout)
        self.assertFalse((self.root / "out").exists())

    def test_default_timeout(self):
        """待つ時間の上限の既定は thresholds.CONVERT_TIMEOUT_SECONDS"""
        result = convert.run_convert(self.source, self.root / "out", blender=self.blender, repo=self.repo)
        self.assertEqual(result.record.settings["timeout_seconds"], thresholds.CONVERT_TIMEOUT_SECONDS)


# MARK: - 動かしたあとの失敗（ok が False）


class FailureTest(FakeKoumakanTestCase):
    def convert_with(self, mode: str) -> ConvertResult:
        self.set_mode(mode)
        result = self.convert()
        self.assertEqual(result.record.results["ok"], result.ok)
        self.assertEqual(result.record.results["problems"], result.problems)
        return result

    def assert_failed(self, result: ConvertResult, message: str) -> None:
        self.assertFalse(result.ok)
        self.assertEqual(result.signal, Signal.RED)
        self.assertTrue(any(message in problem for problem in result.problems), result.problems)
        self.assertEqual(folder_state(self.app_model), self.app_before)

    def test_missing_usdz(self):
        result = self.convert_with("no_usdz")
        self.assert_failed(result, "character.usdz がありません")
        self.assertIsNone(result.usdz)
        self.assertIsNone(result.usdz_contents)
        self.assertIsNotNone(result.rig, "rig は形が正しいので読む")
        self.assertIsNone(result.record.results["outputs"]["usdz"])

    def test_missing_rig(self):
        result = self.convert_with("no_rig")
        self.assert_failed(result, "character_rig.json がありません")
        self.assertIsNone(result.rig)
        self.assertIsNone(result.rig_path)
        self.assertIsNone(result.rig_file)

    def test_usdz_not_zip(self):
        self.assert_failed(self.convert_with("not_zip"), "USDZ（zip）の形ではありません")

    def test_usdz_without_usd(self):
        self.assert_failed(self.convert_with("no_usd_inside"), "USD のファイル")

    def test_bad_json(self):
        result = self.convert_with("bad_json")
        self.assert_failed(result, "JSON として読めません")
        self.assertIsNone(result.rig)
        self.assertEqual(result.rig_path, self.root / "out" / "character_rig.json")

    def test_duplicate_key(self):
        self.assert_failed(self.convert_with("duplicate_key"), "同じキー 'version' が2回あります")

    def test_wrong_version(self):
        result = self.convert_with("version2")
        self.assert_failed(result, "version が 1 ではありません（いまは 2）")
        self.assertIsNone(result.rig)

    def test_version_true(self):
        """version が true（Python では True == 1）でも、1 とはみなさない"""
        result = self.convert_with("version_true")
        self.assert_failed(result, "version が 1 ではありません（いまは True）")
        self.assertIsNone(result.rig)

    def test_rig_with_nan(self):
        """NaN は JSON に書けない（アプリの JSONDecoder が読めない）。Python の json は読めてしまうので、はっきり断る"""
        result = self.convert_with("nan")
        self.assert_failed(result, "character_rig.json が JSON として読めません：NaN")
        self.assertIsNone(result.rig)

    def test_unreadable_output_is_a_problem(self):
        """動かしたあとにファイルが読めない（PermissionError など）ときも、例外にせず失敗として返す"""
        real_digest = provenance.file_digest

        def digest(path):
            if Path(path).name in ("character.usdz", "model.blend") and self.args_file.exists():
                raise PermissionError(13, "Permission denied", str(path))
            return real_digest(path)

        with mock.patch.object(convert.provenance, "file_digest", side_effect=digest):
            result = self.convert()
        self.assert_failed(result, "character.usdz が読めません")
        self.assert_failed(result, "書き出しのあとに、入力のファイルが読めません")
        self.assertIsNone(result.usdz)
        self.assertIsNone(result.input_unchanged)

    def test_missing_bones(self):
        self.assert_failed(self.convert_with("no_bones"), "bones が「骨の名前 → 書き出した骨の名前」の {} ではありません")

    def test_bad_original_names(self):
        self.assert_failed(self.convert_with("bad_original"), "originalNames.bones が")

    def test_bones_missing_from_original_names(self):
        self.assert_failed(self.convert_with("inconsistent"), "originalNames.bones にありません：head")

    def test_nonzero_exit(self):
        result = self.convert_with("fail")
        self.assert_failed(result, "convert_blend.py が失敗しました（終了コード 1）：アーマチュアがありません")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.timed_out)
        self.assertIn("骨格: TestArmature", result.stdout)
        self.assertEqual(result.record.results["returncode"], 1)

    def test_exception(self):
        result = self.convert_with("raise")
        self.assert_failed(result, "（終了コード 1）：RuntimeError: にせの失敗")

    def test_timeout(self):
        self.set_mode("sleep")
        started = time.monotonic()
        result = self.convert(timeout=1.0)
        self.assertLess(time.monotonic() - started, 30.0, "時間切れで止める")
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.returncode)
        self.assert_failed(result, "1 秒で終わらなかったので、止めました")
        # 止めるまでの出力も受け取る
        self.assertIn("骨格: TestArmature（骨 4）", result.stdout)
        self.assertEqual(result.summary["armature_bone_count"], 4)
        self.assertIs(result.record.results["timed_out"], True)

    def test_input_changed(self):
        result = self.convert_with("modify_input")
        self.assert_failed(result, "入力のファイルが変わりました")
        self.assertIs(result.input_unchanged, False)
        # 記録の入力は、動かす前の指紋
        self.assertEqual(result.record.inputs[0].sha256, hashlib.sha256(SOURCE_BYTES).hexdigest())
        self.assertNotEqual(sha256(self.source), result.record.inputs[0].sha256)

    def test_app_model_dir_written(self):
        """convert_blend.py が出力フォルダを使わずにアプリの Model フォルダに書いたら、失敗にする"""
        self.set_mode("write_app")
        result = self.convert()
        self.assertFalse(result.ok)
        self.assertTrue(any("アプリの Model フォルダが変わりました" in p for p in result.problems), result.problems)
        self.assertTrue(any("指定した出力フォルダ" in p for p in result.problems), result.problems)
        self.assertIs(result.record.results["app_model_dir_unchanged"], False)

    def test_cannot_start(self):
        """Blender が起動できない（中身が壊れている）ときは、例外にせず失敗として返す"""
        self.blender.write_bytes(b"\x7fELF broken")
        self.blender.chmod(0o755)
        result = self.convert()
        self.assertFalse(result.ok)
        self.assertIsNone(result.returncode)
        self.assertTrue(result.problems[0].startswith("Blender を起動できませんでした"), result.problems)


# MARK: - 書き出せたが、知らせがあるとき（信号は黄）


class WarningTest(FakeKoumakanTestCase):
    def convert_with(self, mode: str) -> ConvertResult:
        self.set_mode(mode)
        result = self.convert()
        self.assertTrue(result.ok, result.problems)
        self.assertEqual(result.signal, Signal.YELLOW)
        return result

    def test_driver_errors(self):
        result = self.convert_with("pydriver")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("計算できなかったドライバーの式が 1 個あります", result.warnings[0])

    def test_missing_humanoid(self):
        """アプリのヒューマノイドの骨が足りないモデルは、書き出せても緑にしない（アプリの動きが、その骨に当たらない）"""
        result = self.convert_with("missing_humanoid")
        self.assertEqual(result.summary["missing_humanoid"], ["chest", "leftEye"])
        self.assertEqual(result.warnings, [
            "アプリのヒューマノイドの骨のうち、モデルに見つからない物があります：chest、leftEye"
            "（アプリの動きは、この骨に当たりません）"])

    def test_notices_that_the_output_may_differ(self):
        """convert_blend.py が「反映されない」「外す」と知らせたときは緑にしない（表情の知らせは、知らせのまま）"""
        result = self.convert_with("modifier_notice")
        self.assertEqual(result.warnings, [
            "convert_blend.py の知らせ：Body: 骨で選ぶ MASK は焼き込めないので外す"
            "（書き出した物が、Blender で見る物と違うかもしれません）",
            "convert_blend.py の知らせ：Body: モディファイア SOLIDIFY は書き出しに反映されない場合がある"
            "（書き出した物が、Blender で見る物と違うかもしれません）",
        ])

    def test_wrong_source(self):
        result = self.convert_with("wrong_source")
        self.assertEqual(result.warnings, [
            "character_rig.json の source（'other.blend'）が、入力のファイルの名前（model.blend）と違います"])

    def test_influence_limit_changed(self):
        result = self.convert_with("limit5")
        self.assertEqual(result.summary["influence_limit"], 5)
        self.assertEqual(result.warnings, [
            f"convert_blend.py は1点に効く骨を 5 本までにしています（この道具の基準は {thresholds.MAX_BONE_INFLUENCES} 本）"])

    def test_report_format_changed(self):
        result = self.convert_with("old_format")
        self.assertEqual(result.summary["unread"], ["ボーン", "表情プリセット", "アプリの表情", "透過", "腕の下がり角"])
        self.assertIsNone(result.summary["bone_count"])
        self.assertIsNone(result.summary["vertices_limited"])
        self.assertEqual(result.summary["armature_bone_count"], 4)
        self.assertEqual(len(result.warnings), 2)
        self.assertIn("読めなかった物があります：ボーン、表情プリセット", result.warnings[0])
        # 形の違う文は、知らない知らせとしても見せる
        self.assertEqual(result.warnings[1],
                         "convert_blend.py の知らせ：ボーン: 4 本（書き出した物が、Blender で見る物と違うかもしれません）")


# MARK: - 報告の文を読む


class ParseSummaryTest(unittest.TestCase):
    def test_real_output(self):
        """本物の convert_blend.py の出力（このセッションで書き出したときの物）"""
        self.assertEqual(parse_summary(REAL_STDOUT), {
            "armature_name": "TestArmature",
            "armature_bone_count": 96,
            "mesh_count": 2,
            "reported_output_dir": "/work/out",
            "bone_count": 96,
            "humanoid_count": 21,
            "missing_humanoid": [],
            "preset_count": 0,
            "kept_shape_key_count": 0,
            "masked_vertices_removed": 0,
            "influence_limit": 4,
            "vertices_limited": 4,
            "unmapped_emotions": ["angry", "fluster", "happy", "heart", "pout", "sad", "shock", "shy", "sleep",
                                  "sleepy", "smile", "smug", "surprised", "troubled"],
            "transparency": {},
            "rest_arm_down_deg": 37.5,
            "notices": ["Face_Params がないので、表情のアクションは読み取らない"],
            "unread": [],
        })

    def test_blender_version(self):
        self.assertEqual(convert._blender_version(REAL_STDOUT), ("5.2.2 LTS", "d13f752e3b9c"))
        self.assertEqual(convert._blender_version("no version here"), (None, None))

    def test_empty(self):
        summary = parse_summary("")
        self.assertTrue(all(summary[key] is None for key in convert.SUMMARY_KEYS))
        self.assertEqual(summary["notices"], [])
        self.assertEqual(summary["unread"], ["骨格", "書き出し完了", "ボーン", "表情プリセット", "アプリの表情", "透過",
                                             "腕の下がり角"])

    def test_values(self):
        stdout = "\n".join([
            "骨格: Arm（骨）（骨 12）、メッシュ 3 個",
            "  ボーン: 12（ヒューマノイド 2、見つからない: ['chest', 'leftEye']）",
            "  透過: {'m00': 'OPAQUE', 'm01': 'MASK', 'm02': 'BLEND'}",
            "  レスト時の腕の下がり角: None 度",
            "  Body: 骨で選ぶ MASK は焼き込めないので外す",
            "  動きのある表情は使わない: ['Expr_Yawn']",
        ])
        summary = parse_summary(stdout)
        self.assertEqual(summary["armature_name"], "Arm（骨）")
        self.assertEqual(summary["armature_bone_count"], 12)
        self.assertEqual(summary["mesh_count"], 3)
        self.assertEqual(summary["missing_humanoid"], ["chest", "leftEye"])
        self.assertEqual(summary["transparency"], {"m00": "OPAQUE", "m01": "MASK", "m02": "BLEND"})
        # convert_blend.py が None と書いたときは、値は None だが「読めた」
        self.assertIsNone(summary["rest_arm_down_deg"])
        self.assertNotIn("腕の下がり角", summary["unread"])
        self.assertEqual(summary["notices"], ["Body: 骨で選ぶ MASK は焼き込めないので外す",
                                              "動きのある表情は使わない: ['Expr_Yawn']"])
        self.assertEqual(summary["unread"], ["書き出し完了", "表情プリセット", "アプリの表情"])

    def test_negative_angle(self):
        self.assertEqual(parse_summary("  レスト時の腕の下がり角: -12.5 度")["rest_arm_down_deg"], -12.5)

    def test_unknown_forms_are_not_guessed(self):
        """形の違う値は推測せず None にし、unread に入れる"""
        stdout = "\n".join([
            "  ボーン: 12（ヒューマノイド 2、見つからない: [chest]）",
            "  透過: {'m00': OPAQUE}",
            "  レスト時の腕の下がり角: 約40 度",
            "  アプリの表情で対応が無いもの: ['it\\'s']",
        ])
        summary = parse_summary(stdout)
        for key in ("bone_count", "humanoid_count", "missing_humanoid", "transparency", "rest_arm_down_deg",
                    "unmapped_emotions"):
            self.assertIsNone(summary[key], key)
        self.assertEqual(summary["unread"], ["骨格", "書き出し完了", "ボーン", "表情プリセット", "アプリの表情", "透過",
                                             "腕の下がり角"])
        self.assertEqual(summary["notices"], [])

    def test_only_ascii_digits(self):
        """数は ASCII の 0〜9 だけを読む（全角の数字は、形が違う文として扱う）"""
        summary = parse_summary("  ボーン: ９６（ヒューマノイド 21、見つからない: なし）")
        self.assertIsNone(summary["bone_count"])
        self.assertIn("ボーン", summary["unread"])

    def test_reported_dir_keeps_unusual_characters(self):
        """行は print の改行（\\n）だけで分ける。場所の中の U+2028 などや、終わりの空白で切らない"""
        for path in ("/work/out ", "/work/a b", "/work/a\x85b", "/work/a　"):
            with self.subTest(path=path):
                summary = parse_summary(f"書き出し完了: {path}\r\n  ボーン: 1（ヒューマノイド 0、見つからない: なし）\n")
                self.assertEqual(summary["reported_output_dir"], path)
                self.assertEqual(summary["bone_count"], 1)

    def test_first_line_wins(self):
        stdout = "  ボーン: 5（ヒューマノイド 1、見つからない: なし）\n  ボーン: 9（ヒューマノイド 9、見つからない: なし）"
        self.assertEqual(parse_summary(stdout)["bone_count"], 5)

    def test_rejects_non_text(self):
        with self.assertRaises(ValueError):
            parse_summary(b"bytes")


# MARK: - そのほか


class PathsAndCommandTest(FakeKoumakanTestCase):
    def test_koumakan_paths(self):
        paths = koumakan_paths(self.repo)
        self.assertEqual(paths, KoumakanPaths(
            repo=self.repo, converter_dir=self.converter_dir, convert_blend=self.convert_blend,
            convert_pmx=self.convert_pmx, app_model_dir=self.app_model))
        shutil.rmtree(self.app_model)
        self.assertEqual(koumakan_paths(str(self.repo)).app_model_dir, self.app_model, "Model フォルダは無くてもよい")

    def test_build_command(self):
        command = build_command("/B/Blender", Path("/K/convert_blend.py"), Path("/in/m.blend"), Path("/out"))
        self.assertEqual(command, ["/B/Blender", "--background", "--factory-startup", "--disable-autoexec",
                                   "--python-exit-code", "1", "--python", "/K/convert_blend.py",
                                   "--", "/in/m.blend", "/out"])
        self.assertLess(command.index("--python-exit-code"), command.index("--python"))


@unittest.skipIf(REAL_BLENDER is None, "blender が PATH に無いため（本物の Blender でコマンドの形を確かめる試し）")
class RealBlenderTest(FakeKoumakanTestCase):
    """本物の Blender で、にせの convert_blend.py を動かす（コマンドの引数が本物の Blender で通るか）"""

    def setUp(self):
        super().setUp()
        self.real = Path(REAL_BLENDER)

    def test_ok(self):
        result = self.convert(blender=self.real, timeout=300.0)
        self.assertTrue(result.ok, result.problems + [result.stderr])
        self.assertEqual(result.summary["vertices_limited"], 3)
        self.assertRegex(result.record.environment["converter_blender_version"], r"^\d+\.\d+\.\d+")
        self.assertRegex(result.record.environment["converter_blender_build_hash"], r"^[0-9a-f]+$")

    def test_exception_gives_nonzero_exit(self):
        """--python-exit-code 1 が無いと、Blender は例外で止まっても 0 で終わる"""
        self.set_mode("raise")
        result = self.convert(blender=self.real, timeout=300.0)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(result.ok)
        self.assertIn("RuntimeError: にせの失敗", result.problems[0])

    def test_sys_exit_message(self):
        self.set_mode("fail")
        result = self.convert(blender=self.real, timeout=300.0)
        self.assertEqual(result.returncode, 1)
        self.assertIn("（終了コード 1）：アーマチュアがありません", result.problems[0])


if __name__ == "__main__":
    unittest.main()
