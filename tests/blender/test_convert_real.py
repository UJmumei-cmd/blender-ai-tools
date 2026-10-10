"""E1 書き出し：本物の紅魔館の convert_blend.py で、テスト用のモデル（tests/fixtures/mmd_model.py）を書き出す。

紅魔館の場所は、環境変数 BLENDER_AI_TOOLS_KOUMAKAN_REPO から読む。無いときは理由を出して飛ばす
（飛ばしたテストは合格と数えない。CLAUDE.md 8章）。紅魔館のファイルは、このリポジトリにコピーしない。
"""

import hashlib
import os
import tempfile
import unittest
import zipfile
from collections.abc import Callable
from pathlib import Path

import bpy
from pxr import Usd, UsdGeom, UsdSkel

from blender_ai_tools import bones, provenance, thresholds
from blender_ai_tools.export import convert
from blender_ai_tools.measure.results import Signal
from tests.fixtures import mmd_model

KOUMAKAN_ENV = "BLENDER_AI_TOOLS_KOUMAKAN_REPO"

# アプリのヒューマノイドの骨 → テスト用のモデルの骨（convert_pmx.HUMAN_BONES で、最初に見つかる物。CLAUDE.md 4章）
EXPECTED_HUMANOID = {
    "hips": "センター", "spine": "上半身", "chest": "上半身2", "neck": "首", "head": "頭",
    "leftEye": "左目", "rightEye": "右目",
    "leftShoulder": "左肩", "rightShoulder": "右肩",
    "leftUpperArm": "左腕", "rightUpperArm": "右腕",
    "leftLowerArm": "左ひじ", "rightLowerArm": "右ひじ",
    "leftHand": "左手首", "rightHand": "右手首",
    "leftUpperLeg": "左足D", "rightUpperLeg": "右足D",
    "leftLowerLeg": "左ひざD", "rightLowerLeg": "右ひざD",
    "leftFoot": "左足首D", "rightFoot": "右足首D",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder_state(folder: Path):
    """フォルダの中身（場所・種類・中身の sha256）。無いときは None"""
    if not folder.exists():
        return None
    return sorted((str(path.relative_to(folder)), path.is_dir(), None if path.is_dir() else sha256(path))
                  for path in folder.rglob("*"))


def koumakan_repo() -> str:
    """紅魔館のリポジトリの場所。環境変数が無いときは、理由を出して飛ばす"""
    repo = os.environ.get(KOUMAKAN_ENV)
    if not repo:
        raise unittest.SkipTest(f"環境変数 {KOUMAKAN_ENV}（紅魔館のリポジトリの場所）が無いため")
    return repo


def temporary_folder(test_class: type[unittest.TestCase]) -> Path:
    """一時フォルダ。setUpClass が途中で失敗しても消えるように、クラスの片づけに登録する"""
    temp = tempfile.TemporaryDirectory()
    test_class.addClassCleanup(temp.cleanup)
    return Path(temp.name).resolve()


def save_fixture(path: Path, prepare: Callable[[dict], None] | None = None, **options) -> tuple[dict, Path]:
    """テスト用のモデルを作って path に保存し、空のシーンに戻す。prepare(info) で、保存の前にモデルを変えられる"""
    bpy.ops.wm.read_homefile(use_empty=True)
    try:
        info = mmd_model.build_model(**options)
        if prepare is not None:
            prepare(info)
        return info, mmd_model.save_model(path)
    finally:
        # 作ったモデルは片づけ、空のシーンに戻す（ほかのテストに残さない）
        bpy.ops.wm.read_homefile(use_empty=True)


class RealConverterTest(unittest.TestCase):
    """テスト用のモデルを .blend に保存し、本物の convert_blend.py で書き出す（1回だけ書き出して、いくつも確かめる）"""

    @classmethod
    def setUpClass(cls):
        cls.repo = koumakan_repo()
        # 場所が書いてあるのに convert_blend.py が無いときは、飛ばさずに失敗にする（設定のまちがいに気づけるように）
        cls.paths = convert.koumakan_paths(cls.repo)
        cls.app_before = folder_state(cls.paths.app_model_dir)
        cls.root = temporary_folder(cls)
        cls.info, cls.source = save_fixture(cls.root / "model.blend")
        cls.source_sha = sha256(cls.source)
        cls.result = convert.run_convert(cls.source, cls.root / "out", blender=bpy.app.binary_path, repo=cls.repo)

    def test_ok(self):
        result = self.result
        self.assertTrue(result.ok, result.problems + [result.stdout, result.stderr])
        self.assertEqual(result.problems, [])
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.timed_out)
        self.assertEqual(sorted(path.name for path in result.output_dir.iterdir()),
                         ["character.usdz", "character_rig.json"])

    def test_rig_bones(self):
        """ヒューマノイドの骨が書き出され、originalNames を通すと元の日本語の名前に戻る"""
        rig = self.result.rig
        self.assertEqual(rig["version"], 1)
        self.assertEqual(rig["source"], "model.blend")
        self.assertEqual(rig["format"], "blend")
        self.assertEqual(set(rig["bones"]), set(EXPECTED_HUMANOID))
        original = rig["originalNames"]["bones"]
        for humanoid, japanese in EXPECTED_HUMANOID.items():
            with self.subTest(humanoid=humanoid):
                self.assertEqual(rig["bones"][humanoid], humanoid)  # ヒューマノイドの骨は、その名前に変わる
                self.assertEqual(original[rig["bones"][humanoid]], japanese)
        # 骨の名前は全部 ASCII に変わり、元の名前は準標準の 96 本そのまま
        self.assertTrue(all(name.isascii() for name in original))
        self.assertEqual(set(original.values()), set(bones.canonical_names()))
        self.assertEqual(len(original), self.info["bone_count"])

    def test_summary(self):
        summary = self.result.summary
        self.assertEqual(summary["armature_name"], mmd_model.ARMATURE_NAME)
        self.assertEqual(summary["armature_bone_count"], self.info["bone_count"])
        self.assertEqual(summary["mesh_count"], len(self.info["meshes"]))
        self.assertEqual(summary["reported_output_dir"], str(self.result.output_dir))
        self.assertEqual(summary["bone_count"], self.info["bone_count"])
        self.assertEqual(summary["humanoid_count"], len(EXPECTED_HUMANOID))
        self.assertEqual(summary["missing_humanoid"], [])
        self.assertEqual(summary["influence_limit"], thresholds.MAX_BONE_INFLUENCES)
        # 1点4本までに絞られた頂点は、見本で5本の骨を付けた頂点と同じ数
        self.assertEqual(len(self.info["five_influence_vertices"]), 4)
        self.assertEqual(summary["vertices_limited"], len(self.info["five_influence_vertices"]))
        self.assertEqual(summary["masked_vertices_removed"], 0)
        # Face_Params が無いので、表情のプリセットは作られず、シェイプキーも残らない
        self.assertEqual(summary["preset_count"], 0)
        self.assertEqual(summary["kept_shape_key_count"], 0)
        self.assertEqual(summary["transparency"], {})
        self.assertEqual(summary["rest_arm_down_deg"], mmd_model.ARM_DOWN_DEG)
        self.assertEqual(summary["rest_arm_down_deg"], self.result.rig["restArmDownDegrees"])
        self.assertIn("Face_Params がないので、表情のアクションは読み取らない", summary["notices"])
        self.assertEqual(summary["unread"], [])

    def test_input_unchanged(self):
        self.assertEqual(sha256(self.source), self.source_sha)
        self.assertIs(self.result.input_unchanged, True)
        self.assertEqual(self.result.record.inputs[0].sha256, self.source_sha)

    def test_app_model_dir_unchanged(self):
        self.assertEqual(folder_state(self.paths.app_model_dir), self.app_before)
        self.assertIs(self.result.record.results["app_model_dir_unchanged"], True)

    def test_record(self):
        record = self.result.record
        self.assertEqual(record.tool_id, convert.TOOL_ID)
        self.assertEqual([item.sha256 for item in record.inputs],
                         [self.source_sha, sha256(self.paths.convert_blend), sha256(self.paths.convert_pmx)])
        self.assertEqual(record.environment["converter_blender_version"], bpy.app.version_string)
        build_hash = bpy.app.build_hash.decode("ascii") if isinstance(bpy.app.build_hash, bytes) else bpy.app.build_hash
        self.assertEqual(record.environment["converter_blender_build_hash"], build_hash)
        self.assertEqual(record.settings["blender"], bpy.app.binary_path)
        # convert_blend.py のコミット（Blender が __pycache__ を作ると dirty は変わるので、コミットだけ比べる）
        self.assertEqual(record.convert_blend.commit, provenance.git_state(self.paths.repo).commit)
        self.assertEqual(provenance.RunRecord.from_json(record.to_json()), record)

    def test_usdz(self):
        """USDZ が開けて、骨と形が入っている（細かい比べは E2 の往復の確認で行う）"""
        usdz = self.result.output_dir / "character.usdz"
        self.assertEqual(self.result.usdz.sha256, sha256(usdz))
        with zipfile.ZipFile(usdz) as archive:
            names = archive.namelist()
        self.assertEqual(names[0], "character.usdc")
        self.assertEqual(set(self.result.usdz_contents), set(names))
        stage = Usd.Stage.Open(str(usdz))
        self.assertEqual(UsdGeom.GetStageUpAxis(stage), UsdGeom.Tokens.y)
        skeletons = [UsdSkel.Skeleton(prim) for prim in stage.Traverse() if prim.IsA(UsdSkel.Skeleton)]
        self.assertEqual(len(skeletons), 1)
        joint_names = {path.split("/")[-1] for path in skeletons[0].GetJointsAttr().Get()}
        original = self.result.rig["originalNames"]["bones"]
        self.assertLessEqual(joint_names, set(original))
        for humanoid in ("spine", "leftUpperArm", "leftUpperLeg", "rightFoot"):
            self.assertIn(self.result.rig["bones"][humanoid], joint_names)
        points = {prim.GetName(): len(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                  for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)}
        self.assertEqual(points, self.info["vertex_counts"])

    def test_refuses_app_model_dir(self):
        """本物の紅魔館のアプリの Model フォルダには書かない（動かす前に断る）"""
        with self.assertRaises(ValueError):
            convert.run_convert(self.source, self.paths.app_model_dir, blender=bpy.app.binary_path, repo=self.repo)
        with self.assertRaises(ValueError):
            convert.run_convert(self.source, self.paths.app_model_dir / "check", blender=bpy.app.binary_path,
                                repo=self.repo)
        with self.assertRaises(ValueError):
            convert.run_convert(self.source, self.paths.converter_dir / "out", blender=bpy.app.binary_path,
                                repo=self.repo)
        self.assertEqual(folder_state(self.paths.app_model_dir), self.app_before)
        self.assertFalse((self.paths.converter_dir / "out").exists())

    def test_refuses_used_output_dir(self):
        with self.assertRaises(ValueError):
            convert.run_convert(self.source, self.result.output_dir, blender=bpy.app.binary_path, repo=self.repo)

    def test_repeatable(self):
        """同じ入力を別のフォルダにもう1回書き出すと、記録の結果が同じになる（USDZ は中のファイルで比べる）"""
        second = convert.run_convert(self.source, self.root / "second", blender=bpy.app.binary_path, repo=self.repo)
        self.assertTrue(second.ok, second.problems)
        self.assertEqual(second.rig, self.result.rig)
        self.assertEqual(second.usdz_contents, self.result.usdz_contents)
        self.assertEqual(provenance.compare_results(self.result.record.results, second.record.results), [])
        self.assertEqual(sha256(self.source), self.source_sha)



def without_eyes_and_with_modifiers(info: dict) -> None:
    """左目・右目の骨の名前を変え（アプリの leftEye・rightEye が無くなる）、書き出しに反映されないモディファイアを付ける"""
    bone_data = bpy.data.objects[info["armature"]].data.bones
    for name in ("左目", "右目"):
        bone_data[name].name = name + "_old"  # 頂点グループの名前も一緒に変わる
    body = bpy.data.objects[mmd_model.BODY_NAME]
    body.modifiers.new("Thickness", "SOLIDIFY")
    body.modifiers.new("HideByBone", "MASK").mode = "ARMATURE"


class RealConverterYellowTest(unittest.TestCase):
    """書き出せるが緑にしないモデル（本物の convert_blend.py の報告の形で確かめる）：
    ヒューマノイドの骨（左目・右目）が無い。convert_blend.py が「反映されない」「外す」と知らせるモディファイアがある"""

    @classmethod
    def setUpClass(cls):
        repo = koumakan_repo()
        root = temporary_folder(cls)
        _, source = save_fixture(root / "yellow.blend", without_eyes_and_with_modifiers,
                                 with_cloth=False, with_shape_keys=False)
        cls.result = convert.run_convert(source, root / "out", blender=bpy.app.binary_path, repo=repo)

    def test_yellow(self):
        result = self.result
        self.assertTrue(result.ok, result.problems + [result.stderr])
        self.assertEqual(result.summary["missing_humanoid"], ["leftEye", "rightEye"])
        self.assertEqual(result.summary["humanoid_count"], len(EXPECTED_HUMANOID) - 2)
        self.assertEqual(set(result.rig["bones"]), set(EXPECTED_HUMANOID) - {"leftEye", "rightEye"})
        self.assertEqual(result.warnings, [
            "convert_blend.py の知らせ：Body: 骨で選ぶ MASK は焼き込めないので外す"
            "（書き出した物が、Blender で見る物と違うかもしれません）",
            "convert_blend.py の知らせ：Body: モディファイア SOLIDIFY は書き出しに反映されない場合がある"
            "（書き出した物が、Blender で見る物と違うかもしれません）",
            "アプリのヒューマノイドの骨のうち、モデルに見つからない物があります：leftEye、rightEye"
            "（アプリの動きは、この骨に当たりません）",
        ])
        self.assertEqual(result.signal, Signal.YELLOW)


if __name__ == "__main__":
    unittest.main()
