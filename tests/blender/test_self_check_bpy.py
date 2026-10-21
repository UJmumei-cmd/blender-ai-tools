"""A4 測る道具の自己テストを、Blender の試し用モデル（tests/fixtures/mmd_model.py）でも確かめる（Blender の中）。

bpy なしの自己テスト（tests/unit/test_self_check.py）と同じ考え方で、Blender のモデルをわざと壊す（8 種）。
- A4 の測る道具（Blender から取り出した並び：model/extract_bpy.py の mesh_rest・bone_weights）で、面・左右・ウェイトを測る。
- A1 の登録カード（model/card_bpy.build_card → card.summarize_card）。
- A5 の確かめの動き（motion/course_bpy.run_course。壊していないモデルで、どの確かめも緑になるポーズだけ）。
壊していないモデルでは、どれにも赤が出ない（灰は、理由のある物だけ：アプリの動きは、式を移すまで灰）。
わざと壊したモデルでは、見つけるはずの物が決めた信号になり、ほかの物の信号は変わらない（一緒に変わってよい物は also）。
"""

import dataclasses
import unittest
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import bmesh
import bpy

from blender_ai_tools import thresholds
from blender_ai_tools.measure import mesh_check, symmetry, weights
from blender_ai_tools.measure.results import Measurement, Signal
from blender_ai_tools.model import card, card_bpy, extract_bpy
from blender_ai_tools.motion import course_bpy
from blender_ai_tools.motion.course import APP_MOTION_NAME, APP_MOTIONS_ALL, CLOTH_CHECK, KNEE_CHECK, SINK_CHECK
from blender_ai_tools.motion.poses import PoseSpec, pose_by_name
from tests.fixtures import mmd_model

# 壊していないモデルで、確かめの動きのどの確かめも緑になるポーズ（脚を上げるポーズでは、脚がスカートを突き抜ける。
# tests/blender/test_course_bpy.py の test_cloth_penetration_found_where_the_legs_go_through_the_skirt）
CLEAN_POSES = ("head_down", "arms_up", "wave")


def build_model() -> dict:
    """空のシーンに、試し用のモデルを作る。5本の骨を付けた頂点（E2 の確かめ用）は作らない（壊していない見本にする）"""
    bpy.ops.wm.read_homefile(use_empty=True)
    return mmd_model.build_model(five_influences=False)


def course_key(pose: str, check: str) -> str:
    return f"course/{pose}/{check}"


def measure_model(extra_poses: tuple[PoseSpec, ...] = ()) -> dict[str, Measurement]:
    """いまのシーンのモデルを測る。返す物：「どこ/名前」→ Measurement。

    - a4/…：体（Body）の面の点検（閉じているはずの形）・左右のずれ・ウェイト（形を動かす骨だけ）
    - card/…：登録カードのまとめ（summarize_card）
    - course/<ポーズ>/<確かめ>：確かめの動き（CLEAN_POSES と extra_poses）。最後にアプリの動きの灰の出来事
    """
    armature = bpy.data.objects[mmd_model.ARMATURE_NAME]
    body = bpy.data.objects[mmd_model.BODY_NAME]
    results: dict[str, Measurement] = {}
    vertices, triangles = extract_bpy.mesh_rest(body)
    table, _ = extract_bpy.bone_weights(body, armature)
    deform = extract_bpy.bone_rest(armature).deform
    for measurement in mesh_check.summarize_mesh(vertices, triangles, closed_required=True):
        results[f"a4/{measurement.name}"] = measurement
    results[f"a4/{symmetry.ASYMMETRY_NAME}"] = symmetry.summarize_asymmetry(symmetry.asymmetry(vertices))
    for measurement in weights.summarize(table, deform=deform):
        results[f"a4/{measurement.name}"] = measurement
    for measurement in card.summarize_card(card_bpy.build_card()):
        results[f"card/{measurement.name}"] = measurement
    poses = [pose_by_name(name) for name in CLEAN_POSES] + list(extra_poses)
    for event in course_bpy.run_course(poses=poses):
        results[course_key(event.pose, event.check)] = event.measurement
    return results


# MARK: - わざと壊したモデル（どれも、いまのシーンのモデルを壊す。確かめの動きに足すポーズを返す）

def _part(info: dict, name: str) -> range:
    start, stop = info["parts"][mmd_model.BODY_NAME][name]
    return range(start, stop)


def remove_head_weights(info: dict) -> tuple[PoseSpec, ...]:
    """どの骨にも付いていない点：頭の頂点グループ（頭）から、全部の頂点を外す。
    頭と首を混ぜていた頂点は、首の重みだけが残り、合計が 1 でなくなる（weights_not_normalized も黄）"""
    body = bpy.data.objects[mmd_model.BODY_NAME]
    body.vertex_groups["頭"].remove([vertex.index for vertex in body.data.vertices])
    return ()


def flip_torso_faces(info: dict) -> tuple[PoseSpec, ...]:
    """向きの合わない辺：胴の面を2つ裏返す（Blender の面の向きの反転と同じ）。
    胴が向きのそろった閉じた形でなくなるので、確かめの動きの服のめり込み（スカートは胴のまわり）は、中か外かを決められず灰"""
    body = bpy.data.objects[mmd_model.BODY_NAME]
    for polygon in [polygon for polygon in body.data.polygons if polygon.vertices[0] in _part(info, "torso")][:2]:
        polygon.flip()
    body.data.update()
    return ()


def delete_torso_face(info: dict) -> tuple[PoseSpec, ...]:
    """穴：胴の面を1つ消す（bmesh。シェイプキーも一緒に直る）。
    胴が閉じた形でなくなるので、確かめの動きの服のめり込みは、中か外かを決められず灰"""
    body = bpy.data.objects[mmd_model.BODY_NAME]
    mesh = bmesh.new()
    mesh.from_mesh(body.data)
    mesh.faces.ensure_lookup_table()
    face = next(face for face in mesh.faces if face.verts[0].index in _part(info, "torso"))
    bmesh.ops.delete(mesh, geom=[face], context="FACES_ONLY")
    mesh.to_mesh(body.data)
    mesh.free()
    body.data.update()
    return ()


def move_torso_vertex(info: dict) -> tuple[PoseSpec, ...]:
    """左右のずれ：胴の、キャラの左にいちばん出ている頂点を、ASYMMETRY_RED の2倍だけ上へ動かす。
    Body はシェイプキーを持つので、休みの形（基本のキー）も動かす（extract_bpy.mesh_rest は基本のキーを読む）"""
    body = bpy.data.objects[mmd_model.BODY_NAME]
    basis = body.data.shape_keys.key_blocks[0]
    torso = _part(info, "torso")
    index = max(torso, key=lambda i: (basis.data[i].co.x, -i))
    offset = 2.0 * thresholds.ASYMMETRY_RED
    basis.data[index].co.z += offset
    body.data.vertices[index].co.z += offset
    body.data.update()
    return ()


def shrink_skirt(info: dict) -> tuple[PoseSpec, ...]:
    """めり込み：スカートを、体の縦の軸のまわりに半分の太さへ縮める（体の中へ入る）。どのポーズでも、服のめり込みが赤"""
    skirt = bpy.data.objects[mmd_model.SKIRT_NAME]
    for vertex in skirt.data.vertices:
        vertex.co.x *= 0.5
        vertex.co.y *= 0.5
    skirt.data.update()
    return ()


def reversed_knee(info: dict) -> tuple[PoseSpec, ...]:
    """ひざの逆曲がり：左ひざを曲げるポーズ（knee_bend_left）の、ひざの回る向きを逆にし、40 度だけ曲げる
    （動きの中で、ひざが逆に曲がったフレームと同じ）。モデルは変えない"""
    pose = pose_by_name("knee_bend_left")
    rotations = tuple((bone, tuple(-a for a in axis), 40.0) if bone == "左ひざ" else (bone, axis, angle)
                      for bone, axis, angle in pose.rotations)
    return (dataclasses.replace(pose, rotations=rotations),)


def toes_into_floor(info: dict) -> tuple[PoseSpec, ...]:
    """床への沈み：つま先立ちのポーズ（tiptoe）を、床に合わせずに付ける（足首を下へ曲げるので、つま先が床へ沈む）。
    モデルは変えない"""
    return (dataclasses.replace(pose_by_name("tiptoe"), ground=False),)


def rename_ankle_bone(info: dict) -> tuple[PoseSpec, ...]:
    """準標準の骨が足りない：左足首の骨の名前を、辞書に無い名前に変える。足りない骨（semi_standard_bones）と、
    辞書に無い骨（unknown_bones）が、どちらも黄（ウェイトは左足首D にあるので、ほかは変わらない）"""
    bpy.data.objects[mmd_model.ARMATURE_NAME].data.bones["左足首"].name = "ankle_left_old"
    return ()


@dataclass(frozen=True)
class ModelBreakage:
    """わざと壊したモデル1つ。finds・also は measure_model の「どこ/名前」（tests/unit/test_self_check の Breakage と同じ考え方）"""

    apply: Callable[[dict], tuple[PoseSpec, ...]]
    finds: tuple[str, ...]
    expected: Signal
    also: Mapping[str, Signal] = field(default_factory=dict)


def _course(check: str, signal: Signal) -> dict[str, Signal]:
    return {course_key(pose, check): signal for pose in CLEAN_POSES}


BREAKAGES = {
    "remove_head_weights": ModelBreakage(remove_head_weights, (f"a4/{weights.UNWEIGHTED_NAME}",), Signal.RED,
                                         also={f"a4/{weights.NOT_NORMALIZED_NAME}": Signal.YELLOW}),
    "flip_torso_faces": ModelBreakage(flip_torso_faces, (f"a4/{mesh_check.INCONSISTENT_NAME}",), Signal.RED,
                                      also=_course(CLOTH_CHECK, Signal.GRAY)),
    "delete_torso_face": ModelBreakage(delete_torso_face, (f"a4/{mesh_check.HOLES_NAME}",), Signal.RED,
                                       also=_course(CLOTH_CHECK, Signal.GRAY)),
    "move_torso_vertex": ModelBreakage(move_torso_vertex, (f"a4/{symmetry.ASYMMETRY_NAME}",), Signal.RED),
    "shrink_skirt": ModelBreakage(shrink_skirt, tuple(_course(CLOTH_CHECK, Signal.RED)), Signal.RED),
    "reversed_knee": ModelBreakage(reversed_knee, (course_key("knee_bend_left", KNEE_CHECK),), Signal.RED),
    "toes_into_floor": ModelBreakage(toes_into_floor, (course_key("tiptoe", SINK_CHECK),), Signal.RED),
    "rename_ankle_bone": ModelBreakage(rename_ankle_bone, (f"card/{card.SEMI_STANDARD_NAME}",), Signal.YELLOW,
                                       also={f"card/{card.UNKNOWN_BONES_NAME}": Signal.YELLOW}),
}


class SelfCheckModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.info = build_model()
        cls.baseline = measure_model()

    def test_unbroken_model_has_no_red(self):
        """壊していないモデルでは、A4・A1・A5 のどれにも赤が出ない。灰は、理由のある物（アプリの動き）だけ"""
        expected_gray = {course_key(APP_MOTIONS_ALL, APP_MOTION_NAME)}
        for key, measurement in self.baseline.items():
            with self.subTest(measurement=key):
                self.assertNotEqual(measurement.signal, Signal.RED, measurement.reason)
                if measurement.signal == Signal.GRAY:
                    self.assertIn(key, expected_gray)
                    self.assertTrue(measurement.reason)
        for pose in CLEAN_POSES:
            for check in (KNEE_CHECK, SINK_CHECK, CLOTH_CHECK):
                with self.subTest(pose=pose, check=check):
                    self.assertEqual(self.baseline[course_key(pose, check)].signal, Signal.GREEN)

    def test_unbroken_model_measures_everything(self):
        """A4 の面・左右・ウェイトの測る物と、カードと、確かめの動きを、全部測っている"""
        a4 = {key.split("/", 1)[1] for key in self.baseline if key.startswith("a4/")}
        self.assertEqual(a4, {mesh_check.DUPLICATES_NAME, mesh_check.DEGENERATE_NAME, mesh_check.NON_MANIFOLD_NAME,
                              mesh_check.INCONSISTENT_NAME, mesh_check.HOLES_NAME, mesh_check.LOOSE_PARTS_NAME,
                              symmetry.ASYMMETRY_NAME, weights.OVER_LIMIT_NAME, weights.UNWEIGHTED_NAME,
                              weights.NOT_NORMALIZED_NAME, weights.NEGATIVE_NAME, weights.DUPLICATE_NAME})
        self.assertIn(f"card/{card.SEMI_STANDARD_NAME}", self.baseline)
        self.assertEqual(len([key for key in self.baseline if key.startswith("course/")]), 3 * len(CLEAN_POSES) + 1)

    def test_every_breakage_is_found(self):
        """わざと壊したモデルを見つける。ほかの物の信号は変わらない（一緒に変わってよい物は also）"""
        for name, breakage in BREAKAGES.items():
            info = build_model()
            extra = breakage.apply(info)
            bpy.context.view_layer.update()
            results = measure_model(extra)
            with self.subTest(breakage=name):
                for key in breakage.finds:
                    measurement = results[key]
                    self.assertEqual(measurement.signal, breakage.expected, f"{key}：{measurement.reason}")
                    self.assertIsNotNone(measurement.value)
                    self.assertGreater(measurement.value, 0.0, key)
                for key, signal in breakage.also.items():
                    self.assertEqual(results[key].signal, signal, f"{key}：{results[key].reason}")
                    if signal == Signal.GRAY:
                        self.assertTrue(results[key].reason)
                for key, measurement in results.items():
                    if key in breakage.finds or key in breakage.also:
                        continue
                    # 足したポーズの、ほかの確かめは、壊していないポーズと同じく緑
                    before = self.baseline[key].signal if key in self.baseline else Signal.GREEN
                    self.assertEqual(measurement.signal, before, f"{key}：{measurement.reason}")


if __name__ == "__main__":
    unittest.main()
