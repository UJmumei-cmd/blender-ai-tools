"""E2 往復の確認（export/usd_read_bpy.py・export/roundtrip_bpy.py）を Blender の中で確かめる。

(a) pxr で手で書いた USD を読み、読んだ値をぴったり確かめる。USD 自身の計算（UsdSkel の骨の親・スキニング）とも比べる。
(b) Blender の USD の書き出し（convert_blend.py と同じ設定）で書いた物を読み、Blender の形・骨と比べる。
(c) 環境変数 BLENDER_AI_TOOLS_KOUMAKAN_REPO があるとき：テスト用のモデル（tests/fixtures/mmd_model.py）を
    本物の convert_blend.py で書き出し（E1 の export.convert.run_convert で動かす。自分では別の書き出しを作らない）、
    run_roundtrip で比べる。無いときは、理由を出して飛ばす。
ファイルは、どれも一時フォルダに作る。
"""

import hashlib
import math
import os
import tempfile
import unittest
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Quaternion
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdSkel, UsdUtils, Vt

from blender_ai_tools import thresholds
from blender_ai_tools.export import convert, roundtrip, roundtrip_bpy
from blender_ai_tools.export.usd_read_bpy import read_usdz
from blender_ai_tools.measure.weights import WeightTable, linear_blend_skinning, normalize_and_limit
from blender_ai_tools.measure.results import Signal
from blender_ai_tools.model import card, card_bpy, extract_bpy
from tests.fixtures import mmd_model

LIMIT = thresholds.ROUNDTRIP_MAX_DEVIATION
TOLERANCE = thresholds.REPRO_TOLERANCE_LENGTH
KOUMAKAN_ENV = "BLENDER_AI_TOOLS_KOUMAKAN_REPO"
# Z が上から Y が上へ（X のまわりに -90 度）。convert_blend.py の書き出しで /root に入る
Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
# convert_blend.py の export_usdz と同じ書き出しの設定（(b) で、紅魔館が無くても Blender の書き出しを確かめるため）
USD_EXPORT_OPTIONS = dict(
    selected_objects_only=True, export_animation=False, export_materials=True, generate_preview_surface=True,
    export_armatures=True, only_deform_bones=True, export_shapekeys=True, export_uvmaps=True, export_normals=True,
    convert_orientation=True, export_global_forward_selection="NEGATIVE_Z", export_global_up_selection="Y",
    convert_scene_units="METERS", export_textures_mode="NEW", export_lights=False, export_cameras=False,
)


def translation(x, y, z):
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    return matrix


def rotation(axis, degrees, about=(0.0, 0.0, 0.0)):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    t = math.radians(degrees)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    matrix = np.eye(4)
    matrix[:3, :3] = np.eye(3) + math.sin(t) * k + (1.0 - math.cos(t)) * (k @ k)
    c = np.asarray(about, dtype=np.float64)
    matrix[:3, 3] = c - matrix[:3, :3] @ c
    return matrix


def gf(matrix):
    """列ベクトルの形の行列を、Gf（行ベクトルの形）にする"""
    return Gf.Matrix4d(*[float(v) for v in np.asarray(matrix, dtype=np.float64).T.reshape(-1)])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(matrix, points):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def usdskel_points(path, transforms_for):
    """USD 自身（UsdSkel）のスキニングで動かした点（骨格の空間）。transforms_for(骨格の骨の場所) → 骨格の並びの
    スキニングの行列 (J, 4, 4)（列ベクトルの形）。None なら骨格の休みの姿勢（restTransforms）を使う"""
    stage = Usd.Stage.Open(str(path))
    cache = UsdSkel.Cache()
    result = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdSkel.Root):
            continue
        root = UsdSkel.Root(prim)
        cache.Populate(root, Usd.PrimDefaultPredicate)
        for binding in cache.ComputeSkelBindings(root, Usd.PrimDefaultPredicate):
            skeleton = binding.GetSkeleton()
            if transforms_for is None:
                xforms = cache.GetSkelQuery(skeleton).ComputeSkinningTransforms(Usd.TimeCode.Default())
            else:
                joints = [str(t) for t in skeleton.GetJointsAttr().Get()]
                xforms = Vt.Matrix4dArray([gf(m) for m in transforms_for(joints)])
            for query in binding.GetSkinningTargets():
                points = Vt.Vec3fArray(UsdGeom.Mesh(query.GetPrim()).GetPointsAttr().Get())
                if not query.ComputeSkinnedPoints(xforms, points, Usd.TimeCode.Default()):
                    raise AssertionError(f"UsdSkel で動かせませんでした：{query.GetPrim().GetPath()}")
                result[str(query.GetPrim().GetPath())] = np.array(points, dtype=np.float64)
    return result


# MARK: - (a) 手で書いた USD


# 骨格：hips（根）、spine（hips の子）、arm（場所は hips/gap/arm。gap は骨の一覧に無い）
SYN_JOINTS = ["hips", "hips/spine", "hips/gap/arm"]
SYN_BIND = np.stack([translation(0.0, 0.0, 1.0), translation(0.0, 0.0, 1.5),
                     translation(0.25, 0.0, 1.25) @ np.array([[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                                                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])])
SYN_REST = np.stack([SYN_BIND[0], np.linalg.inv(SYN_BIND[0]) @ SYN_BIND[1], np.linalg.inv(SYN_BIND[0]) @ SYN_BIND[2]])
SYN_BODY_POINTS = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 2], [1, 0, 2], [0.5, 0, 2.5]])
SYN_BODY_COUNTS = [3, 5]
SYN_BODY_INDICES = [0, 1, 2, 2, 3, 4, 5, 6]
# 形の骨の並び（skel:joints）：0 = hips/gap/arm（骨格の 2）、1 = hips（骨格の 0）
SYN_BODY_JOINTS = ["hips/gap/arm", "hips"]
SYN_BODY_IDS = np.array([[1, 0], [1, 0], [0, 1], [0, 0], [0, 1], [0, 0], [1, 0]])
SYN_BODY_WEIGHTS = np.array([[1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [1.0, 0.0], [0.25, 0.75], [1.0, 0.0], [1.0, 0.0]])
SYN_GEOM_BIND = translation(0.125, 0.25, 0.5)


def write_synthetic_stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/root")
    root.AddRotateXOp().Set(-90.0)
    stage.SetDefaultPrim(root.GetPrim())

    def name_attr(prim, attr, value):
        prim.CreateAttribute(attr, Sdf.ValueTypeNames.String, custom=True).Set(value)

    skel_root = UsdSkel.Root.Define(stage, "/root/Arm")
    name_attr(skel_root.GetPrim(), "userProperties:blender:object_name", "アーマチュア")
    skeleton = UsdSkel.Skeleton.Define(stage, "/root/Arm/skeleton")
    skeleton.CreateJointsAttr(Vt.TokenArray(SYN_JOINTS))
    skeleton.CreateBindTransformsAttr(Vt.Matrix4dArray([gf(m) for m in SYN_BIND]))
    skeleton.CreateRestTransformsAttr(Vt.Matrix4dArray([gf(m) for m in SYN_REST]))

    # 骨で動く形（skel:joints あり、2本ずつ）とシェイプキー
    body_xform = UsdGeom.Xform.Define(stage, "/root/Arm/Body")
    body_xform.AddTranslateOp().Set(Gf.Vec3d(0.125, 0.25, 0.5))
    name_attr(body_xform.GetPrim(), "userProperties:blender:object_name", "体")
    body = UsdGeom.Mesh.Define(stage, "/root/Arm/Body/Body")
    body.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in SYN_BODY_POINTS]))
    body.CreateFaceVertexCountsAttr(Vt.IntArray(SYN_BODY_COUNTS))
    body.CreateFaceVertexIndicesAttr(Vt.IntArray(SYN_BODY_INDICES))
    name_attr(body.GetPrim(), "userProperties:blender:data_name", "体データ")
    binding = UsdSkel.BindingAPI.Apply(body.GetPrim())
    binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    binding.CreateGeomBindTransformAttr(gf(SYN_GEOM_BIND))
    binding.CreateJointsAttr(Vt.TokenArray(SYN_BODY_JOINTS))
    binding.CreateJointIndicesPrimvar(False, 2).Set(Vt.IntArray([int(v) for v in SYN_BODY_IDS.reshape(-1)]))
    binding.CreateJointWeightsPrimvar(False, 2).Set(Vt.FloatArray([float(v) for v in SYN_BODY_WEIGHTS.reshape(-1)]))
    shape = UsdSkel.BlendShape.Define(stage, "/root/Arm/Body/Body/smile")
    shape.CreateOffsetsAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.25), Gf.Vec3f(0.0, 0.0, 0.5)]))
    shape.CreatePointIndicesAttr(Vt.IntArray([1, 3]))
    binding.CreateBlendShapesAttr(Vt.TokenArray(["smile"]))
    binding.CreateBlendShapeTargetsRel().SetTargets([shape.GetPath()])

    # 形全体で同じ重み（constant）：spine に固く付いた帽子。skel:joints は無い（骨格の並び）
    hat = UsdGeom.Mesh.Define(stage, "/root/Arm/Hat/Hat")
    UsdGeom.Xform.Define(stage, "/root/Arm/Hat")
    hat.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1.75), Gf.Vec3f(0.25, 0, 1.75), Gf.Vec3f(0, 0.25, 1.75)]))
    hat.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    hat.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    hat_binding = UsdSkel.BindingAPI.Apply(hat.GetPrim())
    hat_binding.CreateSkeletonRel().SetTargets([skeleton.GetPath()])
    hat_binding.CreateJointIndicesPrimvar(True, 1).Set(Vt.IntArray([1]))
    hat_binding.CreateJointWeightsPrimvar(True, 1).Set(Vt.FloatArray([1.0]))

    # 骨で動かない形（左手系の面）
    prop_xform = UsdGeom.Xform.Define(stage, "/root/Arm/Prop")
    prop_xform.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
    name_attr(prop_xform.GetPrim(), "userProperties:blender:object_name", "小物")
    prop = UsdGeom.Mesh.Define(stage, "/root/Arm/Prop/Prop")
    prop.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5, 0, 0), Gf.Vec3f(0, 0.5, 0)]))
    prop.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    prop.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    prop.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)

    # 物理の API を当てた部品・カメラ・材質
    rigid = UsdGeom.Xform.Define(stage, "/root/Arm/Rigid")
    rigid.GetPrim().AddAppliedSchema("PhysicsRigidBodyAPI")
    name_attr(rigid.GetPrim(), "userProperties:blender:object_name", "剛体1")
    UsdGeom.Camera.Define(stage, "/root/Cam")
    UsdGeom.Scope.Define(stage, "/root/_materials")
    material = UsdShade.Material.Define(stage, "/root/_materials/m00")
    UsdShade.Shader.Define(stage, "/root/_materials/m00/surface").CreateIdAttr("UsdPreviewSurface")
    stage.GetRootLayer().Save()
    return material


class SyntheticUsdTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        base = Path(cls.folder.name)
        cls.usda = base / "synthetic.usda"
        write_synthetic_stage(cls.usda)
        cls.usdz = base / "synthetic.usdz"
        if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(cls.usda)), str(cls.usdz)):
            raise AssertionError("USDZ を作れませんでした")
        cls.hash = sha256(cls.usdz)
        cls.model = read_usdz(cls.usdz)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def mesh(self, name):
        return next(m for m in self.model.meshes if m.name == name)

    def test_stage_values(self):
        model = self.model
        self.assertEqual(model.up_axis, "Y")
        self.assertEqual(model.meters_per_unit, 1.0)
        np.testing.assert_allclose(model.root_matrix, Z_UP_TO_Y_UP, atol=1e-12)
        self.assertEqual(model.skeleton_path, "/root/Arm/skeleton")
        np.testing.assert_allclose(model.skeleton_world, Z_UP_TO_Y_UP, atol=1e-12)
        self.assertEqual(model.path, str(self.usdz))

    def test_skeleton_values(self):
        model = self.model
        self.assertEqual(model.joint_paths, tuple(SYN_JOINTS))
        self.assertEqual(model.joint_names, ("hips", "spine", "arm"))
        np.testing.assert_array_equal(model.bind, SYN_BIND)
        np.testing.assert_allclose(model.rest_local, SYN_REST, atol=1e-15)
        # 平行移動は右の列（Gf の下の行から転置した）
        np.testing.assert_array_equal(model.bind[2][:3, 3], [0.25, 0.0, 1.25])
        np.testing.assert_array_equal(model.bind[2][3], [0.0, 0.0, 0.0, 1.0])

    def test_parents_match_usdskel_topology(self):
        expected = list(UsdSkel.Topology(Vt.TokenArray(SYN_JOINTS)).GetParentIndices())
        self.assertEqual(expected, [-1, 0, 0])
        np.testing.assert_array_equal(self.model.joint_parents, expected)
        self.assertEqual(roundtrip.path_gaps(self.model.joint_paths), {"hips/gap/arm": ("gap",)})

    def test_skinned_mesh(self):
        body = self.mesh("Body")
        self.assertEqual(body.path, "/root/Arm/Body/Body")
        self.assertEqual(body.source_name, "体")  # 親の Xform の object_name（data_name より先）
        self.assertTrue(body.skinned)
        np.testing.assert_array_equal(body.points, SYN_BODY_POINTS)
        np.testing.assert_array_equal(body.triangles, [[0, 1, 2], [2, 3, 4], [2, 4, 5], [2, 5, 6]])
        np.testing.assert_array_equal(body.geom_bind, SYN_GEOM_BIND)
        np.testing.assert_array_equal(body.joint_indices, SYN_BODY_IDS)
        np.testing.assert_array_equal(body.joint_weights, SYN_BODY_WEIGHTS)
        self.assertEqual(body.joint_names, ("arm", "hips"))
        np.testing.assert_array_equal(body.joint_map, [2, 0])
        self.assertEqual(body.element_size, 2)
        self.assertEqual([s.name for s in body.blend_shapes], ["smile"])
        np.testing.assert_array_equal(body.blend_shapes[0].offsets, [[0.0, 0.0, 0.25], [0.0, 0.0, 0.5]])
        np.testing.assert_array_equal(body.blend_shapes[0].point_indices, [1, 3])

    def test_constant_influences_are_repeated(self):
        hat = self.mesh("Hat")
        self.assertTrue(hat.skinned)
        np.testing.assert_array_equal(hat.joint_indices, [[1], [1], [1]])
        np.testing.assert_array_equal(hat.joint_weights, [[1.0], [1.0], [1.0]])
        self.assertIsNone(hat.joint_map)
        np.testing.assert_array_equal(hat.geom_bind, np.eye(4))

    def test_unskinned_left_handed_mesh(self):
        prop = self.mesh("Prop")
        self.assertFalse(prop.skinned)
        self.assertEqual(prop.source_name, "小物")
        np.testing.assert_array_equal(prop.triangles, [[0, 2, 1]])
        np.testing.assert_allclose(prop.local_to_skeleton, translation(1.0, 0.0, 0.0), atol=1e-12)

    def test_prim_inventory(self):
        model = self.model
        self.assertEqual(model.other_prims, (("/root/Cam", "Camera"),))
        self.assertEqual(model.physics_schemas, (("/root/Arm/Rigid", "PhysicsRigidBodyAPI"),))
        self.assertEqual(model.object_names["/root/Arm/Body"], "体")
        self.assertIn(("/root/_materials/m00/surface", "Shader"), model.prims)
        result = roundtrip.check_no_physics(model, ["剛体1"])
        self.assertEqual(result.signal, Signal.RED)
        self.assertEqual(result.details["physics_named"], {"/root/Arm/Rigid": "剛体1"})

    def test_skinning_matches_usdskel(self):
        """skin_exported を USD 自身のスキニング（UsdSkel）と比べる（行列の向き・skel:joints の読みかえ・geomBind）"""
        posed = np.stack([translation(0.25, 0.0, 0.0) @ SYN_BIND[0],
                          rotation((1, 0, 0), 30.0, about=(0.0, 0.0, 1.5)) @ SYN_BIND[1],
                          rotation((0, 1, 0), 45.0, about=(0.25, 0.0, 1.25)) @ SYN_BIND[2]])
        skinning = posed @ np.linalg.inv(SYN_BIND)
        reference = usdskel_points(self.usdz, lambda joints: skinning)
        for name in ("Body", "Hat"):
            mesh = self.mesh(name)
            mine = roundtrip.skin_exported(mesh, self.model.bind, posed)
            np.testing.assert_allclose(mine, reference[mesh.path], atol=1e-6, err_msg=name)
        # 手で計算した値でも確かめる：点3 は arm だけに付く。arm の根元 (0.25, 0, 1.25) から見た位置を Y のまわりに回す
        p = transform(SYN_GEOM_BIND, SYN_BODY_POINTS[3:4])[0]
        expected = transform(rotation((0, 1, 0), 45.0, about=(0.25, 0.0, 1.25)), [p])[0]
        np.testing.assert_allclose(roundtrip.skin_exported(self.mesh("Body"), self.model.bind, posed)[3], expected,
                                   atol=1e-12)

    def test_rest_consistency_matches_usdskel(self):
        reference = usdskel_points(self.usdz, None)
        body = self.mesh("Body")
        np.testing.assert_allclose(reference[body.path], roundtrip.rest_points(body), atol=1e-6)
        result = roundtrip.check_skeleton(self.model)
        self.assertEqual(result.signal, Signal.GREEN)
        self.assertEqual(result.details["path_gaps"], {"arm": ("gap",)})

    def test_usda_and_usdz_read_the_same(self):
        direct = read_usdz(self.usda)
        np.testing.assert_array_equal(direct.bind, self.model.bind)
        self.assertEqual([m.path for m in direct.meshes], [m.path for m in self.model.meshes])

    def test_file_is_not_changed(self):
        read_usdz(self.usdz)
        self.assertEqual(sha256(self.usdz), self.hash)

    def test_errors(self):
        with self.assertRaises(ValueError):
            read_usdz(Path(self.folder.name) / "none.usdz")
        broken = Path(self.folder.name) / "broken.usda"
        broken.write_text("#usda 1.0\nこわれた\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_usdz(broken)
        empty = Path(self.folder.name) / "empty.usda"
        stage = Usd.Stage.CreateNew(str(empty))  # 骨格の無いファイル
        UsdGeom.Xform.Define(stage, "/root")
        stage.GetRootLayer().Save()
        with self.assertRaises(ValueError):
            read_usdz(empty)


# MARK: - (b) Blender が書いた USD


def build_small_model():
    """日本語の名前・位置をずらしたアーマチュアとメッシュ・シェイプキー・材質の、小さなモデル"""
    bpy.ops.wm.read_homefile(use_empty=True)
    data = bpy.data.armatures.new("骨")
    armature = bpy.data.objects.new("アーマチュア", data)
    bpy.context.scene.collection.objects.link(armature)
    armature.location = (0.5, -0.25, 0.125)
    armature.rotation_euler = (0.0, 0.0, math.radians(90.0))
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    upper = data.edit_bones.new("左腕")
    upper.head, upper.tail, upper.roll = (0.0, 0.0, 1.0), (0.5, 0.0, 1.0), 0.3
    lower = data.edit_bones.new("左ひじ")
    lower.head, lower.tail, lower.parent = (0.5, 0.0, 1.0), (1.0, 0.0, 0.75), upper
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh = bpy.data.meshes.new("スカート")
    mesh.from_pydata([(0, 0, 0.9), (0.4, 0, 0.9), (0.6, 0, 0.9), (0.9, 0, 0.8), (0.9, 0.1, 0.8), (0.4, 0.1, 0.9)], [],
                     [(0, 1, 5), (1, 2, 3, 4)])
    obj = bpy.data.objects.new("スカート", mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent = armature
    obj.location = (0.25, 0.125, 0.0)
    obj.rotation_euler = (0.1, 0.0, 0.0)
    modifier = obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = armature
    first, second = obj.vertex_groups.new(name="左腕"), obj.vertex_groups.new(name="左ひじ")
    first.add([0, 1], 1.0, "REPLACE")
    first.add([2], 0.5, "REPLACE")
    second.add([2], 0.5, "REPLACE")
    second.add([3, 4], 1.0, "REPLACE")
    first.add([5], 0.75, "REPLACE")
    second.add([5], 0.25, "REPLACE")
    obj.shape_key_add(name="Basis")
    key = obj.shape_key_add(name="あ")
    key.data[4].co = (0.9, 0.1, 0.85)
    key.value = 0.0
    obj.data.materials.append(bpy.data.materials.new("肌"))
    bpy.context.view_layer.update()
    return armature, obj


class BlenderWrittenUsdTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.armature, self.obj = build_small_model()
        self.path = Path(self.folder.name) / "small.usdc"
        bpy.ops.object.select_all(action="DESELECT")
        for obj in (self.armature, self.obj):
            obj.select_set(True)
        bpy.ops.wm.usd_export(filepath=str(self.path), **USD_EXPORT_OPTIONS)
        self.model = read_usdz(self.path)

    def tearDown(self):
        self.folder.cleanup()

    def test_values_match_blender(self):
        model, mesh = self.model, self.model.meshes[0]
        self.assertEqual(model.joint_names, ("左腕", "左ひじ"))
        np.testing.assert_array_equal(model.joint_parents, [-1, 0])
        for j, name in enumerate(model.joint_names):
            np.testing.assert_allclose(model.bind[j], np.array(self.armature.data.bones[name].matrix_local),
                                       atol=1e-9, err_msg=name)
        self.assertEqual(mesh.source_name, "スカート")
        arm_world = np.array(self.armature.matrix_world)
        expected = np.linalg.inv(arm_world) @ np.array(self.obj.matrix_world)
        np.testing.assert_allclose(mesh.geom_bind, expected, atol=1e-6)
        np.testing.assert_allclose(model.skeleton_world, Z_UP_TO_Y_UP @ arm_world, atol=1e-6)
        rest, _ = extract_bpy.mesh_rest(self.obj, space="armature")
        np.testing.assert_allclose(roundtrip.rest_points(mesh), rest, atol=1e-6)
        self.assertEqual([s.name for s in mesh.blend_shapes], ["あ"])
        shape = mesh.blend_shapes[0]
        offsets = np.zeros((mesh.vertex_count, 3))
        offsets[shape.point_indices if shape.point_indices is not None else slice(None)] = shape.offsets
        np.testing.assert_allclose(offsets[4], [0.0, 0.0, 0.05], atol=1e-7)
        np.testing.assert_allclose(np.delete(offsets, 4, axis=0), 0.0, atol=1e-7)
        # Blender はシェイプキーの値のために SkelAnimation を書く。知らない種類の部品は無い
        self.assertIn("SkelAnimation", {kind for _, kind in model.prims})
        self.assertEqual(model.other_prims, ())

    def test_posed_skinning_matches_blender(self):
        extract_bpy.set_pose(self.armature, {"左腕": tuple(Quaternion((0, 0, 1), math.radians(20.0))),
                                             "左ひじ": tuple(Quaternion((1, 0, 0), math.radians(50.0)))})
        names = extract_bpy.bone_names(self.armature)
        posed = extract_bpy.pose_matrices(self.armature, space="armature")[0]
        joints = roundtrip.posed_joint_matrices(self.model.joint_names, {n: n for n in names}, names, posed)
        mine = roundtrip.skin_exported(self.model.meshes[0], self.model.bind, joints)
        blender = extract_bpy.mesh_evaluated(self.obj, space="armature", armature=self.armature)
        self.assertGreater(np.abs(blender - roundtrip.rest_points(self.model.meshes[0])).max(), 0.05)  # 動いた
        np.testing.assert_allclose(mine, blender, atol=1e-6)


class BlenderHelpersTest(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        self.info = mmd_model.build_model()
        self.body = bpy.data.objects[mmd_model.BODY_NAME]

    def add_mask(self, *, invert=True, show_render=True, mode="VERTEX_GROUP"):
        start, stop = self.info["parts"][mmd_model.BODY_NAME]["torso"]
        group = self.body.vertex_groups.new(name="隠す")
        group.add(list(range(start, stop)), 1.0, "REPLACE")
        modifier = self.body.modifiers.new("服で隠す", "MASK")
        modifier.mode = mode
        modifier.vertex_group = "隠す"
        modifier.invert_vertex_group = invert
        modifier.show_render = show_render
        expected = np.zeros(len(self.body.data.vertices), dtype=bool)
        expected[start:stop] = True
        return expected

    def test_mask_hidden_vertices(self):
        expected = self.add_mask()
        np.testing.assert_array_equal(roundtrip_bpy.mask_hidden_vertices(self.body), expected)
        self.body.modifiers["服で隠す"].invert_vertex_group = False  # グループの中だけを残す
        np.testing.assert_array_equal(roundtrip_bpy.mask_hidden_vertices(self.body), ~expected)
        self.body.modifiers["服で隠す"].show_render = False
        self.assertFalse(roundtrip_bpy.mask_hidden_vertices(self.body).any())
        self.body.modifiers["服で隠す"].show_render = True
        self.body.modifiers["服で隠す"].mode = "ARMATURE"
        self.assertFalse(roundtrip_bpy.mask_hidden_vertices(self.body).any())
        with self.assertRaises(ValueError):
            roundtrip_bpy.mask_hidden_vertices(bpy.data.objects[mmd_model.ARMATURE_NAME])

    def test_physics_object_names(self):
        empty = bpy.data.objects.new("剛体の親", None)
        bpy.context.scene.collection.objects.link(empty)
        mesh = bpy.data.meshes.new("rb")
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        rigid = bpy.data.objects.new("剛体1", mesh)
        bpy.context.scene.collection.objects.link(rigid)
        bpy.context.view_layer.objects.active = rigid
        rigid.select_set(True)
        bpy.ops.rigidbody.object_add()
        names = roundtrip_bpy.physics_object_names()
        self.assertIn("剛体の親", names)
        self.assertIn("剛体1", names)
        self.assertNotIn(mmd_model.BODY_NAME, names)
        self.assertNotIn(mmd_model.ARMATURE_NAME, names)

    def test_source_shape_keys(self):
        self.assertEqual(roundtrip_bpy.source_shape_keys([self.body]), ("まばたき", "あ"))

    def test_source_weights(self):
        """メッシュごとの Blender のウェイト（bone_weights と同じ）。5本の骨の頂点は、見本の5本の頂点だけ"""
        armature = bpy.data.objects[mmd_model.ARMATURE_NAME]
        skirt = bpy.data.objects[mmd_model.SKIRT_NAME]
        tables, failed = roundtrip_bpy.source_weights([self.body, skirt], armature)
        self.assertEqual((list(tables), failed), ([mmd_model.BODY_NAME, mmd_model.SKIRT_NAME], {}))
        expected, _ = extract_bpy.bone_weights(self.body, armature)
        np.testing.assert_array_equal(tables[mmd_model.BODY_NAME].bone_ids, expected.bone_ids)
        np.testing.assert_array_equal(tables[mmd_model.BODY_NAME].weights, expected.weights)
        deform = extract_bpy.bone_rest(armature, space="armature").deform
        np.testing.assert_array_equal(roundtrip.limited_vertices(tables[mmd_model.BODY_NAME], deform=deform),
                                      sorted(self.info["five_influence_vertices"]))
        np.testing.assert_array_equal(roundtrip.limited_vertices(tables[mmd_model.SKIRT_NAME], deform=deform), [])

    def test_test_poses_bend_knees_and_elbows_normally(self):
        """確かめのポーズを本物の見本に当てると、ひざ・ひじは、ふつうの向きに曲がる（逆曲がりにしない）。

        キャラは -Y を向く。ひざ：もも は前（-Y）へ出て、すねは もも から見て後ろ（+Y）へ 60 度曲がる。
        ひじ：前腕は前（-Y）へ曲がる
        """
        armature = bpy.data.objects[mmd_model.ARMATURE_NAME]
        rest = extract_bpy.bone_rest(armature, space="armature")

        def bend_deg(hip, knee, ankle):
            """もも から すね への曲がり（+X のまわり。正は、すねが後ろへ行く ふつうの曲がり）"""
            thigh, shin = knee - hip, ankle - knee
            return math.degrees(math.atan2(np.cross(thigh, shin)[0], float(np.dot(thigh, shin))))

        def joints(pose, names):
            rotations, missing = roundtrip.resolve_pose(roundtrip.TEST_POSES[pose], rest.names, rest.matrices)
            self.assertEqual(missing, ())
            extract_bpy.set_pose(armature, rotations)
            try:
                return extract_bpy.joint_positions(armature, names, space="armature")[0]
            finally:
                extract_bpy.reset_pose(armature)

        for side in ("左", "右"):
            with self.subTest(side=side):
                names = [f"{side}足D", f"{side}ひざD", f"{side}足首D"]
                rest_bend = bend_deg(*rest.heads[rest.indices(names)])
                self.assertGreater(rest_bend, 0.0)  # 見本のひざは、休みの姿勢でも少し前に出ている
                hip, knee, ankle = joints("knees", names)
                self.assertLess((knee - hip)[1], -0.1)  # もも は前へ
                self.assertAlmostEqual(bend_deg(hip, knee, ankle), rest_bend + 60.0, places=4)  # さらに 60 度、ふつうの向き
                shoulder, elbow, wrist = joints("elbows", [f"{side}腕", f"{side}ひじ", f"{side}手首"])
                self.assertLess((wrist - elbow)[1], -0.1)  # 前腕は前へ
                self.assertAlmostEqual(float((elbow - shoulder)[1]), 0.0, places=6)  # 上腕は動かない

    def test_unreadable_file_is_gray(self):
        results = roundtrip_bpy.run_roundtrip("/nonexistent/character.usdz", {})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].signal, Signal.GRAY)
        self.assertEqual(results[0].name, roundtrip.DEVIATION_NAME)

    def test_other_scene_is_rejected(self):
        with self.assertRaises(ValueError):
            roundtrip_bpy.run_roundtrip("x.usdz", {}, scene=bpy.data.scenes.new("ほか"))


# MARK: - (c) 本物の convert_blend.py


def add_hiding_mask(vertices):
    """Body の頂点 vertices を隠す MASK（convert_blend.py は、その頂点を消して書き出す）。隠れる頂点の印を返す"""
    body = bpy.data.objects[mmd_model.BODY_NAME]
    group = body.vertex_groups.new(name="隠す")
    group.add([int(v) for v in vertices], 1.0, "REPLACE")
    modifier = body.modifiers.new("服で隠す", "MASK")
    modifier.mode = "VERTEX_GROUP"
    modifier.vertex_group = "隠す"
    modifier.invert_vertex_group = True
    hidden = np.zeros(len(body.data.vertices), dtype=bool)
    hidden[list(vertices)] = True
    return hidden


def add_torso_mask(info):
    """胴を隠す MASK。隠れる頂点の印を返す"""
    start, stop = info["parts"][mmd_model.BODY_NAME]["torso"]
    return add_hiding_mask(range(start, stop))


# アーマチュアのオブジェクトの大きさ（スケール）を変えた見本の大きさ。骨格の空間の長さは、アプリでこの倍になる
ARMATURE_SCALE = 2.0


def build_fixture(mask, scale=1.0, *, five_influences=True, hide_five=False):
    """見本を作る。mask：胴を隠す MASK。hide_five：5本の骨の頂点を隠す MASK。five_influences：5本の骨の頂点を作るか"""
    bpy.ops.wm.read_homefile(use_empty=True)
    info = mmd_model.build_model(five_influences=five_influences)
    hidden = add_torso_mask(info) if mask else None
    if hide_five:
        hidden = add_hiding_mask(info["five_influence_vertices"])
    bpy.data.objects[mmd_model.ARMATURE_NAME].scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    return info, hidden


def folder_listing(folder):
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*")) if folder.exists() else None


@unittest.skipUnless(os.environ.get(KOUMAKAN_ENV), f"{KOUMAKAN_ENV} が無いので、本物の convert_blend.py の試しを飛ばします")
class ConverterRoundtripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = os.environ[KOUMAKAN_ENV]
        try:
            paths = convert.koumakan_paths(cls.repo)
        except ValueError as error:
            raise unittest.SkipTest(f"{KOUMAKAN_ENV} の場所に紅魔館の書き出しがありません：{error}") from None
        cls.app_model = paths.app_model_dir
        cls.app_listing = folder_listing(cls.app_model)
        # setUpClass が途中で失敗しても消えるように、クラスの片づけに登録する
        folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        root = Path(folder.name)
        cls.runs = {mask: cls.convert(root / ("masked" if mask else "plain"), mask) for mask in (False, True)}
        cls.runs["scaled"] = cls.convert(root / "scaled", False, ARMATURE_SCALE)
        cls.runs["four"] = cls.convert(root / "four", False, five_influences=False)
        cls.runs["five_hidden"] = cls.convert(root / "five_hidden", False, hide_five=True)

    @classmethod
    def convert(cls, folder, mask, scale=1.0, **options):
        """見本を保存し、E1 の run_convert（本物の convert_blend.py）で書き出す。書き出せなければ失敗にする"""
        info, _ = build_fixture(mask, scale, **options)
        blend = mmd_model.save_model(folder / "model.blend")
        result = convert.run_convert(blend, folder / "out", blender=bpy.app.binary_path, repo=cls.repo)
        if not result.ok:
            raise AssertionError(f"convert_blend.py の書き出しが失敗しました：{result.problems}\n"
                                 f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        usdz = result.output_dir / convert.USDZ_NAME
        return {"info": info, "blend": blend, "usdz": usdz, "rig": result.rig_path, "rig_data": result.rig,
                "convert": result, "blend_hash": sha256(blend), "usdz_hash": sha256(usdz)}

    def setUp(self):
        self.info, self.hidden = build_fixture(False)
        self.armature = bpy.data.objects[mmd_model.ARMATURE_NAME]

    def run_plain(self):
        run = self.runs[False]
        return roundtrip_bpy.run_roundtrip(run["usdz"], run["rig"])

    def deviations(self, results):
        return {m.details["pose"]: m for m in results if m.name == roundtrip.DEVIATION_NAME}

    def body_details(self, measurement):
        return next(d for d in measurement.details["meshes"].values() if d["source_name"] == mmd_model.BODY_NAME)

    def assert_shapes_within_limit(self, results, five):
        deviations = self.deviations(results)
        self.assertEqual(list(deviations), [roundtrip.REST_POSE, *roundtrip.TEST_POSES])
        for pose, measurement in deviations.items():
            with self.subTest(pose=pose):
                self.assertEqual(measurement.details["unmeasured_count"], 0)
                if pose != roundtrip.REST_POSE:
                    self.assertGreater(measurement.details["moved_joint_count"], 0)
                for mesh in measurement.details["meshes"].values():
                    others = ~np.isin(mesh["source_index"], five)
                    self.assertTrue(np.isfinite(mesh["deviations"]).all())
                    self.assertLess(float(mesh["deviations"][others].max()), LIMIT, mesh["source_name"])
        return deviations

    def expected_five_deviation(self, five):
        """5本の骨の頂点の、elbows のポーズでのずれを Blender の値から計算する：
        Blender（5本、合計で割る）と、convert_blend.py（重い順に4本、合計を 1 に直す）の差"""
        rest = extract_bpy.bone_rest(self.armature, space="armature")
        rotations, missing = roundtrip.resolve_pose(roundtrip.TEST_POSES["elbows"], rest.names, rest.matrices)
        self.assertEqual(missing, ())
        extract_bpy.set_pose(self.armature, rotations)
        try:
            skinning = extract_bpy.skinning_matrices(self.armature)[0]
        finally:
            extract_bpy.reset_pose(self.armature)
        body = bpy.data.objects[mmd_model.BODY_NAME]
        table, _ = extract_bpy.bone_weights(body, self.armature)
        points, _ = extract_bpy.mesh_rest(body, space="armature")
        rows = WeightTable(table.bone_ids[five], table.weights[five])
        limited, changed = normalize_and_limit(rows, deform=rest.deform)
        self.assertTrue(changed.all())
        full = linear_blend_skinning(points[five], rows, skinning, deform=rest.deform)
        four = linear_blend_skinning(points[five], limited, skinning, deform=rest.deform)
        return np.linalg.norm(full - four, axis=1)

    def test_plain_roundtrip(self):
        results = self.run_plain()
        self.assertEqual([m.name for m in results],
                         [roundtrip.COVERAGE_NAME] + [roundtrip.DEVIATION_NAME] * (1 + len(roundtrip.TEST_POSES))
                         + [roundtrip.SKELETON_NAME, roundtrip.JOINT_NAMES_NAME, roundtrip.HUMANOID_NAME,
                            roundtrip.SHAPE_KEYS_NAME, roundtrip.INFLUENCES_NAME, roundtrip.WEIGHT_SUMS_NAME,
                            roundtrip.INFLUENCE_EFFECT_NAME, roundtrip.PHYSICS_NAME])
        by_name = {m.name: m for m in results}
        coverage = by_name[roundtrip.COVERAGE_NAME]
        self.assertEqual(coverage.signal, Signal.GREEN, coverage.reason)
        self.assertEqual(set(coverage.details["match_methods"].values()), {"order"})

        five = np.array(self.info["five_influence_vertices"])
        self.assertEqual(len(five), 4)
        deviations = self.assert_shapes_within_limit(results, five)
        self.assertEqual(deviations[roundtrip.REST_POSE].signal, Signal.GREEN)

        # 5本の骨の頂点：elbows で、ずれが見つかり、いちばん悪い点として出る。値は Blender から計算した物と同じ
        elbows = deviations["elbows"]
        body = self.body_details(elbows)
        measured = body["deviations"][np.searchsorted(body["source_index"], five)]
        np.testing.assert_array_equal(body["source_index"][np.searchsorted(body["source_index"], five)], five)
        expected = self.expected_five_deviation(five)
        self.assertTrue((expected > 1e-5).all())
        np.testing.assert_allclose(measured, expected, atol=2e-6)
        self.assertEqual({row["source_index"] for row in elbows.details["worst"][:4]}, set(five.tolist()))
        self.assertAlmostEqual(elbows.value, float(expected.max()), delta=2e-6)
        # 形のずれ（roundtrip_deviation）は、いままでどおり基準だけで決める：0.46mm は 0.5mm 未満なので緑
        self.assertLess(elbows.value, LIMIT)
        self.assertEqual(elbows.signal, Signal.GREEN)

        # 1点4本までにした影響：5本の頂点の形が変わるので黄（ユーザーと決めた）。値は elbows の5本の頂点のずれ
        effect = by_name[roundtrip.INFLUENCE_EFFECT_NAME]
        self.assertEqual(effect.signal, Signal.YELLOW, effect.reason)
        self.assertEqual(effect.value, elbows.value)
        self.assertAlmostEqual(effect.value, 0.00046, delta=0.00002)  # 約 0.46mm
        self.assertEqual(effect.details["limited_count"], 4)
        self.assertEqual(effect.details["changed_count"], 4)
        self.assertEqual(effect.details["unmeasured_count"], 0)
        self.assertEqual(effect.details["poses"], (roundtrip.REST_POSE, *roundtrip.TEST_POSES))
        self.assertEqual(effect.details["hidden_limited"], {})
        limited = effect.details["meshes"][mmd_model.BODY_NAME]
        np.testing.assert_array_equal(limited["limited"], np.sort(five))
        np.testing.assert_allclose(limited["deviations"], expected[np.argsort(five)], atol=2e-6)
        self.assertEqual(limited["worst_pose"], ("elbows",) * 4)
        for pose in (roundtrip.REST_POSE, "arms_up", "knees", "twist"):
            # ほかのポーズでは、5本の骨がいっしょに動く（か、動かない）ので、4本にしても形は変わらない
            self.assertLess(float(limited["poses"][pose].max()), TOLERANCE, pose)
        np.testing.assert_array_equal(effect.details["meshes"][mmd_model.SKIRT_NAME]["limited"], [])

        for name in (roundtrip.JOINT_NAMES_NAME, roundtrip.HUMANOID_NAME, roundtrip.SHAPE_KEYS_NAME,
                     roundtrip.WEIGHT_SUMS_NAME, roundtrip.PHYSICS_NAME):
            self.assertEqual(by_name[name].signal, Signal.GREEN, f"{name}：{by_name[name].reason}")
        influences = by_name[roundtrip.INFLUENCES_NAME]
        self.assertEqual((influences.signal, influences.value), (Signal.GREEN, 4.0))

    def test_agrees_with_e1_and_a1(self):
        """E1（run_convert の報告と記録）・A1（登録カード）・A4（extract_bpy のウェイト）・E2 が、同じ見本で同じ数を出す"""
        run = self.runs[False]
        result = run["convert"]
        rig = run["rig_data"]
        self.assertEqual(result.signal, Signal.GREEN, result.warnings)
        # E1 が記録した指紋は、E2 が読んだファイル・元の .blend の物
        self.assertEqual(result.usdz.sha256, run["usdz_hash"])
        self.assertEqual(result.record.inputs[0].sha256, run["blend_hash"])
        # rig は、ファイルの場所でも、E1 が確かめて読んだ辞書でも、同じ結果
        from_path = roundtrip_bpy.run_roundtrip(run["usdz"], run["rig"])
        from_dict = roundtrip_bpy.run_roundtrip(run["usdz"], rig)
        self.assertEqual([(m.name, m.signal, m.value) for m in from_dict], [(m.name, m.signal, m.value) for m in from_path])
        by_name = {m.name: m for m in from_path}
        summary = result.summary
        # アプリのヒューマノイドの骨：E1 の報告の数 = rig の bones の数 = E2 が調べた数（見つからない骨は無い）
        self.assertEqual(summary["missing_humanoid"], [])
        self.assertEqual(summary["humanoid_count"], len(rig["bones"]))
        humanoid = by_name[roundtrip.HUMANOID_NAME]
        self.assertEqual((humanoid.signal, humanoid.details["humanoid_count"]), (Signal.GREEN, len(rig["bones"])))
        # 1点に効く骨：E1 が絞った頂点の数 = A4 のウェイトで形を動かす骨が上限を超える頂点の数 = 見本の5本の頂点
        limit = thresholds.MAX_BONE_INFLUENCES
        self.assertEqual(summary["influence_limit"], limit)
        rest = extract_bpy.bone_rest(self.armature, space="armature")
        over = 0
        for name in (mmd_model.BODY_NAME, mmd_model.SKIRT_NAME):
            table, names = extract_bpy.bone_weights(bpy.data.objects[name], self.armature)
            self.assertEqual(names, rest.names)  # 骨の番号は同じ並び
            used = (table.weights != 0.0) & (table.bone_ids >= 0) & rest.deform[np.maximum(table.bone_ids, 0)]
            over += int((used.sum(axis=1) > limit).sum())
        self.assertEqual(over, len(self.info["five_influence_vertices"]))
        self.assertEqual(summary["vertices_limited"], over)
        self.assertEqual(by_name[roundtrip.INFLUENCES_NAME].value, float(limit))
        # E2 が「4本にされた」と数えた頂点の数も、E1 が絞った頂点の数と同じ（convert_blend.py と同じ数え方）
        self.assertEqual(by_name[roundtrip.INFLUENCE_EFFECT_NAME].details["limited_count"], summary["vertices_limited"])
        # 腕の下がり角：E1 の報告 = rig の restArmDownDegrees = A1 の登録カード（同じ式・同じ丸め）。
        # アーマチュアの大きさだけを変えた見本でも同じ（向きを変えない大きさでは、世界とアーマチュアの空間で同じ角度）
        model_card = card_bpy.build_card()
        self.assertEqual(summary["rest_arm_down_deg"], rig["restArmDownDegrees"])
        self.assertEqual(model_card.value(card.ARM_REST_DOWN["L"]), rig["restArmDownDegrees"])
        self.assertEqual(self.runs["scaled"]["rig_data"]["restArmDownDegrees"], rig["restArmDownDegrees"])

    def test_skeleton_rest_matches_usdskel(self):
        """骨格の休みの姿勢の確かめを、USD 自身の計算（restTransforms から組み立てたスキニング）と比べる。

        Blender 5.2.2 の書き出しは、形を動かす骨の親のうち 腰キャンセル左 を骨の一覧に入れないので、
        左脚の休みの姿勢がずれる（赤）。直った版でも、USD の計算と同じ答えになることを確かめる
        """
        results = self.run_plain()
        skeleton = next(m for m in results if m.name == roundtrip.SKELETON_NAME)
        run = self.runs[False]
        exported = read_usdz(run["usdz"])
        reference = usdskel_points(run["usdz"], None)
        displacement = max(float(np.linalg.norm(reference[m.path] - roundtrip.rest_points(m), axis=1).max())
                           for m in exported.meshes if m.skinned)
        self.assertAlmostEqual(skeleton.value, displacement, delta=1e-6)
        self.assertEqual(skeleton.signal == Signal.RED, displacement >= LIMIT)
        bones = set(mmd_model.bone_parents())
        for joint, missing in skeleton.details["path_gaps_original"].items():
            self.assertIn(joint, bones)
            self.assertTrue(set(missing) <= bones)
        if skeleton.signal == Signal.RED:
            self.assertTrue(skeleton.details["path_gaps"])
            self.assertTrue(skeleton.details["mismatched_original"])

    def test_state_is_restored(self):
        armature = self.armature
        body = bpy.data.objects[mmd_model.BODY_NAME]
        skirt = bpy.data.objects[mmd_model.SKIRT_NAME]
        scene = bpy.context.scene
        scene.frame_set(7)
        extract_bpy.set_pose(armature, {"頭": tuple(Quaternion((1, 0, 0), math.radians(10.0)))})
        armature.data.pose_position = "REST"
        skirt.modifiers[mmd_model.MODIFIER_NAME].show_viewport = False  # 描画だけで有効（書き出しは描画の設定）
        body.hide_set(True)
        neck = armature.pose.bones["首"]
        neck.rotation_mode = "XZY"  # クォータニオンでない骨も、値と形を戻す
        neck.rotation_euler = (0.1, -0.2, 0.3)
        armature.pose.bones["左腕"].location = (0.01, 0.0, 0.0)
        bpy.ops.object.select_all(action="DESELECT")
        skirt.select_set(True)
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="POSE")

        def pose_state():
            return {bone.name: (tuple(bone.location), tuple(bone.rotation_quaternion), tuple(bone.rotation_euler),
                                tuple(bone.rotation_axis_angle), bone.rotation_mode, tuple(bone.scale))
                    for bone in armature.pose.bones}

        before = pose_state()
        selected = set(bpy.context.selected_objects)
        filepath = bpy.data.filepath
        run = self.runs[False]

        try:
            results = roundtrip_bpy.run_roundtrip(run["usdz"], run["rig"])
            self.assertEqual(armature.mode, "POSE")
            self.assertEqual(bpy.context.view_layer.objects.active, armature)
            self.assertEqual(set(bpy.context.selected_objects), selected)
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

        self.assertEqual(pose_state(), before)
        self.assertEqual(armature.data.pose_position, "REST")
        self.assertFalse(skirt.modifiers[mmd_model.MODIFIER_NAME].show_viewport)
        self.assertTrue(body.hide_get())
        self.assertEqual(scene.frame_current, 7)
        self.assertEqual(bpy.data.filepath, filepath)
        self.assertEqual(sha256(run["blend"]), run["blend_hash"])
        self.assertEqual(sha256(run["usdz"]), run["usdz_hash"])
        self.assertEqual(folder_listing(self.app_model), self.app_listing)
        # スカートは、ビューポートで外したアーマチュアでも、描画の設定で動かして比べた（ずれない）
        self.assert_shapes_within_limit(results, np.array(self.info["five_influence_vertices"]))

    def test_masked_roundtrip(self):
        self.info, self.hidden = build_fixture(True)
        body = bpy.data.objects[mmd_model.BODY_NAME]
        np.testing.assert_array_equal(roundtrip_bpy.mask_hidden_vertices(body), self.hidden)
        run = self.runs[True]
        exported = read_usdz(run["usdz"])
        exported_body = next(m for m in exported.meshes if m.source_name == mmd_model.BODY_NAME)
        self.assertEqual(exported_body.vertex_count, len(self.hidden) - int(self.hidden.sum()))

        results = roundtrip_bpy.run_roundtrip(run["usdz"], run["rig"])

        coverage = results[0]
        self.assertEqual(coverage.name, roundtrip.COVERAGE_NAME)
        self.assertEqual(coverage.signal, Signal.GREEN, coverage.reason)
        self.assertEqual(coverage.details["expected_hidden_counts"][mmd_model.BODY_NAME], int(self.hidden.sum()))
        # E1 の報告の「服で隠れる頂点を消した数」も、E2 が数えた消えるはずの頂点と同じ（同じ決まりで数える）
        self.assertEqual(run["convert"].summary["masked_vertices_removed"], int(self.hidden.sum()))
        self.assertEqual(coverage.details["match_methods"][exported_body.path], "order")
        deviations = self.assert_shapes_within_limit(results, np.array(self.info["five_influence_vertices"]))
        self.assertFalse(np.isin(self.body_details(deviations["rest"])["source_index"], np.flatnonzero(self.hidden))
                         .any())
        self.assertTrue(body.modifiers["服で隠す"].show_viewport)

    def test_edit_mode_changes_are_read(self):
        """編集モードで消した頂点も、データに写してから数える（古い頂点の数で止まらない）。モードは変えない"""
        body = bpy.data.objects[mmd_model.BODY_NAME]
        bpy.context.view_layer.objects.active = body
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            edit = bmesh.from_edit_mesh(body.data)
            edit.verts.ensure_lookup_table()
            bmesh.ops.delete(edit, geom=[edit.verts[5]], context="VERTS")
            bmesh.update_edit_mesh(body.data)
            run = self.runs[False]
            results = roundtrip_bpy.run_roundtrip(run["usdz"], run["rig"])
            self.assertEqual(body.mode, "EDIT")
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
        coverage = results[0]
        self.assertEqual(coverage.name, roundtrip.COVERAGE_NAME)
        path = next(p for p in coverage.details["match_methods"] if p.endswith("/Body"))
        self.assertEqual(coverage.details["unmatched_exported"], {path: 1})  # 書き出した頂点 5 は、元に無い
        self.assertEqual(coverage.signal, Signal.GRAY)
        rest = self.deviations(results)[roundtrip.REST_POSE]
        self.assertEqual(rest.signal, Signal.GRAY)
        self.assertEqual(rest.details["unmeasured_count"], 1)
        self.assertEqual(int(np.flatnonzero(np.isnan(self.body_details(rest)["deviations"]))[0]), 5)

    def test_scaled_armature_is_measured_in_meters(self):
        """アーマチュアのオブジェクトの大きさが 2 なら、アプリでの形も 2 倍。ずれは、アプリで見える長さ（メートル）で測る。

        骨格の空間の形は大きさ 1 の見本と同じなので、どの長さも大きさ 1 の見本のちょうど 2 倍になる。
        5本の骨の頂点のずれ（大きさ 1 では 0.46mm）は 0.92mm になり、基準を超えて赤
        """
        plain = {(m.name, m.details.get("pose")): m for m in self.run_plain()}
        self.info, self.hidden = build_fixture(False, ARMATURE_SCALE)
        run = self.runs["scaled"]
        exported = read_usdz(run["usdz"])
        np.testing.assert_allclose(np.linalg.svd(exported.skeleton_world[:3, :3], compute_uv=False),
                                   [ARMATURE_SCALE] * 3, atol=1e-6)
        results = roundtrip_bpy.run_roundtrip(run["usdz"], run["rig"])
        by_key = {(m.name, m.details.get("pose")): m for m in results}
        self.assertEqual(list(by_key), list(plain))
        for pose in (roundtrip.REST_POSE, *roundtrip.TEST_POSES):
            with self.subTest(pose=pose):
                key = (roundtrip.DEVIATION_NAME, pose)
                self.assertAlmostEqual(by_key[key].value, ARMATURE_SCALE * plain[key].value, delta=1e-7)
        elbows = by_key[(roundtrip.DEVIATION_NAME, "elbows")]
        self.assertGreaterEqual(elbows.value, LIMIT)
        self.assertEqual(elbows.signal, Signal.RED)
        five = set(self.info["five_influence_vertices"])
        self.assertEqual({row["source_index"] for row in elbows.details["worst"][:4]}, five)
        over = self.body_details(elbows)
        self.assertEqual(set(over["source_index"][over["deviations"] >= LIMIT].tolist()), five)
        self.assertEqual(by_key[(roundtrip.DEVIATION_NAME, roundtrip.REST_POSE)].signal, Signal.GREEN)
        key = (roundtrip.SKELETON_NAME, None)
        self.assertAlmostEqual(by_key[key].value, ARMATURE_SCALE * plain[key].value, delta=1e-6)
        # 1点4本までにした影響も、アプリの長さで 0.92mm になり、基準を超えて赤
        key = (roundtrip.INFLUENCE_EFFECT_NAME, None)
        self.assertEqual(plain[key].signal, Signal.YELLOW)
        self.assertEqual(by_key[key].signal, Signal.RED)
        self.assertEqual(by_key[key].value, elbows.value)
        self.assertAlmostEqual(by_key[key].value, ARMATURE_SCALE * plain[key].value, delta=1e-7)

    def test_unexercised_pose_keeps_the_yellow_visible(self):
        """骨が動かなかったポーズ（本物のモデルでは、IK に戻される脚のポーズ など）が混ざると、influence_limit_effect は
        灰になる（灰 > 黄）。ただし elbows で見つけた黄は隠さない：値は elbows のずれのまま、理由の最初の文は黄の文、
        details["measured_signal"] は黄、changed_count は5本の頂点の数。動いたポーズでは、どの頂点も測れている"""
        plain = {m.name: m for m in self.run_plain()}
        poses = dict(roundtrip.TEST_POSES)
        poses["impossible"] = {"存在しない骨": ((0.0, 0.0, 1.0), 30.0)}  # モデルに無い骨だけのポーズ：何も動かない
        run = self.runs[False]
        results = roundtrip_bpy.run_roundtrip(run["usdz"], run["rig"], poses=poses)
        deviations = self.deviations(results)
        self.assertEqual(list(deviations), [roundtrip.REST_POSE, *poses])
        impossible = deviations["impossible"]
        self.assertEqual((impossible.signal, impossible.details["moved_joint_count"]), (Signal.GRAY, 0))
        self.assertEqual(impossible.details["missing_bones"], ("存在しない骨",))
        effect = next(m for m in results if m.name == roundtrip.INFLUENCE_EFFECT_NAME)
        before = plain[roundtrip.INFLUENCE_EFFECT_NAME]
        self.assertEqual(before.signal, Signal.YELLOW)
        self.assertEqual(effect.signal, Signal.GRAY)
        self.assertEqual(effect.details["measured_signal"], Signal.YELLOW)
        self.assertEqual(effect.value, before.value)
        self.assertEqual(effect.value, deviations["elbows"].value)
        self.assertEqual(effect.reason.split("。")[0], before.reason)
        self.assertIn("動かなかったポーズ", effect.reason)
        self.assertEqual(effect.details["unexercised_poses"], ("impossible",))
        self.assertEqual(effect.details["unmeasured_poses"], ())
        self.assertEqual((effect.details["limited_count"], effect.details["changed_count"],
                          effect.details["unmeasured_count"]), (4, 4, 0))
        body = effect.details["meshes"][mmd_model.BODY_NAME]
        np.testing.assert_array_equal(body["limited"], np.sort(np.array(self.info["five_influence_vertices"])))
        np.testing.assert_array_equal(body["unmeasured"], [])
        self.assertEqual(body["worst_pose"], ("elbows",) * 4)
        self.assertLess(float(body["poses"]["impossible"].max()), TOLERANCE)  # 動かないポーズでは、ずれも無い

    def test_four_influences_is_green(self):
        """5本の骨の頂点が無い見本：convert_blend.py は1つも絞らず、E2 も「4本を超える頂点はありません」の緑"""
        self.info, self.hidden = build_fixture(False, five_influences=False)
        self.assertEqual(self.info["five_influence_vertices"], [])
        run = self.runs["four"]
        self.assertEqual(run["convert"].summary["vertices_limited"], 0)
        results = roundtrip_bpy.run_roundtrip(run["usdz"], run["rig"])
        effect = next(m for m in results if m.name == roundtrip.INFLUENCE_EFFECT_NAME)
        self.assertEqual((effect.signal, effect.value, effect.reason), (Signal.GREEN, 0.0, "4本を超える頂点はありません"))
        self.assertEqual(effect.details["limited_count"], 0)
        # 5本の頂点が無いので、elbows のずれも float の誤差だけ
        elbows = self.deviations(results)["elbows"]
        self.assertEqual(elbows.signal, Signal.GREEN)
        self.assertLess(elbows.value, TOLERANCE)

    def test_hidden_five_influence_vertices_are_not_counted(self):
        """5本の骨の頂点を MASK で隠すと、convert_blend.py は頂点を消してから4本にするので、どれも絞らない。
        E2 も、その頂点を数えない（緑。消えた頂点は details["hidden_limited"]）"""
        self.info, self.hidden = build_fixture(False, hide_five=True)
        five = np.sort(np.array(self.info["five_influence_vertices"]))
        run = self.runs["five_hidden"]
        self.assertEqual(run["convert"].summary["vertices_limited"], 0)
        self.assertEqual(run["convert"].summary["masked_vertices_removed"], len(five))
        results = roundtrip_bpy.run_roundtrip(run["usdz"], run["rig"])
        self.assertEqual(results[0].signal, Signal.GREEN, results[0].reason)  # 消えるはずの頂点だけが消えた
        effect = next(m for m in results if m.name == roundtrip.INFLUENCE_EFFECT_NAME)
        self.assertEqual((effect.signal, effect.value), (Signal.GREEN, 0.0))
        self.assertEqual(effect.reason, "4本を超える頂点は、服で隠れて書き出されない物だけです")
        np.testing.assert_array_equal(effect.details["hidden_limited"][mmd_model.BODY_NAME], five)
        self.assertEqual(effect.details["limited_count"], 0)


if __name__ == "__main__":
    unittest.main()
