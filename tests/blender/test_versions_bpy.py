"""A6 版の保存と前の版との比べ（versions/save_bpy.py・snapshot_bpy.py）を、Blender の中で確かめる。

テスト用のモデル（tests/fixtures/mmd_model.py）を空のシーンに作り、一時フォルダの model.blend として保存してから調べる。
ファイルは一時フォルダの中だけに作る。
"""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bmesh
import bpy
import numpy as np

from blender_ai_tools.measure.results import Signal
from blender_ai_tools.model.extract_bpy import bone_rest, bone_weights
from blender_ai_tools.versions import history, save_bpy
from blender_ai_tools.versions.diff import change_lines, diff_snapshots
from blender_ai_tools.versions.save_bpy import compare_with_previous, current_base, previous_version, save_version
from blender_ai_tools.versions.snapshot import ARRAYS_NAME, JSON_NAME, load_snapshot, snapshots_equal
from blender_ai_tools.versions.snapshot_bpy import take_snapshot
from tests.fixtures import mmd_model

# 動かす Body の頂点の番号と、動かす量（m）
VERTEX = 10
STEP = (0.002, 0.0, 0.0)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(change):
    return change.category, change.object, change.text_key, change.count, change.outside_allowed


def move_vertex(obj, index, step):
    """頂点を動かす（メッシュの頂点と、全部のシェイプキーを同じだけ。編集モードで基本の形を動かしたときと同じ）"""
    delta = np.array(step)
    obj.data.vertices[index].co = tuple(np.array(obj.data.vertices[index].co) + delta)
    if obj.data.shape_keys is not None:
        for block in obj.data.shape_keys.key_blocks:
            block.data[index].co = tuple(np.array(block.data[index].co) + delta)
    obj.data.update()


class VersionsCase(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        self.info = mmd_model.build_model()
        self.body = bpy.data.objects[mmd_model.BODY_NAME]
        self.armature = bpy.data.objects[mmd_model.ARMATURE_NAME]
        self._temp = tempfile.TemporaryDirectory()
        self.folder = Path(self._temp.name)
        self.base = self.folder / "model.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(self.base))  # 開いているファイルを model.blend にする
        self.base_digest = digest(self.base)

    def tearDown(self):
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.wm.read_homefile(use_empty=True)
        self._temp.cleanup()

    def folder_names(self):
        return sorted(path.name for path in self.folder.iterdir())

    def objects_in(self, path):
        with bpy.data.libraries.load(str(path)) as (data_from, _):
            return sorted(data_from.objects)


class SaveVersionTest(VersionsCase):
    def test_save_twice(self):
        dirty = bpy.data.is_dirty
        first, first_dir = save_version()
        second, second_dir = save_version()
        self.assertEqual((first, second), (self.folder / "model_v001.blend", self.folder / "model_v002.blend"))
        self.assertEqual((first_dir, second_dir),
                         (self.folder / "model_v001.snapshot", self.folder / "model_v002.snapshot"))
        # 開いているファイルの場所・印は変わらない。元のファイルは上書きしない。.blend1 や一時ファイルは残らない
        self.assertEqual(bpy.data.filepath, str(self.base))
        self.assertEqual(bpy.data.is_dirty, dirty)
        self.assertEqual(digest(self.base), self.base_digest)
        self.assertEqual(self.folder_names(), ["model.blend", "model_v001.blend", "model_v001.snapshot",
                                               "model_v002.blend", "model_v002.snapshot"])
        for folder in (first_dir, second_dir):
            self.assertEqual(sorted(p.name for p in folder.iterdir()), [ARRAYS_NAME, JSON_NAME])
        # 保存した版は、Blender のファイルとして読める
        expected = sorted([mmd_model.ARMATURE_NAME, mmd_model.BODY_NAME, mmd_model.SKIRT_NAME])
        for path in (first, second):
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(self.objects_in(path), expected)
        changes, result = history.compare_versions(first, second)
        self.assertEqual((changes, result.signal, result.value), ([], Signal.GREEN, 0.0))
        self.assertEqual(previous_version(), second)

    def test_move_one_vertex_between_saves(self):
        first, _ = save_version()
        move_vertex(self.body, VERTEX, STEP)
        # 保存する前：いまの中身を前の版と比べる
        changes, result = compare_with_previous()
        self.assertEqual([plain(c) for c in changes], [("vertices", "Body", "versions.vertices_moved", 1, True)])
        self.assertEqual(result.signal, Signal.RED)
        changes, result = compare_with_previous(allowed={"Body": {"vertices"}})
        self.assertEqual((len(changes), result.signal), (1, Signal.GREEN))

        second, _ = save_version()
        changes, result = history.compare_versions(first, second)
        self.assertEqual([plain(c) for c in changes], [("vertices", "Body", "versions.vertices_moved", 1, True)])
        self.assertAlmostEqual(changes[0].values["distance_mm"], 2.0, delta=1e-3)  # float32 の丸めの分
        self.assertEqual(result.signal, Signal.RED)
        self.assertEqual(change_lines(changes, "ja"),
                         ["【変えてよい所の外】Body：頂点が 1 個動きました（いちばん大きい動き 2.000 mm）"])
        # 保存したあと：いまの中身は、いちばん新しい版と同じ
        changes, result = compare_with_previous()
        self.assertEqual((changes, result.signal), ([], Signal.GREEN))
        # 前の版を指定して比べる
        changes, _ = compare_with_previous(version_path=first)
        self.assertEqual([c.count for c in changes], [1])

    def test_edit_mode_move_is_saved_and_mode_is_kept(self):
        first, _ = save_version()
        bpy.context.view_layer.objects.active = self.body
        self.body.active_shape_key_index = 0  # 基本の形（Basis）を動かす
        bpy.ops.object.mode_set(mode="EDIT")
        mesh = bmesh.from_edit_mesh(self.body.data)
        mesh.verts.ensure_lookup_table()
        mesh.verts[VERTEX].co.x += STEP[0]
        bmesh.update_edit_mesh(self.body.data)
        second, _ = save_version()
        self.assertEqual(self.body.mode, "EDIT")
        self.assertIs(bpy.context.view_layer.objects.active, self.body)
        changes, _ = history.compare_versions(first, second)
        # 編集モードで基本の形を動かすと、ほかのキーも一緒に動く（キーの形は変わらない）
        self.assertEqual([plain(c) for c in changes], [("vertices", "Body", "versions.vertices_moved", 1, True)])

    def test_open_version_matches_its_record(self):
        first, first_dir = save_version()
        bpy.ops.wm.open_mainfile(filepath=str(first))
        self.assertEqual(bpy.data.filepath, str(first))
        self.assertTrue(snapshots_equal(load_snapshot(first_dir), take_snapshot()))
        # 版のファイルを開いて保存しても、元の名前で次の番号になる。開いたファイルは変えない
        first_digest = digest(first)
        second, _ = save_version()
        self.assertEqual(second, self.folder / "model_v002.blend")
        self.assertEqual(current_base(), self.base)
        self.assertEqual(bpy.data.filepath, str(first))
        self.assertEqual(digest(first), first_digest)

    def test_never_overwrites(self):
        kept = self.folder / "model_v001.blend"
        kept.write_bytes(b"keep")
        (self.folder / "model_v003.snapshot").mkdir()  # 版のファイルを消して、記録だけ残っている
        path, folder = save_version()
        self.assertEqual((path.name, folder.name), ("model_v004.blend", "model_v004.snapshot"))
        self.assertEqual(kept.read_bytes(), b"keep")
        self.assertEqual(list((self.folder / "model_v003.snapshot").iterdir()), [])
        self.assertEqual(digest(self.base), self.base_digest)

    def test_failure_leaves_no_files_and_gives_the_number_back(self):
        """途中で失敗したら、自分で作った物（取っておいた名前・記録のフォルダ）を消す。次の保存は同じ番号から"""
        for failing in ("save_snapshot", "_save_copy"):
            with self.subTest(failing=failing):
                with mock.patch.object(save_bpy, failing, side_effect=RuntimeError("わざと失敗")):
                    with self.assertRaises(RuntimeError):
                        save_version()
                self.assertEqual(self.folder_names(), ["model.blend"])
                self.assertEqual(bpy.data.filepath, str(self.base))
                self.assertEqual(digest(self.base), self.base_digest)
        path, folder = save_version()
        self.assertEqual((path.name, folder.name), ("model_v001.blend", "model_v001.snapshot"))

    def test_unsaved_file_needs_base_path(self):
        bpy.ops.wm.read_homefile(use_empty=True)
        mmd_model.build_model()
        self.assertEqual(bpy.data.filepath, "")
        with self.assertRaises(ValueError):
            save_version()
        self.assertEqual(self.folder_names(), ["model.blend"])
        self.assertIsNone(previous_version())
        changes, result = compare_with_previous(language="ja")
        self.assertEqual((changes, result.signal, result.reason),
                         ([], Signal.GRAY, "比べる前の版がありません（まだ版を保存していません）"))
        path, folder = save_version(base_path=self.folder / "other.blend")
        self.assertEqual((path.name, folder.name), ("other_v001.blend", "other_v001.snapshot"))
        self.assertEqual(bpy.data.filepath, "")
        self.assertEqual(previous_version(self.folder / "other.blend"), path)

    def test_no_previous_version_is_gray(self):
        changes, result = compare_with_previous(language="en")
        self.assertEqual((changes, result.signal, result.value), ([], Signal.GRAY, None))
        self.assertIsNone(previous_version())

    def test_missing_record_is_gray(self):
        first, first_dir = save_version()
        (first_dir / ARRAYS_NAME).unlink()
        changes, result = compare_with_previous(language="ja")
        self.assertEqual((changes, result.signal), ([], Signal.GRAY))
        self.assertTrue(result.reason.startswith("版 model_v001.blend の記録"))


class TakeSnapshotTest(VersionsCase):
    def test_contents(self):
        snapshot = take_snapshot()
        self.assertEqual(list(snapshot.objects), sorted([mmd_model.ARMATURE_NAME, mmd_model.BODY_NAME,
                                                         mmd_model.SKIRT_NAME]))
        body = snapshot.objects["Body"]
        self.assertEqual((body.type, body.parent, body.parent_bone), ("MESH", mmd_model.ARMATURE_NAME, ""))
        self.assertEqual(body.modifiers, ((mmd_model.MODIFIER_NAME, "ARMATURE", True),))
        np.testing.assert_array_equal(body.matrix, np.array(self.body.matrix_basis))

        mesh = snapshot.meshes["Body"]
        self.assertEqual(mesh.vertex_count, self.info["vertex_counts"]["Body"])
        self.assertEqual(tuple(mesh.shape_keys), mmd_model.SHAPE_KEY_NAMES)
        self.assertEqual(mesh.face_count, len(self.body.data.polygons))
        self.assertEqual(len(mesh.triangles), len(self.body.data.loop_triangles))
        np.testing.assert_array_equal(mesh.vertices, mesh.shape_keys["Basis"])
        self.assertEqual(mesh.face_vertices[:4].tolist(), list(self.body.data.polygons[0].vertices))
        last = self.body.data.polygons[-1]
        self.assertEqual(mesh.face_vertices[-last.loop_total:].tolist(), list(last.vertices))
        # ウェイトは、名前で見ると extract_bpy.bone_weights と同じ（骨の頂点グループだけ比べる）
        table, names = bone_weights(self.body, self.armature)
        for index in (0, VERTEX, mesh.vertex_count - 1):
            by_bone = {names[b]: w for b, w in zip(table.bone_ids[index], table.weights[index]) if b >= 0}
            by_group = {mesh.group_names[g]: w for g, w in zip(mesh.weights.bone_ids[index],
                                                               mesh.weights.weights[index]) if g >= 0}
            self.assertEqual(by_group, by_bone)

        arm = snapshot.armatures[mmd_model.ARMATURE_NAME]
        rest = bone_rest(self.armature, space="armature")
        self.assertEqual(arm.names, rest.names)
        np.testing.assert_array_equal(arm.parents, rest.parents)
        np.testing.assert_array_equal(arm.heads, rest.heads)
        np.testing.assert_array_equal(arm.tails, rest.tails)
        np.testing.assert_array_equal(arm.deform, rest.deform)
        np.testing.assert_array_equal(arm.matrices, rest.matrices)
        self.assertEqual(snapshot.drivers, ())

    def test_drivers_materials_and_state(self):
        key = self.body.data.shape_keys.key_blocks["まばたき"]
        curve = key.driver_add("value")
        curve.driver.expression = "var * 2"
        variable = curve.driver.variables.new()
        variable.name = "var"
        variable.targets[0].id = self.armature
        variable.targets[0].data_path = 'pose.bones["頭"].location[0]'
        self.armature.driver_add('pose.bones["左足D"].rotation_quaternion', 1)
        skin, cloth = bpy.data.materials.new("肌"), bpy.data.materials.new("服")
        self.body.data.materials.append(skin)
        self.body.data.materials.append(cloth)
        self.body.data.polygons[0].material_index = 1

        scene = bpy.context.scene
        scene.frame_set(7)
        bpy.ops.object.select_all(action="DESELECT")
        self.body.select_set(True)
        bpy.context.view_layer.objects.active = self.armature
        before = (scene.frame_current, bpy.context.view_layer.objects.active.name,
                  sorted(o.name for o in bpy.context.selected_objects), self.armature.mode)

        snapshot = take_snapshot()
        self.assertEqual(before, (scene.frame_current, bpy.context.view_layer.objects.active.name,
                                  sorted(o.name for o in bpy.context.selected_objects), self.armature.mode))
        body_drivers = [d for d in snapshot.drivers if d.owner == "Body"]
        self.assertEqual([(d.data_path, d.index, d.kind, d.expression) for d in body_drivers],
                         [('data.shape_keys.key_blocks["まばたき"].value', 0, "SCRIPTED", "var * 2")])
        self.assertEqual(len(body_drivers[0].variables), 1)
        self.assertIn(mmd_model.ARMATURE_NAME, body_drivers[0].variables[0])
        self.assertIn('pose.bones["頭"].location[0]', body_drivers[0].variables[0])
        arm_drivers = [d for d in snapshot.drivers if d.owner == mmd_model.ARMATURE_NAME]
        self.assertEqual([(d.data_path, d.index) for d in arm_drivers],
                         [('pose.bones["左足D"].rotation_quaternion', 1)])
        self.assertEqual(snapshot.objects["Body"].materials, ("肌", "服"))
        self.assertEqual(snapshot.meshes["Body"].face_materials[:2].tolist(), [1, 0])

        # 式を変えると、ドライバーの変化として出る
        curve.driver.expression = "var * 3"
        changes = diff_snapshots(snapshot, take_snapshot())
        self.assertEqual([plain(c) for c in changes], [("drivers", "Body", "versions.drivers_changed", 1, True)])

    def test_linked_object_with_the_same_name_is_kept_apart(self):
        """ほかのファイルからリンクした、同じ名前のオブジェクトが、このファイルの物を記録から消さない
        （消えると、このファイルの物の変化が「変わった所はありません」に見える）"""
        library = self.folder / "lib.blend"
        bpy.ops.wm.read_homefile(use_empty=True)
        mesh = bpy.data.meshes.new("LibMesh")
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        bpy.context.scene.collection.objects.link(bpy.data.objects.new(mmd_model.BODY_NAME, mesh))
        bpy.ops.wm.save_as_mainfile(filepath=str(library))
        bpy.ops.wm.open_mainfile(filepath=str(self.base))
        with bpy.data.libraries.load(str(library), link=True) as (data_from, data_to):
            data_to.objects = [mmd_model.BODY_NAME]
        linked = data_to.objects[0]
        bpy.context.scene.collection.objects.link(linked)
        local = next(o for o in bpy.context.scene.objects if o.name == mmd_model.BODY_NAME and o.library is None)
        self.assertEqual((linked.name, local.name), (mmd_model.BODY_NAME, mmd_model.BODY_NAME))
        self.assertNotEqual(linked.name_full, local.name_full)

        snapshot = take_snapshot()
        self.assertEqual(list(snapshot.objects), sorted([mmd_model.ARMATURE_NAME, mmd_model.BODY_NAME,
                                                         linked.name_full, mmd_model.SKIRT_NAME]))
        self.assertEqual(snapshot.meshes[mmd_model.BODY_NAME].vertex_count,
                         self.info["vertex_counts"][mmd_model.BODY_NAME])
        self.assertEqual(snapshot.objects[mmd_model.BODY_NAME].parent, mmd_model.ARMATURE_NAME)
        self.assertEqual((snapshot.meshes[linked.name_full].vertex_count, snapshot.objects[linked.name_full].parent),
                         (3, None))
        # このファイルの Body の頂点を動かすと、Body の変化として出る
        move_vertex(local, VERTEX, STEP)
        changes = diff_snapshots(snapshot, take_snapshot())
        self.assertEqual([plain(c) for c in changes],
                         [("vertices", mmd_model.BODY_NAME, "versions.vertices_moved", 1, True)])

    def test_editing_a_non_basis_key_changes_only_that_key(self):
        """基本の形でないキー（あ）を編集モードで動かすと、頂点（基本の形）は変わらず、そのキーの形だけが変わる"""
        before = take_snapshot()
        bpy.context.view_layer.objects.active = self.body
        self.body.active_shape_key_index = mmd_model.SHAPE_KEY_NAMES.index("あ")
        bpy.ops.object.mode_set(mode="EDIT")
        mesh = bmesh.from_edit_mesh(self.body.data)
        mesh.verts.ensure_lookup_table()
        mesh.verts[VERTEX].co.x += 0.005
        bmesh.update_edit_mesh(self.body.data)
        for state in ("EDIT", "OBJECT"):
            with self.subTest(mode=state):
                bpy.ops.object.mode_set(mode=state)
                after = take_snapshot()
                self.assertEqual(self.body.mode, state)
                np.testing.assert_array_equal(after.meshes["Body"].vertices, before.meshes["Body"].vertices)
                np.testing.assert_array_equal(after.meshes["Body"].shape_keys["Basis"],
                                              before.meshes["Body"].shape_keys["Basis"])
                changes = diff_snapshots(before, after)
                self.assertEqual([plain(c) for c in changes],
                                 [("shape_keys", "Body", "versions.shape_key_changed", 1, True)])
                self.assertEqual(changes[0].values["name"], "あ")
                self.assertAlmostEqual(changes[0].values["distance_mm"], 5.0, delta=1e-3)
        self.body.active_shape_key_index = 0

    def test_armature_edit_mode_is_read_and_kept(self):
        """アーマチュアの編集モードで動かした骨も、モードを変えずに記録に入る"""
        before = take_snapshot()
        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="EDIT")
        bone = self.armature.data.edit_bones["頭"]
        bone.head.z += 0.01
        bone.tail.z += 0.01
        after = take_snapshot()
        self.assertEqual(self.armature.mode, "EDIT")
        changes = diff_snapshots(before, after)
        self.assertEqual([plain(c) for c in changes],
                         [("bones", mmd_model.ARMATURE_NAME, "versions.bones_moved", 1, True)])
        self.assertEqual(changes[0].values["names"], ("頭",))
        self.assertAlmostEqual(changes[0].values["distance_mm"], 10.0, delta=1e-3)
        bpy.ops.object.mode_set(mode="OBJECT")
        self.assertEqual(diff_snapshots(after, take_snapshot()), [])

    def test_object_driver(self):
        """オブジェクト自身（location など）のドライバーも記録に入る"""
        before = take_snapshot()
        self.body.driver_add("location", 0).driver.expression = "frame"
        after = take_snapshot()
        self.assertEqual([(d.owner, d.data_path, d.index, d.expression) for d in after.drivers],
                         [("Body", "location", 0, "frame")])
        changes = diff_snapshots(before, after)
        self.assertEqual([plain(c) for c in changes], [("drivers", "Body", "versions.drivers_added", 1, True)])
        self.assertEqual(changes[0].values["names"], ("location",))

    def test_edit_mode_is_read_and_kept(self):
        bpy.context.view_layer.objects.active = self.body
        self.body.active_shape_key_index = 0
        bpy.ops.object.mode_set(mode="EDIT")
        mesh = bmesh.from_edit_mesh(self.body.data)
        mesh.verts.ensure_lookup_table()
        before = np.array(mesh.verts[VERTEX].co)
        mesh.verts[VERTEX].co.z += 0.003
        bmesh.update_edit_mesh(self.body.data)
        snapshot = take_snapshot()
        self.assertEqual(self.body.mode, "EDIT")
        self.assertAlmostEqual(snapshot.meshes["Body"].vertices[VERTEX, 2], before[2] + 0.003, delta=1e-6)

    def test_other_scene(self):
        other = bpy.data.scenes.new("Other")
        empty = bpy.data.objects.new("Only", None)
        other.collection.objects.link(empty)
        snapshot = take_snapshot(other)
        self.assertEqual(list(snapshot.objects), ["Only"])
        self.assertEqual(snapshot.objects["Only"].type, "EMPTY")
        with self.assertRaises(ValueError):
            take_snapshot("Scene")


if __name__ == "__main__":
    unittest.main()
