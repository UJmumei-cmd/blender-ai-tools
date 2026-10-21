"""A6 版の記録（versions/snapshot.py）と、記録どうしの違い（versions/diff.py）・保存した版の比べ（versions/history.py）を
確かめる。bpy なしで動く。

記録は、手で作った小さな物（箱の形の "Body" と、骨4本の "Armature"）を使う。答えは手で決めた数字。
ファイルは一時フォルダの中だけに作る。
"""

import hashlib
import json
import pickle
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from blender_ai_tools import texts, thresholds
from blender_ai_tools.measure.results import Signal
from blender_ai_tools.measure.weights import WeightTable
from blender_ai_tools.versions import history
from blender_ai_tools.versions.diff import (
    CATEGORIES, MAX_NAMES_IN_LINE, AllowedChanges, Change, change_lines, diff_snapshots, summarize_changes,
)
from blender_ai_tools.versions.snapshot import (
    ARRAYS_NAME, JSON_NAME, ArmatureState, Driver, MeshState, ObjectState, Snapshot, load_snapshot, save_snapshot,
    snapshots_equal,
)

TOL = thresholds.REPRO_TOLERANCE_LENGTH
BLINK_PATH = 'data.shape_keys.key_blocks["まばたき"].value'

# MARK: - 見本の記録

CUBE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                dtype=np.float64)
QUADS = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def quad_triangles(quads, flip=()):
    """四角形を三角形に分ける。flip に入れた四角形は、もう1つの対角線で分ける"""
    triangles = []
    for i, (a, b, c, d) in enumerate(quads):
        triangles += [(a, b, d), (b, c, d)] if i in flip else [(a, b, c), (a, c, d)]
    return np.array(triangles, dtype=np.int64)


def body_mesh(vertices=CUBE):
    top = np.zeros_like(vertices)
    top[6] = (0.0, 0.0, 0.001)  # 「あ」は頂点 6 を 1mm 上げる
    blink = np.zeros_like(vertices)
    blink[7] = (0.0, -0.002, 0.0)  # 「まばたき」は頂点 7 を 2mm 前へ
    weights = WeightTable.from_lists([[(0, 1.0)]] * 4 + [[(0, 0.5), (1, 0.5)]] * 4)
    return MeshState(vertices=vertices, triangles=quad_triangles(QUADS),
                     shape_keys={"Basis": vertices, "あ": vertices + top, "まばたき": vertices + blink},
                     weights=weights, group_names=("左腕", "左ひじ"), face_sizes=[4] * 6,
                     face_vertices=[v for quad in QUADS for v in quad], face_materials=[0, 0, 1, 1, 0, 0])


BONE_NAMES = ("センター", "上半身", "左腕", "左ひじ")
HEADS = np.array([[0, 0, 0.8], [0, 0, 1.0], [0.15, 0, 1.4], [0.4, 0, 1.2]], dtype=np.float64)
TAILS = np.array([[0, 0, 0.9], [0, 0, 1.3], [0.4, 0, 1.2], [0.6, 0, 1.05]], dtype=np.float64)


def bone_matrices(heads=HEADS):
    matrices = np.tile(np.eye(4), (len(heads), 1, 1))
    matrices[:, :3, 3] = heads
    return matrices


def armature_state():
    return ArmatureState(names=BONE_NAMES, parents=[-1, 0, 1, 2], heads=HEADS, tails=TAILS,
                         deform=np.array([False, True, True, True]), matrices=bone_matrices())


def blink_driver(expression="var"):
    return Driver(owner="Body", data_path=BLINK_PATH, expression=expression,
                  variables=('var ; SINGLE_PROP ; OBJECT|Armature|pose.bones["頭"]["blink"]',))


def sample():
    objects = {
        "Armature": ObjectState("ARMATURE", None, np.eye(4)),
        "Body": ObjectState("MESH", "Armature", np.eye(4), modifiers=(("Armature", "ARMATURE", True),),
                            materials=("肌", "服")),
    }
    return Snapshot(objects=objects, meshes={"Body": body_mesh()}, armatures={"Armature": armature_state()},
                    drivers=(blink_driver(),))


def with_mesh(snapshot, **changes):
    return replace(snapshot, meshes={**snapshot.meshes, "Body": replace(snapshot.meshes["Body"], **changes)})


def with_armature(snapshot, **changes):
    arm = replace(snapshot.armatures["Armature"], **changes)
    return replace(snapshot, armatures={**snapshot.armatures, "Armature": arm})


def with_object(snapshot, name, **changes):
    return replace(snapshot, objects={**snapshot.objects, name: replace(snapshot.objects[name], **changes)})


def moved_vertex(snapshot, index, delta, *, keys=True):
    """頂点 index を delta だけ動かす（keys が True なら、全部のシェイプキーも一緒に動かす。Blender の編集モードと同じ）"""
    mesh = snapshot.meshes["Body"]
    step = np.zeros((mesh.vertex_count, 3))
    step[index] = delta
    shape_keys = {name: points + step if keys or name == "Basis" else points
                  for name, points in mesh.shape_keys.items()}
    return with_mesh(snapshot, vertices=mesh.vertices + step, shape_keys=shape_keys)


def plain(change):
    """比べやすい形（数は別に比べる）"""
    return change.category, change.object, change.text_key, change.count, change.outside_allowed


# MARK: - 記録の形

class SnapshotValidationTest(unittest.TestCase):
    def test_sample_is_valid(self):
        snapshot = sample()
        mesh = snapshot.meshes["Body"]
        self.assertEqual((mesh.vertex_count, mesh.face_count, len(mesh.triangles)), (8, 6, 12))
        self.assertEqual(mesh.vertices.dtype, np.float64)
        self.assertFalse(mesh.vertices.flags.writeable)
        self.assertEqual(snapshot.armatures["Armature"].bone_count, 4)

    def test_triangles_become_faces_when_no_faces_given(self):
        mesh = MeshState(vertices=CUBE, triangles=quad_triangles(QUADS))
        self.assertEqual(mesh.face_sizes.tolist(), [3] * 12)
        self.assertEqual(mesh.face_vertices.tolist(), quad_triangles(QUADS).reshape(-1).tolist())
        self.assertEqual(mesh.face_materials.tolist(), [0] * 12)
        self.assertEqual(mesh.weights.bone_ids.shape, (8, 0))

    def test_rejects_bad_meshes(self):
        good = body_mesh()
        cases = {
            "shape key size": dict(shape_keys={"Basis": CUBE[:7]}),
            "weights size": dict(weights=WeightTable.from_lists([[(0, 1.0)]] * 7)),
            "group id": dict(weights=WeightTable.from_lists([[(2, 1.0)]] * 8)),
            "duplicate groups": dict(group_names=("左腕", "左腕")),
            "face sum": dict(face_sizes=[4] * 5),
            "face index": dict(face_vertices=[8] + [0] * 23),
            "face size": dict(face_sizes=[2, 6, 4, 4, 4, 4]),
            "face materials": dict(face_materials=[0] * 5),
            "nan": dict(vertices=np.where(np.arange(24).reshape(8, 3) == 0, np.nan, CUBE)),
        }
        for label, change in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    replace(good, **change)
        with self.assertRaises(ValueError):
            MeshState(vertices=CUBE, triangles=quad_triangles(QUADS), face_sizes=[4] * 6)

    def test_rejects_bad_armatures(self):
        good = armature_state()
        for label, change in {"duplicate": dict(names=("a", "a", "b", "c")), "parent": dict(parents=[-1, 0, 1, 9]),
                              "self parent": dict(parents=[0, 0, 1, 2]), "deform": dict(deform=[1, 1, 1, 1]),
                              "matrices": dict(matrices=np.eye(4))}.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    replace(good, **change)

    def test_rejects_bad_snapshots(self):
        snapshot = sample()
        with self.assertRaises(ValueError):  # メッシュのオブジェクトの控えが無い
            replace(snapshot, meshes={})
        with self.assertRaises(ValueError):  # アーマチュアの控えが、種類の違うオブジェクトに付いている
            replace(snapshot, armatures={"Armature": armature_state(), "Body": armature_state()})
        with self.assertRaises(ValueError):  # ドライバーの持ち主が無い
            replace(snapshot, drivers=(replace(blink_driver(), owner="Hair"),))
        with self.assertRaises(ValueError):
            replace(snapshot, schema_version=2)
        with self.assertRaises(ValueError):  # モディファイアの名前が重なる
            with_object(snapshot, "Body", modifiers=(("A", "ARMATURE", True), ("A", "MASK", True)))
        with self.assertRaises(ValueError):  # show_render は真偽だけ
            with_object(snapshot, "Body", modifiers=(("A", "ARMATURE", 1),))

    def test_snapshots_equal_is_exact(self):
        self.assertTrue(snapshots_equal(sample(), sample()))
        self.assertFalse(snapshots_equal(sample(), moved_vertex(sample(), 0, (1e-12, 0, 0))))
        self.assertFalse(snapshots_equal(sample(), replace(sample(), drivers=())))


# MARK: - 保存と読み込み

_UNPICKLED = []


def _mark_unpickled():
    _UNPICKLED.append(True)
    return "unpickled"


class Boom:
    """読み込むときに pickle が動けば、_UNPICKLED に印が付く"""

    def __reduce__(self):
        return _mark_unpickled, ()


class SnapshotFileTest(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.folder = Path(self._temp.name) / "model_v001.snapshot"
        _UNPICKLED.clear()

    def tearDown(self):
        self._temp.cleanup()

    def rich_sample(self):
        """どんな文字の名前でも、空の並びでも、そのまま戻るか確かめるための記録"""
        snapshot = sample()
        objects = {**snapshot.objects, "a/b.c 頭\"飾り\"": ObjectState("EMPTY", "Armature", np.diag([2.0, 2.0, 2.0, 1.0]),
                                                                    parent_bone="頭"),
                   "Empty mesh": ObjectState("MESH", None, np.eye(4))}
        meshes = {**snapshot.meshes, "Empty mesh": MeshState(vertices=np.zeros((0, 3)), triangles=np.zeros((0, 3)))}
        arm = replace(armature_state(), heads=HEADS + np.float64(1.0) / 3.0)  # 割り切れない数
        drivers = snapshot.drivers + (Driver("Armature", 'pose.bones["左腕"].rotation_quaternion', "", index=2,
                                             kind="AVERAGE", variables=("a ; TRANSFORMS ; x", "b")),)
        return Snapshot(objects=objects, meshes=meshes, armatures={"Armature": arm}, drivers=drivers)

    def test_round_trip_is_exact(self):
        original = self.rich_sample()
        json_path = save_snapshot(original, self.folder)
        self.assertEqual(json_path, self.folder / JSON_NAME)
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), [ARRAYS_NAME, JSON_NAME])
        for loaded in (load_snapshot(self.folder), load_snapshot(json_path)):
            self.assertTrue(snapshots_equal(original, loaded))
            self.assertEqual(list(loaded.objects), list(original.objects))
            self.assertEqual(loaded.armatures["Armature"].heads.tobytes(), original.armatures["Armature"].heads.tobytes())
            self.assertEqual(loaded.meshes["Body"].weights.weights.dtype, np.float64)
            self.assertEqual(loaded.objects["a/b.c 頭\"飾り\""].parent_bone, "頭")
            self.assertEqual(diff_snapshots(original, loaded, tolerance=0.0, weight_tolerance=0.0,
                                            ratio_tolerance=0.0), [])

    def test_does_not_overwrite(self):
        save_snapshot(sample(), self.folder)
        before = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        with self.assertRaises(FileExistsError):
            save_snapshot(moved_vertex(sample(), 0, (1.0, 0, 0)), self.folder)
        self.assertEqual({p.name: p.read_bytes() for p in self.folder.iterdir()}, before)

    def test_refuses_a_file_as_folder(self):
        self.folder.parent.mkdir(exist_ok=True)
        self.folder.write_bytes(b"x")
        with self.assertRaises(ValueError):
            save_snapshot(sample(), self.folder)

    def rewrite_arrays(self, write):
        """arrays.npz を書き換え、snapshot.json の sha256 を合わせる（わざと作った、悪い記録）"""
        arrays_path = self.folder / ARRAYS_NAME
        arrays_path.unlink()
        write(arrays_path)
        data = json.loads((self.folder / JSON_NAME).read_text(encoding="utf-8"))
        data["arrays_sha256"] = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
        (self.folder / JSON_NAME).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_rejects_object_arrays_without_unpickling(self):
        save_snapshot(sample(), self.folder)
        with np.load(self.folder / ARRAYS_NAME, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        arrays["a0"] = np.array([Boom()], dtype=object)

        def write(path):
            with open(path, "wb") as stream:
                np.savez(stream, **arrays)
        self.rewrite_arrays(write)
        with self.assertRaises(ValueError) as caught:
            load_snapshot(self.folder)
        self.assertIn("pickle", str(caught.exception))
        self.assertEqual(_UNPICKLED, [])

    def test_rejects_pickle_file(self):
        save_snapshot(sample(), self.folder)
        self.rewrite_arrays(lambda path: path.write_bytes(pickle.dumps(Boom())))
        with self.assertRaises(ValueError) as caught:
            load_snapshot(self.folder)
        self.assertIn("pickle", str(caught.exception))
        self.assertEqual(_UNPICKLED, [])

    def test_corrupt_zip_is_a_value_error(self):
        """zip として壊れた arrays.npz（sha256 は合わせてある）は ValueError（zipfile の例外をそのまま出さない）"""
        save_snapshot(sample(), self.folder)
        self.rewrite_arrays(lambda path: path.write_bytes(b"PK\x03\x04" + b"garbage" * 10))
        with self.assertRaises(ValueError):
            load_snapshot(self.folder)

    def test_entries_that_are_not_arrays_are_rejected(self):
        """.npy でない物が入った zip（NpzFile は bytes を返す）は ValueError（AttributeError で止まらない）"""
        save_snapshot(sample(), self.folder)
        for label, entries in {"raw .npy": (("a0.npy", b"not an npy"),), "no extension": (("a0", b"x"),)}.items():
            with self.subTest(case=label):
                def write(path, entries=entries):
                    with zipfile.ZipFile(path, "w") as archive:
                        for name, content in entries:
                            archive.writestr(name, content)
                self.rewrite_arrays(write)
                with self.assertRaises(ValueError):
                    load_snapshot(self.folder)

    def test_rejects_npy_instead_of_npz(self):
        save_snapshot(sample(), self.folder)

        def write(path):
            with open(path, "wb") as stream:
                np.save(stream, np.zeros(3))
        self.rewrite_arrays(write)
        with self.assertRaises(ValueError):
            load_snapshot(self.folder)

    def test_rejects_changed_arrays(self):
        save_snapshot(sample(), self.folder)
        other = Path(self._temp.name) / "other.snapshot"
        save_snapshot(moved_vertex(sample(), 0, (1.0, 0, 0)), other)
        (self.folder / ARRAYS_NAME).write_bytes((other / ARRAYS_NAME).read_bytes())
        with self.assertRaises(ValueError) as caught:
            load_snapshot(self.folder)
        self.assertIn("sha256", str(caught.exception))

    def test_rejects_extra_and_retyped_arrays(self):
        save_snapshot(sample(), self.folder)
        with np.load(self.folder / ARRAYS_NAME, allow_pickle=False) as loaded:
            arrays = {key: loaded[key] for key in loaded.files}
        for label, changed in {"extra": {**arrays, "a999": np.zeros(1)},
                               "float32": {**arrays, "a0": arrays["a0"].astype(np.float32)}}.items():
            with self.subTest(case=label):
                def write(path, changed=changed):
                    with open(path, "wb") as stream:
                        np.savez_compressed(stream, **changed)
                self.rewrite_arrays(write)
                with self.assertRaises(ValueError):
                    load_snapshot(self.folder)

    def test_rejects_bad_json(self):
        save_snapshot(sample(), self.folder)
        json_path = self.folder / JSON_NAME
        good = json_path.read_text(encoding="utf-8")
        data = json.loads(good)
        cases = {
            "schema": json.dumps({**data, "schema_version": 2}),
            "format": json.dumps({**data, "format": "other"}),
            "nan": good.replace('"index": 0', '"index": NaN'),
            "duplicate": good.replace('"index": 0', '"index": 0, "index": 0'),
            "missing key": json.dumps({k: v for k, v in data.items() if k != "drivers"}),
            "arrays name": json.dumps({**data, "arrays": "../other.npz"}),
            "not json": "{",
        }
        for label, text in cases.items():
            with self.subTest(case=label):
                json_path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_snapshot(self.folder)

    def test_missing_files(self):
        with self.assertRaises(ValueError):
            load_snapshot(self.folder)
        save_snapshot(sample(), self.folder)
        (self.folder / ARRAYS_NAME).unlink()
        with self.assertRaises(ValueError):
            load_snapshot(self.folder)


# MARK: - 違い

class DiffTest(unittest.TestCase):
    def test_no_change(self):
        self.assertEqual(diff_snapshots(sample(), sample()), [])

    def test_vertex_moved_by_twice_the_tolerance(self):
        changes = diff_snapshots(sample(), moved_vertex(sample(), 6, (2 * TOL, 0.0, 0.0)))
        self.assertEqual([plain(c) for c in changes], [("vertices", "Body", "versions.vertices_moved", 1, True)])
        self.assertEqual(set(changes[0].values), {"object", "count", "distance_mm"})
        self.assertEqual(changes[0].values["object"], "Body")
        self.assertEqual(changes[0].values["count"], 1)
        self.assertAlmostEqual(changes[0].values["distance_mm"], 0.02, places=9)

    def test_vertex_moved_by_half_the_tolerance(self):
        self.assertEqual(diff_snapshots(sample(), moved_vertex(sample(), 6, (0.5 * TOL, 0.0, 0.0))), [])

    def test_exactly_the_tolerance_is_not_a_change(self):
        self.assertEqual(diff_snapshots(sample(), moved_vertex(sample(), 0, (TOL, 0.0, 0.0))), [])
        self.assertEqual(len(diff_snapshots(sample(), moved_vertex(sample(), 0, (TOL, 0.0, 0.0)), tolerance=0.0)), 1)

    def test_several_vertices(self):
        moved = moved_vertex(moved_vertex(sample(), 1, (0.0, 0.003, 0.0)), 4, (0.0, 0.0, -0.001))
        changes = diff_snapshots(sample(), moved)
        self.assertEqual([plain(c) for c in changes], [("vertices", "Body", "versions.vertices_moved", 2, True)])
        self.assertAlmostEqual(changes[0].values["distance_mm"], 3.0, places=9)

    def test_basis_moved_without_other_keys(self):
        """基本の形だけを動かすと、ほかのキーのずれが変わる（キーの形が変わった）"""
        changes = diff_snapshots(sample(), moved_vertex(sample(), 5, (0.001, 0.0, 0.0), keys=False))
        self.assertEqual([plain(c) for c in changes], [
            ("vertices", "Body", "versions.vertices_moved", 1, True),
            ("shape_keys", "Body", "versions.shape_key_changed", 1, True),
            ("shape_keys", "Body", "versions.shape_key_changed", 1, True),
        ])
        self.assertEqual([c.values["name"] for c in changes[1:]], ["あ", "まばたき"])
        self.assertAlmostEqual(changes[1].values["distance_mm"], 1.0, places=9)

    def test_shape_key_added_removed_changed_reordered(self):
        mesh = sample().meshes["Body"]
        keys = dict(mesh.shape_keys)
        added = with_mesh(sample(), shape_keys={**keys, "い": keys["Basis"]})
        changes = diff_snapshots(sample(), added)
        self.assertEqual([plain(c) for c in changes], [("shape_keys", "Body", "versions.shape_keys_added", 1, True)])
        self.assertEqual(dict(changes[0].values), {"object": "Body", "count": 1, "names": ("い",)})

        removed = with_mesh(sample(), shape_keys={k: v for k, v in keys.items() if k != "まばたき"})
        changes = diff_snapshots(sample(), removed)
        self.assertEqual([plain(c) for c in changes], [("shape_keys", "Body", "versions.shape_keys_removed", 1, True)])
        self.assertEqual(changes[0].values["names"], ("まばたき",))

        changed_points = keys["あ"].copy()
        changed_points[6, 2] += 0.001
        changes = diff_snapshots(sample(), with_mesh(sample(), shape_keys={**keys, "あ": changed_points}))
        self.assertEqual([plain(c) for c in changes], [("shape_keys", "Body", "versions.shape_key_changed", 1, True)])
        self.assertEqual(changes[0].values["name"], "あ")
        self.assertAlmostEqual(changes[0].values["distance_mm"], 1.0, places=9)

        reordered = with_mesh(sample(), shape_keys={k: keys[k] for k in ("Basis", "まばたき", "あ")})
        changes = diff_snapshots(sample(), reordered)
        self.assertEqual([plain(c) for c in changes],
                         [("shape_keys", "Body", "versions.shape_key_order_changed", 1, True)])
        self.assertEqual(changes[0].values["names"], ("Basis", "まばたき", "あ"))

    def test_weights_changed(self):
        mesh = sample().meshes["Body"]
        rows = [[(0, 1.0)], [(0, 0.75), (1, 0.25)], [(0, 1.0)], [(0, 1.0)]] + [[(0, 0.5), (1, 0.5)]] * 4
        changes = diff_snapshots(sample(), with_mesh(sample(), weights=WeightTable.from_lists(rows)))
        self.assertEqual([plain(c) for c in changes], [("weights", "Body", "versions.weights_changed", 1, True)])
        self.assertEqual(dict(changes[0].values), {"object": "Body", "count": 1, "difference": 0.25})
        self.assertEqual(mesh.weights.weights[1, 0], 1.0)

    def test_weights_tolerance(self):
        base = [[(0, 1.0)]] * 4 + [[(0, 0.5), (1, 0.5)]] * 4
        small = list(base)
        small[2] = [(0, 1.0 - 0.5 * thresholds.REPRO_TOLERANCE_WEIGHT)]
        self.assertEqual(diff_snapshots(sample(), with_mesh(sample(), weights=WeightTable.from_lists(small))), [])
        large = list(base)
        large[2] = [(0, 1.0 - 2 * thresholds.REPRO_TOLERANCE_WEIGHT)]
        changes = diff_snapshots(sample(), with_mesh(sample(), weights=WeightTable.from_lists(large)))
        self.assertEqual([c.count for c in changes], [1])
        self.assertAlmostEqual(changes[0].values["difference"], 2 * thresholds.REPRO_TOLERANCE_WEIGHT, places=15)

    def test_weights_compared_by_group_name(self):
        """頂点グループの並びを入れ替えただけなら、ウェイトは変わっていない"""
        rows = [[(1, 1.0)]] * 4 + [[(1, 0.5), (0, 0.5)]] * 4
        swapped = with_mesh(sample(), weights=WeightTable.from_lists(rows), group_names=("左ひじ", "左腕"))
        self.assertEqual(diff_snapshots(sample(), swapped), [])

    def test_vertex_group_added_and_removed(self):
        added = with_mesh(sample(), group_names=("左腕", "左ひじ", "頭"))
        changes = diff_snapshots(sample(), added)
        self.assertEqual([plain(c) for c in changes], [("weights", "Body", "versions.groups_added", 1, True)])
        self.assertEqual(changes[0].values["names"], ("頭",))
        removed = with_mesh(sample(), weights=WeightTable.from_lists([[(0, 1.0)]] * 4 + [[(0, 0.5)]] * 4),
                            group_names=("左腕",))
        changes = diff_snapshots(sample(), removed)
        self.assertEqual([plain(c) for c in changes], [
            ("weights", "Body", "versions.groups_removed", 1, True),
            ("weights", "Body", "versions.weights_changed", 4, True),
        ])
        self.assertEqual(changes[1].values["difference"], 0.5)

    def test_bone_moved(self):
        step = np.zeros((4, 3))
        step[2] = (0.0, 0.0, 2 * TOL)
        moved = with_armature(sample(), heads=HEADS + step, tails=TAILS + step, matrices=bone_matrices(HEADS + step))
        changes = diff_snapshots(sample(), moved)
        self.assertEqual([plain(c) for c in changes], [("bones", "Armature", "versions.bones_moved", 1, True)])
        self.assertEqual(changes[0].values["names"], ("左腕",))
        self.assertAlmostEqual(changes[0].values["distance_mm"], 0.02, places=9)
        half = with_armature(sample(), heads=HEADS + step / 4, tails=TAILS + step / 4)
        self.assertEqual(diff_snapshots(sample(), half), [])

    def test_bone_roll(self):
        matrices = bone_matrices()
        matrices[2, :3, 2] = (0.0, 1.0, 0.0)  # 骨の Z 軸の向きだけを変える（ねじれ）
        changes = diff_snapshots(sample(), with_armature(sample(), matrices=matrices))
        self.assertEqual([plain(c) for c in changes], [("bones", "Armature", "versions.bones_moved", 1, True)])
        length = float(np.linalg.norm(TAILS[2] - HEADS[2]))
        self.assertAlmostEqual(changes[0].values["distance_mm"], length * np.sqrt(2.0) * 1000.0, places=9)

    def test_bone_parent_deform_added_removed(self):
        changes = diff_snapshots(sample(), with_armature(sample(), parents=[-1, 0, 1, 1]))
        self.assertEqual([plain(c) for c in changes], [("bones", "Armature", "versions.bone_parents_changed", 1, True)])
        self.assertEqual(changes[0].values["names"], ("左ひじ",))
        changes = diff_snapshots(sample(), with_armature(sample(), deform=np.array([True, True, True, True])))
        self.assertEqual([plain(c) for c in changes], [("bones", "Armature", "versions.bone_deform_changed", 1, True)])
        self.assertEqual(changes[0].values["names"], ("センター",))
        grown = ArmatureState(names=BONE_NAMES + ("右腕",), parents=[-1, 0, 1, 2, 1],
                              heads=np.vstack([HEADS, [[-0.15, 0, 1.4]]]), tails=np.vstack([TAILS, [[-0.4, 0, 1.2]]]),
                              deform=np.array([False, True, True, True, True]),
                              matrices=bone_matrices(np.vstack([HEADS, [[-0.15, 0, 1.4]]])))
        added = replace(sample(), armatures={"Armature": grown})
        changes = diff_snapshots(sample(), added)
        self.assertEqual([plain(c) for c in changes], [("bones", "Armature", "versions.bones_added", 1, True)])
        self.assertEqual(changes[0].values["names"], ("右腕",))
        changes = diff_snapshots(added, sample())
        self.assertEqual([plain(c) for c in changes], [("bones", "Armature", "versions.bones_removed", 1, True)])

    def test_bone_order(self):
        """骨の並びだけを変えた（親子・位置は同じ）。書き出しの骨の番号が変わるので、変化として出す"""
        order = [0, 1, 3, 2]
        old_to_new = {old: new for new, old in enumerate(order)}
        parents = [-1 if p < 0 else old_to_new[p] for p in np.array([-1, 0, 1, 2])[order]]
        arm = armature_state()
        reordered = ArmatureState(names=tuple(arm.names[i] for i in order), parents=parents, heads=arm.heads[order],
                                  tails=arm.tails[order], deform=arm.deform[order], matrices=arm.matrices[order])
        changes = diff_snapshots(sample(), replace(sample(), armatures={"Armature": reordered}))
        self.assertEqual([plain(c) for c in changes], [("bones", "Armature", "versions.bone_order_changed", 1, True)])

    def test_object_added_removed_type_changed(self):
        hair = ObjectState("MESH", "Armature", np.eye(4))
        added = replace(sample(), objects={**sample().objects, "Hair": hair},
                        meshes={**sample().meshes, "Hair": MeshState(vertices=CUBE, triangles=quad_triangles(QUADS))})
        changes = diff_snapshots(sample(), added)
        self.assertEqual([plain(c) for c in changes], [("object", "Hair", "versions.object_added", 1, True)])
        self.assertEqual(dict(changes[0].values), {"object": "Hair", "type": "MESH"})
        changes = diff_snapshots(added, sample())
        self.assertEqual([plain(c) for c in changes], [("object", "Hair", "versions.object_removed", 1, True)])

        no_body = Snapshot(objects={"Armature": sample().objects["Armature"]}, armatures=sample().armatures)
        changes = diff_snapshots(sample(), no_body)
        self.assertEqual([plain(c) for c in changes], [("object", "Body", "versions.object_removed", 1, True)])

        empty = replace(sample(), objects={**sample().objects, "Body": ObjectState("EMPTY", "Armature", np.eye(4))},
                        meshes={})
        changes = diff_snapshots(sample(), empty)
        self.assertEqual([plain(c) for c in changes], [("object", "Body", "versions.object_type_changed", 1, True)])
        self.assertEqual(dict(changes[0].values), {"object": "Body", "old": "MESH", "new": "EMPTY"})

    def test_object_moved_and_reparented(self):
        matrix = np.eye(4)
        matrix[0, 3] = 0.001
        changes = diff_snapshots(sample(), with_object(sample(), "Body", matrix=matrix))
        self.assertEqual([plain(c) for c in changes], [("object", "Body", "versions.object_moved", 1, True)])
        self.assertEqual(dict(changes[0].values), {"object": "Body", "distance_mm": 1.0, "linear": 0.0})
        rotated = np.eye(4)
        rotated[:2, :2] = [[0.0, -1.0], [1.0, 0.0]]
        changes = diff_snapshots(sample(), with_object(sample(), "Body", matrix=rotated))
        self.assertEqual(dict(changes[0].values), {"object": "Body", "distance_mm": 0.0, "linear": 1.0})
        tiny = np.eye(4)
        tiny[0, 0] = 1.0 + 0.5 * thresholds.REPRO_TOLERANCE_RATIO
        self.assertEqual(diff_snapshots(sample(), with_object(sample(), "Body", matrix=tiny)), [])
        changes = diff_snapshots(sample(), with_object(sample(), "Body", parent=None))
        self.assertEqual([plain(c) for c in changes], [("object", "Body", "versions.object_parent_changed", 1, True)])
        self.assertEqual(dict(changes[0].values), {"object": "Body", "old": "Armature", "new": None})
        changes = diff_snapshots(sample(), with_object(sample(), "Body", parent_bone="頭"))
        self.assertEqual(changes[0].values["new"], "Armature / 頭")

    def test_modifiers(self):
        def mods(*entries):
            return with_object(sample(), "Body", modifiers=entries)
        cases = {
            "added": (mods(("Armature", "ARMATURE", True), ("Mask", "MASK", True)), "versions.modifiers_added",
                      ("Mask",)),
            "removed": (mods(), "versions.modifiers_removed", ("Armature",)),
            "render": (mods(("Armature", "ARMATURE", False)), "versions.modifiers_render_changed", ("Armature",)),
            "type": (mods(("Armature", "MASK", True)), "versions.modifiers_type_changed", ("Armature",)),
        }
        for label, (new, key, names) in cases.items():
            with self.subTest(case=label):
                changes = diff_snapshots(sample(), new)
                self.assertEqual([plain(c) for c in changes], [("modifiers", "Body", key, 1, True)])
                self.assertEqual(changes[0].values["names"], names)
        two = mods(("Armature", "ARMATURE", True), ("Mask", "MASK", True))
        swapped = mods(("Mask", "MASK", True), ("Armature", "ARMATURE", True))
        changes = diff_snapshots(two, swapped)
        self.assertEqual([plain(c) for c in changes], [("modifiers", "Body", "versions.modifier_order_changed", 1, True)])
        self.assertEqual(changes[0].values["names"], ("Mask", "Armature"))

    def test_materials(self):
        changes = diff_snapshots(sample(), with_object(sample(), "Body", materials=("肌", "服", "髪")))
        self.assertEqual([plain(c) for c in changes], [("materials", "Body", "versions.material_slots_changed", 1, True)])
        self.assertEqual(dict(changes[0].values), {"object": "Body", "old": ("肌", "服"), "new": ("肌", "服", "髪")})
        # 枠の並びを入れ替え、面の番号も合わせた：面の材質の名前は同じなので、面の変化は無い
        swapped = with_mesh(with_object(sample(), "Body", materials=("服", "肌")), face_materials=[1, 1, 0, 0, 1, 1])
        changes = diff_snapshots(sample(), swapped)
        self.assertEqual([c.text_key for c in changes], ["versions.material_slots_changed"])
        changes = diff_snapshots(sample(), with_mesh(sample(), face_materials=[1, 0, 1, 1, 0, 0]))
        self.assertEqual([plain(c) for c in changes], [("materials", "Body", "versions.face_materials_changed", 1, True)])
        self.assertEqual(changes[0].values["count"], 1)
        # 枠の外の番号どうしは、番号で比べる
        outside = with_mesh(sample(), face_materials=[5, 0, 1, 1, 0, 0])
        self.assertEqual(diff_snapshots(outside, outside), [])
        self.assertEqual([c.count for c in diff_snapshots(outside, with_mesh(sample(), face_materials=[6, 0, 1, 1, 0, 0]))],
                         [1])

    def test_topology(self):
        fewer = with_mesh(sample(), triangles=quad_triangles(QUADS[:5]), face_sizes=[4] * 5,
                          face_vertices=[v for quad in QUADS[:5] for v in quad], face_materials=[0, 0, 1, 1, 0])
        changes = diff_snapshots(sample(), fewer)
        self.assertEqual([plain(c) for c in changes], [("topology", "Body", "versions.topology_changed", 1, True)])
        self.assertEqual(dict(changes[0].values), {"object": "Body", "old_vertices": 8, "new_vertices": 8,
                                                   "old_faces": 6, "new_faces": 5})

    def test_topology_with_new_vertex_skips_per_vertex_checks(self):
        vertices = np.vstack([CUBE + 0.5, [[2.0, 2.0, 2.0]]])  # 頂点を全部動かしても、頂点の数が違えば比べない
        grown = MeshState(vertices=vertices, triangles=quad_triangles(QUADS),
                          shape_keys={"Basis": vertices, "あ": vertices, "まばたき": vertices},
                          weights=WeightTable.from_lists([[(0, 1.0)]] * 9), group_names=("左腕", "左ひじ"),
                          face_sizes=[4] * 6, face_vertices=[v for quad in QUADS for v in quad],
                          face_materials=[0, 0, 1, 1, 0, 0])
        changes = diff_snapshots(sample(), replace(sample(), meshes={"Body": grown}))
        self.assertEqual([plain(c) for c in changes], [("topology", "Body", "versions.topology_changed", 1, True)])
        self.assertEqual((changes[0].values["old_vertices"], changes[0].values["new_vertices"]), (8, 9))

    def test_quad_split_is_not_a_topology_change(self):
        """四角形を三角形に分ける向きだけが変わった（面は同じ）"""
        flipped = with_mesh(sample(), triangles=quad_triangles(QUADS, flip={0, 3}))
        self.assertEqual(diff_snapshots(sample(), flipped), [])

    def test_drivers(self):
        changes = diff_snapshots(sample(), replace(sample(), drivers=(blink_driver("var * 2"),)))
        self.assertEqual([plain(c) for c in changes], [("drivers", "Body", "versions.drivers_changed", 1, True)])
        self.assertEqual(changes[0].values["names"], (BLINK_PATH,))
        changed_variable = replace(blink_driver(), variables=("var ; SINGLE_PROP ; OBJECT|Armature|other",))
        self.assertEqual([c.text_key for c in diff_snapshots(sample(), replace(sample(), drivers=(changed_variable,)))],
                         ["versions.drivers_changed"])
        extra = Driver("Body", "location", "frame", index=1)
        changes = diff_snapshots(sample(), replace(sample(), drivers=(blink_driver(), extra)))
        self.assertEqual([plain(c) for c in changes], [("drivers", "Body", "versions.drivers_added", 1, True)])
        self.assertEqual(changes[0].values["names"], ("location[1]",))
        changes = diff_snapshots(sample(), replace(sample(), drivers=()))
        self.assertEqual([plain(c) for c in changes], [("drivers", "Body", "versions.drivers_removed", 1, True)])

    def test_order_of_changes(self):
        """オブジェクトの名前の順、その中は CATEGORIES の順"""
        new = moved_vertex(sample(), 6, (0.001, 0.0, 0.0))
        new = with_object(new, "Body", materials=("肌",), modifiers=())
        new = with_armature(new, deform=np.array([True, True, True, True]))
        new = replace(new, drivers=())
        changes = diff_snapshots(sample(), new)
        self.assertEqual([(c.object, c.category) for c in changes], [
            ("Armature", "bones"), ("Body", "vertices"), ("Body", "drivers"), ("Body", "materials"),
            ("Body", "materials"), ("Body", "modifiers"),
        ])
        self.assertEqual(changes[4].text_key, "versions.face_materials_changed")
        self.assertEqual(changes[4].count, 2)  # 枠 1（服）の面が、枠の外になった

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            diff_snapshots(sample(), "x")
        with self.assertRaises(ValueError):
            diff_snapshots(sample(), sample(), tolerance=-1.0)
        with self.assertRaises(ValueError):
            diff_snapshots(sample(), sample(), allowed=["Body"])


# MARK: - 変えてよい所とまとめ

class AllowedTest(unittest.TestCase):
    def test_allowed_change_is_green(self):
        new = moved_vertex(sample(), 6, (2 * TOL, 0.0, 0.0))
        changes = diff_snapshots(sample(), new, allowed=AllowedChanges({"Body": {"vertices"}}))
        self.assertEqual([c.outside_allowed for c in changes], [False])
        result = summarize_changes(changes, language="ja")
        self.assertEqual((result.name, result.value, result.unit, result.signal), ("unexpected_changes", 0.0, "count",
                                                                                   Signal.GREEN))
        self.assertEqual(result.reason, "変えてよい所の外の変化はありません（変えてよい所の中の変化が 1 件）")
        self.assertEqual(result.details["outside_items"], 0)

    def test_not_allowed_change_is_red(self):
        new = moved_vertex(sample(), 6, (2 * TOL, 0.0, 0.0))
        for allowed in (None, AllowedChanges({"Body": {"weights"}}), AllowedChanges({"Skirt": "*"}), {}):
            with self.subTest(allowed=allowed):
                changes = diff_snapshots(sample(), new, allowed=allowed)
                result = summarize_changes(changes, language="ja")
                self.assertEqual((result.value, result.signal), (1.0, Signal.RED))
                self.assertEqual(result.reason, "変えてよい所の外の変化が 1 件あります（変えてよい所の中の変化は 0 件）")
                self.assertEqual(result.details["outside_items"], 1)

    def test_star_allows_every_category(self):
        new = with_object(moved_vertex(sample(), 6, (0.001, 0, 0)), "Body", materials=())
        changes = diff_snapshots(sample(), new, allowed={"Body": "*"})
        self.assertEqual(len(changes), 3)
        self.assertEqual(summarize_changes(changes).signal, Signal.GREEN)

    def test_mixed(self):
        step = np.zeros((4, 3))
        step[3] = (0.001, 0.0, 0.0)
        new = with_armature(moved_vertex(sample(), 6, (0.001, 0, 0)), heads=HEADS + step, tails=TAILS + step)
        changes = diff_snapshots(sample(), new, allowed={"Body": {"vertices"}})
        self.assertEqual([(c.object, c.outside_allowed) for c in changes], [("Armature", True), ("Body", False)])
        result = summarize_changes(changes, language="en")
        self.assertEqual((result.value, result.signal), (1.0, Signal.RED))
        self.assertEqual(result.reason, "1 changes are outside the allowed changes (1 changes inside the allowed changes)")
        self.assertEqual(result.details["outside"], (changes[0],))
        self.assertEqual(result.details["allowed"], (changes[1],))

    def test_no_change_is_green(self):
        result = summarize_changes([], language="ja")
        self.assertEqual((result.value, result.signal, result.reason),
                         (0.0, Signal.GREEN, "前の版から変わった所はありません"))

    def test_allowed_validation(self):
        allowed = AllowedChanges({"Body": ["vertices", "weights"], "Hair": "*"})
        self.assertTrue(allowed.allows("Body", "weights"))
        self.assertFalse(allowed.allows("Body", "bones"))
        self.assertTrue(allowed.allows("Hair", "topology"))
        self.assertFalse(allowed.allows("Skirt", "vertices"))
        for bad in ({"Body": {"vertex"}}, {"Body": "vertices"}, {"": "*"}, ["Body"]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    AllowedChanges(bad)
        with self.assertRaises(ValueError):
            allowed.allows("Body", "colors")


class ChangeLinesTest(unittest.TestCase):
    def tearDown(self):
        texts.set_language("ja")

    def test_lines_in_both_languages(self):
        changes = diff_snapshots(sample(), moved_vertex(sample(), 6, (2 * TOL, 0.0, 0.0)))
        self.assertEqual(change_lines(changes, "ja"),
                         ["【変えてよい所の外】Body：頂点が 1 個動きました（いちばん大きい動き 0.020 mm）"])
        self.assertEqual(change_lines(changes, "en"),
                         ["[Outside the allowed changes] Body: 1 vertices moved (largest move 0.020 mm)"])
        self.assertEqual(change_lines(changes, "ja", mark_outside=False),
                         ["Body：頂点が 1 個動きました（いちばん大きい動き 0.020 mm）"])
        texts.set_language("en")
        self.assertEqual(change_lines(changes, mark_outside=False), ["Body: 1 vertices moved (largest move 0.020 mm)"])

    def test_allowed_lines_have_no_mark(self):
        changes = diff_snapshots(sample(), moved_vertex(sample(), 6, (0.001, 0.0, 0.0)), allowed={"Body": {"vertices"}})
        self.assertEqual(change_lines(changes, "ja"), ["Body：頂点が 1 個動きました（いちばん大きい動き 1.000 mm）"])

    def test_no_changes_line(self):
        self.assertEqual(change_lines([], "ja"), ["前の版から変わった所はありません"])
        self.assertEqual(change_lines([], "en"), ["Nothing changed since the previous version"])

    def test_none_and_name_lists(self):
        changes = diff_snapshots(sample(), with_object(sample(), "Body", parent=None, materials=("肌", "")))
        # 枠 1 の材質が無くなったので、枠 1 の面（2つ）の材質も変わった
        self.assertEqual(change_lines(changes, "ja", mark_outside=False), [
            "Body：親が変わりました（Armature → （なし））",
            "Body：材質の枠が変わりました（肌、服 → 肌、（なし））",
            "Body：材質が変わった面が 2 個あります",
        ])
        self.assertEqual(change_lines(changes, "en", mark_outside=False), [
            "Body: the parent changed (Armature → (none))",
            "Body: the material slots changed (肌, 服 → 肌, (none))",
            "Body: the material changed on 2 faces",
        ])

    def test_long_name_lists_are_shortened(self):
        extra = [f"b{i}" for i in range(1, 8)]
        count = len(BONE_NAMES) + len(extra)
        heads = np.vstack([HEADS, np.zeros((len(extra), 3))])
        tails = np.vstack([TAILS, np.ones((len(extra), 3))])
        grown = ArmatureState(names=BONE_NAMES + tuple(extra), parents=[-1, 0, 1, 2] + [0] * len(extra), heads=heads,
                              tails=tails, deform=np.ones(count, dtype=bool))
        changes = diff_snapshots(sample(), replace(sample(), armatures={"Armature": grown}))
        self.assertEqual(MAX_NAMES_IN_LINE, 5)
        self.assertEqual(change_lines(changes, "ja", mark_outside=False)[0],
                         "Armature：骨が 7 本増えました（b1、b2、b3、b4、b5 ほか 2 個）")
        self.assertEqual(change_lines(changes, "en", mark_outside=False)[0],
                         "Armature: 7 bones were added (b1, b2, b3, b4, b5 and 2 more)")

    def test_every_change_text_has_both_languages(self):
        """どの種類の変化の文も、日本語と英語で作れる"""
        new = moved_vertex(sample(), 6, (0.001, 0.0, 0.0), keys=False)
        new = with_mesh(new, weights=WeightTable.from_lists([[(0, 1.0)]] * 8))
        new = with_object(new, "Body", parent=None, materials=("肌",), modifiers=(("Mask", "MASK", False),))
        new = with_armature(new, parents=[-1, 0, 1, 1], deform=np.ones(4, dtype=bool))
        new = replace(new, drivers=(blink_driver("var * 2"),))
        changes = diff_snapshots(sample(), new)
        self.assertEqual({c.category for c in changes}, set(CATEGORIES) - {"topology"})
        for language in ("ja", "en"):
            lines = change_lines(changes, language)
            self.assertEqual(len(lines), len(changes))
            for line in lines:
                self.assertNotIn("{", line)

    def test_change_validation(self):
        good = dict(category="vertices", object="Body", text_key="versions.vertices_moved",
                    values={"object": "Body", "count": 1, "distance_mm": 0.5}, count=1, outside_allowed=True)
        self.assertEqual(Change(**good).text("ja"), "Body：頂点が 1 個動きました（いちばん大きい動き 0.500 mm）")
        for label, change in {"category": dict(category="color"), "key": dict(text_key="screen.tab"),
                              "unknown key": dict(text_key="versions.nothing"),
                              "values": dict(values={"object": "Body", "count": 1}),
                              "nan": dict(values={"object": "Body", "count": 1, "distance_mm": float("nan")}),
                              "bool value": dict(values={"object": "Body", "count": True, "distance_mm": 0.5}),
                              "count": dict(count=-1), "flag": dict(outside_allowed=1),
                              "object": dict(object="")}.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    Change(**{**good, **change})


# MARK: - 保存した版の比べ

class HistoryTest(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.folder = Path(self._temp.name)
        self.base = self.folder / "model.blend"

    def tearDown(self):
        self._temp.cleanup()

    def save(self, number, snapshot):
        path = self.folder / f"model_v{number:03d}.blend"
        path.write_bytes(b"blend")
        if snapshot is not None:
            save_snapshot(snapshot, self.folder / f"model_v{number:03d}.snapshot")
        return path

    def test_compare_versions(self):
        first = self.save(1, sample())
        second = self.save(2, moved_vertex(sample(), 6, (0.001, 0.0, 0.0)))
        changes, result = history.compare_versions(first, second, language="ja")
        self.assertEqual([plain(c) for c in changes], [("vertices", "Body", "versions.vertices_moved", 1, True)])
        self.assertEqual((result.value, result.signal), (1.0, Signal.RED))
        self.assertEqual((result.details["previous_version"], result.details["version"]), (str(first), str(second)))
        changes, result = history.compare_versions(first, second, allowed={"Body": {"vertices"}})
        self.assertEqual(result.signal, Signal.GREEN)

    def test_compare_latest(self):
        changes, result = history.compare_latest(self.base, language="ja")
        self.assertEqual((changes, result.signal, result.reason),
                         ([], Signal.GRAY, "比べる前の版がありません（まだ版を保存していません）"))
        self.save(1, sample())
        self.assertEqual(history.compare_latest(self.base)[1].signal, Signal.GRAY)
        self.save(2, sample())
        changes, result = history.compare_latest(self.base)
        self.assertEqual((changes, result.signal), ([], Signal.GREEN))
        self.save(3, moved_vertex(sample(), 2, (0.0, 0.0, 0.002)))
        (self.folder / "model_v004.blend").write_bytes(b"")  # 保存の途中の物は飛ばす
        changes, result = history.compare_latest(self.base)
        self.assertEqual([c.count for c in changes], [1])
        self.assertTrue(result.details["version"].endswith("model_v003.blend"))
        self.assertTrue(result.details["previous_version"].endswith("model_v002.blend"))

    def test_missing_record_is_gray(self):
        first = self.save(1, None)
        second = self.save(2, sample())
        changes, result = history.compare_versions(first, second, language="ja")
        self.assertEqual((changes, result.signal, result.value), ([], Signal.GRAY, None))
        self.assertTrue(result.reason.startswith("版 model_v001.blend の記録（形・骨・ウェイトなどの控え）が見つからないか、読めません："))

    def test_unreadable_record_is_gray(self):
        """記録の並びのファイルが zip として壊れていても（sha256 は合っている）、比べは灰になる（例外で止まらない）"""
        first = self.save(1, sample())
        second = self.save(2, sample())
        folder = self.folder / "model_v001.snapshot"
        (folder / ARRAYS_NAME).write_bytes(b"PK\x03\x04" + b"garbage" * 10)
        data = json.loads((folder / JSON_NAME).read_text(encoding="utf-8"))
        data["arrays_sha256"] = hashlib.sha256((folder / ARRAYS_NAME).read_bytes()).hexdigest()
        (folder / JSON_NAME).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        changes, result = history.compare_versions(first, second, language="ja")
        self.assertEqual((changes, result.signal, result.value), ([], Signal.GRAY, None))
        self.assertIn("model_v001.blend", result.reason)
        changes, result = history.compare_snapshot_with_version(sample(), first)
        self.assertEqual((changes, result.signal), ([], Signal.GRAY))

    def test_compare_snapshot_with_version(self):
        version = self.save(1, sample())
        changes, result = history.compare_snapshot_with_version(sample(), version)
        self.assertEqual((changes, result.signal), ([], Signal.GREEN))
        changes, result = history.compare_snapshot_with_version(sample(), None, language="en")
        self.assertEqual((result.signal, result.reason),
                         (Signal.GRAY, "There is no previous version to compare with (no version has been saved yet)"))

    def test_version_lines(self):
        self.save(1, sample())
        self.save(2, sample())
        (self.folder / "model_v003.blend").write_bytes(b"")
        self.assertEqual(history.version_lines(self.base, "ja"), ["第 1 版：model_v001.blend", "第 2 版：model_v002.blend"])
        self.assertEqual(history.version_lines(self.base, "en"),
                         ["Version 1: model_v001.blend", "Version 2: model_v002.blend"])
        self.assertTrue(history.is_version_file(self.folder / "model_v002.blend"))
        self.assertFalse(history.is_version_file(self.base))


if __name__ == "__main__":
    unittest.main()
