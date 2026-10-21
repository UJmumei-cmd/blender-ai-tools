"""A6 版のファイルの名前（versions/naming.py）を確かめる。bpy なしで動く。

ファイルは一時フォルダの中だけに作る。
"""

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from blender_ai_tools.versions import naming
from blender_ai_tools.versions.naming import (
    base_path, base_stem, latest_version, list_versions, next_version_path, parse_version, release_reservation,
    snapshot_dir, version_number, version_path,
)


class BaseStemTest(unittest.TestCase):
    def test_plain_name(self):
        self.assertEqual(base_stem("model.blend"), "model")
        self.assertEqual(base_stem(Path("/a/b/キャラ.blend")), "キャラ")

    def test_strips_one_version_suffix(self):
        self.assertEqual(base_stem("model_v003.blend"), "model")
        self.assertEqual(base_stem("/a/b/キャラ_v012.blend"), "キャラ")
        self.assertEqual(base_stem("model_v999.blend"), "model")
        self.assertEqual(base_stem("model_v1000.blend"), "model")
        self.assertEqual(base_stem("model_v12345.blend"), "model")
        # 1つだけ外す
        self.assertEqual(base_stem("a_v003_v002.blend"), "a_v003")

    def test_names_that_are_not_versions(self):
        """この道具が作らない形は、元の名前の一部のまま"""
        for name in ("model_v1", "model_v01", "model_v0001", "model_v000", "model_V003", "model-v003", "_v003",
                     "model_v003x", "model_v 003", "model_v０１２", "model_v003\n"):  # 終わりが改行の名前も版ではない
            with self.subTest(name=name):
                self.assertEqual(base_stem(f"{name}.blend"), name)
                self.assertIsNone(parse_version(f"{name}.blend"))

    def test_uppercase_extension(self):
        self.assertEqual(base_stem("Model_v004.BLEND"), "Model")
        self.assertEqual(parse_version("Model_v004.BLEND"), ("Model", 4))
        self.assertEqual(base_path("/x/Model_v004.Blend"), Path("/x/Model.blend"))

    def test_rejects_non_blend(self):
        for value in ("model.blend1", "model.txt", "model", ".blend", "", "model.blend/", b"model.blend", 3, None):
            with self.subTest(value=value):
                if value == "model.blend/":
                    # Path は終わりの / を外すので、これは .blend として受け付ける
                    self.assertEqual(base_stem(value), "model")
                    continue
                with self.assertRaises(ValueError):
                    base_stem(value)

    def test_parse_version(self):
        self.assertEqual(parse_version("/x/model_v003.blend"), ("model", 3))
        self.assertIsNone(parse_version("/x/model.blend"))
        self.assertIsNone(parse_version("/x/model_v003.blend1"))
        self.assertIsNone(parse_version("/x/model_v003.snapshot"))
        self.assertEqual(version_number("/x/model_v1000.blend"), 1000)
        self.assertIsNone(version_number("/x/model.blend"))

    def test_base_path(self):
        self.assertEqual(base_path("/x/model_v003.blend"), Path("/x/model.blend"))
        self.assertEqual(base_path("/x/model.blend"), Path("/x/model.blend"))


class VersionPathTest(unittest.TestCase):
    def test_three_digits(self):
        self.assertEqual(version_path("/x/model.blend", 1), Path("/x/model_v001.blend"))
        self.assertEqual(version_path("/x/model.blend", 42), Path("/x/model_v042.blend"))
        self.assertEqual(version_path("/x/model.blend", 999), Path("/x/model_v999.blend"))

    def test_more_digits_after_999(self):
        self.assertEqual(version_path("/x/model.blend", 1000), Path("/x/model_v1000.blend"))
        self.assertEqual(version_path("/x/model.blend", 12345), Path("/x/model_v12345.blend"))

    def test_from_version_file(self):
        """版のファイルから作っても、元の名前で作る"""
        self.assertEqual(version_path("/x/model_v003.blend", 12), Path("/x/model_v012.blend"))

    def test_uppercase_extension_gives_lowercase(self):
        self.assertEqual(version_path("/x/M.BLEND", 2), Path("/x/M_v002.blend"))

    def test_round_trip(self):
        for n in (1, 7, 99, 999, 1000, 4321):
            with self.subTest(n=n):
                self.assertEqual(parse_version(version_path("/x/model.blend", n)), ("model", n))

    def test_bad_numbers(self):
        for n in (0, -1, True, 1.0, "1", None):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    version_path("/x/model.blend", n)

    def test_rejects_non_blend(self):
        with self.assertRaises(ValueError):
            version_path("/x/model.fbx", 1)

    def test_snapshot_dir(self):
        self.assertEqual(snapshot_dir("/x/model_v003.blend"), Path("/x/model_v003.snapshot"))
        self.assertEqual(snapshot_dir("/x/model_v1000.BLEND"), Path("/x/model_v1000.snapshot"))
        with self.assertRaises(ValueError):
            snapshot_dir("/x/model.blend")
        with self.assertRaises(ValueError):
            snapshot_dir("/x/model_v003.txt")


class FolderCase(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.folder = Path(self._temp.name)
        self.base = self.folder / "model.blend"
        self.base.write_bytes(b"original")

    def tearDown(self):
        self._temp.cleanup()

    def touch(self, name, content=b"blend"):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path


class ListVersionsTest(FolderCase):
    def test_only_matching_files(self):
        self.touch("model_v001.blend")
        self.touch("model_v003.blend")
        self.touch("model_v010.BLEND")
        for name in ("model_v002.blend1", "model_v004.txt", "other_v005.blend", "model_v1.blend",
                     "model_v0006.blend", "Model_v007.blend", "sub/model_v009.blend", "model_v011.blend.tmp",
                     "xmodel_v012.blend", ".model_v013.abc.saving.blend"):
            self.touch(name)
        (self.folder / "model_v008.blend").mkdir()  # .blend の名前のフォルダは版ではない
        (self.folder / "model_v006.snapshot").mkdir()
        self.assertEqual(list_versions(self.base), [
            (1, self.folder / "model_v001.blend"),
            (3, self.folder / "model_v003.blend"),
            (10, self.folder / "model_v010.BLEND"),
        ])

    def test_numeric_order(self):
        for name in ("model_v1000.blend", "model_v999.blend", "model_v002.blend"):
            self.touch(name)
        self.assertEqual([n for n, _ in list_versions(self.base)], [2, 999, 1000])

    def test_from_version_file(self):
        self.touch("model_v001.blend")
        self.touch("model_v002.blend")
        self.assertEqual(list_versions(self.folder / "model_v002.blend"), list_versions(self.base))

    def test_missing_folder(self):
        self.assertEqual(list_versions(self.folder / "none" / "model.blend"), [])

    def test_latest_skips_empty_files(self):
        self.assertIsNone(latest_version(self.base))
        self.touch("model_v001.blend")
        self.touch("model_v002.blend", b"")  # 保存の途中か、失敗した物
        self.assertEqual(latest_version(self.base), self.folder / "model_v001.blend")
        self.touch("model_v003.blend")
        self.assertEqual(latest_version(self.base), self.folder / "model_v003.blend")


class NextVersionTest(FolderCase):
    def test_first_version(self):
        path = next_version_path(self.base)
        self.assertEqual(path, self.folder / "model_v001.blend")
        self.assertTrue(path.is_file())
        self.assertEqual(path.stat().st_size, 0)
        self.assertEqual(self.base.read_bytes(), b"original")

    def test_two_reservations_in_a_row(self):
        first = next_version_path(self.base)
        second = next_version_path(self.base)
        self.assertEqual((first.name, second.name), ("model_v001.blend", "model_v002.blend"))
        self.assertTrue(first.is_file() and second.is_file())

    def test_after_highest_even_with_gaps(self):
        self.touch("model_v001.blend")
        self.touch("model_v005.blend")
        self.assertEqual(next_version_path(self.base).name, "model_v006.blend")

    def test_uppercase_and_snapshot_names_take_numbers(self):
        self.touch("model_v002.BLEND")
        (self.folder / "model_v004.snapshot").mkdir()  # 版のファイルを消して、記録だけ残っている
        self.assertEqual(next_version_path(self.base).name, "model_v005.blend")

    def test_other_files_do_not_take_numbers(self):
        for name in ("model_v007.blend1", "other_v009.blend", "model_v1.blend"):
            self.touch(name)
        self.assertEqual(next_version_path(self.base).name, "model_v001.blend")

    def test_from_version_file(self):
        """いまのファイルが版のファイル（model_v003.blend）でも、元の名前で次の番号を取る"""
        current = self.touch("model_v003.blend")
        path = next_version_path(current)
        self.assertEqual(path, self.folder / "model_v004.blend")
        self.assertEqual(current.read_bytes(), b"blend")

    def test_base_ending_with_version_and_no_original(self):
        """元の名前のファイル（model.blend）が無くても、版のファイルから数える"""
        self.base.unlink()
        current = self.touch("model_v003.blend")
        self.assertEqual(next_version_path(current).name, "model_v004.blend")

    def test_reserve_false_does_not_create(self):
        self.touch("model_v001.blend")
        path = next_version_path(self.base, reserve=False)
        self.assertEqual(path.name, "model_v002.blend")
        self.assertFalse(path.exists())
        self.assertEqual(next_version_path(self.base, reserve=False), path)

    def test_skips_names_taken_after_the_scan(self):
        """フォルダを見たあとで、ほかのプロセスが先に作った名前は飛ばす（上書きしない）"""
        taken = self.touch("model_v001.blend", b"other process")
        (self.folder / "model_v002.snapshot").mkdir()
        with mock.patch.object(naming, "_taken_numbers", return_value=set()):
            path = next_version_path(self.base)
        self.assertEqual(path.name, "model_v003.blend")
        self.assertEqual(taken.read_bytes(), b"other process")

    def test_gives_up_after_many_tries(self):
        with mock.patch.object(naming, "_taken_numbers", return_value=set()), \
                mock.patch.object(naming, "MAX_RESERVE_TRIES", 3):
            for n in (1, 2, 3):
                self.touch(f"model_v{n:03d}.blend")
            with self.assertRaises(FileExistsError):
                next_version_path(self.base)

    def test_concurrent_reservations_get_different_names(self):
        count = 8
        barrier = threading.Barrier(count)
        results = []
        lock = threading.Lock()

        def work():
            barrier.wait()
            path = next_version_path(self.base)
            with lock:
                results.append(path.name)

        threads = [threading.Thread(target=work) for _ in range(count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(results), [f"model_v{n:03d}.blend" for n in range(1, count + 1)])

    def test_missing_folder(self):
        with self.assertRaises(ValueError):
            next_version_path(self.folder / "none" / "model.blend")

    def test_bad_reserve_flag(self):
        with self.assertRaises(ValueError):
            next_version_path(self.base, reserve=1)

    def test_release_reservation(self):
        reserved = next_version_path(self.base)
        self.assertTrue(release_reservation(reserved))
        self.assertFalse(reserved.exists())
        saved = self.touch("model_v002.blend")
        self.assertFalse(release_reservation(saved))  # 中身のある版は消さない
        self.assertEqual(saved.read_bytes(), b"blend")
        self.assertFalse(release_reservation(self.folder / "model_v009.blend"))

    @unittest.skipIf(not hasattr(os, "symlink"), "シンボリックリンクを作れません")
    def test_does_not_follow_dangling_symlink(self):
        """壊れたリンクの名前も使われているとみなし、リンクの先にファイルを作らない"""
        outside = self.folder / "outside.blend"
        try:
            os.symlink(outside, self.folder / "model_v001.blend")
        except OSError as error:
            self.skipTest(f"シンボリックリンクを作れません：{error}")
        with mock.patch.object(naming, "_taken_numbers", return_value=set()):
            path = next_version_path(self.base)
        self.assertEqual(path.name, "model_v002.blend")
        self.assertFalse(outside.exists())


if __name__ == "__main__":
    unittest.main()
