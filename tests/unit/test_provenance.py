"""A15 道具の版と再現性の記録を確かめる（bpy なし）"""

import datetime
import enum
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from blender_ai_tools import provenance, thresholds
from blender_ai_tools.measure.results import Measurement, Signal
from blender_ai_tools.provenance import (Difference, FileDigest, GitState, RunRecord, addon_version,
                                         compare_results, file_digest, git_state, is_reproducible, load_record,
                                         make_rng, now_iso, python_environment, save_record)

PACKAGE_DIR = Path(__file__).resolve().parents[2] / "blender_ai_tools"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class Side(str, enum.Enum):
    """str と Enum を合わせた型（str() すると "Side.LEFT" になり、中身の "L" にならない）"""

    LEFT = "L"
    RIGHT = "R"


def make_record(**changes) -> RunRecord:
    """テスト用の記録"""
    values = dict(
        tool_id="measure.penetration",
        tool_version="1.0",
        started_at="2026-10-10T02:30:00Z",
        finished_at="2026-10-10T02:30:05Z",
        addon_version="0.1.0",
        environment={"python_version": "3.13.0", "numpy_version": "2.3.0", "platform": "test"},
        inputs=(FileDigest("model.blend", "/work/model.blend", EMPTY_SHA256, 0),),
        settings={"depth_red": thresholds.PENETRATION_RED, "frames": [1, 2, 3]},
        seed=0,
        results={"depth": 0.0012, "pose": {"knee_L": 12.5, "knee_R": 13.0}},
    )
    values.update(changes)
    return RunRecord(**values)


# MARK: - 入力ファイルの指紋


class FileDigestTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def write(self, name: str, data: bytes) -> Path:
        path = self.dir / name
        path.write_bytes(data)
        return path

    def test_known_bytes(self):
        data = "左ひざD\nhello\n".encode("utf-8")
        path = self.write("a.bin", data)
        digest = file_digest(path)
        self.assertEqual(digest.sha256, hashlib.sha256(data).hexdigest())
        self.assertEqual(digest.size, len(data))
        self.assertEqual(digest.name, "a.bin")
        self.assertEqual(digest.path, os.path.abspath(path))
        self.assertTrue(os.path.isabs(digest.path))

    def test_str_path_is_accepted(self):
        path = self.write("b.bin", b"xyz")
        self.assertEqual(file_digest(str(path)), file_digest(path))

    def test_empty_file(self):
        digest = file_digest(self.write("empty.bin", b""))
        self.assertEqual(digest.sha256, EMPTY_SHA256)
        self.assertEqual(digest.size, 0)

    def test_files_across_chunk_boundaries(self):
        """1 MiB ずつ読んでも、全体を一度に読んだときと同じハッシュになる（最後の半端な塊も入る）"""
        chunk = 1024 * 1024
        generator = np.random.Generator(np.random.PCG64(7))
        for size in (chunk - 1, chunk, chunk + 1, 2 * chunk + 12345):
            with self.subTest(size=size):
                data = generator.bytes(size)
                digest = file_digest(self.write(f"r{size}.bin", data))
                self.assertEqual(digest.sha256, hashlib.sha256(data).hexdigest())
                self.assertEqual(digest.size, size)

    def test_same_content_same_digest_different_content_different_digest(self):
        first = file_digest(self.write("1.bin", b"abc"))
        second = file_digest(self.write("2.bin", b"abc"))
        third = file_digest(self.write("3.bin", b"abd"))
        self.assertEqual(first.sha256, second.sha256)
        self.assertNotEqual(first.sha256, third.sha256)

    def test_missing_file(self):
        with self.assertRaises(ValueError):
            file_digest(self.dir / "nothing.blend")

    def test_directory(self):
        with self.assertRaises(ValueError):
            file_digest(self.dir)

    def test_validation(self):
        with self.assertRaises(ValueError):
            FileDigest("a", "/a", "XYZ", 0)
        with self.assertRaises(ValueError):
            FileDigest("a", "/a", EMPTY_SHA256.upper(), 0)
        with self.assertRaises(ValueError):
            FileDigest("a", "/a", EMPTY_SHA256, -1)
        with self.assertRaises(ValueError):
            FileDigest("a", "/a", EMPTY_SHA256, True)
        with self.assertRaises(ValueError):
            FileDigest("", "/a", EMPTY_SHA256, 0)
        self.assertEqual(FileDigest("a", "/a", EMPTY_SHA256, np.int64(3)).size, 3)


# MARK: - git の状態

GIT = shutil.which("git")


def run_git(directory: Path, *args: str) -> str:
    """テスト用のリポジトリを作る git。ユーザーの設定（署名など）を読まない"""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    result = subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false",
         "-c", "init.defaultBranch=main", "-C", str(directory), *args],
        capture_output=True, text=True, timeout=60, env=env, check=True)
    return result.stdout.strip()


@unittest.skipIf(GIT is None, "git が入っていないため（git_state の試しは、git がある所で行う）")
class GitStateTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.repo = self.root / "koumakan"
        (self.repo / "tools" / "model-converter").mkdir(parents=True)
        self.script = self.repo / "tools" / "model-converter" / "convert_blend.py"
        self.script.write_text("print('convert')\n", encoding="utf-8")
        run_git(self.repo, "init", "-q")
        run_git(self.repo, "add", ".")
        run_git(self.repo, "commit", "-q", "-m", "first")
        self.head = run_git(self.repo, "rev-parse", "HEAD")
        # テストのフォルダより上のリポジトリを見に行かないようにする
        patcher = mock.patch.dict(os.environ, {"GIT_CEILING_DIRECTORIES": str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self._dir.cleanup()

    def test_clean_repo(self):
        state = git_state(self.repo)
        self.assertEqual(state, GitState(commit=self.head, dirty=False))
        self.assertRegex(state.commit, COMMIT_PATTERN)

    def test_modified_file_is_dirty(self):
        self.script.write_text("print('changed')\n", encoding="utf-8")
        self.assertEqual(git_state(self.repo), GitState(commit=self.head, dirty=True))

    def test_untracked_file_is_dirty(self):
        (self.repo / "new.py").write_text("x = 1\n", encoding="utf-8")
        self.assertEqual(git_state(self.repo), GitState(commit=self.head, dirty=True))

    def test_new_commit_changes_commit(self):
        self.script.write_text("print('v2')\n", encoding="utf-8")
        run_git(self.repo, "commit", "-q", "-am", "second")
        state = git_state(self.repo)
        self.assertNotEqual(state.commit, self.head)
        self.assertEqual(state.commit, run_git(self.repo, "rev-parse", "HEAD"))
        self.assertFalse(state.dirty)

    def test_subdirectory_and_script_path(self):
        """convert_blend.py のフォルダや、ファイルそのものを渡してもよい"""
        self.assertEqual(git_state(self.script.parent), GitState(self.head, False))
        self.assertEqual(git_state(self.script), GitState(self.head, False))
        self.assertEqual(git_state(str(self.script)), GitState(self.head, False))

    def test_not_a_repo(self):
        plain = self.root / "plain"
        plain.mkdir()
        self.assertEqual(git_state(plain), GitState(None, None))

    def test_missing_directory(self):
        self.assertEqual(git_state(self.root / "nothing"), GitState(None, None))

    def test_repo_without_commits(self):
        empty = self.root / "empty"
        empty.mkdir()
        run_git(empty, "init", "-q")
        self.assertEqual(git_state(empty), GitState(None, False))
        (empty / "a.txt").write_text("a", encoding="utf-8")
        self.assertEqual(git_state(empty), GitState(None, True))

    def test_ignores_git_dir_environment(self):
        """GIT_DIR などが別のリポジトリを指していても、渡したフォルダのリポジトリを調べる"""
        other = self.root / "other"
        other.mkdir()
        run_git(other, "init", "-q")
        (other / "o.txt").write_text("o", encoding="utf-8")
        run_git(other, "add", ".")
        run_git(other, "commit", "-q", "-m", "other")
        with mock.patch.dict(os.environ, {"GIT_DIR": str(other / ".git"), "GIT_WORK_TREE": str(other)}):
            self.assertEqual(git_state(self.repo), GitState(self.head, False))

    def test_runs_git_without_shell_and_with_timeout(self):
        with mock.patch("blender_ai_tools.provenance.subprocess.run", wraps=subprocess.run) as run:
            git_state(self.repo)
        self.assertGreaterEqual(run.call_count, 1)
        for call in run.call_args_list:
            command = call.args[0]
            self.assertIsInstance(command, list)
            self.assertEqual(command[:3], ["git", "-C", str(self.repo)])
            self.assertFalse(call.kwargs.get("shell", False))
            self.assertGreater(call.kwargs["timeout"], 0)

    def test_git_missing(self):
        with mock.patch("blender_ai_tools.provenance.subprocess.run", side_effect=FileNotFoundError("git")):
            self.assertEqual(git_state(self.repo), GitState(None, None))

    def test_git_timeout(self):
        error = subprocess.TimeoutExpired(["git"], provenance.GIT_TIMEOUT_SECONDS)
        with mock.patch("blender_ai_tools.provenance.subprocess.run", side_effect=error):
            self.assertEqual(git_state(self.repo), GitState(None, None))

    def test_untracked_file_is_dirty_even_when_config_hides_untracked_files(self):
        """git の設定（status.showUntrackedFiles=no）で隠れていても、まだ git に入れていないファイルを変更に数える"""
        run_git(self.repo, "config", "status.showUntrackedFiles", "no")
        (self.repo / "tools" / "model-converter" / "helper.py").write_text("x = 1\n", encoding="utf-8")
        # 設定が効いていて、ふつうの status には出てこないことを先に確かめる
        self.assertEqual(run_git(self.repo, "status", "--porcelain"), "")
        self.assertEqual(git_state(self.script), GitState(self.head, True))


class GitStateValidationTest(unittest.TestCase):
    def test_empty_path_is_rejected(self):
        """空の文字は "."（いまの作業フォルダ）になり、別のリポジトリを調べてしまうので ValueError"""
        with mock.patch("blender_ai_tools.provenance.subprocess.run") as run:
            with self.assertRaises(ValueError):
                git_state("")
        run.assert_not_called()

    def test_validation(self):
        self.assertEqual(GitState(None, None).commit, None)
        with self.assertRaises(ValueError):
            GitState("abc", False)
        with self.assertRaises(ValueError):
            GitState("a" * 40, 1)
        with self.assertRaises(ValueError):
            GitState("A" * 40, False)
        self.assertTrue(GitState("a" * 64, True).dirty)


# MARK: - 版と環境・乱数・時刻


class EnvironmentTest(unittest.TestCase):
    def test_addon_version_matches_manifest(self):
        manifest = tomllib.loads((PACKAGE_DIR / "blender_manifest.toml").read_text(encoding="utf-8"))
        self.assertEqual(addon_version(), manifest["version"])

    def test_python_environment(self):
        environment = python_environment()
        self.assertEqual(set(environment), {"python_version", "numpy_version", "platform"})
        self.assertEqual(environment["python_version"], platform.python_version())
        self.assertEqual(environment["numpy_version"], np.__version__)
        self.assertIsInstance(environment["platform"], str)
        self.assertTrue(environment["platform"])

    def test_provenance_does_not_import_catalog(self):
        """部品どうしを別々に保つ：provenance は catalog を読み込まない"""
        source = (PACKAGE_DIR / "provenance" / "__init__.py").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?m)^\s*(from|import)\s+[.\w]*catalog")


class RngTest(unittest.TestCase):
    def test_same_seed_same_numbers(self):
        np.testing.assert_array_equal(make_rng(42).random(100), make_rng(42).random(100))
        np.testing.assert_array_equal(make_rng(42).normal(size=10), make_rng(42).normal(size=10))

    def test_different_seed_different_numbers(self):
        self.assertFalse(np.array_equal(make_rng(1).random(10), make_rng(2).random(10)))

    def test_default_seed(self):
        np.testing.assert_array_equal(make_rng().random(5), make_rng(thresholds.DEFAULT_SEED).random(5))

    def test_uses_pcg64_with_known_first_number(self):
        """方式は PCG64。種 0 の最初の数は、numpy 2.3 と 2.5 で同じ値"""
        rng = make_rng(0)
        self.assertIsInstance(rng.bit_generator, np.random.PCG64)
        self.assertEqual(rng.random(), 0.6369616873214543)

    def test_numpy_integer_seed(self):
        np.testing.assert_array_equal(make_rng(np.int64(5)).random(3), make_rng(5).random(3))

    def test_rejects_bad_seeds(self):
        for seed in (True, False, np.bool_(True), -1, np.int64(-3), 1.0, 1.5, "1", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                make_rng(seed)


class NowIsoTest(unittest.TestCase):
    def test_format_and_time(self):
        before = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
        text = now_iso()
        after = datetime.datetime.now(datetime.UTC)
        self.assertRegex(text, r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
        moment = datetime.datetime.fromisoformat(text)
        self.assertEqual(moment.utcoffset(), datetime.timedelta(0))
        self.assertLessEqual(before, moment)
        self.assertLessEqual(moment, after)

    @unittest.skipUnless(hasattr(time, "tzset"), "time.tzset が無い OS のため")
    def test_is_utc_even_in_another_time_zone(self):
        """機械の時差が日本（+9時間）でも、UTC の時刻を書く（クラウドは UTC なので、時差を変えて確かめる）"""
        def restore_time_zone():
            time.tzset()

        patcher = mock.patch.dict(os.environ, {"TZ": "JST-9"})
        patcher.start()
        self.addCleanup(restore_time_zone)
        self.addCleanup(patcher.stop)
        time.tzset()
        self.assertEqual(time.timezone, -9 * 3600)  # 時差が本当に変わったか
        before = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
        moment = datetime.datetime.fromisoformat(now_iso())
        after = datetime.datetime.now(datetime.UTC)
        self.assertLessEqual(before, moment)
        self.assertLessEqual(moment, after)


# MARK: - 記録と JSON


class RunRecordTest(unittest.TestCase):
    def test_round_trip(self):
        record = make_record(
            catalog_fingerprint="f" * 64,
            convert_blend=GitState("a" * 40, True),
            notes="基準用モデル",
        )
        text = record.to_json()
        again = RunRecord.from_json(text)
        self.assertEqual(again, record)
        self.assertEqual(again.to_json(), text)
        self.assertIsInstance(again.inputs, tuple)
        self.assertIsInstance(again.convert_blend, GitState)

    def test_round_trip_without_optional_parts(self):
        record = make_record(seed=None)
        again = RunRecord.from_json(record.to_json())
        self.assertIsNone(again.convert_blend)
        self.assertIsNone(again.catalog_fingerprint)
        self.assertIsNone(again.seed)
        self.assertEqual(again.to_json(), record.to_json())

    def test_numpy_values_nan_and_paths(self):
        record = make_record(results={
            "float": np.float64(1.5),
            "float32": np.float32(0.5),
            "int": np.int64(3),
            "flag": np.bool_(True),
            "array": np.arange(6, dtype=np.float64).reshape(2, 3),
            "int_array": np.array([1, 2], dtype=np.int32),
            "nan": math.nan,
            "np_nan": np.float64("nan"),
            "inf": math.inf,
            "minus_inf": -np.inf,
            "array_with_nan": np.array([1.0, np.nan, np.inf]),
            "tuple": (1, (2, 3)),
            "path": Path("/work/model.blend"),
            "signal": Signal.RED,
            "nested": {"empty": {}, "list": []},
        })
        text = record.to_json()
        data = json.loads(text)["results"]
        self.assertEqual(data["float"], 1.5)
        self.assertEqual(data["float32"], 0.5)
        self.assertEqual(data["int"], 3)
        self.assertIs(type(data["int"]), int)
        self.assertIs(data["flag"], True)
        self.assertEqual(data["array"], [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        self.assertEqual(data["int_array"], [1, 2])
        self.assertIsNone(data["nan"])
        self.assertIsNone(data["np_nan"])
        self.assertIsNone(data["inf"])
        self.assertIsNone(data["minus_inf"])
        self.assertEqual(data["array_with_nan"], [1.0, None, None])
        self.assertEqual(data["tuple"], [1, [2, 3]])
        self.assertEqual(data["path"], "/work/model.blend")
        self.assertEqual(data["signal"], 3)
        self.assertEqual(data["nested"], {"empty": {}, "list": []})
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        self.assertEqual(RunRecord.from_json(text).to_json(), text)

    def test_canonical_text(self):
        first = make_record(results={"b": 1, "a": {"y": 2, "x": 1}}, settings={"z": 1, "a": 2})
        second = make_record(results={"a": {"x": 1, "y": 2}, "b": 1}, settings={"a": 2, "z": 1})
        self.assertEqual(first.to_json(), second.to_json())
        text = make_record(notes="左ひざD の補正").to_json()
        self.assertIn("左ひざD の補正", text)  # 日本語は \\u にしない
        self.assertIn('\n  "addon_version": "0.1.0"', text)  # 字下げ 2
        top_keys = list(json.loads(text))
        self.assertEqual(top_keys, sorted(top_keys))

    def test_unknown_types_are_rejected(self):
        measurement = Measurement.from_value("depth", 0.001, "m", red=thresholds.PENETRATION_RED)
        for value in (object(), {1, 2}, b"bytes", 1 + 2j, np.array([1 + 2j]), measurement,
                      datetime.datetime(2026, 1, 1), [1, object()], np.datetime64("2026-01-01T00:00:00", "ns"),
                      np.array([b"ab"])):
            with self.subTest(value=type(value).__name__), self.assertRaises(TypeError):
                make_record(results={"x": {"y": value}}).to_json()

    def test_arrays_of_times_and_records_are_rejected(self):
        """時刻・時間の配列は ns の整数に、構造体の配列は名前の無いリストに、勝手に変わってしまうので TypeError
        （時刻1つ（np.datetime64）が TypeError になるのと合わせる）"""
        for value in (np.array(["2026-01-01"], dtype="datetime64[ns]"), np.array([5], dtype="timedelta64[ns]"),
                      np.array([(1, 2.0)], dtype=[("frame", "i4"), ("depth", "f8")])):
            with self.subTest(dtype=str(value.dtype)):
                with self.assertRaises(TypeError):
                    make_record(results={"x": value}).to_json()
                with self.assertRaises(TypeError):
                    compare_results({"x": value}, {"x": value})

    def test_error_names_the_key(self):
        with self.assertRaises(TypeError) as context:
            make_record(results={"pose": {"knee_L": {1, 2}}}).to_json()
        self.assertIn("results.pose.knee_L", str(context.exception))

    def test_str_enum_is_written_by_its_value(self):
        """str と Enum を合わせた値・キーは、中身の文字で書く（str() の "Side.LEFT" にしない）"""
        record = make_record(results={"side": Side.LEFT, "by_side": {Side.RIGHT: 1.0}, "sides": [Side.LEFT]})
        data = json.loads(record.to_json())["results"]
        self.assertEqual(data, {"side": "L", "by_side": {"R": 1.0}, "sides": ["L"]})
        loaded = RunRecord.from_json(record.to_json())
        self.assertEqual(compare_results(record.results, loaded.results), [])

    def test_non_string_keys_are_rejected(self):
        with self.assertRaises(TypeError):
            make_record(results={1: "a"}).to_json()
        with self.assertRaises(TypeError):
            make_record(settings={"a": {(1, 2): 3}}).to_json()

    def test_validation(self):
        bad_values = {
            "tool_id": "",
            "tool_version": 1,
            "started_at": "yesterday",
            "finished_at": "2026-10-10T02:30:00",  # 時差が無い
            "addon_version": None,
            "environment": [],
            "inputs": ("model.blend",),
            "settings": None,
            "seed": True,
            "results": "ok",
            "catalog_fingerprint": "",
            "convert_blend": {"commit": None, "dirty": None},
            "notes": None,
            "schema_version": 2,
        }
        for name, value in bad_values.items():
            with self.subTest(field=name), self.assertRaises(ValueError):
                make_record(**{name: value})
        with self.assertRaises(ValueError):
            make_record(seed=-1)
        with self.assertRaises(ValueError):
            make_record(schema_version=True)

    def test_inputs_list_becomes_tuple_and_numpy_seed_becomes_int(self):
        record = make_record(inputs=[FileDigest("a", "/a", EMPTY_SHA256, 0)], seed=np.int64(7))
        self.assertIsInstance(record.inputs, tuple)
        self.assertIs(type(record.seed), int)
        self.assertEqual(record.seed, 7)

    def test_from_json_rejects_bad_text(self):
        good = json.loads(make_record().to_json())
        cases = {
            "not json": "{",
            "not an object": "[]",
            "unknown key": json.dumps({**good, "extra": 1}),
            "missing key": json.dumps({key: value for key, value in good.items() if key != "results"}),
            "new schema": json.dumps({**good, "schema_version": 2}),
            "bool schema": json.dumps({**good, "schema_version": True}),
            "nan literal": make_record().to_json().replace('"depth": 0.0012', '"depth": NaN'),
            "duplicate key": make_record().to_json().replace('"notes": ""', '"notes": "", "notes": "x"'),
            "bad digest": json.dumps({**good, "inputs": [{"name": "a", "path": "/a", "sha256": "x", "size": 0}]}),
            "digest extra key": json.dumps({**good, "inputs": [{"name": "a", "path": "/a", "sha256": EMPTY_SHA256,
                                                                "size": 0, "x": 1}]}),
            "float seed": json.dumps({**good, "seed": 1.0}),
            "bool seed": json.dumps({**good, "seed": True}),
            "results list": json.dumps({**good, "results": []}),
            "convert_blend extra": json.dumps({**good, "convert_blend": {"commit": None, "dirty": None, "x": 1}}),
        }
        for name, text in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                RunRecord.from_json(text)
        with self.assertRaises(ValueError):
            RunRecord.from_json(b"{}")

    def test_from_json_fills_optional_parts(self):
        data = json.loads(make_record().to_json())
        for key in ("catalog_fingerprint", "convert_blend", "notes"):
            del data[key]
        record = RunRecord.from_json(json.dumps(data))
        self.assertEqual((record.catalog_fingerprint, record.convert_blend, record.notes), (None, None, ""))


# MARK: - 保存と読み込み


class SaveRecordTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def test_save_and_load(self):
        record = make_record()
        path = save_record(record, self.dir)
        self.assertEqual(path, self.dir / "2026-10-10T02-30-00Z_measure.penetration.json")
        self.assertTrue(path.is_file())
        self.assertEqual(load_record(path), record)
        text = path.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("}\n"))
        self.assertEqual(json.loads(text)["tool_id"], "measure.penetration")

    def test_never_overwrites(self):
        first = save_record(make_record(notes="1回目"), self.dir)
        second = save_record(make_record(notes="2回目"), self.dir)
        third = save_record(make_record(notes="3回目"), self.dir)
        self.assertEqual(second.name, first.stem + "_2.json")
        self.assertEqual(third.name, first.stem + "_3.json")
        self.assertEqual([load_record(p).notes for p in (first, second, third)], ["1回目", "2回目", "3回目"])

    def test_keeps_existing_file_with_the_same_name(self):
        existing = self.dir / "2026-10-10T02-30-00Z_measure.penetration.json"
        existing.write_text("ユーザーの物", encoding="utf-8")
        path = save_record(make_record(), self.dir)
        self.assertNotEqual(path, existing)
        self.assertEqual(existing.read_text(encoding="utf-8"), "ユーザーの物")

    def test_only_the_record_is_left(self):
        """一時ファイルが残らない"""
        save_record(make_record(), self.dir)
        save_record(make_record(), self.dir)
        names = sorted(p.name for p in self.dir.iterdir())
        self.assertEqual(len(names), 2)
        self.assertTrue(all(name.endswith(".json") and not name.startswith(".") for name in names))

    def test_creates_directory(self):
        folder = self.dir / "records" / "2026"
        path = save_record(make_record(), folder)
        self.assertEqual(path.parent, folder)
        self.assertTrue(path.is_file())

    def test_unsafe_tool_id_stays_in_directory(self):
        path = save_record(make_record(tool_id="../../evil/x y:z"), self.dir)
        self.assertEqual(path.parent, self.dir)
        self.assertRegex(path.name, r"\A[A-Za-z0-9._-]+\.json\Z")
        self.assertEqual(load_record(path).tool_id, "../../evil/x y:z")

    def test_record_that_cannot_be_json_leaves_no_file(self):
        with self.assertRaises(TypeError):
            save_record(make_record(results={"x": object()}), self.dir / "new")
        self.assertFalse((self.dir / "new").exists())

    def test_failed_replace_leaves_no_file(self):
        with mock.patch("blender_ai_tools.provenance.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                save_record(make_record(), self.dir)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_directory_is_a_file(self):
        not_a_folder = self.dir / "file.txt"
        not_a_folder.write_text("x", encoding="utf-8")
        with self.assertRaises(ValueError):
            save_record(make_record(), not_a_folder)

    def test_not_a_record(self):
        with self.assertRaises(ValueError):
            save_record({"tool_id": "x"}, self.dir)

    def test_load_errors(self):
        with self.assertRaises(ValueError):
            load_record(self.dir / "nothing.json")
        with self.assertRaises(ValueError):
            load_record(self.dir)
        broken = self.dir / "broken.json"
        broken.write_text("{", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_record(broken)
        latin = self.dir / "latin.json"
        latin.write_bytes(b"\xff\xfe{}")
        with self.assertRaises(ValueError):
            load_record(latin)


# MARK: - 結果の比べ


class CompareResultsTest(unittest.TestCase):
    def assertOneDifference(self, differences, key, kind):
        self.assertEqual([(d.key, d.kind) for d in differences], [(key, kind)])
        return differences[0]

    def test_identical(self):
        results = {"depth": 0.001, "pose": {"knee_L": 12.0}, "name": "左足D", "ok": True, "none": None,
                   "array": np.arange(4.0)}
        self.assertEqual(compare_results(results, results), [])
        self.assertTrue(is_reproducible(results, results))
        self.assertEqual(compare_results({}, {}), [])

    def test_tolerance_boundary(self):
        """幅と同じ差は違いにしない。幅より大きい差は違い（2進で正確に表せる数で確かめる）"""
        self.assertEqual(compare_results({"x": 1.0}, {"x": 1.25}, default_tolerance=0.25), [])
        difference = self.assertOneDifference(compare_results({"x": 1.0}, {"x": 1.5}, default_tolerance=0.25),
                                              "x", "value")
        self.assertEqual(difference.delta, 0.5)
        self.assertEqual(difference.tolerance, 0.25)
        self.assertEqual((difference.a, difference.b), (1.0, 1.5))

    def test_delta_is_positive_both_ways(self):
        for a, b in ((2.0, 1.0), (1.0, 2.0), (-1.0, 1.0)):
            with self.subTest(a=a, b=b):
                difference = compare_results({"x": a}, {"x": b})[0]
                self.assertEqual(difference.delta, abs(a - b))
                self.assertGreater(difference.delta, 0)

    def test_zero_tolerance_by_default(self):
        self.assertOneDifference(compare_results({"x": 0.1 + 0.2}, {"x": 0.3}), "x", "value")
        self.assertEqual(compare_results({"x": 1}, {"x": 1.0}), [])

    def test_repro_tolerances(self):
        a = {"leg_length": 0.8, "pose": {"knee_L": 10.0}, "volume_ratio": 0.5}
        b = {"leg_length": 0.8 + thresholds.REPRO_TOLERANCE_LENGTH / 2,
             "pose": {"knee_L": 10.0 + thresholds.REPRO_TOLERANCE_ANGLE_DEG / 2},
             "volume_ratio": 0.5 + thresholds.REPRO_TOLERANCE_RATIO / 2}
        tolerances = {"leg_length": thresholds.REPRO_TOLERANCE_LENGTH,
                      "pose.knee_L": thresholds.REPRO_TOLERANCE_ANGLE_DEG,
                      "volume_ratio": thresholds.REPRO_TOLERANCE_RATIO}
        self.assertTrue(is_reproducible(a, b, tolerances))
        self.assertEqual([d.key for d in compare_results(a, b)], ["leg_length", "pose.knee_L", "volume_ratio"])
        c = dict(b, leg_length=0.8 + thresholds.REPRO_TOLERANCE_LENGTH * 2)
        self.assertOneDifference(compare_results(a, c, tolerances), "leg_length", "value")

    def test_per_key_tolerance_overrides_default(self):
        """キーごとの幅は、既定の幅より小さくても、そちらを使う"""
        a, b = {"x": 1.0, "y": 1.0}, {"x": 1.5, "y": 1.5}
        differences = compare_results(a, b, tolerances={"x": 0.0}, default_tolerance=1.0)
        difference = self.assertOneDifference(differences, "x", "value")
        self.assertEqual(difference.tolerance, 0.0)
        # x には幅 1.0、y には既定の幅 0.0
        difference = self.assertOneDifference(compare_results(a, b, tolerances={"x": 1.0}), "y", "value")
        self.assertEqual(difference.tolerance, 0.0)

    def test_nested_keys(self):
        a = {"pose": {"knee_L": 1.0, "knee_R": 2.0}, "foot": {"L": {"sink": 0.0}}}
        b = {"pose": {"knee_L": 1.0, "knee_R": 2.5}, "foot": {"L": {"sink": 0.001}}}
        differences = compare_results(a, b)
        self.assertEqual([d.key for d in differences], ["foot.L.sink", "pose.knee_R"])
        self.assertEqual(compare_results(a, b, tolerances={"pose.knee_R": 0.5, "foot.L.sink": 0.001}), [])

    def test_tolerance_of_parent_key_does_not_apply(self):
        """幅はキーが同じときだけ使う（親のキーの幅は、子に使わない）"""
        self.assertOneDifference(compare_results({"pose": {"knee_L": 1.0}}, {"pose": {"knee_L": 1.5}},
                                                 tolerances={"pose": 1.0}), "pose.knee_L", "value")

    def test_dotted_key_equals_nested_key(self):
        self.assertEqual(compare_results({"a.b": 1.0}, {"a": {"b": 1.0}}), [])

    def test_key_collision(self):
        with self.assertRaises(ValueError):
            compare_results({"a.b": 1.0, "a": {"b": 2.0}}, {})

    def test_arrays(self):
        a = {"v": np.zeros(3)}
        b = {"v": np.array([0.1, -0.3, 0.2])}
        difference = self.assertOneDifference(compare_results(a, b), "v", "value")
        self.assertAlmostEqual(difference.delta, 0.3, places=15)
        self.assertEqual(compare_results(a, b, default_tolerance=0.3), [])
        self.assertEqual(len(compare_results(a, b, default_tolerance=0.29)), 1)

    def test_list_and_array_compare_elementwise(self):
        self.assertEqual(compare_results({"v": [1.0, 2.0, 3.0]}, {"v": np.array([1.0, 2.0, 3.0])}), [])
        self.assertEqual(compare_results({"v": [[1, 2], [3, 4]]}, {"v": np.array([[1.0, 2.0], [3.0, 4.0]])}), [])
        difference = self.assertOneDifference(compare_results({"v": [1, 2, 3]}, {"v": [1, 2, 5]}), "v", "value")
        self.assertEqual(difference.delta, 2.0)

    def test_shape_mismatch(self):
        """要素の数が同じでも、形が違えば "shape"（平らにして比べない）"""
        a = {"v": np.arange(6.0).reshape(2, 3)}
        b = {"v": np.arange(6.0).reshape(3, 2)}
        difference = self.assertOneDifference(compare_results(a, b, default_tolerance=100.0), "v", "shape")
        self.assertIsNone(difference.delta)
        self.assertOneDifference(compare_results({"v": [1.0, 2.0, 3.0]}, {"v": [1.0, 2.0]}), "v", "shape")
        self.assertOneDifference(compare_results({"v": np.zeros(3)}, {"v": np.zeros((3, 1))}), "v", "shape")

    def test_empty_arrays(self):
        """(0, 3) の配列は JSON で [] になるので、[] と同じに扱う"""
        self.assertEqual(compare_results({"v": np.zeros((0, 3))}, {"v": []}), [])
        self.assertEqual(compare_results({"v": np.zeros((0, 3))}, {"v": np.zeros((0, 3))}), [])
        self.assertOneDifference(compare_results({"v": np.zeros((3, 0))}, {"v": []}), "v", "shape")
        # 空の配列のリスト（例：フレームごとの、めり込んだ点。どのフレームにも無い）も JSON の [[], []] と同じ
        self.assertEqual(compare_results({"v": [np.zeros((0, 3)), np.zeros((0, 3))]}, {"v": [[], []]}), [])
        self.assertOneDifference(compare_results({"v": [np.zeros((0, 3))]}, {"v": [[], []]}), "v", "shape")

    def test_numbers_inside_lists_of_dicts_use_the_tolerance(self):
        """出来事の一覧（辞書のリスト）の中の数も、幅の中の差は違いにしない（== で比べると、小さな誤差で違いになる）"""
        tolerance = thresholds.REPRO_TOLERANCE_LENGTH
        a = {"events": [{"frame": 12, "bone": "左ひざD", "depth": 0.0012, "points": [[0.0, 0.1, 0.9]]},
                        {"frame": 30, "bone": "右ひざD", "depth": None, "points": []}]}

        def changed(depth=0.0012, bone="左ひざD", frame=12, point_z=0.9, second_depth=None):
            return {"events": [{"frame": frame, "bone": bone, "depth": depth, "points": [[0.0, 0.1, point_z]]},
                               {"frame": 30, "bone": "右ひざD", "depth": second_depth, "points": []}]}

        self.assertEqual(compare_results(a, changed(depth=0.0012 + tolerance / 2), {"events": tolerance}), [])
        self.assertEqual(compare_results(a, changed(point_z=0.9 - tolerance / 2), default_tolerance=tolerance), [])
        difference = self.assertOneDifference(
            compare_results(a, changed(depth=0.0012 + 3 * tolerance), {"events": tolerance}), "events", "value")
        self.assertAlmostEqual(difference.delta, 3 * tolerance, delta=1e-15)
        self.assertEqual(difference.tolerance, tolerance)
        # delta は、いちばん大きい差（点の差と深さの差のうち大きいほう）
        difference = self.assertOneDifference(
            compare_results(a, changed(depth=0.0012 + 3 * tolerance, point_z=0.9 + 5 * tolerance),
                            {"events": tolerance}), "events", "value")
        self.assertAlmostEqual(difference.delta, 5 * tolerance, delta=1e-15)
        # 整数も幅で比べる（12 と 13 の差 1 は、幅 0.5 を超える）
        self.assertOneDifference(compare_results(a, changed(frame=13), default_tolerance=0.5), "events", "value")
        # 片方だけ測れなかった：幅がいくら大きくても違い（delta は None）
        difference = self.assertOneDifference(
            compare_results(a, changed(second_depth=0.001), default_tolerance=math.inf), "events", "value")
        self.assertIsNone(difference.delta)
        self.assertEqual(compare_results(a, changed(second_depth=math.nan)), [])  # NaN と None は同じ
        # 数でない所の違いは、幅に関係なく "other"（数の差があっても "other" を先にする）
        for b in (changed(bone="右ひざD"), changed(bone="右ひざD", depth=1.0),
                  {"events": a["events"][:1]}, {"events": [dict(a["events"][0], note="x"), a["events"][1]]},
                  {"events": [dict(a["events"][0], depth="0.0012"), a["events"][1]]},
                  {"events": [dict(a["events"][0], depth=True), a["events"][1]]}):
            with self.subTest(b=b):
                difference = self.assertOneDifference(
                    compare_results(a, b, default_tolerance=math.inf), "events", "other")
                self.assertIsNone(difference.delta)
                self.assertIsNone(difference.tolerance)
        # 幅を超えた数の差が先（1つ目の出来事）に見つかっても、あとに数でない違いがあれば "other"
        b = {"events": [dict(a["events"][0], depth=1.0), dict(a["events"][1], bone="左ひざD")]}
        difference = self.assertOneDifference(compare_results(a, b), "events", "other")
        self.assertIsNone(difference.delta)

    def test_ragged_number_lists_use_the_tolerance(self):
        """長さのそろわない数のリスト（四角い配列にならない）も、要素ごとに幅を使って比べる"""
        a = {"v": [[1.0, 2.0], [3.0]]}
        self.assertEqual(compare_results(a, {"v": [[1.0, 2.0], [3.0 + 1e-12]]}, default_tolerance=1e-9), [])
        difference = self.assertOneDifference(compare_results(a, {"v": [[1.0, 2.5], [3.0]]}, default_tolerance=0.25),
                                              "v", "value")
        self.assertEqual(difference.delta, 0.5)
        self.assertOneDifference(compare_results(a, {"v": [[1.0, 2.0], [3.0, 4.0]]}, default_tolerance=10.0),
                                 "v", "type")  # 四角い配列と、四角くないリスト
        self.assertOneDifference(compare_results(a, {"v": [[1.0, 2.0], [3.0], [4.0]]}, default_tolerance=10.0),
                                 "v", "other")

    def test_nan_cases(self):
        nan = math.nan
        self.assertEqual(compare_results({"x": nan}, {"x": nan}), [])
        self.assertEqual(compare_results({"x": nan}, {"x": np.float64("nan")}), [])
        self.assertEqual(compare_results({"x": None}, {"x": nan}), [])  # JSON にすると NaN は null
        self.assertEqual(compare_results({"x": math.inf}, {"x": None}), [])
        for a, b in ((nan, 1.0), (1.0, nan), (None, 0.0), (0, None)):
            with self.subTest(a=a, b=b):
                difference = self.assertOneDifference(
                    compare_results({"x": a}, {"x": b}, default_tolerance=math.inf), "x", "value")
                self.assertIsNone(difference.delta)

    def test_nan_in_arrays(self):
        a = {"v": np.array([1.0, nan := math.nan, 3.0])}
        self.assertEqual(compare_results(a, {"v": [1.0, None, 3.0]}), [])
        self.assertEqual(compare_results(a, {"v": [1.4, nan, 3.0]}, default_tolerance=0.5), [])
        difference = self.assertOneDifference(compare_results(a, {"v": [1.0, 2.0, 3.0]}, default_tolerance=10.0),
                                              "v", "value")
        self.assertIsNone(difference.delta)
        self.assertEqual(compare_results({"v": [None, None]}, {"v": np.full(2, np.nan)}), [])

    def test_missing_keys(self):
        differences = compare_results({"a": 1, "b": 2}, {"a": 1, "c": 3})
        self.assertEqual([(d.key, d.kind, d.a, d.b) for d in differences], [("b", "missing", 2, None),
                                                                              ("c", "missing", None, 3)])
        differences = compare_results({"pose": {"x": 1, "y": 2}}, {})
        self.assertEqual([(d.key, d.kind) for d in differences], [("pose.x", "missing"), ("pose.y", "missing")])
        self.assertOneDifference(compare_results({"a": {}}, {}), "a", "missing")

    def test_bools_are_not_numbers(self):
        self.assertOneDifference(compare_results({"x": True}, {"x": 1}), "x", "type")
        self.assertOneDifference(compare_results({"x": False}, {"x": 0.0}), "x", "type")
        self.assertOneDifference(compare_results({"x": True}, {"x": False}, default_tolerance=10.0), "x", "other")
        self.assertEqual(compare_results({"x": np.bool_(True)}, {"x": True}), [])
        self.assertOneDifference(compare_results({"x": [True, False]}, {"x": [1, 0]}), "x", "type")
        self.assertOneDifference(compare_results({"x": [True, False]}, {"x": [True, True]}), "x", "other")
        self.assertEqual(compare_results({"x": np.array([True, False])}, {"x": [True, False]}), [])
        # 数でないリストの中でも、True と 1 は同じにしない（Python の == では True == 1 になる）
        self.assertOneDifference(compare_results({"x": ["a", True]}, {"x": ["a", 1]}), "x", "other")
        self.assertOneDifference(compare_results({"x": [{"ok": True}]}, {"x": [{"ok": 1}]}), "x", "other")

    def test_strings_and_none(self):
        self.assertEqual(compare_results({"x": "左足D"}, {"x": "左足D"}), [])
        self.assertOneDifference(compare_results({"x": "左足D"}, {"x": "右足D"}), "x", "other")
        self.assertEqual(compare_results({"x": Path("a/b")}, {"x": str(Path("a/b"))}), [])
        self.assertEqual(compare_results({"x": None}, {"x": None}), [])
        self.assertOneDifference(compare_results({"x": None}, {"x": "a"}), "x", "type")
        self.assertEqual(compare_results({"x": ["左足D", "右足D"]}, {"x": ("左足D", "右足D")}), [])
        self.assertOneDifference(compare_results({"x": ["左足D", "右足D"]}, {"x": ["右足D", "左足D"]}), "x", "other")

    def test_type_mismatch(self):
        for a, b in ((1.0, "1.0"), ([1.0, 2.0], 1.0), ({}, 1.0), ("a", ["a"]), (np.array([1.0]), 1.0)):
            with self.subTest(a=a, b=b):
                difference = self.assertOneDifference(compare_results({"x": a}, {"x": b}), "x", "type")
                self.assertIsNone(difference.delta)

    def test_zero_dimensional_array_is_a_number(self):
        self.assertEqual(compare_results({"x": np.array(2.0)}, {"x": 2.0}), [])

    def test_large_integers_are_exact(self):
        difference = self.assertOneDifference(compare_results({"n": 2 ** 60}, {"n": 2 ** 60 + 1}), "n", "value")
        self.assertEqual(difference.delta, 1.0)
        self.assertEqual(compare_results({"n": np.int64(2 ** 60)}, {"n": 2 ** 60}), [])

    def test_overflowing_difference(self):
        difference = compare_results({"x": 1e308}, {"x": -1e308})[0]
        self.assertEqual(difference.delta, math.inf)
        # float にできないほど大きな整数と小数：OverflowError にせず、差は inf
        difference = self.assertOneDifference(compare_results({"x": 10 ** 400}, {"x": 1.0}), "x", "value")
        self.assertEqual(difference.delta, math.inf)
        self.assertOneDifference(compare_results({"x": [{"n": 1.0}]}, {"x": [{"n": 10 ** 400}]}), "x", "value")
        difference = compare_results({"x": np.array([1e308])}, {"x": np.array([-1e308])})[0]
        self.assertEqual(difference.delta, math.inf)

    def test_float32_against_float64(self):
        a, b = {"x": np.float32(0.1)}, {"x": 0.1}
        self.assertEqual(len(compare_results(a, b)), 1)
        self.assertEqual(compare_results(a, b, default_tolerance=thresholds.REPRO_TOLERANCE_RATIO), [])

    def test_sorted_and_deterministic(self):
        a = {"z": 1.0, "a": 1.0, "m": {"b": 1.0, "a": 1.0}}
        b = {"m": {"a": 2.0, "b": 2.0}, "a": 2.0, "z": 2.0}
        keys = [d.key for d in compare_results(a, b)]
        self.assertEqual(keys, ["a", "m.a", "m.b", "z"])
        reordered = {"m": {"b": 1.0, "a": 1.0}, "z": 1.0, "a": 1.0}
        self.assertEqual([d.key for d in compare_results(reordered, b)], keys)

    def test_bad_tolerances(self):
        for kwargs in ({"default_tolerance": -0.1}, {"default_tolerance": math.nan}, {"default_tolerance": True},
                       {"tolerances": {"x": -1e-9}}, {"tolerances": {"unused": -1.0}}, {"tolerances": {"x": "1"}},
                       {"tolerances": {1: 1.0}}, {"tolerances": [("x", 1.0)]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                compare_results({"x": 1.0}, {"x": 1.0}, **kwargs)

    def test_infinite_tolerance_ignores_size_but_not_unmeasured(self):
        self.assertEqual(compare_results({"x": 0.0}, {"x": 1e300}, default_tolerance=math.inf), [])
        self.assertEqual(len(compare_results({"x": 0.0}, {"x": math.nan}, default_tolerance=math.inf)), 1)

    def test_bad_inputs(self):
        with self.assertRaises(TypeError):
            compare_results([1.0], {"x": 1.0})
        with self.assertRaises(TypeError):
            compare_results({"x": {1, 2}}, {"x": {1, 2}})
        with self.assertRaises(TypeError):
            compare_results({1: 1.0}, {1: 1.0})
        measurement = Measurement.from_value("depth", 0.001, "m", red=thresholds.PENETRATION_RED)
        with self.assertRaises(TypeError):
            compare_results({"x": measurement}, {"x": measurement})

    def test_difference_kind_is_checked(self):
        with self.assertRaises(ValueError):
            Difference("x", 1, 2, 1.0, 0.0, "bigger")

    def test_record_round_trip_is_reproducible(self):
        """記録を JSON にして読み戻した結果は、元の結果と同じ（NaN・配列・numpy の数を含む）"""
        results = {
            "depth": np.float64(0.0012),
            "count": np.int64(4),
            "missing": math.nan,
            "far": math.inf,
            "flags": np.array([True, False]),
            "points": np.arange(12.0).reshape(4, 3) / 7.0,
            "empty": np.zeros((0, 3)),
            "with_nan": np.array([[1.0, np.nan], [np.nan, 2.0]]),
            "pose": {"knee_L": np.float32(12.5), "names": ("左足D", "右足D")},
            "bone_names": np.array(["左足D", "右足D"]),
            "path": Path("/work/model.blend"),
            "per_frame_points": [np.zeros((0, 3)), np.zeros((0, 3))],
            "events": [{"frame": np.int64(3), "depth": np.float32(0.1), "side": Side.LEFT, "missing": np.nan}],
            "side": Side.RIGHT,
        }
        loaded = RunRecord.from_json(make_record(results=results).to_json())
        self.assertEqual(compare_results(results, loaded.results), [])
        self.assertTrue(is_reproducible(loaded.results, results))


if __name__ == "__main__":
    unittest.main()
