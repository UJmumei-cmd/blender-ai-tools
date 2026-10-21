"""裏の係の中身（bpy なし）：仕事を1つ動かし、progress.json と result.json を書く。

動かすのは目録（A8）の道具だけ（catalog.call() を通す。値はスキーマで確かめ直される）。
入力の .blend は開くだけで保存しない。前後で指紋（sha256）を比べ、変わっていれば結果を使えないものとして失敗にする。
例外は中で受け止めて、failed か cancelled として書く（裏の Blender が黙って消えないように）。
"""

from __future__ import annotations

import dataclasses
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import provenance
from ..catalog import Catalog
from ..measure.results import Measurement
from .progress import RESULT_FILE, JobCancelled, Progress, Reporter, current_reporter
from .spec import SCHEMA_VERSION, JobSpec, strict_json_dumps, strict_json_loads, write_text_atomic


def jsonable(value: Any) -> Any:
    """道具の戻り値を、JSON に書ける形に近づける。Measurement は to_data()、dataclass は辞書、タプルはリスト。
    numpy の配列などは provenance.to_plain() が引き受ける（できない型はそこで TypeError）"""
    if isinstance(value, Measurement):
        return value.to_data()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def _digest(path: Path) -> provenance.FileDigest | None:
    try:
        return provenance.file_digest(path)
    except (ValueError, OSError):
        return None


def result_path(job_dir: str | os.PathLike[str]) -> Path:
    return Path(job_dir) / RESULT_FILE


def read_result(job_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """result.json を読む。まだ無ければ None。形が違えば ValueError"""
    path = result_path(job_dir)
    if not path.is_file():
        return None
    data = strict_json_loads(path.read_text(encoding="utf-8"), "結果の JSON")
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"結果の形が違います：{path}")
    return data


def run_job(job: JobSpec, catalog: Catalog, job_dir: str | os.PathLike[str], *,
            open_input: Callable[[Path], None] | None = None,
            environment: dict[str, Any] | None = None) -> Progress:
    """仕事を1つ動かす。終わりの進み具合を返す（done・failed・cancelled のどれか）。

    open_input: 入力の .blend を開く関数（bpy の側が渡す。bpy なしのテストでは None）。
    environment: 記録に残す環境（Blender の版など。bpy の側が渡す）。
    """
    folder = Path(job_dir)
    folder.mkdir(parents=True, exist_ok=True)
    reporter = Reporter(folder, pid=os.getpid())
    reporter.start()

    input_path = None if job.input_blend is None else Path(job.input_blend)
    before = None
    if input_path is not None:
        before = _digest(input_path)
        if before is None:
            return reporter.finish("failed", error_key="jobs.error_input_missing", path=str(input_path))
    try:
        reporter.check_cancel()  # もう中止されていれば、重い .blend を開く前に止まる
        if open_input is not None and input_path is not None:
            open_input(input_path)
        reporter.check_cancel()
        with current_reporter(reporter):
            result = catalog.call(job.tool_id, job.params, confirmed=job.confirmed, plan_approved=job.plan_approved)
        tool_version = catalog.get(job.tool_id).version
        plain = provenance.to_plain(jsonable(result), "result")
    except JobCancelled:
        return reporter.finish("cancelled", error_key="jobs.error_cancelled")
    except Exception as error:  # noqa: BLE001  裏の係は、どんな例外でも「失敗」と書いて終わる（黙って消えない）
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return reporter.finish("failed", error=f"{type(error).__name__}: {error}")

    if input_path is not None:
        after = _digest(input_path)
        if after is None or before is None or after.sha256 != before.sha256:
            return reporter.finish("failed", error_key="jobs.error_input_changed", path=str(input_path))

    finished_at = provenance.now_iso()
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job.job_id,
        "tool_id": job.tool_id,
        "tool_version": tool_version,
        "started_at": reporter.started_at,
        "finished_at": finished_at,
        "input": None if before is None else {"path": before.path, "sha256": before.sha256, "size": before.size},
        "catalog_fingerprint": catalog.fingerprint(),
        "addon_version": provenance.addon_version(),
        "environment": provenance.to_plain({**provenance.python_environment(), **(environment or {})}, "environment"),
        "result": plain,
    }
    write_text_atomic(result_path(folder), strict_json_dumps(envelope))
    return reporter.finish("done")
