"""裏の Blender の入口。画面の Blender が、仕事ごとにこのファイルを別の Blender で動かす。

使い方（queue.build_worker_command が作る）:
    blender --background --factory-startup --disable-autoexec --python-exit-code 1 \
        --python blender_ai_tools/jobs/worker_main_bpy.py -- <仕事のフォルダ>

- 仕事のフォルダの job.json を読み、目録（A8）の道具を catalog.call() で動かし、progress.json と result.json を書く。
- 入力の .blend は開くだけで、保存はしない（run_job が前後の指紋を比べる）。
- 終了コード：done なら 0、failed・cancelled なら 1、始められなかったときは 2。

この入口だけは sys.path を書き換える（別の Blender では、拡張機能のパッケージを探す道がほかに無い）。
パッケージの中のコードは sys.path を書き換えない（CLAUDE.md 7章）。
"""

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import bpy  # noqa: E402

from blender_ai_tools import registry  # noqa: E402
from blender_ai_tools.jobs import spec, worker  # noqa: E402
from blender_ai_tools.provenance import environment_bpy  # noqa: E402


def open_input(path: Path) -> None:
    """入力の .blend を開く（画面の設定は読まない）。開くだけで、このあと保存はしない"""
    result = bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False)
    if result != {"FINISHED"}:
        raise ValueError(f"入力の .blend を開けませんでした：{path}（{result}）")


def main(argv: list[str]) -> int:
    if "--" not in argv or len(argv) - argv.index("--") != 2:
        print("使い方: blender --background --python worker_main_bpy.py -- <仕事のフォルダ>")
        return 2
    job_dir = Path(argv[argv.index("--") + 1])
    try:
        job = spec.load_job(job_dir)
    except ValueError as error:
        print(f"仕事を読めませんでした：{error}")
        return 2
    catalog = registry.build_catalog()
    progress = worker.run_job(job, catalog, job_dir, open_input=open_input,
                              environment=environment_bpy.blender_environment())
    return 0 if progress.status == "done" else 1


if __name__ == "__main__":
    code = main(sys.argv)
    sys.stdout.flush()
    sys.exit(code)
