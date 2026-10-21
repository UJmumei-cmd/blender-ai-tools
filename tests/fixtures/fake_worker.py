"""テスト用の裏の係（Blender なし）。python3 -I tests/fixtures/fake_worker.py <仕事のフォルダ> で動く。

試し用の道具だけを載せた目録で、jobs.worker.run_job を動かす。順番待ち（JobQueue）の、始める・進み具合を読む・
中止する・消えたときの扱いを、本物の別プロセスで確かめるための物。build_fake_catalog() は、同じプロセスの
テストからも使う。

このファイルはテストの入口のスクリプトなので、sys.path を書き換えてリポジトリを探す（パッケージの中のコードはしない）。
"""

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blender_ai_tools.catalog import Catalog, Effect, ToolSpec  # noqa: E402
from blender_ai_tools.jobs import progress, spec, worker  # noqa: E402
from blender_ai_tools.measure.results import Measurement, Signal  # noqa: E402

_INT = {"type": "integer", "minimum": 0, "maximum": 1000000}
_NUMBER = {"type": "number", "minimum": 0, "maximum": 3600}
_TEXT = {"type": "string", "maxLength": 4096}


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def sleep_tool(params: dict) -> dict:
    """steps 段に分けて、1段ごとに seconds 秒待つ。段ごとに進み具合を書く（中止はそこで止まる）"""
    steps, seconds = params["steps"], params["seconds"]
    for step in range(1, steps + 1):
        progress.report(step, steps, "jobs.step_of", {"step": step, "total": steps})
        time.sleep(seconds)
    return {"steps": steps}


def echo_tool(params: dict) -> dict:
    """値をそのまま返す。Measurement も入れて、JSON にできる形になるかを確かめる"""
    return {"value": params["value"],
            "measurement": Measurement("echo_length", float(len(params["value"])), "count", Signal.GREEN, "試し",
                                       {"text": params["value"]})}


def crash_tool(params: dict) -> dict:
    """「終わり」を書かずに、プロセスごと消える（裏の Blender が落ちたときの試し）"""
    sys.stdout.flush()
    os._exit(params["code"])


def touch_input_tool(params: dict) -> dict:
    """入力のファイルを書き換える（してはいけないこと。見つかるかの試し）"""
    with open(params["path"], "ab") as handle:
        handle.write(b"x")
    return {"touched": True}


def fail_tool(params: dict) -> dict:
    raise ValueError("わざと失敗しました")


def swallow_tool(params: dict) -> dict:
    """何でも握りつぶす道具（except Exception）。cancel_at_step 段で、自分で中止を頼む（画面の側の代わり）。
    中止が Exception の子だと、ここで握りつぶされて最後まで動いてしまう"""
    steps, cancel_at_step = params["steps"], params["cancel_at_step"]
    for step in range(1, steps + 1):
        try:
            progress.report(step, steps, "jobs.step_of", {"step": step, "total": steps})
            if step == cancel_at_step:
                progress.request_cancel(progress.current_job_dir())
        except Exception:  # noqa: BLE001  わざと何でも受け止める
            pass
    return {"steps": steps}


def modify_tool(params: dict) -> dict:
    """「変更」の道具（確認が要る）"""
    return {"modified": True}


def build_fake_catalog() -> Catalog:
    catalog = Catalog()
    tools = [
        ("fake.sleep", "待つ", Effect.READ, _schema({"steps": _INT, "seconds": _NUMBER}, ["steps", "seconds"]), sleep_tool),
        ("fake.echo", "返す", Effect.READ, _schema({"value": _TEXT}, ["value"]), echo_tool),
        ("fake.crash", "落ちる", Effect.READ, _schema({"code": _INT}, ["code"]), crash_tool),
        ("fake.touch_input", "入力を書き換える", Effect.READ, _schema({"path": _TEXT}, ["path"]), touch_input_tool),
        ("fake.fail", "失敗する", Effect.READ, _schema({}, []), fail_tool),
        ("fake.swallow", "何でも握りつぶす", Effect.READ, _schema({"steps": _INT, "cancel_at_step": _INT}, ["steps", "cancel_at_step"]),
         swallow_tool),
        ("fake.modify", "変更する", Effect.MODIFY, _schema({}, []), modify_tool),
    ]
    for tool_id, title, effect, schema, func in tools:
        catalog.register(ToolSpec(id=tool_id, version="1", feature="A10", category="fake", title=title,
                                  summary="テスト用の道具", effect=effect, heavy=True, input_schema=schema), func)
    return catalog


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("使い方: python3 -I fake_worker.py <仕事のフォルダ>")
        return 2
    job_dir = Path(argv[1])
    try:
        job = spec.load_job(job_dir)
    except ValueError as error:
        print(f"仕事を読めませんでした：{error}")
        return 2
    result = worker.run_job(job, build_fake_catalog(), job_dir, environment={"fake_worker": True})
    return 0 if result.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
