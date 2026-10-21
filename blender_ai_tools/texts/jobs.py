"""裏で動かす係（A10）の文。キー → {"ja": 日本語, "en": 英語}。数字などは {name} の形で入れる"""

TEXTS: dict[str, dict[str, str]] = {
    "jobs.status_queued": {"ja": "順番待ち", "en": "Queued"},
    "jobs.status_running": {"ja": "動いています", "en": "Running"},
    "jobs.status_done": {"ja": "終わりました", "en": "Done"},
    "jobs.status_failed": {"ja": "失敗しました", "en": "Failed"},
    "jobs.status_cancelled": {"ja": "中止しました", "en": "Cancelled"},
    "jobs.error_cancelled": {"ja": "中止の頼みを受けて止めました", "en": "Stopped on request"},
    "jobs.error_process_died": {
        "ja": "裏の Blender が途中で止まりました（終了コード {code}）",
        "en": "The background Blender stopped unexpectedly (exit code {code})",
    },
    "jobs.error_orphaned": {
        "ja": "前の回の裏の Blender が終わる前に、画面が閉じられました。結果はありません",
        "en": "The screen was closed before the previous background Blender finished. There is no result",
    },
    "jobs.error_input_changed": {
        "ja": "入力のファイルが途中で変わりました。結果は使えません（{path}）",
        "en": "The input file changed during the job. The result cannot be used ({path})",
    },
    "jobs.error_input_missing": {
        "ja": "入力のファイルがありません（{path}）",
        "en": "The input file is missing ({path})",
    },
    "jobs.error_timeout": {
        "ja": "時間の上限（{seconds} 秒）を過ぎたので止めました",
        "en": "Stopped after the time limit ({seconds} s)",
    },
    "jobs.error_start_failed": {
        "ja": "裏の Blender を始められませんでした（{reason}）",
        "en": "Could not start the background Blender ({reason})",
    },
    "jobs.step_of": {"ja": "{step} / {total}", "en": "{step} / {total}"},
    "jobs.progress_percent": {"ja": "{percent}%", "en": "{percent}%"},
    "jobs.remaining_seconds": {"ja": "あと約 {seconds} 秒", "en": "About {seconds} s left"},
}
