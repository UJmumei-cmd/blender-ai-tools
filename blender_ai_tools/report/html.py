"""HTML の報告。1つのファイルで、外の物（スクリプト・CSS・画像）は読まない。文はすべて texts から。

項目の文（題・理由・メモ）は html.escape で必ず逃がす（モデルのファイル名や材質名が入ることがあるため）。
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

from .. import texts
from ..jobs.spec import write_text_atomic
from ..measure.results import Measurement, Signal
from .model import Report, decision_label, headline, signal_label, sort_items, value_text

# details を HTML に入れる長さの上限（文字）。長い物は入れない（画像や印つきの一覧は A7 で別に作る）
MAX_DETAILS_CHARS = 2000

_STYLE = """
body { font-family: sans-serif; margin: 1.5em; color: #222; background: #fff; }
h1 { font-size: 1.4em; } .headline { font-size: 1.2em; margin: 0.5em 0 1em; padding: 0.6em 0.8em; border-radius: 6px; }
.headline.red { background: #ffe3e3; } .headline.gray { background: #ececec; } .headline.yellow { background: #fff6cc; }
.headline.green { background: #e3f7e3; } .meta { color: #666; font-size: 0.9em; }
table { border-collapse: collapse; width: 100%; } th, td { border: 1px solid #ddd; padding: 0.4em 0.6em; vertical-align: top; }
th { background: #f4f4f4; text-align: left; } td.signal { white-space: nowrap; font-weight: bold; }
tr.red td.signal { color: #b00020; } tr.yellow td.signal { color: #9a6b00; } tr.gray td.signal { color: #666; }
tr.green td.signal { color: #1a7f37; } tr.accepted { opacity: 0.6; }
details { margin-top: 0.3em; } pre { font-size: 0.8em; white-space: pre-wrap; word-break: break-all; margin: 0; }
"""


def _headline_class(report: Report) -> str:
    counted = [item.counted_signal for item in report.items if item.counted_signal is not None]
    if not counted:
        return "green"
    return max(counted).name.lower()


def to_html(report: Report, *, language: str | None = None) -> str:
    """報告を HTML の文字にする"""
    lang = language or texts.current_language()
    rows = []
    for item in sort_items(report.items):
        measurement = Measurement.from_data(item.measurement)
        row_class = "accepted" if item.accepted else measurement.signal.name.lower()
        details_html = ""
        if measurement.details:
            text = json.dumps(measurement.details, ensure_ascii=False, sort_keys=True, allow_nan=False)
            if len(text) <= MAX_DETAILS_CHARS:
                details_html = (f'<details><summary>{html.escape(texts.text("report.details", language=lang))}</summary>'
                                f"<pre>{html.escape(text)}</pre></details>")
        decision = html.escape(decision_label(item.decision, language=lang))
        if item.decision_note:
            decision += "<br>" + html.escape(texts.text("report.decision_note", language=lang, note=item.decision_note))
        rows.append(
            f'<tr class="{row_class}" id="{html.escape(item.item_id, quote=True)}">'
            f"<td>{html.escape(item.group)}</td>"
            f"<td>{html.escape(item.titles.get(lang) or item.titles['ja'])}</td>"
            f'<td class="signal">{html.escape(signal_label(measurement.signal, language=lang))}</td>'
            f"<td>{html.escape(value_text(item.measurement, language=lang))}</td>"
            f"<td>{html.escape(measurement.reason)}{details_html}</td>"
            f"<td>{decision}</td></tr>")
    columns = "".join(f"<th>{html.escape(texts.text(f'report.column_{name}', language=lang))}</th>"
                      for name in ("group", "title", "signal", "value", "reason", "decision"))
    title = texts.text("report.title", language=lang)
    return (
        f'<!DOCTYPE html>\n<html lang="{html.escape(lang)}"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f'<p class="meta">{html.escape(texts.text("report.kind", language=lang, kind=report.kind))} / '
        f'{html.escape(texts.text("report.generated_at", language=lang, time=report.created_at))}</p>'
        f'<div class="headline {_headline_class(report)}">{html.escape(headline(report.items, language=lang))}</div>'
        f"<table><thead><tr>{columns}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "</body></html>\n")


def write_html(report: Report, path: str | os.PathLike[str], *, language: str | None = None) -> Path:
    """HTML の報告をファイルに書く（一時ファイルから置き換える）"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(target, to_html(report, language=language))
    return target
