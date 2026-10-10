"""E1 書き出し：紅魔館の convert_blend.py を、別の Blender のプロセスで動かす。

この道具は、自分では書き出さない。アプリ用の書き出しは、紅魔館の convert_blend.py の1本だけ（CLAUDE.md 2章6・4章）。
ここでは、決まった形のコマンドで別の Blender（--background）を動かし、出てきた物を確かめて、記録（A15）を作る。

    <Blender> --background --factory-startup --disable-autoexec --python-exit-code 1 \\
        --python <紅魔館>/tools/model-converter/convert_blend.py -- <入力.blend> <出力フォルダ>

コマンドの各部の理由：
- --factory-startup：ユーザーの設定・アドオンを読まない（どの Mac でも同じ書き出しになるように）。
- --disable-autoexec：.blend に入っているスクリプトを動かさない（安全な形のドライバーの式は、これでも計算される）。
  --factory-startup の既定でも止まっているので、書き出しの結果は変わらない。はっきりさせるために付ける。
- --python-exit-code 1：convert_blend.py が例外で止まったとき、終了コードを 1 にする（付けないと 0 のまま終わる）。

守ること：
- 出力フォルダは必ず指定する。指定しないと、convert_blend.py はアプリの Model フォルダに直接書くため。
- 出力フォルダが、アプリの Model フォルダか、convert_blend.py のあるフォルダ（tools/model-converter）か、
  そのどちらかの中なら断る（ValueError）。シンボリックリンク・".."・大文字と小文字の違い（Mac のディスクは
  大文字と小文字を区別しない）でそこへ着く場所も断る。断るのはこの2つだけ。紅魔館のリポジトリのほかの場所は断らない。
- 出力フォルダは、まだ無いか、空のフォルダにする（前の書き出しの残りを、今回の物とまちがえないため）。
  無ければ作る。入力のファイルがあるフォルダは空ではないので、そこにも書かない。
- 入力の .blend には触らない。前後の sha256 を比べ、変わっていたら失敗にする（元のファイルを上書きしない。2章4）。
- アプリの Model フォルダの中身も前後で比べ、変わっていたら失敗にする（9章：ユーザーの確認なしに書かない）。
- アプリのフォルダへの置き換えは、ここではしない（ユーザーの確認が要る。あとの段階で作る）。

信号（ConvertResult.signal）：書き出せなかった・出てきた物の形が違う・入力やアプリのフォルダが変わったときは赤。
書き出せても、知っておくことがあるときは黄（報告の文が読めない、アプリのヒューマノイドの骨が足りない、
convert_blend.py が「反映されない」などと知らせた、ドライバーの式が計算できなかった など）。どれも無いときだけ緑。

失敗しても、出力フォルダに途中までの物が残ることがある。次の書き出しは、別の（空の）フォルダにする。

このモジュールは bpy を使わない。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import provenance, thresholds
from ..measure.results import Signal
from ..provenance import FileDigest, GitState, RunRecord

__all__ = [
    "APP_MODEL_DIR",
    "CONVERTER_DIR",
    "CONVERT_BLEND_NAME",
    "CONVERT_PMX_NAME",
    "RIG_NAME",
    "RIG_VERSION",
    "SUMMARY_KEYS",
    "TOOL_ID",
    "TOOL_VERSION",
    "USDZ_NAME",
    "ConvertResult",
    "KoumakanPaths",
    "build_command",
    "koumakan_paths",
    "parse_summary",
    "run_convert",
]

# MARK: - 決まり

# 道具の id と版（目録 A8 に載せるときも、この id を使う）
TOOL_ID = "export.convert"
TOOL_VERSION = "1"

# 紅魔館のリポジトリの中の場所（CLAUDE.md 4章）
CONVERTER_DIR = ("tools", "model-converter")
CONVERT_BLEND_NAME = "convert_blend.py"
CONVERT_PMX_NAME = "convert_pmx.py"
APP_MODEL_DIR = ("KoumakanApp", "KoumakanApp", "Model")

# convert_blend.py が出力フォルダに書く物
USDZ_NAME = "character.usdz"
RIG_NAME = "character_rig.json"
# この道具が読める character_rig.json の版
RIG_VERSION = 1

# USDZ の中に、少なくとも1つは入っているはずの USD のファイルの拡張子
_USD_SUFFIXES = (".usdc", ".usda", ".usd")
# アプリの Model フォルダの中身を比べるときに見る、ファイルとフォルダの数の上限。超えたら「比べられなかった」にする
_MAX_FOLDER_ENTRIES = 10000
# ファイルを読むときの1回の大きさ（1 MiB）
_CHUNK_SIZE = 1024 * 1024
# 問題の文に入れる、convert_blend.py のエラーの文の長さの上限（文字）
_MAX_MESSAGE_LENGTH = 300
# Blender が、ドライバーの式を計算できなかったときに出す文
_PY_DRIVER_ERROR = "Error in PyDriver"
# Blender が、--python のスクリプトが例外で止まったときに、例外の文のあとに足す文
_BLENDER_SCRIPT_FAILED = "Error: script failed"


# MARK: - 紅魔館の場所


@dataclass(frozen=True)
class KoumakanPaths:
    """紅魔館のリポジトリの中の、書き出しに使う場所（どれも絶対パスで、シンボリックリンクをたどった物）。

    repo: リポジトリのいちばん上のフォルダ
    converter_dir: tools/model-converter（convert_blend.py のあるフォルダ。ここには書かない）
    convert_blend: convert_blend.py
    convert_pmx: convert_pmx.py（convert_blend.py が読み込む）
    app_model_dir: アプリの Model フォルダ（KoumakanApp/KoumakanApp/Model）。無いこともある。ここには書かない
    """

    repo: Path
    converter_dir: Path
    convert_blend: Path
    convert_pmx: Path
    app_model_dir: Path


def _path_argument(value: Any, what: str) -> Path:
    """文字・Path を受け取る。None・空の文字・ほかの型は ValueError（空の文字は "." といまのフォルダになってしまうため）"""
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
        raise ValueError(f"{what}を指定してください（いまは {value!r}）")
    return Path(value)


def koumakan_paths(repo: str | os.PathLike[str]) -> KoumakanPaths:
    """紅魔館のリポジトリの場所から、書き出しに使う場所を作る。

    repo/tools/model-converter/convert_blend.py と convert_pmx.py が無いときは ValueError。
    アプリの Model フォルダは無くてもよい（そこには書かないので）。
    """
    root = _path_argument(repo, "紅魔館のリポジトリの場所").resolve()
    if not root.is_dir():
        raise ValueError(f"紅魔館のリポジトリのフォルダがありません：{root}")
    converter_dir = root.joinpath(*CONVERTER_DIR).resolve()
    missing = [name for name in (CONVERT_BLEND_NAME, CONVERT_PMX_NAME) if not (converter_dir / name).is_file()]
    if missing:
        raise ValueError(f"紅魔館のリポジトリに {'・'.join(missing)} がありません：{converter_dir}"
                         "（拡張機能の設定の、紅魔館の場所を確かめてください）")
    return KoumakanPaths(
        repo=root,
        converter_dir=converter_dir,
        convert_blend=(converter_dir / CONVERT_BLEND_NAME).resolve(),
        convert_pmx=(converter_dir / CONVERT_PMX_NAME).resolve(),
        app_model_dir=root.joinpath(*APP_MODEL_DIR).resolve(),
    )


# MARK: - 入力と出力の場所を確かめる


def _check_input(input_blend: Any) -> Path:
    """入力の .blend を確かめ、絶対パスにする"""
    path = _path_argument(input_blend, "入力の .blend ").resolve()
    if path.suffix.lower() != ".blend":
        raise ValueError(f"入力は .blend のファイルにしてください：{path}")
    if not path.is_file():
        raise ValueError(f"入力の .blend がありません（またはファイルではありません）：{path}")
    return path


def _folded(path: Path) -> Path:
    """大文字・小文字と、Unicode の書き方（濁点を分けて書くか）の違いを消した場所。Mac のディスクは、これらを区別しない"""
    return Path(unicodedata.normalize("NFC", str(path)).casefold())


def _is_inside(path: Path, folder: Path) -> bool:
    """path が folder そのものか、その中か。どちらも絶対パスで、シンボリックリンクと ".." をたどった物を渡す。

    大文字・小文字などの違いだけの場所も「同じ」とみなす（Linux では断りすぎになるが、安全なほうにする）。
    folder がすでにあるときは、path のすでにある部分を実物（samefile）でも比べる。
    """
    if path.is_relative_to(folder) or _folded(path).is_relative_to(_folded(folder)):
        return True
    if not folder.exists():
        return False
    for candidate in (path, *path.parents):
        try:
            if candidate.exists() and os.path.samefile(candidate, folder):
                return True
        except OSError:
            continue
    return False


def _refuse_protected(folder: Path, paths: KoumakanPaths) -> None:
    """書いてはいけない場所なら ValueError"""
    if _is_inside(folder, paths.app_model_dir):
        raise ValueError(f"出力フォルダが、アプリの Model フォルダ（またはその中）です：{folder}。"
                         "まず作業用のフォルダに書き出して確かめます。アプリのフォルダの置き換えは、確認のあとで別に行います")
    if _is_inside(folder, paths.converter_dir):
        raise ValueError(f"出力フォルダが、convert_blend.py のあるフォルダ（またはその中）です：{folder}。"
                         "紅魔館の書き出しのスクリプトの横には書きません")


def _prepare_output_dir(output_dir: Any, paths: KoumakanPaths) -> Path:
    """出力フォルダを確かめ、無ければ作る。シンボリックリンクと ".." をたどった絶対パスを返す"""
    if output_dir is None or (isinstance(output_dir, (str, os.PathLike)) and not os.fspath(output_dir)):
        raise ValueError("出力フォルダを必ず指定してください（指定しないと、convert_blend.py はアプリの Model フォルダに"
                         "直接書きます）")
    folder = _path_argument(output_dir, "出力フォルダ").resolve()
    _refuse_protected(folder, paths)
    if folder.exists():
        if not folder.is_dir():
            raise ValueError(f"出力フォルダの場所に、フォルダではない物があります：{folder}")
        if any(folder.iterdir()):
            raise ValueError(f"出力フォルダが空ではありません：{folder}。前の書き出しの残りとまちがえないように、"
                             "まだ無いフォルダか、空のフォルダを指定してください")
    else:
        try:
            folder.mkdir(parents=True)
        except OSError as error:
            raise ValueError(f"出力フォルダを作れませんでした：{folder}（{error}）") from error
    # 作ったあとに、もう一度確かめる（途中でリンクを差し替えられても、守る場所には書かない）
    checked = folder.resolve()
    _refuse_protected(checked, paths)
    return checked


def _resolve_blender(blender: Any) -> str:
    """Blender の場所。フォルダの区切りを含むときはそのファイル、含まないときは PATH から探す"""
    text = os.fspath(_path_argument(blender, "Blender の場所"))
    separators = [os.sep] + ([os.altsep] if os.altsep else [])
    if any(separator in text for separator in separators):
        path = os.path.abspath(text)
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            raise ValueError(f"Blender が見つからないか、動かせるファイルではありません：{path}")
        return path
    found = shutil.which(text)
    if found is None:
        raise ValueError(f"Blender が見つかりません：{text}（拡張機能の設定に、Blender の場所を書いてください）")
    return os.path.abspath(found)


def _digest_before_run(path: Path, what: str) -> FileDigest:
    """動かす前の指紋。読めないとき（PermissionError など）も ValueError にする"""
    try:
        return provenance.file_digest(path)
    except (ValueError, OSError) as error:
        raise ValueError(f"{what} が読めません：{path}（{error}）") from error


def _check_timeout(timeout: Any) -> float:
    """待つ時間の上限（秒）。正の有限の数だけ"""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"待つ時間の上限（秒）は正の数にしてください（いまは {timeout!r}）")
    return float(timeout)


# MARK: - コマンド


def build_command(blender: str | os.PathLike[str], convert_blend: str | os.PathLike[str],
                  input_blend: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> list[str]:
    """convert_blend.py を動かすコマンド（shell を通さず、このリストのまま動かす）。

    "--" より後ろは Blender が読まず、convert_blend.py に渡る。--python-exit-code は --python より前に置く
    （Blender は引数を前から順に処理するため）。
    """
    return [os.fspath(blender), "--background", "--factory-startup", "--disable-autoexec",
            "--python-exit-code", "1", "--python", os.fspath(convert_blend),
            "--", os.fspath(input_blend), os.fspath(output_dir)]


def _decode(output: bytes | str | None) -> str:
    """プロセスの出力を文字にする（UTF-8。読めないバイトは置き換える）"""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return output.decode("utf-8", errors="replace")


def _run(command: list[str], timeout: float, cwd: Path) -> tuple[int | None, bool, str, str, str | None]:
    """コマンドを動かす。（終了コード、時間切れか、stdout、stderr、起動できなかった理由）を返す"""
    try:
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
                                   cwd=cwd, check=False)
    except subprocess.TimeoutExpired as error:
        # 時間切れのときは、プロセスを止めたうえで、それまでの出力を受け取る
        return None, True, _decode(error.stdout), _decode(error.stderr), None
    except OSError as error:
        return None, False, "", "", str(error)
    return completed.returncode, False, _decode(completed.stdout), _decode(completed.stderr), None


# MARK: - convert_blend.py の報告の文を読む

# 報告の項目（parse_summary が返す辞書のキー）。読めなかった項目は None
#   armature_name・armature_bone_count・mesh_count：「骨格: …（骨 N）、メッシュ M 個」
#   reported_output_dir：「書き出し完了: …」の場所
#   bone_count・humanoid_count・missing_humanoid：「ボーン: N（ヒューマノイド H、見つからない: […]）」
#   preset_count・kept_shape_key_count・masked_vertices_removed・influence_limit・vertices_limited：
#       「表情プリセット: P 個、残したシェイプキー: K 個、服で隠れる頂点を R 個消した、骨の影響をL本までに絞った頂点 V 個」
#   unmapped_emotions：「アプリの表情で対応が無いもの: […]」
#   transparency：「透過: {材質: 種類}」
#   rest_arm_down_deg：「レスト時の腕の下がり角: A 度」（convert_blend.py が None と書いたときも None）
# ほかに、notices（そのほかの、字下げした知らせの文）と unread（見つからなかった文の名前）を入れる。
SUMMARY_KEYS = (
    "armature_name", "armature_bone_count", "mesh_count", "reported_output_dir",
    "bone_count", "humanoid_count", "missing_humanoid",
    "preset_count", "kept_shape_key_count", "masked_vertices_removed", "influence_limit", "vertices_limited",
    "unmapped_emotions", "transparency", "rest_arm_down_deg",
)

# 報告の文（convert_blend.py の print の形。前後の空白は除いてから比べる）。名前 → 形。
# 数字は ASCII の 0〜9 だけを読む（re.ASCII。全角の数字などを数として読まない）
_SUMMARY_LINES = {
    "骨格": re.compile(r"骨格: (?P<armature_name>.+)（骨 (?P<armature_bone_count>\d+)）、"
                     r"メッシュ (?P<mesh_count>\d+) 個", re.ASCII),
    "書き出し完了": re.compile(r"書き出し完了: (?P<reported_output_dir>.+)", re.ASCII),
    "ボーン": re.compile(r"ボーン: (?P<bone_count>\d+)（ヒューマノイド (?P<humanoid_count>\d+)、"
                      r"見つからない: (?P<missing_humanoid>.*)）", re.ASCII),
    "表情プリセット": re.compile(r"表情プリセット: (?P<preset_count>\d+) 個、残したシェイプキー: (?P<kept_shape_key_count>\d+) 個、"
                          r"服で隠れる頂点を (?P<masked_vertices_removed>\d+) 個消した、"
                          r"骨の影響を(?P<influence_limit>\d+)本までに絞った頂点 (?P<vertices_limited>\d+) 個", re.ASCII),
    "アプリの表情": re.compile(r"アプリの表情で対応が無いもの: (?P<unmapped_emotions>.*)", re.ASCII),
    "透過": re.compile(r"透過: (?P<transparency>.*)", re.ASCII),
    "腕の下がり角": re.compile(r"レスト時の腕の下がり角: (?P<rest_arm_down_deg>\S+) 度", re.ASCII),
}
_INT_KEYS = frozenset({
    "armature_bone_count", "mesh_count", "bone_count", "humanoid_count", "preset_count", "kept_shape_key_count",
    "masked_vertices_removed", "influence_limit", "vertices_limited",
})
_LIST_KEYS = frozenset({"missing_humanoid", "unmapped_emotions"})
# Python の文字の書き方（'…'。中に ' と \ が無い物だけ読む）
_QUOTED = r"'[^'\\]*'"
_STR_LIST = re.compile(rf"\[(?:{_QUOTED}(?:, {_QUOTED})*)?\]")
_STR_DICT = re.compile(rf"\{{(?:{_QUOTED}: {_QUOTED}(?:, {_QUOTED}: {_QUOTED})*)?\}}")
_QUOTED_ITEM = re.compile(r"'([^'\\]*)'")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?", re.ASCII)
# convert_blend.py の知らせのうち、書き出した物が Blender で見る物と違う、とは言っていない物（知らせとして残すだけ）。
# ほかの知らせ（「モディファイア … は書き出しに反映されない場合がある」「骨で選ぶ MASK は焼き込めないので外す」、
# 知らない文）は、緑にしない
_INFORMATIONAL_NOTICES = (
    re.compile(r"Face_Params がないので、表情のアクションは読み取らない"),
    re.compile(r"動きのある表情は使わない: .*"),
)
# Blender が最初に出す版の文。例 "Blender 5.2.2 LTS (hash d13f752e3b9c built 2026-09-15 01:34:58)"
_BLENDER_LINE = re.compile(r"Blender (?P<version>\d[^()]*?) \(hash (?P<hash>[0-9a-f]+) built [^()]*\)", re.ASCII)


def _parse_str_list(text: str) -> list[str] | None:
    """"なし" → []、"['a', 'b']" → ["a", "b"]。それ以外の形は None（分からない）"""
    if text == "なし":
        return []
    if not _STR_LIST.fullmatch(text):
        return None
    return _QUOTED_ITEM.findall(text)


def _parse_str_dict(text: str) -> dict[str, str] | None:
    """"{'m00': 'OPAQUE'}" → {"m00": "OPAQUE"}。それ以外の形は None（分からない）"""
    if not _STR_DICT.fullmatch(text):
        return None
    items = _QUOTED_ITEM.findall(text)
    return dict(zip(items[0::2], items[1::2], strict=True))


def _parse_angle(text: str) -> tuple[float | None, bool]:
    """腕の下がり角。（値、読めたか）を返す。"None" は「読めたが値が無い」"""
    if text == "None":
        return None, True
    if _NUMBER.fullmatch(text):
        return float(text), True
    return None, False


def parse_summary(stdout: str) -> dict[str, Any]:
    """convert_blend.py の stdout から、書き出しの報告を読む。

    SUMMARY_KEYS の項目と、notices・unread を返す。文が無い・形が違う項目は None にし、その文の名前を
    unread に入れる（convert_blend.py の版が変わって文の形が変わったときに、気づけるように）。
    数を推測で埋めることはしない。
    """
    if not isinstance(stdout, str):
        raise ValueError(f"stdout は文字で渡してください（いまは {type(stdout).__name__}）")
    summary: dict[str, Any] = dict.fromkeys(SUMMARY_KEYS)
    read: set[str] = set()
    notices: list[str] = []
    # 行は print の改行（\n）だけで分ける（splitlines は U+2028 などでも分け、場所の文字を切ってしまう）。
    # 前の字下げだけを除き、後ろの空白は残す（「書き出し完了: 」の場所は、フォルダの名前の終わりの空白も含む）
    for raw_line in stdout.split("\n"):
        line = raw_line.removesuffix("\r").lstrip()
        if not line:
            continue
        for label, pattern in _SUMMARY_LINES.items():
            match = pattern.fullmatch(line)
            if match is None:
                continue
            if label not in read and _store(summary, match.groupdict()):
                read.add(label)
            break
        else:
            # convert_blend.py の知らせは、2文字の字下げで書かれる（Blender の文は字下げしない）
            if raw_line.startswith("  "):
                notices.append(line)
    summary["notices"] = notices
    summary["unread"] = [label for label in _SUMMARY_LINES if label not in read]
    return summary


def _store(summary: dict[str, Any], groups: dict[str, str]) -> bool:
    """1つの文から読んだ値を入れる。1つでも読めない値があれば、その文の値は入れずに False"""
    values: dict[str, Any] = {}
    for key, text in groups.items():
        if key in _INT_KEYS:
            values[key] = int(text)
        elif key in _LIST_KEYS:
            values[key] = _parse_str_list(text)
            if values[key] is None:
                return False
        elif key == "transparency":
            values[key] = _parse_str_dict(text)
            if values[key] is None:
                return False
        elif key == "rest_arm_down_deg":
            values[key], ok = _parse_angle(text)
            if not ok:
                return False
        else:
            values[key] = text
    summary.update(values)
    return True


def _blender_version(stdout: str) -> tuple[str | None, str | None]:
    """書き出しに使った Blender の版とビルドのハッシュ（Blender が stdout に出す文から）。分からないときは None"""
    for raw_line in stdout.splitlines():
        match = _BLENDER_LINE.fullmatch(raw_line.strip())
        if match:
            return match["version"], match["hash"]
    return None, None


# MARK: - 出てきた物を確かめる


def _digest_or_none(path: Path, problems: list[str]) -> FileDigest | None:
    try:
        return provenance.file_digest(path)
    except (ValueError, OSError) as error:
        problems.append(f"{path.name} が読めません：{error}")
        return None


def _usdz_contents(path: Path, problems: list[str]) -> dict[str, str] | None:
    """USDZ（zip）の中のファイルごとの sha256。zip の時刻が違っても、中身が同じなら同じになる（再現性の比べに使う）"""
    if not zipfile.is_zipfile(path):
        problems.append(f"{USDZ_NAME} が USDZ（zip）の形ではありません")
        return None
    contents: dict[str, str] = {}
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                digest = hashlib.sha256()
                with archive.open(info) as stream:
                    while chunk := stream.read(_CHUNK_SIZE):
                        digest.update(chunk)
                contents[info.filename] = digest.hexdigest()
    except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError, EOFError) as error:
        problems.append(f"{USDZ_NAME} の中が読めません：{error}")
        return None
    if not any(name.lower().endswith(_USD_SUFFIXES) for name in contents):
        problems.append(f"{USDZ_NAME} に USD のファイル（.usdc・.usda・.usd）が入っていません")
    return contents


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"同じキー {key!r} が2回あります")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    """NaN・Infinity は JSON に書けない（Python の json は読めてしまうが、アプリの JSONDecoder は読めない）"""
    raise ValueError(f"{name} は JSON に書けません（アプリが読めません）")


def _is_name_map(value: Any) -> bool:
    """{空でない文字: 空でない文字} の辞書か"""
    return isinstance(value, dict) and all(
        isinstance(key, str) and key and isinstance(item, str) and item for key, item in value.items())


def _rig_problems(data: Any) -> list[str]:
    """character_rig.json の形の問題（この道具と E2 が頼る項目だけを見る。ほかの項目は無くてもよい）"""
    if not isinstance(data, dict):
        return [f"{RIG_NAME} のいちばん外が {{}} の形ではありません"]
    problems: list[str] = []
    version = data.get("version")
    if type(version) is not int or version != RIG_VERSION:
        problems.append(f"{RIG_NAME} の version が {RIG_VERSION} ではありません（いまは {version!r}）")
    bones = data.get("bones")
    if not _is_name_map(bones):
        problems.append(f"{RIG_NAME} の bones が「骨の名前 → 書き出した骨の名前」の {{}} ではありません")
    original_names = data.get("originalNames")
    original_bones = original_names.get("bones") if isinstance(original_names, dict) else None
    if not _is_name_map(original_bones):
        problems.append(f"{RIG_NAME} の originalNames.bones が「書き出した名前 → 元の名前」の {{}} ではありません")
    if _is_name_map(bones) and _is_name_map(original_bones):
        unknown = sorted(name for name in bones.values() if name not in original_bones)
        if unknown:
            problems.append(f"{RIG_NAME} の bones の骨が originalNames.bones にありません：{'、'.join(unknown)}")
    return problems


def _read_rig(path: Path, problems: list[str]) -> dict[str, Any] | None:
    """character_rig.json を読む。形が正しいときだけ中身を返す"""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        problems.append(f"{RIG_NAME} が UTF-8 ではありません")
        return None
    except OSError as error:
        problems.append(f"{RIG_NAME} が読めません：{error}")
        return None
    try:
        data = json.loads(text, object_pairs_hook=_object_without_duplicates, parse_constant=_reject_constant)
    except ValueError as error:
        problems.append(f"{RIG_NAME} が JSON として読めません：{error}")
        return None
    found = _rig_problems(data)
    problems.extend(found)
    return None if found else data


def _folder_state(folder: Path) -> tuple[bool, tuple[tuple[str, int, int, int], ...]] | None:
    """フォルダの中身（あるか、中の物の 場所・種類・大きさ・更新の時刻）。読めない・多すぎるときは None"""
    if not folder.exists():
        return False, ()
    entries: list[tuple[str, int, int, int]] = []
    errors: list[OSError] = []
    for directory, folders, files in os.walk(folder, onerror=errors.append):
        for name in folders + files:
            path = Path(directory) / name
            try:
                status = path.lstat()
            except OSError:
                return None
            entries.append((str(path.relative_to(folder)), status.st_mode, status.st_size, status.st_mtime_ns))
            if len(entries) > _MAX_FOLDER_ENTRIES:
                return None
    if errors:
        return None
    return True, tuple(sorted(entries))


def _last_line(text: str) -> str:
    """いちばん後ろの、空でない行（エラーの文）。Blender が足す決まった文は飛ばす。長いときは切る"""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line and line != "Blender quit" and not line.startswith(_BLENDER_SCRIPT_FAILED):
            return line if len(line) <= _MAX_MESSAGE_LENGTH else line[:_MAX_MESSAGE_LENGTH] + "…"
    return ""


def _record_outputs(usdz: FileDigest | None, usdz_contents: dict[str, str] | None,
                    rig_file: FileDigest | None) -> dict[str, Any]:
    """記録に入れる出力の指紋。同じ入力から2回書き出すと同じになる物だけを入れる（再現性の比べに使う）。

    - 場所は入れない（出力フォルダは settings にある）。
    - USDZ は zip の時刻が毎回変わるので、ファイル全体の sha256 は入れず、中のファイルごとの sha256 を入れる。
      ファイル全体の sha256 は ConvertResult.usdz にある。
    """
    return {
        "usdz": None if usdz is None else {"name": usdz.name, "size": usdz.size, "contents": usdz_contents},
        "rig": None if rig_file is None else {"name": rig_file.name, "sha256": rig_file.sha256, "size": rig_file.size},
    }


# MARK: - 結果


@dataclass(frozen=True)
class ConvertResult:
    """convert_blend.py を1回動かした結果。

    ok: 書き出せて、出てきた物の形が正しく、入力のファイルとアプリの Model フォルダが変わっていないか
    returncode: 終了コード（時間切れ・起動できなかったときは None）
    timed_out: 待つ時間の上限を過ぎて止めたか
    stdout・stderr: Blender の出力（UTF-8 として読んだ物）
    output_dir: 出力フォルダ（絶対パス）
    usdz: character.usdz の指紋（無いときは None）
    rig_path: character_rig.json の場所（無いときは None）
    rig: character_rig.json の中身（形が正しいときだけ。ほかは None）
    summary: convert_blend.py の報告の文から読んだ数（parse_summary）。数はプログラムが読んだ物だけ
    problems: ok でない理由（日本語）
    record: 道具の版と再現性の記録（A15）
    warnings: 書き出せたが、知っておくこと（日本語）。あると信号は黄
    command: 動かしたコマンド
    input_unchanged: 入力の .blend が前後で同じか（読めなかったときは None）
    rig_file: character_rig.json の指紋
    usdz_contents: USDZ の中のファイルごとの sha256（再現性の比べに使う。zip の時刻は入らない）
    """

    ok: bool
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    output_dir: Path
    usdz: FileDigest | None
    rig_path: Path | None
    rig: dict[str, Any] | None
    summary: dict[str, Any]
    problems: list[str]
    record: RunRecord
    warnings: list[str] = field(default_factory=list)
    command: tuple[str, ...] = ()
    input_unchanged: bool | None = None
    rig_file: FileDigest | None = None
    usdz_contents: dict[str, str] | None = None

    @property
    def signal(self) -> Signal:
        """赤：ok でない。黄：ok だが知らせがある。緑：ok で知らせも無い"""
        if not self.ok:
            return Signal.RED
        return Signal.YELLOW if self.warnings else Signal.GREEN


# MARK: - 動かす


def run_convert(input_blend: str | os.PathLike[str], output_dir: str | os.PathLike[str], *,
                blender: str | os.PathLike[str], repo: str | os.PathLike[str],
                timeout: float = thresholds.CONVERT_TIMEOUT_SECONDS) -> ConvertResult:
    """紅魔館の convert_blend.py を別の Blender で動かし、input_blend を output_dir に書き出す。

    input_blend: 入力の .blend（この道具が保存した版のファイルにする。開くだけで、変えない）
    output_dir: 出力フォルダ。必ず指定する。まだ無いか空のフォルダ。アプリの Model フォルダと
        convert_blend.py のあるフォルダ（とその中）は断る
    blender: Blender の場所（例 /Applications/Blender.app/Contents/MacOS/Blender）。区切りの無い名前なら PATH から探す
    repo: 紅魔館のリポジトリの場所
    timeout: 待つ時間の上限（秒）。過ぎたら Blender を止めて、失敗にする

    場所や値がまちがっているとき（動かす前に分かること）は ValueError。動かしたあとの失敗は、
    ConvertResult の ok・problems で返す（例外にしない）。
    """
    paths = koumakan_paths(repo)
    source = _check_input(input_blend)
    blender_path = _resolve_blender(blender)
    limit = _check_timeout(timeout)
    started_at = provenance.now_iso()
    # 動かす前に、入力と convert_blend.py の指紋を取る。読めないときは、出力フォルダを作る前に断る
    source_before = _digest_before_run(source, "入力の .blend")
    converter_files = (_digest_before_run(paths.convert_blend, CONVERT_BLEND_NAME),
                       _digest_before_run(paths.convert_pmx, CONVERT_PMX_NAME))
    folder = _prepare_output_dir(output_dir, paths)
    command = build_command(blender_path, paths.convert_blend, source, folder)

    # Blender が convert_pmx の __pycache__ を作るので、git の状態は動かす前に見る
    git: GitState = provenance.git_state(paths.repo)
    app_before = _folder_state(paths.app_model_dir)

    returncode, timed_out, stdout, stderr, start_error = _run(command, limit, folder)

    problems: list[str] = []
    warnings: list[str] = []
    if start_error is not None:
        problems.append(f"Blender を起動できませんでした：{start_error}")
    elif timed_out:
        problems.append(f"convert_blend.py が {limit:g} 秒で終わらなかったので、止めました")
    elif returncode != 0:
        message = _last_line(stderr) or _last_line(stdout)
        problems.append(f"convert_blend.py が失敗しました（終了コード {returncode}）"
                        + (f"：{message}" if message else ""))

    # 入力のファイルとアプリの Model フォルダが、変わっていないか
    try:
        input_unchanged: bool | None = provenance.file_digest(source).sha256 == source_before.sha256
    except (ValueError, OSError):
        input_unchanged = None
    if input_unchanged is None:
        problems.append(f"書き出しのあとに、入力のファイルが読めません：{source}")
    elif not input_unchanged:
        problems.append(f"書き出しの間に、入力のファイルが変わりました：{source}（元のファイルは変えない決まりです）")
    app_after = _folder_state(paths.app_model_dir)
    app_unchanged: bool | None = None if app_before is None or app_after is None else app_before == app_after
    if app_unchanged is False:
        problems.append(f"書き出しの間に、アプリの Model フォルダが変わりました：{paths.app_model_dir}")
    elif app_unchanged is None:
        warnings.append(f"アプリの Model フォルダが変わっていないかを、確かめられませんでした：{paths.app_model_dir}")

    # 出てきた物
    usdz_path = folder / USDZ_NAME
    usdz: FileDigest | None = None
    usdz_contents: dict[str, str] | None = None
    if usdz_path.is_file():
        usdz = _digest_or_none(usdz_path, problems)
        if usdz is not None:
            usdz_contents = _usdz_contents(usdz_path, problems)
    else:
        problems.append(f"{USDZ_NAME} がありません")
    rig_path = folder / RIG_NAME
    rig: dict[str, Any] | None = None
    rig_file: FileDigest | None = None
    if rig_path.is_file():
        rig_file = _digest_or_none(rig_path, problems)
        rig = _read_rig(rig_path, problems)
    else:
        problems.append(f"{RIG_NAME} がありません")

    summary = parse_summary(stdout)
    reported = summary["reported_output_dir"]
    if reported is not None and Path(reported) != folder:
        problems.append(f"convert_blend.py が書いたと言う場所（{reported}）が、指定した出力フォルダ（{folder}）と違います")
    warnings.extend(_summary_warnings(summary, rig, source, stdout + "\n" + stderr, finished=not problems))

    ok = not problems
    blender_version, blender_hash = _blender_version(stdout)
    record = RunRecord(
        tool_id=TOOL_ID,
        tool_version=TOOL_VERSION,
        started_at=started_at,
        finished_at=provenance.now_iso(),
        addon_version=provenance.addon_version(),
        environment={**provenance.python_environment(), "converter_blender_version": blender_version,
                     "converter_blender_build_hash": blender_hash},
        inputs=(source_before, *converter_files),
        settings={"output_dir": str(folder), "timeout_seconds": limit, "blender": blender_path, "command": command},
        seed=None,
        # 結果の中の辞書・リストは写しにする（ConvertResult の値をあとで変えても、記録が変わらないように）
        results=copy.deepcopy({
            "ok": ok,
            "returncode": returncode,
            "timed_out": timed_out,
            "input_unchanged": input_unchanged,
            "app_model_dir_unchanged": app_unchanged,
            # 書いた場所は出力フォルダごとに違うので入れない（settings.output_dir にある）
            "summary": {key: value for key, value in summary.items() if key != "reported_output_dir"},
            "outputs": _record_outputs(usdz, usdz_contents, rig_file),
            "problems": problems,
            "warnings": warnings,
        }),
        convert_blend=git,
    )
    return ConvertResult(
        ok=ok, returncode=returncode, timed_out=timed_out, stdout=stdout, stderr=stderr, output_dir=folder,
        usdz=usdz, rig_path=rig_path if rig_path.is_file() else None, rig=rig, summary=summary,
        problems=problems, record=record, warnings=warnings, command=tuple(command),
        input_unchanged=input_unchanged, rig_file=rig_file, usdz_contents=usdz_contents,
    )


def _summary_warnings(summary: dict[str, Any], rig: dict[str, Any] | None, source: Path, output: str, *,
                      finished: bool) -> list[str]:
    """書き出せたときの知らせ（報告の文が読めない、基準と違う、ドライバーが計算できなかった など）"""
    warnings: list[str] = []
    if finished and summary["unread"]:
        warnings.append("convert_blend.py の報告の文のうち、読めなかった物があります：" + "、".join(summary["unread"])
                        + "（convert_blend.py の版が変わったのかもしれません）")
    for notice in summary["notices"]:
        if not any(pattern.fullmatch(notice) for pattern in _INFORMATIONAL_NOTICES):
            warnings.append(f"convert_blend.py の知らせ：{notice}（書き出した物が、Blender で見る物と違うかもしれません）")
    limit = summary["influence_limit"]
    if limit is not None and limit != thresholds.MAX_BONE_INFLUENCES:
        warnings.append(f"convert_blend.py は1点に効く骨を {limit} 本までにしています"
                        f"（この道具の基準は {thresholds.MAX_BONE_INFLUENCES} 本）")
    missing = summary["missing_humanoid"]
    if missing:
        # 前提が足りない物は緑にしない（CLAUDE.md 2章5）。足りない骨を足すのは M2（段階1.5）
        warnings.append("アプリのヒューマノイドの骨のうち、モデルに見つからない物があります：" + "、".join(missing)
                        + "（アプリの動きは、この骨に当たりません）")
    humanoid_count = summary["humanoid_count"]
    if rig is not None and humanoid_count is not None and humanoid_count != len(rig["bones"]):
        warnings.append(f"報告のヒューマノイドの骨の数（{humanoid_count}）と、{RIG_NAME} の bones の数"
                        f"（{len(rig['bones'])}）が違います")
    if rig is not None and rig.get("source") != source.name:
        warnings.append(f"{RIG_NAME} の source（{rig.get('source')!r}）が、入力のファイルの名前（{source.name}）と違います")
    driver_errors = output.count(_PY_DRIVER_ERROR)
    if driver_errors:
        warnings.append(f"計算できなかったドライバーの式が {driver_errors} 個あります（.blend のスクリプトは動かさないため）。"
                        "焼き込んだ形が、ふだんの Blender で見る形と違うかもしれません")
    return warnings
