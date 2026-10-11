#!/bin/sh
# Blender の中でテストを動かす（Mac・Linux）。
#
# 使い方:
#   tools/run_tests.sh                       # 全部のテスト
#   tools/run_tests.sh tests.unit.test_bones # 選んだテストだけ（モジュール名かファイルの場所）
#
# Blender の場所：環境変数 BLENDER → /Applications/Blender.app → PATH の blender の順に探す。
# 紅魔館の本物の書き出しを使うテストは、環境変数 BLENDER_AI_TOOLS_KOUMAKAN_REPO に
# 紅魔館のリポジトリの場所を入れると動く（無いときは理由を出して飛ばす）。
set -eu
cd "$(dirname "$0")/.."
if [ -n "${BLENDER:-}" ]; then
  blender="$BLENDER"
elif [ -x /Applications/Blender.app/Contents/MacOS/Blender ]; then
  blender=/Applications/Blender.app/Contents/MacOS/Blender
elif command -v blender >/dev/null 2>&1; then
  blender=blender
else
  echo "Blender が見つかりません。環境変数 BLENDER に Blender の場所を入れてください" >&2
  exit 2
fi
if [ "$#" -gt 0 ]; then
  exec "$blender" --background --factory-startup --python-exit-code 1 --python tests/run_in_blender.py -- "$@"
fi
exec "$blender" --background --factory-startup --python-exit-code 1 --python tests/run_in_blender.py
