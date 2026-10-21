#!/bin/sh
# 拡張機能の zip を dist/ に作る（Mac・Linux）。できた zip は Blender の
# 「編集 → プリファレンス → 機能拡張 → ディスクからインストール」で入れる。
# Blender の場所の探し方は tools/run_tests.sh と同じ。
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
mkdir -p dist
"$blender" --command extension validate blender_ai_tools
exec "$blender" --command extension build --source-dir blender_ai_tools --output-dir dist
