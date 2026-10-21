#!/bin/sh
# 画面のボタンを手で試すための、試し用の人型モデル（MMD 準標準の骨・体・スカート）を .blend に保存する（Mac・Linux）。
#
# 使い方:
#   tools/make_sample_model.sh                    # samples/sample_model.blend に保存する
#   tools/make_sample_model.sh ~/work/model.blend # 場所を決めて保存する
#
# .blend はコミットしない（.gitignore で外してある）。Blender の場所の探し方は tools/run_tests.sh と同じ。
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
output="${1:-samples/sample_model.blend}"
exec "$blender" --background --factory-startup --python tests/fixtures/mmd_model.py -- "$output" --four-influences
