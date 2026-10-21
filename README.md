# AI の道具（blender-ai-tools）

人型のモデル（MMD 準標準の骨）の服の補正、動きでの確かめ、アプリ（紅魔館）への書き出しを扱う、Blender 5.2 の拡張機能です。
計算はすべてプログラムで行い、LLM や Jeff が無くても、日本語のボタンだけで動くように作っています。

- 計画：[`docs/roadmap.md`](docs/roadmap.md)（機能の番号 A1・E1 などは、ここの番号）
- いまの進み具合と次にやること：[`docs/handoff.md`](docs/handoff.md)（作業を始める前に読む）
- 作業の決まり（Claude 向け）：[`CLAUDE.md`](CLAUDE.md)

## いまできること（段階1の途中。ボタンだけで動く。LLM と Jeff は段階2）

| 番号 | 中身 |
|---|---|
| A3 | 骨の名前の辞書（MMD 準標準 96 本。VRM・VRoid・Mixamo・mmd_tools の名前を対応させる。揺れ物の骨は聞かない） |
| A8 | 道具の目録（LLM の返事を厳しく読み、載っている道具だけを動かす） |
| A15 | 版と再現性の記録（入力のハッシュ、convert_blend.py のコミット、環境、乱数の種） |
| A4 | 測る計算（めり込み・すき間・左右・面の点検・見える見えない・形のくずれ・ウェイト・関節と足） |
| A1 | モデルの登録カード（身長・脚の長さ・腕の下がり角・関節の位置、形と骨の指紋、部品と材質の種類） |
| E1 | 書き出し（紅魔館の `convert_blend.py` を別の Blender で動かす。アプリのフォルダには書かない） |
| E2 | 往復の確認（USDZ を読み戻して、いくつかのポーズで形のずれ・骨の名前・1点4本までを確かめる） |
| A5 | 確かめの動き（約 40 のポーズと、ポーズごとの確かめ） |
| A6 | 版の管理（`_v001` の保存、前の版との違いの一覧） |
| A10 | 重い作業（書き出し・確かめの動き）を別の Blender で裏で動かす。順番待ち・進み具合・中止 |
| A11 | 信号の報告（緑・黄・赤・灰の一覧、直す・このままでよい・あとで の記録、赤の数の基準、HTML） |
| 画面 | 3D ビューの右の欄「AI の道具」：登録カード、骨、版の保存と比べ、書き出して確かめる、確かめの動き、裏の仕事、報告 |

## 入れ方（Mac）

1. Blender 5.2 を `/Applications/Blender.app` に入れる。
2. このリポジトリで `tools/build_extension.sh` を動かす → `dist/blender_ai_tools-<版>.zip` ができる。
3. Blender の「編集 → プリファレンス → 機能拡張」の右上のメニューから「ディスクからインストール」で zip を選ぶ。
4. プリファレンスの「AI の道具」で、紅魔館のリポジトリの場所などを設定する。
5. 3D ビューで N キーを押し、右の欄の「AI の道具」を開く。

## 手で試す（画面のボタン）

1. 試し用のモデルを作る：`tools/make_sample_model.sh`（`samples/sample_model.blend` ができる。コミットしない）。
2. Blender でそのファイルを開き、右の欄「AI の道具」で、登録する → 骨の名前を調べる → 版を保存する → 確かめの動き → 書き出して確かめる（紅魔館の場所を設定したとき）の順に押す。
3. 「裏の仕事」に進み具合が出て、終わると「報告」に結果が出る。作業フォルダ（初めは .blend の隣の `<名前>_ai_tools`）に、仕事・書き出し・報告の HTML・答えの記録ができる。

くわしい手順と、確かめること：[`docs/handoff.md`](docs/handoff.md) の「7. デスクトップでの始め方」。

## Windows・Linux

- スクリプト（`tools/*.sh`）は Mac と Linux 用です。Windows では、Blender のコマンドをそのまま使います（PowerShell の例）：

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --factory-startup --python-exit-code 1 --python tests/run_in_blender.py
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --command extension build --source-dir blender_ai_tools --output-dir dist
& "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --factory-startup --python tests/fixtures/mmd_model.py -- samples/sample_model.blend --four-influences
```

- Windows では、まだ一度も試していません（中止の扱い・ファイルの置き換えに Windows 用の道がありますが、確かめていない）。

## テスト

```sh
tools/run_tests.sh                          # 全部（Blender の中で動かす）
tools/run_tests.sh tests.unit.test_bones    # 選んだものだけ
BLENDER_AI_TOOLS_KOUMAKAN_REPO=~/path/to/Koumakan tools/run_tests.sh   # 本物の書き出しも使う
```

- テストは Blender の中の Python で動かします（numpy と USD が入っているため）。
- 紅魔館の場所を渡さないと、本物の書き出しを使うテストは理由を出して飛ばします。
- コミットの前には、全部のテストを通します。

## ライセンス

GPL-3.0-or-later
