# AI の道具（blender-ai-tools）

人型のモデル（MMD 準標準の骨）の服の補正、動きでの確かめ、アプリ（紅魔館）への書き出しを扱う、Blender 5.2 の拡張機能です。
計算はすべてプログラムで行い、LLM や Jeff が無くても、日本語のボタンだけで動くように作っています。

- 計画：[`docs/roadmap.md`](docs/roadmap.md)（機能の番号 A1・E1 などは、ここの番号）
- いまの進み具合と次にやること：[`docs/handoff.md`](docs/handoff.md)（作業を始める前に読む）
- 作業の決まり（Claude 向け）：[`CLAUDE.md`](CLAUDE.md)

## いまできること（段階1の途中）

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
| 画面 | 3D ビューの右の欄「AI の道具」（A1・A3 のボタン。書き出しなどの重い作業は A10 のあとでつなぐ） |

## 入れ方（Mac）

1. Blender 5.2 を `/Applications/Blender.app` に入れる。
2. このリポジトリで `tools/build_extension.sh` を動かす → `dist/blender_ai_tools-<版>.zip` ができる。
3. Blender の「編集 → プリファレンス → 機能拡張」の右上のメニューから「ディスクからインストール」で zip を選ぶ。
4. プリファレンスの「AI の道具」で、紅魔館のリポジトリの場所などを設定する。
5. 3D ビューで N キーを押し、右の欄の「AI の道具」を開く。

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
