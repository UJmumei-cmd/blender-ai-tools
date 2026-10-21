"""A10 重い作業を裏で動かす係（順番待ち・進み具合・中止）。

重い作業（E1 書き出しと E2 の確認、A5 の全部のポーズ、見え方の多くの向き、あとで M6 の焼き込み）は、
別の Blender（--background）で動かす。画面の Blender は固まらず、進み具合（何％・残り時間）が見える。
途中で止めても、元のファイルは壊れない（裏の Blender は、この道具が保存した版の写しを開くだけで、保存しない。
終わったあとに、入力のファイルが変わっていないことも確かめる）。

部品：
- spec.py       仕事の形（JobSpec：目録の道具の id と値、開く .blend、確認の印）と job.json
- progress.py   進み具合（progress.json）、中止の頼み（cancel ファイル）、道具の中から呼ぶ report()
- worker.py     裏の係の中身（bpy なし）：仕事を1つ動かし、progress.json と result.json を書く
- worker_main_bpy.py  裏の Blender の入口（blender --background --python で動かす）
- queue.py      画面の側：順番待ち、1つずつ始める、進み具合を読む、中止、片付け（bpy なし）
- launcher_bpy.py     画面の側の bpy の部分：Blender の場所、設定、タイマーで進み具合を読む

決まり：動くのは目録（A8）に載った道具だけ。裏の係も catalog.call() を通して動かす（CLAUDE.md 2章1）。
"""
