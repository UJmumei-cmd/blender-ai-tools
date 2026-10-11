"""AI の道具：人型モデルの服の補正・動きでの確かめ・アプリへの書き出しを行う、Blender 5.2 の拡張機能。

この階層では bpy を import しない（Blender の外の Python でもテストできるようにするため）。
bpy を使う部品（ui/：設定・パネル・ボタン）は、register() と unregister() の中で読み込む。
"""


def register() -> None:
    """設定・パネル・ボタンを Blender に登録する（画面の言語も、設定から決める）"""
    from . import ui

    ui.register()


def unregister() -> None:
    """register() で登録した物を全部外す"""
    from . import ui

    ui.unregister()
