"""画面の文と報告の文（日本語が元。英語にも切り替えられる）。

文の形は、この texts パッケージだけに置く（CLAUDE.md 7章「文の形は1か所にまとめます」）。
分野ごとに1つのファイル（screen.py：画面・versions.py・motion.py など）に TEXTS = {キー: {"ja": ..., "en": ...}} を書き、
ここでまとめる。ファイルを足したら、下の _MODULES にも足す（文字列からファイルを探すことはしない）。
"""

from . import motion, screen, versions

_MODULES = (screen, versions, motion)
