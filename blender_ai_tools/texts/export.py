"""書き出し（E1）と往復の確認（E2）の文。キー → {"ja": 日本語, "en": 英語}。数字などは {name} の形で入れる"""

TEXTS: dict[str, dict[str, str]] = {
    "export.step_convert": {"ja": "convert_blend.py で書き出しています", "en": "Exporting with convert_blend.py"},
    "export.step_roundtrip": {"ja": "読み戻して、元の形と比べています", "en": "Reading back and comparing with the source"},
    "export.step_done": {"ja": "書き出しと確かめが終わりました", "en": "Export and check finished"},
    "export.convert_failed": {
        "ja": "書き出しに失敗したので、読み戻して比べられません",
        "en": "The export failed, so nothing could be read back and compared",
    },
}
