"""画面と報告の文（texts）の決まりを確かめる"""

import unittest

from blender_ai_tools import texts


class TextsTest(unittest.TestCase):
    def tearDown(self):
        texts.set_language("ja")

    def test_all_texts_are_complete(self):
        """どの文にも日本語と英語があり、入れる名前がそろっている"""
        self.assertEqual(texts.problems(), [])

    def test_keys_start_with_their_area(self):
        for key in texts.TEXTS:
            with self.subTest(key=key):
                self.assertEqual(len(key.split(".")), 2)

    def test_placeholders(self):
        self.assertEqual(texts.placeholders("体の穴が {count} 個（{name}）"), {"count", "name"})
        self.assertEqual(texts.placeholders("なし"), set())

    def test_language_switch_and_values(self):
        fake = {"test.hello": {"ja": "こんにちは {name}", "en": "Hello {name}"}}
        original = texts.TEXTS
        texts.TEXTS = {**original, **{k: dict(v) for k, v in fake.items()}}
        try:
            self.assertEqual(texts.text("test.hello", name="A"), "こんにちは A")
            texts.set_language("en")
            self.assertEqual(texts.text("test.hello", name="A"), "Hello A")
            self.assertEqual(texts.text("test.hello", language="ja", name="B"), "こんにちは B")
            with self.assertRaises(ValueError):
                texts.text("test.hello")  # 値が足りない
            with self.assertRaises(ValueError):
                texts.text("test.hello", name="A", extra=1)  # 値が多い
            with self.assertRaises(KeyError):
                texts.text("test.nothing")
        finally:
            texts.TEXTS = original

    def test_bad_language(self):
        with self.assertRaises(ValueError):
            texts.set_language("fr")


if __name__ == "__main__":
    unittest.main()
