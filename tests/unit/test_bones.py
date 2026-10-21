"""A3 骨の名前の対応（辞書）を確かめる"""

import dataclasses
import json
import random
import re
import shutil
import tempfile
import unicodedata
import unittest
from pathlib import Path
from unittest import mock

from blender_ai_tools import bones
from blender_ai_tools.bones import (
    BoneMapping, CanonicalBone, canonical_bone, canonical_bones, canonical_names, chains, is_ambiguous,
    is_known_extra, is_sway_name, lookup, lookup_candidates, map_bones, mirror_name, name_tokens, normalize,
)

# MARK: - 正の一覧（作業の指示の字のまま。全角は \u で書いて、字の幅をはっきりさせる）

FW = {str(i): chr(0xFF10 + i) for i in range(10)}  # 全角の数字
IK_FW = "ＩＫ"  # 全角の ＩＫ

TRUNK = ["全ての親", "センター", "グルーブ", "腰", "上半身", "上半身2", "下半身", "首", "頭"]
TRUNK_STANDARD = {"センター", "上半身", "下半身", "首", "頭"}
ARM_LEFT = ["左肩P", "左肩", "左肩C", "左腕", "左腕捩", "左腕捩1", "左腕捩2", "左腕捩3", "左ひじ", "左手捩",
            "左手捩1", "左手捩2", "左手捩3", "左手首", "左ダミー"]
ARM_STANDARD = {"左肩", "左腕", "左ひじ", "左手首"}
ARM_HELPER = {"左肩C", "左腕捩1", "左腕捩2", "左腕捩3", "左手捩1", "左手捩2", "左手捩3"}
FINGER_LEFT = ["左親指" + FW["0"], "左親指" + FW["1"], "左親指" + FW["2"]] + [
    f"左{finger}{FW[d]}" for finger in ("人指", "中指", "薬指", "小指") for d in "123"]
LEG_LEFT = ["左足", "左ひざ", "左足首", "左つま先", "左足" + IK_FW, "左つま先" + IK_FW, "左足IK親", "腰キャンセル左",
            "左足D", "左ひざD", "左足首D", "左足先EX"]
LEG_STANDARD = {"左足", "左ひざ", "左足首", "左つま先", "左足" + IK_FW, "左つま先" + IK_FW}


def right_of(name: str) -> str:
    return name.replace("左", "右")


def expected_bones() -> dict[str, tuple]:
    """名前 → (side, group, standard, helper)"""
    out = {}
    for name in TRUNK:
        out[name] = (None, "trunk", name in TRUNK_STANDARD, False)
    out["両目"] = (None, "eye", True, False)
    out["左目"] = ("L", "eye", True, False)
    out["右目"] = ("R", "eye", True, False)
    for names, group, standard, helper in (
            (ARM_LEFT, "arm", ARM_STANDARD, ARM_HELPER),
            (FINGER_LEFT, "finger", set(FINGER_LEFT[1:]), set()),
            (LEG_LEFT, "leg", LEG_STANDARD, set())):
        for name in names:
            out[name] = ("L", group, name in standard, name in helper)
            out[right_of(name)] = ("R", group, name in standard, name in helper)
    return out


def all_aliases():
    """data/aliases_*.json の (ファイル, 表, 別名, 共通の名前) を全部"""
    for file_name in bones.ALIAS_FILES:
        data = json.loads((bones.DATA_DIR / file_name).read_text(encoding="utf-8"))
        for table, spec in data["tables"].items():
            for alias, canonical in spec.get("aliases", {}).items():
                yield file_name, table, alias, canonical


def all_known_extras():
    for file_name in bones.ALIAS_FILES:
        data = json.loads((bones.DATA_DIR / file_name).read_text(encoding="utf-8"))
        for table, spec in data["tables"].items():
            for extra in spec.get("known_extras", []):
                yield file_name, table, extra


# 表によって意味が違う名前（比べる形）。ここに無い名前が2つの骨を指したら、辞書のまちがい
AMBIGUOUS_KEYS = {"leftthumbproximal", "rightthumbproximal", "leftleg", "rightleg"}

# MARK: - 字の書きかえ（テスト用）


def to_fullwidth(text: str) -> str:
    """半角の英数字・記号を全角にする"""
    return "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in text)


def to_halfwidth(text: str) -> str:
    """全角の英数字・記号を半角にする"""
    return "".join(chr(ord(c) - 0xFEE0) if "！" <= c <= "～" else c for c in text)


def _halfwidth_kana_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for code in range(0xFF66, 0xFF9E):
        half = chr(code)
        full = unicodedata.normalize("NFKC", half)
        table.setdefault(full, half)
        for mark in ("ﾞ", "ﾟ"):  # 濁点・半濁点
            composed = unicodedata.normalize("NFKC", half + mark)
            if len(composed) == 1 and composed != full:
                table.setdefault(composed, half + mark)
    return table


_HALF_KANA = _halfwidth_kana_table()


def to_halfwidth_kana(text: str) -> str:
    """全角のカタカナを半角カナにする（ｾﾝﾀｰ など）"""
    return "".join(_HALF_KANA.get(c, c) for c in text)


def swap_ascii_case(text: str) -> str:
    return "".join(c.lower() if c.isupper() else c.upper() for c in text)


def lr_form(name: str, separator: str = ".", lower: bool = False) -> str:
    """mmd_tools の左右の書き方（左腕 → 腕.L、腰キャンセル右 → 腰キャンセル.R）"""
    side = "L" if "左" in name else "R"
    base = name.replace("左" if side == "L" else "右", "")
    return base + separator + (side.lower() if lower else side)


# MARK: - 共通の名前の一覧


class CanonicalListTest(unittest.TestCase):
    def test_exact_names_and_flags(self):
        expected = expected_bones()
        self.assertEqual(len(expected), 96)
        self.assertEqual(set(canonical_names()), set(expected))
        self.assertEqual(len(canonical_names()), 96)
        for bone in canonical_bones():
            with self.subTest(bone=bone.name):
                self.assertEqual((bone.side, bone.group, bone.standard, bone.helper), expected[bone.name])

    def test_widths_are_the_reference_forms(self):
        """全角・半角の正の形（表の字そのもの）"""
        names = set(canonical_names())
        self.assertIn("左親指０", names)  # 全角の０
        self.assertNotIn("左親指0", names)
        self.assertIn("左足ＩＫ", names)  # 全角の ＩＫ
        self.assertNotIn("左足IK", names)
        self.assertIn("左つま先ＩＫ", names)
        self.assertIn("左足IK親", names)  # 半角の IK
        self.assertNotIn("左足ＩＫ親", names)
        for name in ("上半身2", "左腕捩1", "左手捩3", "左足D", "左ひざD", "左足首D", "左足先EX", "左肩P", "左肩C"):
            self.assertIn(name, names)
            self.assertTrue(any(c.isascii() and c.isalnum() for c in name))
            self.assertFalse(any("！" <= c <= "～" for c in name), name)

    def test_counts(self):
        bones_ = canonical_bones()
        self.assertEqual(sum(b.standard for b in bones_), 5 + 3 + 2 * (4 + 14 + 6))
        self.assertEqual(sum(b.helper for b in bones_), 14)
        self.assertEqual(sum(b.side is None for b in bones_), 10)
        self.assertEqual(sum(b.side == "L" for b in bones_), 43)
        self.assertEqual(sum(b.side == "R" for b in bones_), 43)
        # 補助の骨は、どれも準標準で足された骨
        self.assertTrue(all(not b.standard for b in bones_ if b.helper))

    def test_canonical_order_starts_with_trunk(self):
        self.assertEqual(canonical_names()[:9], tuple(TRUNK))
        self.assertEqual(canonical_names()[9:12], ("両目", "左目", "右目"))

    def test_cached_and_frozen(self):
        self.assertIs(canonical_bones(), canonical_bones())
        bone = canonical_bones()[0]
        self.assertIsInstance(bone, CanonicalBone)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            bone.name = "x"

    def test_canonical_bone(self):
        self.assertEqual(canonical_bone("左足IK"), CanonicalBone("左足" + IK_FW, "L", "leg", True, False))
        self.assertIsNone(canonical_bone("腕.L"))  # 共通の名前そのものではない
        self.assertIsNone(canonical_bone("髪1"))

    def test_json_notes_mention_width(self):
        data = json.loads((bones.DATA_DIR / bones.CANONICAL_FILE).read_text(encoding="utf-8"))
        self.assertTrue(any("全角" in note and "半角" in note for note in data["_notes"]))
        for file_name in bones.ALIAS_FILES:
            data = json.loads((bones.DATA_DIR / file_name).read_text(encoding="utf-8"))
            self.assertTrue(data["_notes"] and data["_sources"], file_name)


class MirrorTest(unittest.TestCase):
    def test_every_side_bone_has_a_mirror(self):
        by_name = {b.name: b for b in canonical_bones()}
        for bone in canonical_bones():
            with self.subTest(bone=bone.name):
                mirrored = mirror_name(bone.name)
                if bone.side is None:
                    self.assertIsNone(mirrored)
                    continue
                other = by_name[mirrored]
                self.assertEqual(other.side, "R" if bone.side == "L" else "L")
                self.assertEqual((other.group, other.standard, other.helper), (bone.group, bone.standard, bone.helper))
                self.assertEqual(mirror_name(mirrored), bone.name)

    def test_examples(self):
        self.assertEqual(mirror_name("左腕"), "右腕")
        self.assertEqual(mirror_name("右手捩2"), "左手捩2")
        self.assertEqual(mirror_name("腰キャンセル左"), "腰キャンセル右")
        self.assertEqual(mirror_name("左足IK"), "右足" + IK_FW)  # 半角で渡しても、正の形で返す
        self.assertIsNone(mirror_name("両目"))
        self.assertIsNone(mirror_name("全ての親"))

    def test_not_canonical(self):
        for name in ("腕.L", "LeftArm", "髪1", ""):
            with self.assertRaises(ValueError):
                mirror_name(name)


class ChainsTest(unittest.TestCase):
    def test_contents(self):
        c = chains()
        self.assertEqual(c["spine"], ("下半身", "上半身", "上半身2", "首", "頭"))
        self.assertEqual(c["arm_L"], ("左肩", "左腕", "左ひじ", "左手首"))
        self.assertEqual(c["arm_R"], ("右肩", "右腕", "右ひじ", "右手首"))
        self.assertEqual(c["leg_L"], ("左足", "左ひざ", "左足首", "左つま先"))
        self.assertEqual(c["leg_d_R"], ("右足D", "右ひざD", "右足首D", "右足先EX"))
        self.assertEqual(c["thumb_L"], ("左親指０", "左親指１", "左親指２"))
        self.assertEqual(c["little_R"], ("右小指１", "右小指２", "右小指３"))

    def test_all_keys_and_names(self):
        c = chains()
        kinds = ("arm", "leg", "leg_d", "thumb", "index", "middle", "ring", "little")
        expected = {"spine"} | {f"{k}_{s}" for k in kinds for s in "LR"}
        self.assertEqual(set(c), expected)
        names = set(canonical_names())
        for key, chain in c.items():
            with self.subTest(chain=key):
                self.assertTrue(set(chain) <= names)
                self.assertEqual(len(set(chain)), len(chain))
                if key.endswith("_L"):
                    self.assertEqual(c[key[:-2] + "_R"], tuple(mirror_name(n) for n in chain))
                    self.assertTrue(all(n.startswith("左") for n in chain))

    def test_returns_a_copy(self):
        c = chains()
        c["spine"] = ()
        del c["arm_L"]
        self.assertEqual(len(chains()["spine"]), 5)
        self.assertIn("arm_L", chains())


# MARK: - 正規化と、1つの名前を引く


class NormalizeTest(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(normalize("左足ＩＫ"), "左足ik")
        self.assertEqual(normalize("左足IK"), "左足ik")
        self.assertEqual(normalize("左親指０"), "左親指0")
        self.assertEqual(normalize("ｾﾝﾀｰ"), "センター")
        self.assertEqual(normalize("ＬｅｆｔＡｒｍ"), "leftarm")
        self.assertEqual(normalize("mixamorig12:LeftArm"), "leftarm")
        self.assertEqual(normalize("ns：LeftArm"), "leftarm")  # 全角のコロン
        self.assertEqual(normalize(" leg IK_L "), "legikl")
        self.assertEqual(normalize("Leg-IK.l"), "legikl")
        self.assertEqual(normalize("上半身　2"), "上半身2")  # 全角の空白
        self.assertEqual(normalize(""), "")

    def test_type_errors(self):
        for bad in (None, 1, 1.5, True, b"arm", ["左腕"]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    normalize(bad)
                with self.assertRaises(ValueError):
                    lookup(bad)
                with self.assertRaises(ValueError):
                    is_known_extra(bad)


class LookupCanonicalTest(unittest.TestCase):
    def test_every_canonical_maps_to_itself(self):
        for name in canonical_names():
            with self.subTest(name=name):
                self.assertEqual(lookup(name), name)
                self.assertEqual(lookup_candidates(name), (name,))
                self.assertFalse(is_known_extra(name))

    def test_width_and_case_variants(self):
        for name in canonical_names():
            variants = {to_fullwidth(name), to_halfwidth(name), to_halfwidth_kana(name),
                        to_halfwidth_kana(to_fullwidth(name)), swap_ascii_case(name),
                        to_fullwidth(swap_ascii_case(name)), " " + name + " "}
            for variant in variants:
                with self.subTest(name=name, variant=variant):
                    self.assertEqual(lookup(variant), name)

    def test_halfwidth_kana_examples(self):
        """テスト用の書きかえが、本当に半角カナを作っているか"""
        self.assertEqual(to_halfwidth_kana("センター"), "ｾﾝﾀｰ")
        self.assertEqual(to_halfwidth_kana("グルーブ"), "ｸﾞﾙｰﾌﾞ")
        self.assertEqual(lookup("ｸﾞﾙｰﾌﾞ"), "グルーブ")
        self.assertEqual(lookup("左ﾀﾞﾐｰ"), "左ダミー")
        self.assertEqual(lookup("腰ｷｬﾝｾﾙ左"), "腰キャンセル左")

    def test_mmd_tools_suffix_forms(self):
        for bone in canonical_bones():
            if bone.side is None:
                continue
            forms = [lr_form(bone.name, sep, lower) for sep in (".", "_", " ", "-", "") for lower in (False, True)]
            forms += [to_halfwidth(lr_form(bone.name)), to_fullwidth(lr_form(bone.name, "_"))]
            for form in forms:
                with self.subTest(name=bone.name, form=form):
                    self.assertEqual(lookup(form), bone.name)
            # 左右を逆にした書き方は、反対側の骨になる（同じ骨にはならない）
            wrong = lr_form(mirror_name(bone.name))
            self.assertEqual(lookup(wrong), mirror_name(bone.name))

    def test_suffix_examples(self):
        self.assertEqual(lookup("腕.L"), "左腕")
        self.assertEqual(lookup("腕.R"), "右腕")
        self.assertEqual(lookup("足ＩＫ.L"), "左足" + IK_FW)
        self.assertEqual(lookup("足IK_R"), "右足" + IK_FW)
        self.assertEqual(lookup("肩P.R"), "右肩P")
        self.assertEqual(lookup("目.L"), "左目")
        self.assertEqual(lookup("腰キャンセル.L"), "腰キャンセル左")
        self.assertEqual(lookup("足IK親_L"), "左足IK親")
        self.assertEqual(lookup("親指０.R"), "右親指０")

    def test_suffix_on_centre_or_unsuffixed_base_does_not_map(self):
        for name in ("首.L", "上半身_R", "センター.L", "両目.R", "腕", "足IK", "左腕.L", "腕.L.001", "左腕.001",
                     "L_腕", "腕.Left", "腕.X"):
            with self.subTest(name=name):
                self.assertIsNone(lookup(name))
                self.assertFalse(is_known_extra(name))


class LookupAliasTest(unittest.TestCase):
    def test_japanese_spelling_variants(self):
        """漢字・カタカナの書き方の違い（肘・ヒジ＝ひじ、膝・ヒザ＝ひざ、人差指・人差し指＝人指、捻＝捩）。
        左右の書き換えの形（肘.L など）も対応する"""
        expected = {
            "左肘": "左ひじ", "右肘": "右ひじ", "左ヒジ": "左ひじ", "右ヒジ": "右ひじ",
            "左膝": "左ひざ", "右膝": "右ひざ", "左ヒザ": "左ひざ", "右ヒザ": "右ひざ",
            "左膝D": "左ひざD", "右ヒザD": "右ひざD",
            "左人差指１": "左人指１", "左人差指2": "左人指２", "右人差し指３": "右人指３",
            "左腕捻": "左腕捩", "右手捻": "右手捩", "左腕捻1": "左腕捩1", "右手捻3": "右手捩3",
            "肘.L": "左ひじ", "膝_R": "右ひざ", "膝D.L": "左ひざD", "人差指２.L": "左人指２", "腕捻.R": "右腕捩",
        }
        for name, canonical in expected.items():
            with self.subTest(name=name):
                self.assertEqual(lookup(name), canonical)
        for tip in ("左人差指先", "右人差し指先"):
            with self.subTest(name=tip):
                self.assertIsNone(lookup(tip))
                self.assertTrue(is_known_extra(tip))
        # 書き方の違う名前と、共通の名前そのものが両方あるときは、共通の名前が勝つ（違う方は shadowed）
        result = map_bones(["左ひじ", "左肘"])
        self.assertEqual(result.mapping, {"左ひじ": "左ひじ"})
        self.assertEqual(result.shadowed, {"左肘": "左ひじ"})
        # 準標準の外の名前（肘あて など）は対応させない
        for name in ("肘あて", "膝当て.L", "左ひじ飾り"):
            with self.subTest(name=name):
                self.assertIsNone(lookup(name))

    def test_every_alias_points_to_an_existing_canonical(self):
        names = set(canonical_names())
        count = 0
        for file_name, table, alias, canonical in all_aliases():
            count += 1
            with self.subTest(file=file_name, table=table, alias=alias):
                self.assertIn(canonical, names)
                if normalize(alias) in AMBIGUOUS_KEYS:
                    self.assertIn(canonical, lookup_candidates(alias))
                    self.assertEqual(len(lookup_candidates(alias)), 2)
                    self.assertIsNone(lookup(alias))
                    self.assertTrue(is_ambiguous(alias))
                else:
                    self.assertEqual(lookup_candidates(alias), (canonical,))
                    self.assertEqual(lookup(alias), canonical)
                    self.assertFalse(is_ambiguous(alias))
                self.assertFalse(is_known_extra(alias))
        self.assertGreater(count, 350)

    def test_only_expected_names_are_ambiguous(self):
        found = {normalize(alias) for _, _, alias, _ in all_aliases() if len(lookup_candidates(alias)) > 1}
        self.assertEqual(found, AMBIGUOUS_KEYS)

    def test_every_known_extra(self):
        count = 0
        for file_name, table, extra in all_known_extras():
            count += 1
            with self.subTest(file=file_name, table=table, extra=extra):
                self.assertTrue(is_known_extra(extra))
                self.assertIsNone(lookup(extra))
                self.assertEqual(lookup_candidates(extra), ())
        self.assertGreater(count, 40)

    def test_table_names_are_unique(self):
        tables = [table for file_name in bones.ALIAS_FILES
                  for table in json.loads((bones.DATA_DIR / file_name).read_text(encoding="utf-8"))["tables"]]
        self.assertEqual(len(tables), len(set(tables)))

    def test_vrm(self):
        expected = {
            "hips": "センター", "spine": "上半身", "chest": "上半身2", "neck": "首", "head": "頭",
            "leftShoulder": "左肩", "leftUpperArm": "左腕", "leftLowerArm": "左ひじ", "leftHand": "左手首",
            "rightHand": "右手首", "leftUpperLeg": "左足", "leftLowerLeg": "左ひざ", "leftFoot": "左足首",
            "leftToes": "左つま先", "rightToes": "右つま先", "leftEye": "左目", "rightEye": "右目",
            "leftThumbIntermediate": "左親指１", "leftThumbMetacarpal": "左親指０",
            "leftThumbDistal": "左親指２", "rightThumbDistal": "右親指２",
            "leftIndexProximal": "左人指１", "leftIndexIntermediate": "左人指２",
            "leftIndexDistal": "左人指３", "rightMiddleProximal": "右中指１",
            "leftRingIntermediate": "左薬指２", "leftLittleDistal": "左小指３",
            "rightLittleIntermediate": "右小指２",
            "LeftUpperArm": "左腕",  # 大文字・小文字は問わない
        }
        for name, canonical in expected.items():
            with self.subTest(name=name):
                self.assertEqual(lookup(name), canonical)
        for name in ("upperChest", "jaw"):
            self.assertIsNone(lookup(name))
            self.assertTrue(is_known_extra(name))
        self.assertEqual(lookup_candidates("leftThumbProximal"), ("左親指０", "左親指１"))
        self.assertEqual(lookup_candidates("rightThumbProximal"), ("右親指０", "右親指１"))

    def test_vroid(self):
        expected = {
            "J_Bip_C_Hips": "センター", "J_Bip_C_Spine": "上半身", "J_Bip_C_Chest": "上半身2", "J_Bip_C_Neck": "首",
            "J_Bip_C_Head": "頭", "J_Bip_L_Shoulder": "左肩", "J_Bip_R_UpperArm": "右腕", "J_Bip_L_LowerArm": "左ひじ",
            "J_Bip_L_Hand": "左手首", "J_Bip_L_Thumb1": "左親指０", "J_Bip_L_Thumb2": "左親指１",
            "J_Bip_R_Thumb3": "右親指２", "J_Bip_L_Index1": "左人指１", "J_Bip_L_Middle2": "左中指２",
            "J_Bip_R_Ring3": "右薬指３", "J_Bip_L_Little1": "左小指１", "J_Bip_L_UpperLeg": "左足",
            "J_Bip_R_LowerLeg": "右ひざ", "J_Bip_L_Foot": "左足首", "J_Bip_L_ToeBase": "左つま先",
            "J_Adj_L_FaceEye": "左目", "J_Adj_R_FaceEye": "右目",
        }
        for name, canonical in expected.items():
            with self.subTest(name=name):
                self.assertEqual(lookup(name), canonical)
        for name in ("J_Bip_C_UpperChest", "J_Sec_L_Bust1", "J_Sec_R_Bust2", "J_Sec_Hair1_01", "J_Adj_L_FaceEyeSet",
                     "J_Adj_C_Tongue1", "J_Opt_C_Something", "j_sec_l_bust1", "J.Sec.L.Bust1"):
            with self.subTest(name=name):
                self.assertIsNone(lookup(name))
                self.assertTrue(is_known_extra(name))
        # 前置きは区切りまで見る（区切りを消して比べない）
        for name in ("JSecret", "J_Second", "J_Secret_L", "J_Sec", "J_Bip_L_Thumb4", "Root"):
            with self.subTest(name=name):
                self.assertFalse(is_known_extra(name))
                self.assertIsNone(lookup(name))

    def test_mixamo(self):
        expected = {
            "Hips": "センター", "Spine": "上半身", "Spine1": "上半身2", "Neck": "首", "Head": "頭",
            "LeftShoulder": "左肩", "LeftArm": "左腕", "LeftForeArm": "左ひじ", "LeftHand": "左手首",
            "RightForeArm": "右ひじ", "LeftHandThumb1": "左親指０", "LeftHandThumb2": "左親指１",
            "LeftHandThumb3": "左親指２", "LeftHandIndex1": "左人指１", "RightHandMiddle2": "右中指２",
            "LeftHandRing3": "左薬指３", "LeftHandPinky1": "左小指１", "RightHandPinky3": "右小指３",
            "LeftUpLeg": "左足", "RightUpLeg": "右足", "LeftFoot": "左足首", "LeftToeBase": "左つま先",
            "LeftEye": "左目",
        }
        for prefix in ("", "mixamorig:", "mixamorig1:", "mixamorig12:", "Character:mixamorig:"):
            for name, canonical in expected.items():
                with self.subTest(name=prefix + name):
                    self.assertEqual(lookup(prefix + name), canonical)
            for name in ("Spine2", "HeadTop_End", "LeftHandThumb4", "LeftHandIndex4", "RightHandPinky4",
                         "LeftToe_End", "RightToe_End"):
                with self.subTest(name=prefix + name):
                    self.assertIsNone(lookup(prefix + name))
                    self.assertTrue(is_known_extra(prefix + name))
            # 名前空間 mixamorig は Mixamo の印：LeftLeg は Mixamo の意味（ひざ）に決まる
            if prefix:
                with self.subTest(name=prefix + "LeftLeg"):
                    self.assertEqual(lookup(prefix + "LeftLeg"), "左ひざ")
                    self.assertEqual(lookup(prefix + "RightLeg"), "右ひざ")
        # 名前空間が無い LeftLeg：Mixamo（＝ひざ）と mmd_tools（＝足）のどちらか、名前だけでは決めない
        self.assertIsNone(lookup("LeftLeg"))
        self.assertEqual(lookup_candidates("LeftLeg"), ("左足", "左ひざ"))

    def test_mixamo_namespace_decides_left_leg(self):
        """名前空間 mixamorig（後ろに数字が付いてもよい）の名前は、Mixamo の表の意味で読む"""
        for name, canonical in (("mixamorig:LeftLeg", "左ひざ"), ("mixamorig1:RightLeg", "右ひざ"),
                                ("mixamorig12:LeftLeg", "左ひざ"), ("Character:mixamorig:RightLeg", "右ひざ"),
                                ("MixamoRig：LeftLeg", "左ひざ")):  # 大文字、全角のコロン
            with self.subTest(name=name):
                self.assertEqual(lookup(name), canonical)
                self.assertEqual(lookup_candidates(name), (canonical,))
                self.assertFalse(is_ambiguous(name))
        # Mixamo でない名前空間は印にならない（mixamorig の後ろは数字だけ）
        for name in ("foo:LeftLeg", "mixamorigX:LeftLeg", "mixamorig_extra:LeftLeg", "rig:LeftLeg",
                     "mixamorig:Character:LeftLeg"):
            with self.subTest(name=name):
                self.assertIsNone(lookup(name))
                self.assertEqual(lookup_candidates(name), ("左足", "左ひざ"))
        # Mixamo の表に無い名前は、名前空間があっても今までどおり
        self.assertEqual(lookup("mixamorig:LeftKnee"), "左ひざ")
        self.assertEqual(lookup("mixamorig:Leg_L"), "左足")
        self.assertEqual(lookup("mixamorig:左腕"), "左腕")
        self.assertEqual(lookup("mixamorig:Thumb0_L"), "左親指０")
        self.assertIsNone(lookup("mixamorig:leftThumbProximal"))
        self.assertEqual(lookup_candidates("mixamorig:leftThumbProximal"), ("左親指０", "左親指１"))

    def test_mmd_english(self):
        expected = {
            "center": "センター", "upper body": "上半身", "lower body": "下半身", "neck": "首", "head": "頭",
            "eyes": "両目", "eye_L": "左目", "shoulder_L": "左肩", "arm_L": "左腕", "elbow_L": "左ひじ",
            "wrist_L": "左手首", "wrist_R": "右手首", "arm twist_L": "左腕捩", "wrist twist_R": "右手捩",
            "leg_L": "左足", "knee_L": "左ひざ", "ankle_L": "左足首", "toe_L": "左つま先",
            "leg IK_L": "左足" + IK_FW, "toe IK_L": "左つま先" + IK_FW, "leg IK_R": "右足" + IK_FW,
            "thumb1_L": "左親指１", "thumb2_R": "右親指２", "fore1_L": "左人指１",
            "middle1_L": "左中指１", "third1_L": "左薬指１", "little3_R": "右小指３",
            # mmd_tools の英語名
            "ParentNode": "全ての親", "Groove": "グルーブ", "Waist": "腰", "UpperBody2": "上半身2",
            "upper body2": "上半身2", "LeftShoulderP": "左肩P", "ShoulderP_L": "左肩P", "Thumb0_L": "左親指０",
            "LeftThumb0": "左親指０", "LegIKParent_R": "右足IK親", "WaistCancelLeft": "腰キャンセル左",
            "WaistCancel_R": "腰キャンセル右", "LegD_L": "左足D", "KneeD_R": "右ひざD", "AnkleD_L": "左足首D",
            "LegTipEX_L": "左足先EX", "LeftDummy": "左ダミー", "ArmTwist2_L": "左腕捩2", "HandTwist3_R": "右手捩3",
            "LeftToeTipIK": "左つま先" + IK_FW, "Leg_L": "左足", "LeftKnee": "左ひざ",
        }
        for name, canonical in expected.items():
            with self.subTest(name=name):
                self.assertEqual(lookup(name), canonical)
        for name in ("操作中心", "上半身3", "頭先", "左親指先", "右小指先", "親指先.L", "小指先_R", "ControlNode",
                     "UpperBody3", "LeftThumbTip", "ThumbTip_L"):
            with self.subTest(name=name):
                self.assertIsNone(lookup(name))
                self.assertTrue(is_known_extra(name))

    def test_mmd_tools_shadow_bones_are_known_extras(self):
        """mmd_tools が読み込みで付与（回転・移動の付与）のために作る _dummy_〇〇・_shadow_〇〇 の骨は、聞かなくてよい骨。
        共通の骨（〇〇）に対応させてはいけない"""
        for name in ("_dummy_腰キャンセル左", "_shadow_腰キャンセル左", "_dummy_左目", "_shadow_右目", "_dummy_腕捩1.L",
                     "_shadow_WaistCancel_L", "_dummy_LegD_R", "_shadow_左足D"):
            with self.subTest(name=name):
                self.assertIsNone(lookup(name))
                self.assertEqual(lookup_candidates(name), ())
                self.assertTrue(is_known_extra(name))
        # 前置きは区切りまで見る。ダミー（左ダミー）の英語名は、今までどおり共通の骨
        for name in ("dummy_左腕", "_dummy", "_dummyX", "_shadowing_腕", "shadow", "Shadow_L"):
            with self.subTest(name=name):
                self.assertFalse(is_known_extra(name))
                self.assertIsNone(lookup(name))
        self.assertEqual(lookup("Dummy_L"), "左ダミー")
        self.assertFalse(is_known_extra("Dummy_L"))


class NegativeExamplesTest(unittest.TestCase):
    """人型の骨でない名前を、対応させない（正規化しすぎない）"""

    def test_unknown(self):
        for name in ("LeftEarring", "RightEarring", "Leg", "Rib.R", "Root", "Armature", "左腕.001", "腕捩", "親指０", ""):
            with self.subTest(name=name):
                self.assertIsNone(lookup(name))
                self.assertEqual(lookup_candidates(name), ())
                self.assertFalse(is_known_extra(name))
                self.assertFalse(is_sway_name(name))

    def test_sway_names_do_not_map_but_are_known(self):
        """揺れ物の骨の名前は、L/R の字や 腕・足 の字があっても共通の名前に対応しない。揺れ物の骨として知っている骨になる"""
        for name in ("スカート_L", "髪1", "ribbon.L", "Ribbon", "足飾り.L", "Leg_Belt", "LegBelt.L", "左腕飾り",
                     "LeftLegBelt", "upper_body_ribbon", "skirt_L", "hair_L", "Bust_L", "左ひじ飾り"):
            with self.subTest(name=name):
                self.assertIsNone(lookup(name))
                self.assertEqual(lookup_candidates(name), ())
                self.assertTrue(is_known_extra(name))
                self.assertTrue(is_sway_name(name))

    def test_bust_is_known_extra_not_canonical(self):
        self.assertTrue(is_known_extra("J_Sec_L_Bust1"))
        self.assertIsNone(lookup("J_Sec_L_Bust1"))
        mapping = map_bones(["J_Sec_L_Bust1"])
        self.assertEqual(mapping.known_extras, ("J_Sec_L_Bust1",))
        self.assertEqual(mapping.sway_extras, ("J_Sec_L_Bust1",))
        self.assertEqual(mapping.mapping, {})


# MARK: - 揺れ物・飾りの骨（髪・スカート・リボン など）

# 作業の指示で決めた一覧（ユーザーの承認）。data/extras_sway.json と同じでなければならない
SPEC_SUBSTRINGS = ("髪", "毛", "前髪", "後髪", "横髪", "もみあげ", "スカート", "裾", "リボン", "袖", "ネクタイ", "マフラー",
                   "尻尾", "しっぽ", "耳", "羽", "翼", "帽子", "紐", "ひも", "フリル", "胸", "ケープ", "マント", "エプロン",
                   "ベルト", "飾り", "アクセ", "ポニテ", "ツイン", "揺れ", "物理")
SPEC_TOKENS = ("hair", "bang", "bangs", "ponytail", "skirt", "ribbon", "sleeve", "tie", "necktie", "scarf", "tail",
               "ear", "ears", "wing", "wings", "hat", "cap", "string", "frill", "bust", "breast", "cape", "cloak", "coat",
               "apron", "belt", "accessory", "acc", "chain", "sway", "physics", "phys")

SWAY_NAMES = (
    # 作業の指示の例
    "髪1_2", "スカート_前_0_1", "Hair_L_01", "SkirtBack2", "J_Sec_Hair1_03", "ribbon.L", "Tail_03", "EarL",
    # 日本語（字の並び）
    "前髪1", "後髪_2", "後ろ髪1_1", "横髪.L", "アホ毛", "もみあげ毛_L", "もみあげ.R", "スカート_0_0", "前スカート", "裾_後",
    "リボン1", "左袖", "袖.R", "ネクタイ", "マフラー2", "尻尾1", "しっぽ_3", "左耳", "耳.R", "羽_L_1", "翼2", "帽子", "紐_1",
    "ひも.L", "フリル3", "左胸", "胸上", "ケープ", "マント1", "エプロン", "ベルト_前", "首飾り", "アクセ1", "ポニテ1",
    "ツインテ_L", "揺れ_1", "物理_00",
    # 全角・半角（NFKC にしてから比べる）
    "ｽｶｰﾄ_0", "ﾘﾎﾞﾝ.L", "髪１", "ＨａｉｒＬ", "ＳＫＩＲＴ＿０",
    # 英語（語まるごと）
    "BackHair2", "Bang_01", "Bangs", "Ponytail_1", "PonyTail2", "Skirt_F", "Sleeve_R", "Tie", "NeckTie", "Necktie",
    "Scarf_1", "Ears", "Wing_L", "Wings", "Hat", "Cap_1", "String_2", "Frill01", "Bust_L", "BreastR", "Cape", "Cloak_2",
    "CoatBack", "Apron", "Belt_F", "Accessory1", "Acc_02", "Chain3", "SwayBone", "PhysicsBone1", "Phys_L", "HAIR_01",
    "hair.001",
    # 名前空間は外して比べる
    "mixamorig:Hair1", "Character:Skirt_B",
)

# 人型の骨らしい分からない名前：揺れ物の決まりで隠さない（ユーザーに聞く）
HUMANOID_LIKE_UNKNOWN = ("LeftForeArmTwist", "LeftForeArmRoll", "ForeArm_L", "forearm", "UpperArmTwist_L", "Spine3",
                         "Spine4", "Chest2", "Neck2", "左腕捩4", "左腕捻4", "左手捩4",
                         # ほかの書き方の人型の骨らしい名前（英語・日本語）
                         "LeftArmRoll", "Neck1", "Head_End", "Pelvis", "LeftThigh", "LeftCalf", "LeftClavicle", "Sternum",
                         "Tongue", "LeftEyelid", "Hips2", "LeftHandThumb0", "首2", "頭2", "左手首2", "両目先", "左目先",
                         "左腕捩り", "左ひじ補助", "肩甲骨.L", "鎖骨.L", "上半身4", "左足IK親2")
# 字の並びとしては揺れ物の英語を含むが、語としては含まない名前（ear・tail・cap・hat・acc・hair）
LOOKALIKE_UNKNOWN = ("Heart", "Spear", "Bearing", "Earring_L", "LeftEarring", "Detail_01", "Retail", "Capsule_L",
                     "Chat_1", "Accent_1", "Chair_L")


class SwayDataTest(unittest.TestCase):
    def test_data_file_is_the_approved_list(self):
        data = json.loads((bones.DATA_DIR / bones.SWAY_FILE).read_text(encoding="utf-8"))
        self.assertEqual(tuple(data["substrings"]), SPEC_SUBSTRINGS)
        self.assertEqual(tuple(data["tokens"]), SPEC_TOKENS)
        self.assertTrue(data["_notes"] and data["_sources"])
        self.assertTrue(any("forearm" in note for note in data["_notes"]))  # 語まるごとで比べる理由

    def test_every_listed_pattern_is_used(self):
        """一覧のどの決まりも、それだけで揺れ物の名前を見分ける（書いても効かない決まりが無い）"""
        for text in SPEC_SUBSTRINGS:
            for name in (text, f"{text}1", f"左{text}_0_1", f"{text}.R"):
                with self.subTest(name=name):
                    self.assertTrue(is_sway_name(name))
                    self.assertTrue(is_known_extra(name))
                    self.assertIsNone(lookup(name))
        for word in SPEC_TOKENS:
            for name in (word, word.capitalize() + "_L", "Left" + word.capitalize() + "2", word.upper() + "_01",
                         f"x_{word}.R", "J_Sec_" + word.capitalize() + "1"):
                with self.subTest(name=name):
                    self.assertTrue(is_sway_name(name))
                    self.assertTrue(is_known_extra(name))
                    self.assertIsNone(lookup(name))

    def test_tokens_match_only_whole_words(self):
        """英語の語は、ほかの語の一部としては当たらない（小文字で続けた forehair・hairx は1語）"""
        for word in SPEC_TOKENS:
            for name in ("fore" + word, word + "x", "Fore" + word, word.capitalize() + "x_L"):
                with self.subTest(name=name):
                    self.assertFalse(is_sway_name(name))
                    self.assertFalse(is_known_extra(name))


class SwayNameTest(unittest.TestCase):
    def test_positive_table(self):
        for name in SWAY_NAMES:
            with self.subTest(name=name):
                self.assertTrue(is_sway_name(name))
                self.assertTrue(is_known_extra(name))
                self.assertIsNone(lookup(name))
                self.assertEqual(lookup_candidates(name), ())
                self.assertFalse(is_ambiguous(name))

    def test_positive_table_in_map_bones(self):
        names = list(SWAY_NAMES)
        result = map_bones(names)
        check_partition(self, names, result)
        self.assertEqual((result.mapping, result.unknown, result.conflicts), ({}, (), {}))
        self.assertEqual(result.known_extras, SWAY_NAMES)
        self.assertEqual(result.sway_extras, SWAY_NAMES)

    def test_every_canonical_name_and_alias_still_maps(self):
        """人型の骨の名前（共通の名前・左右の書き方・aliases）は、揺れ物の決まりで隠れない"""
        for bone in canonical_bones():
            forms = [bone.name] + ([] if bone.side is None else [lr_form(bone.name), lr_form(bone.name, "_")])
            for form in forms:
                with self.subTest(name=form):
                    self.assertEqual(lookup(form), bone.name)
                    self.assertFalse(is_sway_name(form))
                    self.assertFalse(is_known_extra(form))
        for file_name, table, alias, canonical in all_aliases():
            with self.subTest(file=file_name, table=table, alias=alias):
                self.assertIn(canonical, lookup_candidates(alias))
                self.assertFalse(is_sway_name(alias))
                self.assertFalse(is_known_extra(alias))

    def test_alias_variants_are_never_sway(self):
        """aliases の全部と、左右の書き換えの形（derive_lr_suffix の表：肘.L・膝_R・腕捻 r など）は、全角・大文字小文字・
        半角カナ・名前空間の書き方を変えても、揺れ物の決まりで隠れない（人型の骨が先）"""
        count = 0
        for file_name in bones.ALIAS_FILES:
            data = json.loads((bones.DATA_DIR / file_name).read_text(encoding="utf-8"))
            for table, spec in data["tables"].items():
                for alias, canonical in spec.get("aliases", {}).items():
                    forms = [alias]
                    if spec.get("derive_lr_suffix") and ("左" in alias) != ("右" in alias):
                        forms += [lr_form(alias, sep, lower) for sep in (".", "_", " ", "-", "") for lower in (False, True)]
                    for form in forms:
                        # 名前空間は、表を示さない物（Armature:）にする。mixamorig: は LeftLeg の意味を Mixamo に決めるため
                        for variant in (form, to_fullwidth(form), swap_ascii_case(form), to_halfwidth_kana(form),
                                        "Armature:" + form, " " + form + " "):
                            count += 1
                            with self.subTest(file=file_name, table=table, name=variant):
                                self.assertIn(canonical, lookup_candidates(variant))
                                self.assertFalse(is_sway_name(variant))
                                self.assertFalse(is_known_extra(variant))
        self.assertGreater(count, 3000)

    def test_named_humanoid_examples_map_first(self):
        expected = {"左肩": "左肩", "LeftForeArm": "左ひじ", "RightUpLeg": "右足", "upper body": "上半身",
                    "上半身2": "上半身2", "首": "首", "頭": "頭", "左腕捻3": "左腕捩3"}
        for name, canonical in {**expected, "mixamorig:LeftForeArm": "左ひじ", "J_Bip_L_LowerArm": "左ひじ"}.items():
            with self.subTest(name=name):
                self.assertEqual(lookup(name), canonical)
                self.assertFalse(is_sway_name(name))
                self.assertFalse(is_known_extra(name))
        result = map_bones(list(expected))
        self.assertEqual(result.mapping, expected)
        self.assertEqual((result.known_extras, result.sway_extras, result.unknown), ((), (), ()))
        # 耳は共通の名前に無いので、揺れ物の骨でよい
        self.assertIsNone(lookup("左耳"))
        self.assertTrue(is_sway_name("左耳"))

    def test_mapping_wins_over_a_matching_pattern(self):
        """比べる形が人型の骨の名前と同じなら、名前の中に揺れ物の語（ear）があっても人型の骨（順番の1が先）"""
        self.assertIn("ear", name_tokens("LeftFor_Ear_m"))
        self.assertEqual(normalize("LeftFor_Ear_m"), normalize("LeftForeArm"))
        self.assertEqual(lookup("LeftFor_Ear_m"), "左ひじ")
        self.assertFalse(is_sway_name("LeftFor_Ear_m"))
        self.assertFalse(is_known_extra("LeftFor_Ear_m"))
        result = map_bones(["LeftFor_Ear_m"])
        self.assertEqual(result.mapping, {"LeftFor_Ear_m": "左ひじ"})
        self.assertEqual((result.known_extras, result.sway_extras), ((), ()))
        # 表によって意味が違う名前も、揺れ物にはしない（ユーザーに聞く）
        self.assertFalse(is_sway_name("LeftLeg"))
        self.assertEqual(map_bones(["LeftLeg"]).ambiguous, {"LeftLeg": ("左足", "左ひざ")})

    def test_humanoid_like_unknowns_stay_unknown(self):
        for name in HUMANOID_LIKE_UNKNOWN + LOOKALIKE_UNKNOWN:
            with self.subTest(name=name):
                self.assertIsNone(lookup(name))
                self.assertFalse(is_sway_name(name))
                self.assertFalse(is_known_extra(name))
        names = list(HUMANOID_LIKE_UNKNOWN + LOOKALIKE_UNKNOWN)
        result = map_bones(names)
        self.assertEqual(result.unknown, tuple(names))
        self.assertEqual((result.known_extras, result.sway_extras, result.mapping), ((), (), {}))

    def test_other_known_extras_are_not_sway(self):
        """揺れ物でない知っている骨（指先・UpperChest・mmd_tools の _dummy_ など）は、sway_extras に入らない"""
        for name in ("upperChest", "jaw", "上半身3", "頭先", "操作中心", "左親指先", "HeadTop_End", "Spine2",
                     "J_Bip_C_UpperChest", "J_Adj_C_Tongue1", "J_Adj_L_FaceEyeSet", "_dummy_左足D", "_shadow_腰キャンセル左"):
            with self.subTest(name=name):
                self.assertTrue(is_known_extra(name))
                self.assertFalse(is_sway_name(name))
        swaying = [extra for _, _, extra in all_known_extras() if is_sway_name(extra)]
        self.assertEqual(swaying, [])
        # 書き方を変えても同じ（全角・大文字小文字・名前空間）
        for _, _, extra in all_known_extras():
            for variant in (to_fullwidth(extra), swap_ascii_case(extra), "Armature:" + extra):
                with self.subTest(name=variant):
                    self.assertTrue(is_known_extra(variant))
                    self.assertFalse(is_sway_name(variant))
        # 前置きで知っている骨でも、揺れ物の決まりに当たれば揺れ物の骨
        for name in ("J_Sec_Hair1_03", "J_Sec_L_Bust1", "_shadow_髪1"):
            with self.subTest(name=name):
                self.assertTrue(is_known_extra(name))
                self.assertTrue(is_sway_name(name))

    def test_bad_types(self):
        for bad in (None, 1, b"hair", ["髪1"]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    is_sway_name(bad)
                with self.assertRaises(ValueError):
                    name_tokens(bad)

    def test_deterministic(self):
        names = list(SWAY_NAMES + HUMANOID_LIKE_UNKNOWN + LOOKALIKE_UNKNOWN) + list(canonical_names())
        flags = [is_sway_name(name) for name in names]
        self.assertEqual(flags, [is_sway_name(name) for name in names])
        first = map_bones(names)
        self.assertEqual(map_bones(names), first)
        bones._dictionary.cache_clear()  # 辞書を読み直しても同じ
        self.assertEqual([is_sway_name(name) for name in names], flags)
        self.assertEqual(map_bones(names), first)
        reversed_result = map_bones(list(reversed(names)))
        self.assertEqual(reversed_result.sway_extras, tuple(reversed(first.sway_extras)))
        self.assertEqual(reversed_result.unknown, tuple(reversed(first.unknown)))


class NameTokensTest(unittest.TestCase):
    def test_examples(self):
        expected = {
            "SkirtBack2": ("skirt", "back"),
            "J_Sec_Hair1_03": ("j", "sec", "hair"),
            "LeftForeArmTwist": ("left", "fore", "arm", "twist"),
            "EarL": ("ear", "l"),
            "IKParent": ("ik", "parent"),
            "HAIR_01_L": ("hair", "l"),
            "hairBack3Tail": ("hair", "back", "tail"),
            "Hair1Ribbon": ("hair", "ribbon"),
            "PonyTail2": ("pony", "tail"),
            "forearm": ("forearm",),
            "LEFTFOREARM": ("leftforearm",),
            "ＨａｉｒＬ": ("hair", "l"),  # 全角
            "mixamorig:LeftEar": ("left", "ear"),  # 名前空間を外す
            "Leg-IK.l": ("leg", "ik", "l"),
            "ribbon.L": ("ribbon", "l"),
            "Tail_03": ("tail",),
            "スカートFront_0": ("front",),  # 日本語の字は区切り
            "髪1_2": (),
            "": (),
        }
        for name, tokens in expected.items():
            with self.subTest(name=name):
                self.assertEqual(name_tokens(name), tokens)


# MARK: - モデルの骨の名前をまとめて対応させる

VROID = (["Root", "J_Bip_C_Hips", "J_Bip_C_Spine", "J_Bip_C_Chest", "J_Bip_C_UpperChest", "J_Bip_C_Neck",
          "J_Bip_C_Head", "J_Adj_L_FaceEye", "J_Adj_R_FaceEye", "J_Sec_L_Bust1", "J_Sec_L_Bust2", "J_Sec_R_Bust1",
          "J_Sec_R_Bust2", "J_Sec_Hair1_01", "J_Sec_Hair2_01"]
         + [f"J_Bip_{s}_{part}" for s in "LR" for part in ("Shoulder", "UpperArm", "LowerArm", "Hand", "UpperLeg",
                                                           "LowerLeg", "Foot", "ToeBase")]
         + [f"J_Bip_{s}_{finger}{i}" for s in "LR" for finger in ("Thumb", "Index", "Middle", "Ring", "Little")
            for i in (1, 2, 3)])

MIXAMO = (["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "HeadTop_End"]
          + [f"{s}{part}" for s in ("Left", "Right") for part in ("Shoulder", "Arm", "ForeArm", "Hand", "UpLeg", "Leg",
                                                                   "Foot", "ToeBase", "Toe_End")]
          + [f"{s}Hand{finger}{i}" for s in ("Left", "Right")
             for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky") for i in (1, 2, 3, 4)])

VRM1 = (["hips", "spine", "chest", "upperChest", "neck", "head", "jaw", "leftEye", "rightEye"]
        + [f"{s}{part}" for s in ("left", "right") for part in ("Shoulder", "UpperArm", "LowerArm", "Hand", "UpperLeg",
                                                                 "LowerLeg", "Foot", "Toes")]
        + [f"{s}Thumb{p}" for s in ("left", "right") for p in ("Metacarpal", "Proximal", "Distal")]
        + [f"{s}{f}{p}" for s in ("left", "right") for f in ("Index", "Middle", "Ring", "Little")
           for p in ("Proximal", "Intermediate", "Distal")])
VRM0 = [name.replace("ThumbProximal", "ThumbIntermediate").replace("ThumbMetacarpal", "ThumbProximal")
        for name in VRM1]

STANDARD_LEG_IK = ("左足" + IK_FW, "左つま先" + IK_FW, "右足" + IK_FW, "右つま先" + IK_FW)


def check_partition(test: unittest.TestCase, names, result: BoneMapping) -> None:
    """入力の名前は、どれか1つの入れ物だけに入る"""
    buckets = [set(result.mapping), set(result.unknown), set(result.known_extras), set(result.shadowed),
               {n for members in result.conflicts.values() for n in members}]
    total = sum(len(b) for b in buckets)
    test.assertEqual(total, len(set(names)))
    test.assertEqual(set().union(*buckets), set(names))
    test.assertTrue(set(result.ambiguous) <= set(result.unknown))
    # 揺れ物の骨は known_extras の中の印（入れ物は増えない）。並びは known_extras と同じ（入力の順）
    test.assertEqual(result.sway_extras, tuple(n for n in result.known_extras if n in set(result.sway_extras)))
    test.assertEqual(result.sway_extras, tuple(n for n in result.known_extras if is_sway_name(n)))


class MapBonesTest(unittest.TestCase):
    def test_full_semi_standard_is_compliant(self):
        names = list(canonical_names())
        result = map_bones(names)
        self.assertEqual(result.mapping, {n: n for n in names})
        self.assertEqual(list(result.mapping), names)  # 入力の順
        self.assertEqual((result.unknown, result.known_extras, result.conflicts, result.shadowed, result.ambiguous,
                          result.sway_extras), ((), (), {}, {}, {}, ()))
        self.assertEqual((result.missing_standard, result.missing_semi_standard), ((), ()))
        self.assertTrue(result.is_semi_standard_compliant)

    def test_compliant_with_extras_and_without_helpers(self):
        names = [b.name for b in canonical_bones() if not b.helper] + ["髪1", "Root", "スカート_L", "操作中心", "左親指先"]
        result = map_bones(names)
        self.assertTrue(result.is_semi_standard_compliant)
        self.assertEqual(result.unknown, ("Root",))
        self.assertEqual(result.known_extras, ("髪1", "スカート_L", "操作中心", "左親指先"))
        self.assertEqual(result.sway_extras, ("髪1", "スカート_L"))
        check_partition(self, names, result)

    def test_missing_one_semi_or_standard_bone(self):
        names = [n for n in canonical_names() if n != "左ダミー"]
        result = map_bones(names)
        self.assertFalse(result.is_semi_standard_compliant)
        self.assertEqual(result.missing_semi_standard, ("左ダミー",))
        self.assertEqual(result.missing_standard, ())
        names = [n for n in canonical_names() if n not in ("センター", "右足" + IK_FW)]
        result = map_bones(names)
        self.assertFalse(result.is_semi_standard_compliant)
        self.assertEqual(result.missing_standard, ("センター", "右足" + IK_FW))
        self.assertEqual(result.missing_semi_standard, ())

    def test_mmd_tools_renamed_model_is_compliant(self):
        for separator in (".", "_"):
            names = [n if b.side is None else lr_form(n, separator)
                     for b in canonical_bones() for n in [b.name]]
            result = map_bones(names)
            with self.subTest(separator=separator):
                self.assertTrue(result.is_semi_standard_compliant)
                self.assertEqual(result.mapping["腕" + separator + "L"], "左腕")
                self.assertEqual(result.mapping["腰キャンセル" + separator + "R"], "腰キャンセル右")
                self.assertEqual(len(result.mapping), 96)

    def test_mmd_tools_english_model_resolves_left_leg(self):
        """mmd_tools で英語にした名前だけのモデル：LeftLeg は 左足（Mixamo のひざではない）"""
        data = json.loads((bones.DATA_DIR / "aliases_mmd.json").read_text(encoding="utf-8"))
        table = data["tables"]["mmd_en_mmd_tools"]["aliases"]
        plain = [alias for alias in table if "_" not in alias]
        self.assertEqual(len(plain), 96)
        result = map_bones(plain)
        self.assertEqual(result.mapping["LeftLeg"], "左足")
        self.assertEqual(result.mapping["RightLeg"], "右足")
        self.assertEqual(result.mapping["LeftKnee"], "左ひざ")
        self.assertTrue(result.is_semi_standard_compliant)
        lr = [alias for alias in table if "_" in alias] + [a for a in plain if not a.startswith(("Left", "Right"))
                                                           and table[a] in TRUNK + ["両目"]]
        result = map_bones(lr)
        self.assertTrue(result.is_semi_standard_compliant, result)

    def test_vroid_is_not_compliant(self):
        result = map_bones(VROID)
        check_partition(self, VROID, result)
        self.assertFalse(result.is_semi_standard_compliant)
        self.assertEqual(result.unknown, ("Root",))
        self.assertEqual(set(result.known_extras), {"J_Bip_C_UpperChest", "J_Sec_L_Bust1", "J_Sec_L_Bust2",
                                                    "J_Sec_R_Bust1", "J_Sec_R_Bust2", "J_Sec_Hair1_01",
                                                    "J_Sec_Hair2_01"})
        # 前置き J_Sec_ で知っている骨のうち、胸（bust）と髪（hair）は揺れ物の骨。UpperChest は揺れ物ではない
        self.assertEqual(result.sway_extras, ("J_Sec_L_Bust1", "J_Sec_L_Bust2", "J_Sec_R_Bust1", "J_Sec_R_Bust2",
                                              "J_Sec_Hair1_01", "J_Sec_Hair2_01"))
        self.assertEqual(result.mapping["J_Bip_C_Hips"], "センター")
        self.assertEqual(result.mapping["J_Bip_L_Thumb1"], "左親指０")
        self.assertEqual(result.mapping["J_Bip_R_Little3"], "右小指３")
        self.assertEqual(len(result.mapping), 5 + 2 + 2 * (8 + 15))
        self.assertEqual(result.missing_standard, ("下半身", "両目") + STANDARD_LEG_IK)
        self.assertEqual(result.missing_semi_standard, (
            "全ての親", "グルーブ", "腰",
            "左肩P", "左腕捩", "左手捩", "左ダミー", "右肩P", "右腕捩", "右手捩", "右ダミー",
            "左足IK親", "腰キャンセル左", "左足D", "左ひざD", "左足首D", "左足先EX",
            "右足IK親", "腰キャンセル右", "右足D", "右ひざD", "右足首D", "右足先EX"))
        # 補助の骨（肩C・腕捩1〜3・手捩1〜3）は、無くても missing に出ない
        self.assertNotIn("左肩C", result.missing_semi_standard)
        self.assertNotIn("左腕捩1", result.missing_semi_standard)

    def test_mixamo_model(self):
        for prefix in ("mixamorig:", "mixamorig1:", ""):
            names = [prefix + n for n in MIXAMO]
            result = map_bones(names)
            with self.subTest(prefix=prefix):
                check_partition(self, names, result)
                self.assertEqual(result.unknown, ())
                self.assertEqual(result.ambiguous, {})
                self.assertEqual(result.mapping[prefix + "LeftLeg"], "左ひざ")
                self.assertEqual(result.mapping[prefix + "RightLeg"], "右ひざ")
                self.assertEqual(result.mapping[prefix + "LeftUpLeg"], "左足")
                self.assertEqual(result.mapping[prefix + "Spine1"], "上半身2")
                self.assertEqual(result.mapping[prefix + "LeftHandThumb1"], "左親指０")
                self.assertEqual(len(result.known_extras), 2 + 2 * (1 + 5))
                self.assertEqual(len(result.mapping), 5 + 2 * (8 + 15))
                self.assertFalse(result.is_semi_standard_compliant)
                self.assertEqual(result.missing_standard, ("下半身", "両目", "左目", "右目") + STANDARD_LEG_IK)

    def test_vrm1_and_vrm0_models(self):
        for names, version in ((VRM1, "1.0"), (VRM0, "0.x")):
            result = map_bones(names)
            with self.subTest(version=version):
                check_partition(self, names, result)
                self.assertEqual(result.unknown, ())
                self.assertEqual(result.known_extras, ("upperChest", "jaw"))
                self.assertEqual(len(result.mapping), 5 + 2 + 2 * (8 + 15))
                thumbs = {result.mapping[n] for n in names if "Thumb" in n}
                self.assertEqual(thumbs, {f"{s}親指{FW[d]}" for s in "左右" for d in "012"})
        self.assertEqual(map_bones(VRM1).mapping["leftThumbProximal"], "左親指１")
        self.assertEqual(map_bones(VRM0).mapping["leftThumbProximal"], "左親指０")
        self.assertEqual(map_bones(VRM0).mapping["rightThumbIntermediate"], "右親指１")

    def test_vrm_thumb_resolution_small_inputs(self):
        result = map_bones(["leftThumbMetacarpal", "leftThumbProximal", "leftThumbDistal"])
        self.assertEqual(result.mapping, {"leftThumbMetacarpal": "左親指０", "leftThumbProximal": "左親指１",
                                          "leftThumbDistal": "左親指２"})
        result = map_bones(["leftThumbProximal", "leftThumbIntermediate", "leftThumbDistal"])
        self.assertEqual(result.mapping, {"leftThumbProximal": "左親指０", "leftThumbIntermediate": "左親指１",
                                          "leftThumbDistal": "左親指２"})
        # 右手の印からでも決まる（同じモデル）
        result = map_bones(["rightThumbMetacarpal", "leftThumbProximal"])
        self.assertEqual(result.mapping["leftThumbProximal"], "左親指１")
        # 印が無い・両方ある：決めない
        for names in (["leftThumbProximal", "leftThumbDistal", "hips"],
                      ["leftThumbProximal", "leftThumbMetacarpal", "rightThumbIntermediate"]):
            result = map_bones(names)
            with self.subTest(names=names):
                self.assertNotIn("leftThumbProximal", result.mapping)
                self.assertIn("leftThumbProximal", result.unknown)
                self.assertEqual(result.ambiguous, {"leftThumbProximal": ("左親指０", "左親指１")})
                check_partition(self, names, result)

    def test_mmd_tools_imported_model_with_shadow_bones(self):
        """mmd_tools で読み込んだ準標準のモデル（rename_bones=False）：_dummy_・_shadow_ の骨は聞かない"""
        shadow = ["_dummy_腰キャンセル左", "_shadow_腰キャンセル左", "_dummy_腰キャンセル右", "_shadow_腰キャンセル右",
                  "_dummy_左足D", "_shadow_左足D"]
        names = list(canonical_names()) + shadow + ["髪1", "Root"]
        result = map_bones(names)
        check_partition(self, names, result)
        self.assertEqual(result.mapping, {n: n for n in canonical_names()})
        self.assertEqual(result.known_extras, tuple(shadow) + ("髪1",))
        self.assertEqual(result.sway_extras, ("髪1",))
        self.assertEqual(result.unknown, ("Root",))
        self.assertTrue(result.is_semi_standard_compliant)

    def test_mixamo_namespace_in_map_bones(self):
        result = map_bones(["mixamorig:LeftLeg"])
        self.assertEqual(result.mapping, {"mixamorig:LeftLeg": "左ひざ"})
        self.assertEqual((result.unknown, result.ambiguous), ((), {}))
        # 頭だけのような、どの表にもある名前でも、名前空間 mixamorig が付けば Mixamo の印になる
        result = map_bones(["mixamorig:Head", "LeftLeg"])
        self.assertEqual(result.mapping, {"mixamorig:Head": "頭", "LeftLeg": "左ひざ"})
        # mmd_tools の印（LeftKnee）と Mixamo の印が両方あるときは、名前空間の無い LeftLeg を決めない
        result = map_bones(["mixamorig:Hips", "LeftKnee", "LeftLeg"])
        self.assertEqual(result.ambiguous, {"LeftLeg": ("左足", "左ひざ")})
        self.assertNotIn("LeftLeg", result.mapping)

    def test_left_leg_resolution(self):
        result = map_bones(["LeftLeg"])
        self.assertEqual(result.mapping, {})
        self.assertEqual(result.unknown, ("LeftLeg",))
        self.assertEqual(result.ambiguous, {"LeftLeg": ("左足", "左ひざ")})
        self.assertEqual(map_bones(["LeftLeg", "LeftUpLeg"]).mapping["LeftLeg"], "左ひざ")
        self.assertEqual(map_bones(["mixamorig:LeftLeg", "mixamorig:Hips"]).mapping["mixamorig:LeftLeg"], "左ひざ")
        self.assertEqual(map_bones(["LeftLeg", "LeftKnee"]).mapping["LeftLeg"], "左足")
        self.assertEqual(map_bones(["LeftLeg", "LowerBody"]).mapping["LeftLeg"], "左足")
        # どちらの印もある・両方に載っている名前（LeftArm）は印にならない
        for names in (["LeftLeg", "LeftUpLeg", "LeftKnee"], ["LeftLeg", "LeftArm", "Head", "LeftEye"]):
            with self.subTest(names=names):
                result = map_bones(names)
                self.assertNotIn("LeftLeg", result.mapping)
                self.assertIn("LeftLeg", result.ambiguous)

    def test_exact_canonical_name_wins(self):
        names = ["腕.L", "左腕", "arm_L", "LeftArm"]
        result = map_bones(names)
        self.assertEqual(result.mapping, {"左腕": "左腕"})
        self.assertEqual(result.shadowed, {"腕.L": "左腕", "arm_L": "左腕", "LeftArm": "左腕"})
        self.assertEqual(result.conflicts, {})
        self.assertNotIn("左腕", result.missing_standard)
        check_partition(self, names, result)
        # 半角で書いた共通の名前も「そのもの」として勝つ
        result = map_bones(["足IK.L", "左足IK"])
        self.assertEqual(result.mapping, {"左足IK": "左足" + IK_FW})
        self.assertEqual(result.shadowed, {"足IK.L": "左足" + IK_FW})

    def test_conflict_without_exact_name(self):
        names = ["arm_R", "腕.R", "arm_L", "腕.L", "腕_L", "head"]
        result = map_bones(names)
        self.assertEqual(result.conflicts, {"左腕": ("arm_L", "腕.L", "腕_L"), "右腕": ("arm_R", "腕.R")})
        self.assertEqual(list(result.conflicts), ["左腕", "右腕"])  # 共通の名前の順
        self.assertEqual(result.mapping, {"head": "頭"})
        self.assertNotIn("左腕", result.missing_standard)
        self.assertNotIn("右腕", result.missing_standard)
        self.assertIn("左ひじ", result.missing_standard)
        self.assertEqual(result.shadowed, {})
        check_partition(self, names, result)

    def test_conflict_between_two_exact_names(self):
        names = ["左足" + IK_FW, "左足IK", "足IK.L"]
        result = map_bones(names)
        self.assertEqual(result.conflicts, {"左足" + IK_FW: tuple(names)})
        self.assertEqual(result.mapping, {})
        self.assertEqual(result.shadowed, {})

    def test_conflict_makes_full_list_not_compliant(self):
        names = list(canonical_names()) + ["上半身２"]  # 全角の２：共通の名前と同じ形
        result = map_bones(names)
        self.assertEqual(result.conflicts, {"上半身2": ("上半身2", "上半身２")})
        self.assertEqual(result.missing_semi_standard, ())
        self.assertFalse(result.is_semi_standard_compliant)

    def test_duplicates_count_once(self):
        result = map_bones(["左腕", "左腕", "Root", "Root", "髪1", "髪1", "upperChest", "upperChest"])
        self.assertEqual(result.mapping, {"左腕": "左腕"})
        self.assertEqual(result.conflicts, {})
        self.assertEqual(result.unknown, ("Root",))
        self.assertEqual(result.known_extras, ("髪1", "upperChest"))
        self.assertEqual(result.sway_extras, ("髪1",))

    def test_empty_input(self):
        result = map_bones([])
        self.assertEqual(result.mapping, {})
        self.assertEqual(result.missing_standard, tuple(b.name for b in canonical_bones() if b.standard))
        self.assertEqual(result.missing_semi_standard,
                         tuple(b.name for b in canonical_bones() if not b.standard and not b.helper))
        self.assertEqual(len(result.missing_standard) + len(result.missing_semi_standard), 96 - 14)
        self.assertFalse(result.is_semi_standard_compliant)

    def test_single_element(self):
        result = map_bones(["J_Bip_C_Head"])
        self.assertEqual(result.mapping, {"J_Bip_C_Head": "頭"})
        self.assertEqual(result.by_canonical, {"頭": "J_Bip_C_Head"})
        self.assertNotIn("頭", result.missing_standard)
        self.assertIn("首", result.missing_standard)

    def test_generator_and_tuple_input(self):
        self.assertEqual(map_bones(n for n in ["腕.L"]).mapping, {"腕.L": "左腕"})
        self.assertEqual(map_bones(("腕.L",)).mapping, {"腕.L": "左腕"})

    def test_bad_input(self):
        for bad in ("左腕", b"arm", 3, None, ["左腕", None], ["左腕", True], [1], [["左腕"]]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    map_bones(bad)

    def test_order_independent(self):
        names = VROID + ["腕.L", "左腕", "arm_R", "腕.R", "LeftLeg", "髪1", "スカート_0_0", "LeftForeArmTwist"]
        base = map_bones(names)
        self.assertEqual(set(base.sway_extras), {"髪1", "スカート_0_0", "J_Sec_L_Bust1", "J_Sec_L_Bust2",
                                                 "J_Sec_R_Bust1", "J_Sec_R_Bust2", "J_Sec_Hair1_01", "J_Sec_Hair2_01"})
        rng = random.Random(0)
        for _ in range(5):
            shuffled = names[:]
            rng.shuffle(shuffled)
            result = map_bones(shuffled)
            self.assertEqual(result.mapping, base.mapping)  # dict の比べは順を見ない
            self.assertEqual(list(result.mapping), [n for n in shuffled if n in base.mapping])  # 入力の順
            self.assertEqual(set(result.unknown), set(base.unknown))
            self.assertEqual(set(result.known_extras), set(base.known_extras))
            self.assertEqual(set(result.sway_extras), set(base.sway_extras))
            self.assertEqual(list(result.sway_extras), [n for n in shuffled if n in base.sway_extras])  # 入力の順
            self.assertEqual({k: set(v) for k, v in result.conflicts.items()},
                             {k: set(v) for k, v in base.conflicts.items()})
            self.assertEqual(list(result.conflicts), list(base.conflicts))  # 共通の名前の順
            self.assertEqual(result.shadowed, base.shadowed)
            self.assertEqual(result.ambiguous, base.ambiguous)
            self.assertEqual(result.missing_standard, base.missing_standard)
            self.assertEqual(result.missing_semi_standard, base.missing_semi_standard)

    def test_humanoid_names_match_app_human_bones(self):
        """VRM のヒューマノイドの名前は、紅魔館 convert_pmx.py の HUMAN_BONES が最初に探す骨と同じ骨に対応する
        （hips ← センター。脚は足D より先に、VMD で動かす 足・ひざ・足首 に対応させる）"""
        expected = {
            "hips": "センター", "spine": "上半身", "chest": "上半身2", "neck": "首", "head": "頭",
            "leftEye": "左目", "rightEye": "右目", "leftShoulder": "左肩", "rightShoulder": "右肩",
            "leftUpperArm": "左腕", "rightUpperArm": "右腕", "leftLowerArm": "左ひじ", "rightLowerArm": "右ひじ",
            "leftHand": "左手首", "rightHand": "右手首",
            "leftUpperLeg": "左足", "rightUpperLeg": "右足", "leftLowerLeg": "左ひざ", "rightLowerLeg": "右ひざ",
            "leftFoot": "左足首", "rightFoot": "右足首",
        }
        for name, canonical in expected.items():
            with self.subTest(name=name):
                self.assertEqual(lookup(name), canonical)
        for name in ("hips", "J_Bip_C_Hips", "mixamorig:Hips"):
            with self.subTest(name=name):
                self.assertNotEqual(lookup(name), "下半身")

    def test_by_canonical_is_in_canonical_order(self):
        result = map_bones(["J_Bip_C_Head", "J_Bip_L_Hand", "J_Bip_C_Hips", "J_Bip_R_Hand"])
        self.assertEqual(list(result.by_canonical), ["センター", "頭", "左手首", "右手首"])
        self.assertEqual(result.by_canonical["左手首"], "J_Bip_L_Hand")

    def test_mmd_model_with_sway_bones(self):
        """よくある MMD のモデル：揺れ物の骨は聞かず（sway_extras）、人型の骨らしい名前と分からない名前だけを聞く"""
        sway = ["髪1", "前髪_1", "後ろ髪_1_1", "スカート_0_0", "スカート_0_1", "左胸", "右胸", "リボン"]
        names = list(canonical_names()) + sway + ["Root", "謎の骨", "LeftForeArmTwist", "操作中心", "_dummy_左足D"]
        result = map_bones(names)
        check_partition(self, names, result)
        self.assertTrue(result.is_semi_standard_compliant)
        self.assertEqual(result.mapping, {n: n for n in canonical_names()})
        self.assertEqual(result.unknown, ("Root", "謎の骨", "LeftForeArmTwist"))
        self.assertEqual(result.known_extras, tuple(sway) + ("操作中心", "_dummy_左足D"))
        self.assertEqual(result.sway_extras, tuple(sway))
        self.assertEqual(len(result.sway_extras), 8)

    def test_result_is_frozen(self):
        result = map_bones(["左腕"])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.unknown = ()


# MARK: - 辞書のファイルのまちがいを、読み込みのときに見つける


class BrokenDataTest(unittest.TestCase):
    """data/ の JSON をわざと壊した写しで読み込み、ValueError になるかを確かめる"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.original_dir = bones.DATA_DIR
        for path in bones.DATA_DIR.glob("*.json"):
            shutil.copy(path, self.dir / path.name)
        patcher = mock.patch.object(bones, "DATA_DIR", self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        bones._dictionary.cache_clear()
        self.addCleanup(bones._dictionary.cache_clear)  # ほかのテストは本物の辞書で動かす

    def reset(self, file_name):
        """写しのファイルを、本物の辞書のファイルに戻す（subTest ごとに、壊し方が重ならないように）"""
        shutil.copy(self.original_dir / file_name, self.dir / file_name)

    def edit(self, file_name, change):
        path = self.dir / file_name
        data = json.loads(path.read_text(encoding="utf-8"))
        change(data)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def assert_broken(self, file_name, change, message):
        """壊した写しを読むと、message を含む ValueError になる"""
        self.edit(file_name, change)
        bones._dictionary.cache_clear()
        with self.assertRaisesRegex(ValueError, message):
            canonical_names()

    def test_copy_without_changes_loads(self):
        self.assertEqual(len(canonical_names()), 96)

    def test_alias_to_missing_canonical(self):
        self.assert_broken("aliases_mixamo.json",
                           lambda d: d["tables"]["mixamo"]["aliases"].update({"LeftHandThumb0": "左親指0"}),
                           "対応先 左親指0 は共通の名前にありません")

    def test_side_bone_without_mirror(self):
        def drop_right_arm(d):
            d["bones"] = [b for b in d["bones"] if b["name"] != "右腕"]
        self.assert_broken(bones.CANONICAL_FILE, drop_right_arm, "左腕 の反対側の骨がありません")

    def test_mirror_with_different_kind(self):
        def make_helper(d):
            next(b for b in d["bones"] if b["name"] == "右腕捩")["helper"] = True
        self.assert_broken(bones.CANONICAL_FILE, make_helper, "反対側の骨がありません（または種類が違います）")

    def test_side_does_not_match_name(self):
        def wrong_side(d):
            next(b for b in d["bones"] if b["name"] == "左腕")["side"] = "R"
        self.assert_broken(bones.CANONICAL_FILE, wrong_side, "左腕 の名前の左右の字と side")

    def test_flag_must_be_bool(self):
        def int_flag(d):
            next(b for b in d["bones"] if b["name"] == "首")["standard"] = 1
        self.assert_broken(bones.CANONICAL_FILE, int_flag, "首 の standard・helper は true か false")

    def test_width_variant_duplicate(self):
        # 後ろに空白があるだけの「首 」は、比べる形が「首」と同じ
        self.assert_broken(bones.CANONICAL_FILE, lambda d: d["bones"].append(
            {"name": "首 ", "side": None, "group": "trunk", "standard": True, "helper": False}), "2回あります")

    def test_same_key_two_bones_in_one_table(self):
        self.assert_broken("aliases_vrm.json",
                           lambda d: d["tables"]["vrm"]["aliases"].update({"Hips ": "上半身"}),
                           "表 vrm の中で、同じ形 hips が")

    def test_alias_shadows_a_canonical_name(self):
        self.assert_broken("aliases_mixamo.json",
                           lambda d: d["tables"]["mixamo"]["aliases"].update({"左腕": "左ひじ"}),
                           "共通の名前 左腕 と同じ形の名前が、ほかの骨を指しています")

    def test_extra_overlaps_alias(self):
        self.assert_broken("aliases_mixamo.json",
                           lambda d: d["tables"]["mixamo"]["known_extras"].append("LeftArm"),
                           "known_extras と aliases の両方")

    def test_duplicate_table_name(self):
        self.assert_broken("aliases_mixamo.json",
                           lambda d: d["tables"].update({"vrm": {"aliases": {}, "known_extras": []}}),
                           "表の名前 vrm が、ほかの表と重なっています")

    def test_chain_with_unknown_bone(self):
        self.assert_broken(bones.CANONICAL_FILE, lambda d: d["chains"].update({"tail": ["しっぽ"]}),
                           "骨の並び tail の しっぽ は共通の名前にありません")

    def test_left_chain_with_centre_bone(self):
        self.assert_broken(bones.CANONICAL_FILE, lambda d: d["chains"].update({"neck_L": ["首"]}),
                           "左の骨だけを書いてください")

    # 知っている骨の並びのまちがいは、知らない骨を「聞かなくてよい骨」として隠してしまう（ユーザーに聞かれない）

    def test_extra_prefixes_given_as_a_string(self):
        # "J_Sec_" を [] で囲み忘れると、1字ずつ（j・_・s・e・c）の前置きになり、skirt なども隠れてしまう
        self.assert_broken("aliases_vrm.json",
                           lambda d: d["tables"]["vroid"].update({"known_extra_prefixes": "J_Sec_"}),
                           "表 vroid の known_extra_prefixes は、名前（文字列）のリスト")

    def test_empty_extra_prefix(self):
        # 空の前置きは、どの名前にも当たる
        for prefix in ("", "_", " . "):
            with self.subTest(prefix=prefix):
                self.assert_broken("aliases_vrm.json",
                                   lambda d: d["tables"]["vroid"].update({"known_extra_prefixes": ["J_Sec_", prefix]}),
                                   "表 vroid の known_extra_prefixes に、比べる形が空になる名前があります")

    def test_known_extras_given_as_a_string(self):
        self.assert_broken("aliases_vrm.json", lambda d: d["tables"]["vrm"].update({"known_extras": "upperChest"}),
                           "表 vrm の known_extras は、名前（文字列）のリスト")

    def test_known_extra_with_empty_key(self):
        self.assert_broken("aliases_vrm.json", lambda d: d["tables"]["vrm"].update({"known_extras": ["jaw", "_"]}),
                           "表 vrm の known_extras に、比べる形が空になる名前があります")

    def test_namespaces_must_be_a_list_of_names(self):
        self.assert_broken("aliases_mixamo.json",
                           lambda d: d["tables"]["mixamo"].update({"namespaces": "mixamorig"}),
                           "表 mixamo の namespaces は、名前（文字列）のリスト")
        self.assert_broken("aliases_mixamo.json",
                           lambda d: d["tables"]["mixamo"].update({"namespaces": ["mixamorig:"]}),
                           "表 mixamo の namespaces に、比べる形が空になる名前があります")

    # 揺れ物の名前の決まり（extras_sway.json）

    def test_sway_lists_must_be_lists_of_names(self):
        for key in ("substrings", "tokens"):
            for bad in ("hair", None, [1], {"a": "b"}):
                with self.subTest(key=key, bad=bad):
                    self.reset(bones.SWAY_FILE)
                    self.assert_broken(bones.SWAY_FILE, lambda d: d.update({key: bad}),
                                       f"extras_sway.json: {key} は、名前（文字列）のリスト")

    def test_sway_list_missing(self):
        self.assert_broken(bones.SWAY_FILE, lambda d: d.pop("tokens"), "extras_sway.json: tokens は、名前（文字列）のリスト")

    def test_sway_token_must_be_one_lowercase_word(self):
        # 数字・区切り・大文字の切れ目・日本語の字を含む語は、name_tokens() の語にならないので、黙って効かなくなる
        for bad in ("hair1", "j_sec", "Hair", "hairTie", "hair tie", "髪", "", "ｈａｉｒ"):
            with self.subTest(bad=bad):
                self.reset(bones.SWAY_FILE)
                self.assert_broken(bones.SWAY_FILE, lambda d: d["tokens"].append(bad),
                                   f"tokens の {re.escape(repr(bad))} は、英字の小文字（a〜z）だけの1語")

    def test_sway_substring_must_be_japanese_only(self):
        # 英字を字の並びとして比べると、forearm の中の ear のように人型の骨の名前にも当たる
        for bad in ("ear", "髪1", "髪_", "", "　", "ＩＫ"):
            with self.subTest(bad=bad):
                self.reset(bones.SWAY_FILE)
                self.assert_broken(bones.SWAY_FILE, lambda d: d["substrings"].append(bad),
                                   f"substrings の {re.escape(repr(bad))} は、日本語の字だけで書いてください")

    def test_sway_pattern_must_not_hit_a_humanoid_name(self):
        # 人型の骨の名前に当たる決まりは広すぎる（LeftForeArmTwist・左腕捩4 なども隠してしまう）
        cases = (
            ("substrings", "首", "揺れ物の名前の決まり（首）が、人型の骨の名前 首（mmd_semi_standard.json）に当たります"),
            ("substrings", "肘", "揺れ物の名前の決まり（肘）が、人型の骨の名前 左肘（aliases_mmd.json の表 mmd_ja_extras）"),
            ("tokens", "l", "揺れ物の名前の決まり（l）が、人型の骨の名前 目.L（mmd_semi_standard.json）"),
            ("tokens", "arm", "揺れ物の名前の決まり（arm）が、人型の骨の名前 "),
            ("tokens", "fore", "揺れ物の名前の決まり（fore）が、人型の骨の名前 "),
            ("tokens", "spine", "揺れ物の名前の決まり（spine）が、人型の骨の名前 "),
        )
        for key, value, message in cases:
            with self.subTest(key=key, value=value):
                self.reset(bones.SWAY_FILE)
                self.assert_broken(bones.SWAY_FILE, lambda d: d[key].append(value), re.escape(message))

    def test_exact_known_extra_matching_a_pattern_is_also_sway(self):
        """known_extras に字のまま載っている骨が、揺れ物の決まりにも当たるときは、知っている骨であり揺れ物の骨でもある
        （sway_extras は known_extras の中の印。入れ物は増えない）"""
        self.edit("aliases_vrm.json", lambda d: d["tables"]["vrm"]["known_extras"].append("hairRoot"))
        bones._dictionary.cache_clear()
        self.assertTrue(is_known_extra("hairRoot"))
        self.assertTrue(is_sway_name("hairRoot"))
        result = map_bones(["hairRoot", "upperChest", "謎の骨"])
        check_partition(self, ["hairRoot", "upperChest", "謎の骨"], result)
        self.assertEqual(result.known_extras, ("hairRoot", "upperChest"))
        self.assertEqual(result.sway_extras, ("hairRoot",))
        self.assertEqual(result.unknown, ("謎の骨",))

    def test_sway_list_is_data_driven(self):
        """揺れ物の名前は、プログラムを変えずに data/extras_sway.json に書き足せる"""
        self.assertFalse(is_sway_name("Hairpin_1"))
        self.assertFalse(is_sway_name("かんざし1"))
        self.assertEqual(map_bones(["Hairpin_1"]).unknown, ("Hairpin_1",))

        def add(d):
            d["tokens"].append("hairpin")
            d["substrings"].append("かんざし")
        self.edit(bones.SWAY_FILE, add)
        bones._dictionary.cache_clear()
        self.assertTrue(is_sway_name("Hairpin_1"))
        self.assertTrue(is_sway_name("かんざし1"))
        self.assertEqual(map_bones(["Hairpin_1"]).sway_extras, ("Hairpin_1",))
        # 決まりを空にすると、揺れ物の骨は無くなる（髪1 はユーザーに聞く）
        self.edit(bones.SWAY_FILE, lambda d: d.update({"substrings": [], "tokens": []}))
        bones._dictionary.cache_clear()
        self.assertFalse(is_sway_name("髪1"))
        self.assertEqual(map_bones(["髪1", "J_Sec_Hair1_01"]).unknown, ("髪1",))
        self.assertEqual(map_bones(["髪1", "J_Sec_Hair1_01"]).known_extras, ("J_Sec_Hair1_01",))
        self.assertEqual(map_bones(["髪1", "J_Sec_Hair1_01"]).sway_extras, ())


if __name__ == "__main__":
    unittest.main()
