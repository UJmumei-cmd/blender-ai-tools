"""A1 モデルの登録カード（bpy なし・numpy だけ）。

モデルごとに1回作ってファイルの中に保存し、あとの確かめ（A4・A5・E1・B5 など）が使い回す。カードに入れる物：

- 部品の種類（体・服・髪・アクセサリー・羽・その他・分からない）。名前から推測した物（kind_source "guess"）は
  ただの案で、ユーザーが一覧から選んで決める（"user"）。服の種類もユーザーが一覧（CLOTH_TYPES）から選ぶ。
  材質が2つ以上ある部品（MMD・PMX のモデルは、1つのメッシュに肌・髪・服の材質が入っていることが多い）は、
  材質ごとにも種類を持つ（MaterialPart。決め方は部品と同じ）。
  名前（オブジェクト・材質）は「データ」として決まった言葉と比べるだけで、指示としては読まない。
- 骨の名前の対応（A3 の map_bones の結果に、辞書に無い骨へのユーザーの答え bone_answers を足した物）。
  準標準の骨が足りないときは、ここに一覧で出す（CLAUDE.md 3章）。
- 測った値（背の高さ・脚の長さ・腕の下がり角・肩幅・頭身の目安）と関節の位置。休みの姿勢・モディファイアを
  通さない形で測る。測れない値は None にして理由を書く。数を作り出さない。
- 形と骨の指紋（sha256）。ユーザーが手で直すと指紋が変わり、「測り直しが必要」と分かる（remeasure_reasons）。
- 置き場所（placement：アーマチュアの matrix_world）。
- 素材の出所と利用条件（ユーザーが書く。書いた文字はそのまま残す）。

座標の決まり（カードの版 2）：
- 関節の位置・骨の位置（skeleton）・指紋は「モデルの中の座標」＝アーマチュアの中の座標（Blender の armature 空間）。
  モデル全体を動かす・Z まわりに回すだけでは、指紋も値も変わらない（置き場所が変わったことは、測り直しの理由ではなく
  別の知らせにする。check_against）。
- 長さ（背の高さ・脚・肩幅）は、置き方（placement）の回転と大きさを通した世界の長さ（メートル）。
  高さは世界の上向き（+Z）で測る：世界の +Z を置き方でアーマチュアの中の向きにした物（placement の3行目）を使う。
  そのため、FBX のように回して置いたアーマチュアでも、背の高さは世界の上向きの高さになる。
  置き方の平行移動と Z まわりの回転では、どの値も変わらない。
- 腕の下がり角は、アーマチュアの中の座標で、convert_pmx.rest_arm_angle とまったく同じ計算をする
  （アプリの restArmDownDegrees と同じ値になる）。

版 1 のカード（世界の座標で作った物）も読める。読めるだけで、変えられない（一度測り直す。check_against がそう言う）。

計算はここ（numpy の並びを受け取る）、Blender から並びを取り出すのは card_bpy.py。
画面と報告に出す文は TEXTS（texts パッケージの分野のファイルと同じ形）にまとめ、card_text() で言語を選んで作る。
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .. import texts, thresholds
from ..bones import BoneMapping, canonical_bone, canonical_bones, canonical_names, map_bones, mirror_name, normalize
from ..measure.geometry import as_array, as_points, as_triangles
from ..measure.results import Measurement, Signal

__all__ = [
    "ARM_REST_DOWN", "BODY_MEASUREMENTS_NAME", "BODY_SOURCE_ALL", "BODY_SOURCE_KIND", "BONE_MAPPING_KEYS",
    "BoneArrays", "CARD_CURRENT_NAME", "CLOTH_TYPES", "CLOTH_TYPE_IDS", "CLOTH_TYPE_LABELS", "EXTRA_ANSWER", "HEAD_RATIO",
    "HEAD_RATIO_METHOD", "HEIGHT", "KIND_SOURCES", "LEG_LENGTH", "LEGACY_SCHEMA_VERSION", "MATERIAL_NONE",
    "MEASUREMENT_NAMES", "MaterialPart", "MeshArrays", "ModelCard", "NON_BODY_KINDS", "PART_KINDS", "PART_KIND_LABELS",
    "PLACEMENT_CHANGED", "PLACEMENT_MOVED", "PLACEMENT_SAME", "Part", "REQUIRED_SOURCE_FIELDS", "SCHEMA_VERSION",
    "SEMI_STANDARD_NAME", "SHOULDER_WIDTH", "SOURCE_FIELDS", "SOURCE_LABELS", "SOURCE_TERMS_NAME", "SUPPORTED_SCHEMA_VERSIONS",
    "TEXTS", "UNCONFIRMED_PARTS_NAME", "UNKNOWN_BONES_NAME", "answerable_bones", "bone_mapping_to_json",
    "bones_fingerprint", "card_text", "carry_user_choices", "check_against", "empty_source", "fingerprints_of", "guess_material",
    "guess_part_kind",
    "keep_user_choices", "make_card", "map_with_answers", "measure_body", "placement_change", "remeasure_reasons",
    "same_model", "set_bone_answer", "set_material_kind", "set_part_kind", "set_source", "shape_fingerprint",
    "summarize_card",
]

# MARK: - 決まり

# カードの形の版。項目の意味を変えたら上げる。
#   1：位置と指紋が世界の座標（読めるだけ。変えられない）
#   2：位置と指紋がアーマチュアの中の座標。置き場所（placement）・骨の位置（skeleton）・骨の答え（bone_answers）・
#      材質ごとの種類（Part.materials）を足した
SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = (LEGACY_SCHEMA_VERSION, SCHEMA_VERSION)

# 部品の種類
PART_KINDS = ("body", "cloth", "hair", "accessory", "wing", "other", "unknown")
PART_KIND_LABELS = {
    "body": "体",
    "cloth": "服",
    "hair": "髪",
    "accessory": "アクセサリー",
    "wing": "羽・翼",
    "other": "その他",
    "unknown": "分からない",
}
# 体ではないと決まっている種類（材質がこの種類なら、背の高さを測るときに外す。"unknown" は外さない）
NON_BODY_KINDS = ("cloth", "hair", "accessory", "wing", "other")
# 種類を決めたのは誰か："guess" は名前からの推測（ただの案）、"user" はユーザーが選んだ物
KIND_SOURCES = ("guess", "user")

# 服の種類（ユーザーが一覧から選ぶ。名前からは決めない）
CLOTH_TYPES = (
    ("top", "上着"),
    ("shirt", "シャツ・ブラウス"),
    ("skirt", "スカート"),
    ("pants", "ズボン"),
    ("dress", "ワンピース"),
    ("socks", "靴下"),
    ("shoes", "靴"),
    ("gloves", "手袋"),
    ("hat", "帽子"),
    ("other", "その他"),
)
CLOTH_TYPE_IDS = tuple(type_id for type_id, _ in CLOTH_TYPES)
CLOTH_TYPE_LABELS = dict(CLOTH_TYPES)

# 素材の出所と利用条件（ユーザーが書く。書いた文字はそのまま残す）
SOURCE_FIELDS = ("origin", "author", "url", "terms")
SOURCE_LABELS = {"origin": "素材の出所", "author": "作者", "url": "URL", "terms": "利用条件"}
# これが空だと、カードのまとめ（source_terms）を黄にする
REQUIRED_SOURCE_FIELDS = ("origin", "terms")

# 骨の答え：共通の名前（MMD 準標準）のどれかか、これ（人型の骨ではない骨）
EXTRA_ANSWER = "extra"

# 材質の無い三角形（材質の枠が空・枠が無い）の、material_up_range のキー
MATERIAL_NONE = ""

# 置き場所の比べ（placement_change）の答え
PLACEMENT_SAME = "same"  # 同じ（再現性の幅の中）
PLACEMENT_MOVED = "moved"  # 平行移動・Z まわりの回転だけ（値は変わらない。知らせだけ）
PLACEMENT_CHANGED = "changed"  # 大きさ・傾き・裏返しが変わった（世界の長さ・高さが変わるので測り直す）

# 測る値の名前（measurements["values"] のキー）。長さはメートル、角度は度
HEIGHT = "height_m"
LEG_LENGTH = {"L": "leg_length_L_m", "R": "leg_length_R_m"}
ARM_REST_DOWN = {"L": "arm_rest_down_L_deg", "R": "arm_rest_down_R_deg"}
SHOULDER_WIDTH = "shoulder_width_m"
HEAD_RATIO = "head_ratio"
MEASUREMENT_NAMES = (HEIGHT, LEG_LENGTH["L"], LEG_LENGTH["R"], ARM_REST_DOWN["L"], ARM_REST_DOWN["R"],
                     SHOULDER_WIDTH, HEAD_RATIO)

# 測り方の説明（カードの中に記録として残す。カードの JSON が言語で変わらないように、日本語のまま持つ）
HEIGHT_METHOD = ("体の部品（材質ごとに種類があるときは、体ではない材質の三角形を外す）の、世界の上向き（+Z）での"
                 "いちばん上 − いちばん下（休みの姿勢・モディファイアを通さない形。アーマチュアの大きさを入れた長さ）")
LEG_METHOD = "足 → ひざ → 足首 の根元をつないだ長さ（アーマチュアの大きさを入れた、世界の長さ）"
LEG_D_METHOD = ("足D → ひざD → 足首D の根元をつないだ長さ（足・ひざ・足首 が無いので D の骨で測った。"
                "アーマチュアの大きさを入れた、世界の長さ）")
ARM_METHOD = ("アーマチュアの中の座標で、腕 の根元から ひじ の根元への向きの、水平からの下がり角"
              "（convert_pmx.py の rest_arm_angle と同じ式・同じ float32 の引き算。0.1 度に丸める）")
SHOULDER_METHOD = "左腕 の根元と 右腕 の根元の距離（アーマチュアの大きさを入れた、世界の長さ）"
HEAD_RATIO_METHOD = "頭の骨の根元から頭のてっぺんまで（世界の上向きで測る。目安。A2 の目印で直す）"

# 腕の下がり角を丸める桁（convert_pmx.rest_arm_angle が round(..., 1) で丸め、アプリの restArmDownDegrees になる）
APP_ANGLE_DIGITS = 1

# 測るのに使う骨（左。右は辞書の mirror_name で作る）
_LEG_BONES = ("左足", "左ひざ", "左足首")
_LEG_D_BONES = ("左足D", "左ひざD", "左足首D")
_UPPER_ARM = "左腕"
_ELBOW = "左ひじ"
_HEAD = "頭"

# measurements の中の、高さをどのメッシュで測ったか
BODY_SOURCE_KIND = "kind"  # 種類が体の部品
BODY_SOURCE_ALL = "all_meshes"  # 体の部品が無いので、全部のメッシュ

# bone_mapping（JSON の形）のキー
BONE_MAPPING_KEYS = ("mapping", "unknown", "known_extras", "conflicts", "ambiguous", "shadowed",
                     "missing_standard", "missing_semi_standard", "semi_standard_compliant")
_MEASUREMENT_KEYS = {
    1: ("space", "joints", "missing_joints", "mesh_z_range", "body_meshes", "body_meshes_source", "values"),
    2: ("space", "joints", "missing_joints", "mesh_up_range", "material_up_range", "excluded_materials",
        "body_meshes", "body_meshes_source", "values"),
}
_MEASUREMENT_SPACE = {1: "world", 2: "armature"}
_VALUE_KEYS = ("value", "unit", "method", "reason", "bones")
_PART_KEYS = {
    1: ("object_name", "kind", "kind_source", "cloth_type", "vertex_count", "triangle_count", "material_names",
        "matched_keyword"),
    2: ("object_name", "kind", "kind_source", "cloth_type", "vertex_count", "triangle_count", "material_names",
        "matched_keyword", "materials"),
}
_MATERIAL_KEYS = ("name", "kind", "kind_source", "cloth_type", "matched_keyword")
_CARD_KEYS = {
    1: ("schema_version", "blend_name", "armature_name", "parts", "bone_mapping", "measurements", "fingerprints",
        "source", "created_at", "tool_version", "blender_version"),
    2: ("schema_version", "blend_name", "armature_name", "parts", "bone_mapping", "bone_answers", "skeleton",
        "measurements", "fingerprints", "placement", "source", "created_at", "tool_version", "blender_version"),
}
_SKELETON_KEYS = ("names", "heads")

# 指紋の形の印（中身の作り方を変えたら v の数を上げる）。版 2 のカードも同じ作り方（入れる座標の空間だけが違う）。
# 版 1 のカードと同じモデルかを確かめる（世界の座標で作った骨の指紋と比べる）ために、作り方は変えない
_SHAPE_TAG = "blender_ai_tools.card.shape.v1"
_BONES_TAG = "blender_ai_tools.card.bones.v1"
# 指紋の位置をそろえる幅（再現性の幅。これより小さい差は、変化として数えない）
_QUANTUM = thresholds.REPRO_TOLERANCE_LENGTH
# 整数にした座標の上限（int64 に入る大きさより十分小さい）
_MAX_QUANTIZED = float(2 ** 62)
_SHA256 = re.compile(r"[0-9a-f]{64}")


# MARK: - 文

# 画面と報告に出す文（日本語が元。英語にも切り替える）。texts パッケージの分野のファイルと同じ形
# （キー "card.名前" → {"ja": 日本語, "en": 英語}。数字などは {name} の形で入れる）。
# texts/card.py に移して texts の _MODULES に足せば、card_text() はそちらを使う（中身はそのまま移せる）
TEXTS: dict[str, dict[str, str]] = {
    "card.bones_changed": {
        "ja": "骨（名前・並び・親子・位置・ねじれ・形を動かすかどうか）が変わりました",
        "en": "The bones changed (names, order, parents, positions, roll or deform flags)",
    },
    "card.shape_changed": {"ja": "{name} の形が変わりました", "en": "The shape of {name} changed"},
    "card.mesh_removed": {"ja": "{name} がなくなりました", "en": "{name} is gone"},
    "card.mesh_added": {"ja": "{name} が増えました", "en": "{name} was added"},
    "card.armature_changed": {
        "ja": "アーマチュアが変わりました（{before} → {after}）",
        "en": "The armature changed ({before} → {after})",
    },
    "card.no_card": {
        "ja": "登録カードがありません。先にモデルを登録してください",
        "en": "There is no registration card. Please register the model first",
    },
    "card.unreadable_card": {"ja": "登録カードが読めません：{error}", "en": "The registration card cannot be read: {error}"},
    "card.unreadable_model": {"ja": "いまのモデルを読めません：{error}", "en": "The current model cannot be read: {error}"},
    "card.legacy_card": {
        "ja": "古い形（版 {version}）の登録カードです。位置が世界の座標で入っているので、一度測り直してください",
        "en": "This registration card has an old format (version {version}) with world positions. "
              "Please measure the model once more",
    },
    "card.placement_moved": {
        "ja": "置き場所が変わりました（形はそのまま）",
        "en": "The model was moved (its shape is unchanged)",
    },
    "card.placement_resized": {
        "ja": "モデルの大きさか傾き（アーマチュアの置き方）が変わりました。背の高さと長さを測り直してください",
        "en": "The size or tilt of the model (the armature placement) changed. Please measure the height and "
              "lengths again",
    },
    "card.choices_not_carried": {
        "ja": "前のカード（アーマチュア {armature}）が、いまのモデルと同じモデルの物か確かめられない"
              "（アーマチュアの名前か骨が違う）ので、部品の種類・骨の答え・素材の出所と利用条件を写しませんでした",
        "en": "The previous card (armature {armature}) may belong to a different model (the armature name or the "
              "bones differ), so part kinds, bone answers and the source and terms of use were not copied",
    },
    "card.part_choice_dropped": {
        "ja": "部品 {name} がなくなったので、選んだ種類を写しませんでした",
        "en": "The part {name} is gone, so its chosen kind was not copied",
    },
    "card.material_choice_dropped": {
        "ja": "部品 {name} の材質 {material} がなくなったので、選んだ種類を写しませんでした",
        "en": "The material {material} of the part {name} is gone, so its chosen kind was not copied",
    },
    "card.answer_dropped": {
        "ja": "骨 {bone} の答え（{answer}）は、いまの辞書では使えないので外しました",
        "en": "The answer for the bone {bone} ({answer}) cannot be used with the current dictionary, so it was removed",
    },
    "card.semi_standard_missing": {
        "ja": "MMD 準標準の骨が足りないか、決まらない骨があります（その骨が要る確かめは灰になります）",
        "en": "Some MMD semi-standard bones are missing or undecided (checks that need them become gray)",
    },
    "card.unknown_bones": {
        "ja": "辞書に無い骨・1つに決まらない骨があります（ユーザーに聞きます）",
        "en": "Some bones are not in the dictionary or are undecided (the user is asked)",
    },
    "card.no_parts": {"ja": "部品（メッシュ）が1つもありません", "en": "There are no parts (meshes)"},
    "card.unconfirmed_parts": {
        "ja": "種類（服なら服の種類も）をまだ決めていない部品か材質があります",
        "en": "Some parts or materials have no confirmed kind yet (or no clothing type for clothes)",
    },
    "card.unmeasured_values": {"ja": "測れなかった値があります：{names}", "en": "Some values could not be measured: {names}"},
    "card.all_meshes_height": {
        "ja": "体の部品が決まっていないので、背の高さを全部のメッシュで測りました",
        "en": "No part is marked as body, so the height was measured with all meshes",
    },
    "card.source_missing": {
        "ja": "素材の出所と利用条件が書かれていません：{fields}",
        "en": "The source and terms of use are not written: {fields}",
    },
    "card.stale": {"ja": "測り直しが必要です：{reasons}", "en": "The card needs to be measured again: {reasons}"},
    "card.source_origin": {"ja": "素材の出所", "en": "Source"},
    "card.source_author": {"ja": "作者", "en": "Author"},
    "card.source_url": {"ja": "URL", "en": "URL"},
    "card.source_terms": {"ja": "利用条件", "en": "Terms of use"},
    "card.list_separator": {"ja": "、", "en": ", "},
    "card.sentence_separator": {"ja": "。", "en": ". "},
}


def card_text(key: str, **values: Any) -> str:
    """文を、いまの言語（texts.current_language）で作る。texts に同じキーがあれば、そちらを使う。
    文が無い言語のときは日本語。キーが無い・入れる値が合わないときは KeyError・ValueError（texts.text と同じ）"""
    if key in texts.TEXTS:
        return texts.text(key, **values)
    entry = TEXTS[key]
    template = entry.get(texts.current_language()) or entry[texts.FALLBACK_LANGUAGE]
    needed = texts.placeholders(template)
    if needed != set(values):
        raise ValueError(f"{key}: 入れる値が合いません（いる：{sorted(needed)}、渡した：{sorted(values)}）")
    return template.format(**values)


# MARK: - 入力の確かめ


def _describe(value: Any) -> str:
    """エラーの文に入れる、値の短い説明"""
    text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return f"{text}（{type(value).__name__}）"


def _is_int(value: Any) -> bool:
    """整数か（bool は入れない）"""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _is_number(value: Any) -> bool:
    """有限の数か（bool は入れない）"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return math.isfinite(float(value))


def _names(value: Any, what: str) -> tuple[str, ...]:
    """名前（文字）の並びにする。1つの文字列は受け付けない"""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{what} は名前（文字）の並びにしてください（いまは {_describe(value)}）")
    names = tuple(value)
    for name in names:
        if not isinstance(name, str):
            raise ValueError(f"{what} の中に文字でない物があります（{_describe(name)}）")
    return names


def _placement_array(value: Any) -> np.ndarray:
    """置き方（アーマチュアの matrix_world）を (4, 4) の float64 にする。None は単位行列（世界 = アーマチュアの中）。
    形が違う・NaN・無限大は ValueError"""
    if value is None:
        return np.eye(4)
    array = as_array(value, "placement")
    if array.dtype.kind not in "iuf" or array.shape != (4, 4):
        raise ValueError(f"placement は (4, 4) の数の並びにしてください（いまの型・形：{array.dtype}・{array.shape}）")
    array = np.array(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("placement に NaN か無限大が入っています")
    return array


# MARK: - 部品の種類の推測


# 種類ごとの言葉（上にある種類ほど先に決める）。
#   日本語の言葉：名前の中にあれば当たる（"前髪" は "髪" で当たる）。
#   英語の言葉：名前を英語の単語に分け（"FrontHair" → front・hair、"Hair_001" → hair）、単語がその言葉か、
#     その後ろに s・es を付けた物と同じなら当たる（"Shoes" は shoe。"Hatsune" は hat に当たらない）。
# 順の理由：
#   - アクセサリーを髪より先にする（"髪飾り"・"ヘアピン" は髪ではなくアクセサリー）
#   - 服を羽・体より先にする（"羽織"・"体操服"・"肌着" は服）
#   - 体をいちばん後にする（"体" は1字なので、ほかの言葉の中によく出てくる）
_KEYWORDS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("accessory",
     ("髪飾り", "首飾り", "飾り", "リボン", "アクセ", "イヤリング", "ピアス", "ネックレス", "ヘアピン", "ヘアバンド",
      "髪留め", "カチューシャ"),
     ("accessory", "accessories", "ribbon", "earring", "necklace", "hairpin", "hairband")),
    ("cloth",
     ("服", "スカート", "シャツ", "ブラウス", "ワンピ", "ズボン", "パンツ", "上着", "靴下", "靴", "手袋", "帽子",
      "衣装", "制服", "ジャケット", "コート", "ブーツ", "ソックス", "ニーソ", "タイツ", "ストッキング", "ドレス",
      "羽織", "着物", "袴", "肌着", "エプロン"),
     ("cloth", "clothes", "skirt", "shirt", "dress", "shoe", "sock", "glove", "hat", "pants", "jacket", "coat",
      "boot", "stocking", "tights", "costume", "uniform", "outfit", "trousers", "shorts")),
    ("wing", ("翼", "羽", "ウイング", "ウィング"), ("wing",)),
    ("hair", ("髪", "ヘア", "アホ毛"), ("hair",)),
    ("body", ("素体", "ボディ", "体", "肌", "顔"), ("body", "skin", "face")),
)
# 英語の単語に分ける（"TShirt" → T・Shirt、"HAIRBack" → HAIR・Back、"hair2" → hair）
_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")


def _match_keyword(text: str) -> tuple[str, str] | None:
    """名前1つから (種類, 当たった言葉)。どれにも当たらなければ None"""
    normalized = unicodedata.normalize("NFKC", text)
    words = {word.lower() for word in _WORD.findall(normalized)}
    lowered = normalized.lower()
    for kind, japanese, english in _KEYWORDS:
        for keyword in japanese:
            if keyword in lowered:
                return kind, keyword
        for keyword in english:
            if keyword in words or keyword + "s" in words or keyword + "es" in words:
                return kind, keyword
    return None


def guess_part_kind(object_name: str, material_names: Iterable[str] = ()) -> tuple[str, str | None]:
    """オブジェクトの名前と材質の名前から、部品の種類を推測する。返す物は (種類, 当たった言葉 | None)。

    これはただの案（kind_source "guess"）。ユーザーが一覧から選んで決める。
    名前はデータとして、決まった言葉（_KEYWORDS）と比べるだけ。名前に何が書いてあっても、指示としては読まない。
    当たった言葉は、いつも _KEYWORDS の中の言葉（名前の一部をそのまま返すことはない）。
    材質1つの種類を推測するときも、これに材質の名前を object_name として渡す。

    1. オブジェクトの名前が言葉に当たれば、その種類。
    2. 当たらなければ、材質の名前を見る。当たった材質の種類が1つならその種類。
       いくつもあるときは、体があれば体（MMD のモデルは、1つのメッシュに肌・髪・服の材質が入っていることが多い）、
       無ければ、当たった材質がいちばん多い種類（同じ数なら決めずに "unknown"）。
    3. どれにも当たらなければ ("unknown", None)。
    """
    if not isinstance(object_name, str):
        raise ValueError(f"object_name は文字にしてください（いまは {_describe(object_name)}）")
    materials = _names(material_names, "material_names")
    found = _match_keyword(object_name)
    if found is not None:
        return found
    hits: dict[str, list[str]] = {}
    for material in materials:
        match = _match_keyword(material)
        if match is not None:
            hits.setdefault(match[0], []).append(match[1])
    if not hits:
        return "unknown", None
    if "body" in hits:
        return "body", hits["body"][0]
    most = max(len(keywords) for keywords in hits.values())
    best = [kind for kind, keywords in hits.items() if len(keywords) == most]
    if len(best) != 1:
        return "unknown", None
    return best[0], hits[best[0]][0]


# MARK: - 部品と材質


def _check_kind(kind: Any) -> str:
    if kind not in PART_KINDS:
        raise ValueError(f"部品の種類は {'・'.join(PART_KINDS)} のどれかにしてください（いまは {_describe(kind)}）")
    return kind


def _check_cloth_type(kind: str, cloth_type: Any) -> str | None:
    if cloth_type is None:
        return None
    if cloth_type not in CLOTH_TYPE_IDS:
        raise ValueError(f"服の種類は {'・'.join(CLOTH_TYPE_IDS)} のどれかにしてください（いまは {_describe(cloth_type)}）")
    if kind != "cloth":
        raise ValueError(f"服の種類（{cloth_type}）を付けられるのは、種類が服（cloth）の物だけです（いまの種類：{kind}）")
    return cloth_type


def _check_choice(owner: str, kind: Any, kind_source: Any, cloth_type: Any, matched_keyword: Any) -> None:
    """種類・決めた人・服の種類・当たった言葉を確かめる（部品と材質で同じ決まり）"""
    _check_kind(kind)
    if kind_source not in KIND_SOURCES:
        raise ValueError(f"{owner}：kind_source は \"guess\" か \"user\" にしてください（いまは {_describe(kind_source)}）")
    _check_cloth_type(kind, cloth_type)
    if matched_keyword is not None and not isinstance(matched_keyword, str):
        raise ValueError(f"{owner}：matched_keyword は文字か None にしてください")


def _confirmed(kind: str, kind_source: str, cloth_type: str | None) -> bool:
    """ユーザーが決め終わったか：種類を選び（"unknown" ではない）、服なら服の種類も選んだ"""
    if kind_source != "user" or kind == "unknown":
        return False
    return kind != "cloth" or cloth_type is not None


@dataclass(frozen=True)
class MaterialPart:
    """部品（メッシュ）の中の、材質1つの種類。材質が2つ以上ある部品だけが持つ。

    name: 材質の名前
    kind: 種類（PART_KINDS のどれか）
    kind_source: "guess"（名前からの推測。ただの案）か "user"（ユーザーが選んだ）
    cloth_type: 服の種類（CLOTH_TYPE_IDS のどれか）。ユーザーが選ぶまでは None。服でない材質はいつも None
    matched_keyword: 推測で当たった言葉（推測の記録。当たらなかったときは None）
    """

    name: str
    kind: str
    kind_source: str
    cloth_type: str | None
    matched_keyword: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"材質の name は空でない文字にしてください（いまは {_describe(self.name)}）")
        _check_choice(f"材質 {self.name}", self.kind, self.kind_source, self.cloth_type, self.matched_keyword)

    @property
    def is_confirmed(self) -> bool:
        """ユーザーが決め終わったか：種類を選び（"unknown" ではない）、服なら服の種類も選んだ"""
        return _confirmed(self.kind, self.kind_source, self.cloth_type)

    def to_data(self) -> dict[str, Any]:
        """JSON にする形"""
        return {"name": self.name, "kind": self.kind, "kind_source": self.kind_source, "cloth_type": self.cloth_type,
                "matched_keyword": self.matched_keyword}

    @classmethod
    def from_data(cls, data: Any) -> MaterialPart:
        """to_data() の形から作る。項目が違うときは ValueError"""
        if not isinstance(data, dict) or set(data) != set(_MATERIAL_KEYS):
            raise ValueError(f"材質の項目は {'・'.join(_MATERIAL_KEYS)} にしてください")
        return cls(**data)


def guess_material(name: str) -> MaterialPart:
    """材質の名前から、材質の種類を推測する（部品と同じ言葉の決まり。kind_source "guess"、服の種類は None）"""
    kind, keyword = guess_part_kind(name)
    return MaterialPart(name=name, kind=kind, kind_source="guess", cloth_type=None, matched_keyword=keyword)


@dataclass(frozen=True)
class Part:
    """モデルの部品（メッシュのオブジェクト1つ）。

    object_name: オブジェクトの名前
    kind: 種類（PART_KINDS のどれか）
    kind_source: "guess"（名前からの推測。ただの案）か "user"（ユーザーが選んだ）
    cloth_type: 服の種類（CLOTH_TYPE_IDS のどれか）。ユーザーが選ぶまでは None。服でない部品はいつも None
    vertex_count・triangle_count: 頂点と三角形の数
    material_names: 材質の名前（材質の枠の順。同じ名前は1つ。空の枠は入れない）
    matched_keyword: 推測で当たった言葉（推測の記録。当たらなかったときは None）
    materials: 材質ごとの種類（material_names と同じ順・同じ名前）。材質が2つ以上ある部品だけが持ち、
        1つ以下の部品は空（部品の種類が、そのまま材質の種類）
    """

    object_name: str
    kind: str
    kind_source: str
    cloth_type: str | None
    vertex_count: int
    triangle_count: int
    material_names: tuple[str, ...] = ()
    matched_keyword: str | None = None
    materials: tuple[MaterialPart, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.object_name, str) or not self.object_name:
            raise ValueError(f"部品の object_name は空でない文字にしてください（いまは {_describe(self.object_name)}）")
        _check_choice(self.object_name, self.kind, self.kind_source, self.cloth_type, self.matched_keyword)
        for name in ("vertex_count", "triangle_count"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{self.object_name}：{name} は 0 以上の整数にしてください（いまは {_describe(value)}）")
            object.__setattr__(self, name, int(value))
        names = _names(self.material_names, f"{self.object_name} の material_names")
        object.__setattr__(self, "material_names", names)
        if isinstance(self.materials, (str, bytes, Mapping)) or not isinstance(self.materials, Iterable):
            raise ValueError(f"{self.object_name}：materials は MaterialPart の並びにしてください")
        materials = tuple(self.materials)
        if not all(isinstance(item, MaterialPart) for item in materials):
            raise ValueError(f"{self.object_name}：materials の中は MaterialPart にしてください")
        if materials and (len(names) < 2 or tuple(item.name for item in materials) != names):
            raise ValueError(f"{self.object_name}：materials は、材質が2つ以上あるときだけ、material_names と同じ順・"
                             f"同じ名前で持ってください")
        object.__setattr__(self, "materials", materials)

    @property
    def is_confirmed(self) -> bool:
        """部品の種類を、ユーザーが決め終わったか（材質の種類は見ない。unconfirmed_materials を見る）"""
        return _confirmed(self.kind, self.kind_source, self.cloth_type)

    @property
    def unconfirmed_materials(self) -> tuple[str, ...]:
        """ユーザーがまだ決めていない材質の名前（materials の順）"""
        return tuple(item.name for item in self.materials if not item.is_confirmed)

    def material(self, name: str) -> MaterialPart:
        """名前で材質を引く。無い名前は ValueError"""
        for item in self.materials:
            if item.name == name:
                return item
        raise ValueError(f"部品 {self.object_name!r} に、材質 {name!r} の種類はありません"
                         f"（材質が1つ以下の部品は、部品の種類を選んでください）")

    def to_data(self, schema_version: int = SCHEMA_VERSION) -> dict[str, Any]:
        """JSON にする形（カードの版の形）"""
        data = {
            "object_name": self.object_name,
            "kind": self.kind,
            "kind_source": self.kind_source,
            "cloth_type": self.cloth_type,
            "vertex_count": self.vertex_count,
            "triangle_count": self.triangle_count,
            "material_names": list(self.material_names),
            "matched_keyword": self.matched_keyword,
        }
        if schema_version >= 2:
            data["materials"] = [item.to_data() for item in self.materials]
        elif self.materials:
            raise ValueError(f"{self.object_name}：版 {schema_version} のカードには、材質ごとの種類を入れられません")
        return data

    @classmethod
    def from_data(cls, data: Any, schema_version: int = SCHEMA_VERSION) -> Part:
        """to_data() の形から作る。項目が違うときは ValueError"""
        keys = _PART_KEYS.get(schema_version)
        if keys is None or not isinstance(data, dict) or set(data) != set(keys):
            raise ValueError(f"部品の項目は {'・'.join(keys or ())} にしてください")
        if not isinstance(data["material_names"], list):
            raise ValueError("部品の material_names は [] の形にしてください")
        values = dict(data)
        if "materials" in values:
            if not isinstance(values["materials"], list):
                raise ValueError("部品の materials は [] の形にしてください")
            values["materials"] = tuple(MaterialPart.from_data(item) for item in values["materials"])
        return cls(**values)


def _guess_part(mesh: MeshArrays) -> Part:
    """メッシュから、推測だけの部品を作る（材質が2つ以上なら、材質ごとの推測も）"""
    kind, keyword = guess_part_kind(mesh.name, mesh.material_names)
    materials = tuple(guess_material(name) for name in mesh.material_names) if len(mesh.material_names) >= 2 else ()
    return Part(object_name=mesh.name, kind=kind, kind_source="guess", cloth_type=None,
                vertex_count=len(mesh.vertices), triangle_count=len(mesh.triangles),
                material_names=mesh.material_names, matched_keyword=keyword, materials=materials)


# MARK: - 入力の並び


def _readonly(array: np.ndarray) -> np.ndarray:
    array = np.array(array, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class BoneArrays:
    """骨の休みの形の並び（model.extract_bpy.BoneRest と同じ中身。並びは armature.data.bones の順）。

    カードを作るときは、アーマチュアの中の座標（extract_bpy.bone_rest(..., space="armature")）で渡す。

    names: 骨の名前 (B,)。同じ名前は不可
    parents: 親の骨の番号 (B,) int64。根の骨は -1
    heads・tails: 骨の根元・先 (B, 3)
    deform: 形を動かす骨か (B,) bool
    matrices: 骨の休みの行列 (B, 4, 4)（列ベクトルに左から掛ける形。列 2 が骨の Z 軸＝ねじれの向き）
    """

    names: tuple[str, ...]
    parents: np.ndarray
    heads: np.ndarray
    tails: np.ndarray
    deform: np.ndarray
    matrices: np.ndarray

    def __post_init__(self) -> None:
        names = _names(self.names, "骨の名前")
        if len(set(names)) != len(names):
            raise ValueError("骨の名前に、同じ名前が2つあります")
        count = len(names)
        parents = as_array(self.parents, "parents")
        if parents.size == 0:
            parents = np.zeros(0, dtype=np.int64)
        if parents.dtype.kind not in "iu" or parents.shape != (count,):
            raise ValueError(f"parents は ({count},) の整数の並びにしてください（いまの型・形：{parents.dtype}・{parents.shape}）")
        parents = parents.astype(np.int64)
        if count and (parents.min() < -1 or parents.max() >= count):
            raise ValueError("parents に、骨の番号の外の番号があります（根の骨は -1）")
        heads = as_points(self.heads, "heads")
        tails = as_points(self.tails, "tails")
        if len(heads) != count or len(tails) != count:
            raise ValueError(f"heads・tails は ({count}, 3) の並びにしてください（いまの形：{heads.shape}・{tails.shape}）")
        deform = as_array(self.deform, "deform")
        if deform.size == 0:
            deform = np.zeros(0, dtype=bool)
        if deform.dtype.kind != "b" or deform.shape != (count,):
            raise ValueError(f"deform は ({count},) の真偽（bool）の並びにしてください（いまの型・形：{deform.dtype}・{deform.shape}）")
        matrices = as_array(self.matrices, "matrices")
        if matrices.size == 0:
            matrices = np.zeros((0, 4, 4))
        if matrices.dtype.kind not in "iuf" or matrices.shape != (count, 4, 4):
            raise ValueError(f"matrices は ({count}, 4, 4) の数の並びにしてください（いまの形：{matrices.shape}）")
        matrices = np.asarray(matrices, dtype=np.float64)
        if not np.isfinite(matrices).all():
            raise ValueError("matrices に NaN か無限大が入っています")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "parents", _readonly(parents))
        object.__setattr__(self, "heads", _readonly(heads))
        object.__setattr__(self, "tails", _readonly(tails))
        object.__setattr__(self, "deform", _readonly(deform))
        object.__setattr__(self, "matrices", _readonly(matrices))


@dataclass(frozen=True, eq=False)
class MeshArrays:
    """メッシュ1つの休みの形（モディファイアを通さない形。座標は骨と同じ空間）。

    name: オブジェクトの名前
    vertices: 頂点 (N, 3)
    triangles: 三角形 (M, 3) int64
    material_names: 材質の名前（材質の枠の順。同じ名前は1つ。空の枠は入れない）
    triangle_materials: 三角形ごとの材質 (M,) int64。material_names の番号で、材質の無い三角形は -1。
        材質が1つ以下なら省いてよい（0 個：全部 -1、1 個：全部 0）。2つ以上あるときは必ず渡す
        （どの三角形がどの材質か分からないと、材質ごとの種類で高さを測れないため）
    """

    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    material_names: tuple[str, ...] = ()
    triangle_materials: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"メッシュの名前は空でない文字にしてください（いまは {_describe(self.name)}）")
        vertices = as_points(self.vertices, f"{self.name} の vertices")
        triangles = as_triangles(self.triangles, len(vertices), f"{self.name} の triangles")
        names = _names(self.material_names, f"{self.name} の material_names")
        if len(set(names)) != len(names) or MATERIAL_NONE in names:
            raise ValueError(f"{self.name} の material_names に、同じ名前か空の名前があります")
        if self.triangle_materials is None:
            if len(names) >= 2:
                raise ValueError(f"{self.name}：材質が2つ以上あるので、triangle_materials（三角形ごとの材質の番号）を"
                                 f"渡してください")
            materials = np.full(len(triangles), 0 if names else -1, dtype=np.int64)
        else:
            materials = as_array(self.triangle_materials, f"{self.name} の triangle_materials")
            if materials.size == 0:
                materials = np.zeros(0, dtype=np.int64)
            if materials.dtype.kind not in "iu" or materials.shape != (len(triangles),):
                raise ValueError(f"{self.name} の triangle_materials は ({len(triangles)},) の整数の並びにしてください"
                                 f"（いまの型・形：{materials.dtype}・{materials.shape}）")
            materials = materials.astype(np.int64)
            if materials.size and (materials.min() < -1 or materials.max() >= len(names)):
                raise ValueError(f"{self.name} の triangle_materials に、材質の番号の外の番号があります"
                                 f"（材質の無い三角形は -1）")
        object.__setattr__(self, "vertices", _readonly(vertices))
        object.__setattr__(self, "triangles", _readonly(triangles))
        object.__setattr__(self, "material_names", names)
        object.__setattr__(self, "triangle_materials", _readonly(materials))


# MARK: - 指紋


def _put_text(digest: Any, text: str) -> None:
    """文字を、長さを前に付けて入れる（つなぎ目で別の文字の並びと同じにならないように）"""
    data = text.encode("utf-8")
    digest.update(len(data).to_bytes(8, "little"))
    digest.update(data)


def _put_ints(digest: Any, array: np.ndarray) -> None:
    """整数の並びを、形を前に付けて入れる（little endian の int64）"""
    values = np.ascontiguousarray(array, dtype="<i8")
    digest.update(len(values.shape).to_bytes(8, "little"))
    for length in values.shape:
        digest.update(int(length).to_bytes(8, "little"))
    digest.update(values.tobytes())


def _quantize(points: np.ndarray) -> np.ndarray:
    """位置を _QUANTUM（再現性の幅）の何倍かの整数にする（いちばん近い整数。ちょうど半分なら偶数の方）"""
    scaled = np.rint(np.asarray(points, dtype=np.float64) / _QUANTUM)
    if scaled.size and float(np.abs(scaled).max()) >= _MAX_QUANTIZED:
        raise ValueError("座標が大きすぎて、指紋を作れません")
    return scaled.astype(np.int64)


def _new_digest(tag: str) -> Any:
    digest = hashlib.sha256()
    _put_text(digest, tag)
    _put_text(digest, repr(float(_QUANTUM)))  # 幅を変えたら、別の指紋になる
    return digest


def shape_fingerprint(name: str, vertices: Any, triangles: Any) -> str:
    """メッシュの形の指紋（sha256 の16進）。名前・頂点の数・三角形の並び・頂点の位置から作る。

    位置は thresholds.REPRO_TOLERANCE_LENGTH の何倍かの整数にそろえてから入れる。そのため、それよりずっと小さい
    差（計算の誤差）では変わらない。ただし、そろえる境い目のすぐ近くの点は、ごく小さな差でも隣の整数になり、
    指紋が変わることがある。そのときは「測り直しが必要」になる（安全な向きのまちがい）。
    同じ入力からは、いつも同じ指紋になる。位置の空間は呼ぶ側が決める（カードの版 2 はアーマチュアの中の座標）。
    """
    if not isinstance(name, str):
        raise ValueError(f"name は文字にしてください（いまは {_describe(name)}）")
    v = as_points(vertices, "vertices")
    t = as_triangles(triangles, len(v))
    digest = _new_digest(_SHAPE_TAG)
    _put_text(digest, name)
    _put_ints(digest, np.array([len(v), len(t)]))
    _put_ints(digest, t)
    _put_ints(digest, _quantize(v))
    return digest.hexdigest()


def _roll_points(heads: np.ndarray, tails: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    """骨のねじれ（ロール）を表す点：根元 + 骨の Z 軸の向き × 骨の長さ。
    向きを長さ（メートル）にして、位置と同じ幅でそろえられるようにする"""
    z_axes = matrices[:, :3, 2]
    norms = np.linalg.norm(z_axes, axis=1)
    unit = np.zeros_like(z_axes)
    good = norms > 0.0
    unit[good] = z_axes[good] / norms[good, None]
    lengths = np.linalg.norm(tails - heads, axis=1)
    return heads + unit * lengths[:, None]


def bones_fingerprint(names: Sequence[str], parents: Any, heads: Any, tails: Any, deform: Any,
                      matrices: Any) -> str:
    """骨の指紋（sha256 の16進）。骨の名前と並び・親子・根元と先の位置・ねじれ（Z 軸）・形を動かすかどうかから作る。

    位置とねじれの点は shape_fingerprint と同じ幅でそろえる（境い目の近くの決まりも同じ）。
    骨の並びを変えるだけでも変わる（ウェイトの骨の番号が変わるため）。位置の空間は呼ぶ側が決める。
    """
    bones = BoneArrays(names=tuple(_names(names, "names")), parents=parents, heads=heads, tails=tails,
                       deform=deform, matrices=matrices)
    return _bones_digest(bones)


def _bones_digest(bones: BoneArrays) -> str:
    digest = _new_digest(_BONES_TAG)
    _put_ints(digest, np.array([len(bones.names)]))
    for name in bones.names:
        _put_text(digest, name)
    _put_ints(digest, bones.parents)
    _put_ints(digest, _quantize(bones.heads))
    _put_ints(digest, _quantize(bones.tails))
    _put_ints(digest, _quantize(_roll_points(bones.heads, bones.tails, bones.matrices)))
    _put_ints(digest, bones.deform.astype(np.int64))
    return digest.hexdigest()


def fingerprints_of(bones: BoneArrays, meshes: Sequence[MeshArrays]) -> dict[str, Any]:
    """カードに入れる指紋：{"shapes": {メッシュの名前: 指紋}, "bones": 指紋}"""
    if not isinstance(bones, BoneArrays):
        raise ValueError(f"bones は BoneArrays にしてください（いまは {_describe(bones)}）")
    meshes = _check_meshes(meshes)
    return {
        "shapes": {mesh.name: shape_fingerprint(mesh.name, mesh.vertices, mesh.triangles) for mesh in meshes},
        "bones": _bones_digest(bones),
    }


def _check_meshes(meshes: Any) -> tuple[MeshArrays, ...]:
    if isinstance(meshes, (str, bytes, Mapping)) or not isinstance(meshes, Iterable):
        raise ValueError("meshes は MeshArrays の並びにしてください")
    items = tuple(meshes)
    for mesh in items:
        if not isinstance(mesh, MeshArrays):
            raise ValueError(f"meshes の中は MeshArrays にしてください（いまは {_describe(mesh)}）")
    names = [mesh.name for mesh in items]
    if len(set(names)) != len(names):
        raise ValueError("meshes に、同じ名前のメッシュが2つあります")
    return items


# MARK: - 骨の名前の対応と、ユーザーの答え


def answerable_bones(names: Iterable[str]) -> tuple[str, ...]:
    """ユーザーに聞く（答えを覚えてよい）骨の名前：辞書だけでは決まらない骨（辞書に無い・1つに決まらない・
    取り合いになった骨）。並びは入力の順。辞書で決まった骨・知っている骨（揺れ物など）は入らない"""
    unique = list(dict.fromkeys(_names(names, "names")))
    raw = map_bones(unique)
    claimed = {name for claimants in raw.conflicts.values() for name in claimants}
    wanted = set(raw.unknown) | claimed
    return tuple(name for name in unique if name in wanted)


def _answer_problem(raw: BoneMapping, names: frozenset[str], name: str, answer: str,
                    others: Mapping[str, str]) -> str | None:
    """答え1つが使えない理由（日本語。エラーの文に使う）。使えるときは None。

    answer は "extra" か、共通の名前の字のまま（canonical_bone で直した後の形）。
    """
    if name not in names:
        return f"骨 {name!r} は、このモデルにありません"
    claimed = {other for claimants in raw.conflicts.values() for other in claimants}
    if name not in raw.unknown and name not in claimed:
        return f"骨 {name!r} は辞書で決まる骨なので、答えは要りません（聞くのは、辞書に無い骨と決まらない骨だけ）"
    if answer == EXTRA_ANSWER:
        return None
    bone = canonical_bone(answer) if isinstance(answer, str) else None
    if bone is None or bone.name != answer:
        return f"答え {answer!r} は、共通の名前（MMD 準標準）か {EXTRA_ANSWER!r} にしてください"
    winner = next((original for original, canonical in raw.mapping.items() if canonical == answer), None)
    if winner is not None:
        return f"{answer} には、もう骨 {winner!r} が対応しています"
    for other, other_answer in others.items():
        if other != name and other_answer == answer:
            return f"{answer} は、もう骨 {other!r} の答えにしています"
    return None


def _split_answers(names: Sequence[str], answers: Mapping[str, str]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """答えを、使える物と使えない物に分ける（名前の順に1つずつ確かめ、先に使えた答えを優先する）"""
    raw = map_bones(names)
    name_set = frozenset(names)
    valid: dict[str, str] = {}
    dropped: list[tuple[str, str]] = []
    for name in sorted(answers):
        answer = answers[name]
        if _answer_problem(raw, name_set, name, answer, valid) is None:
            valid[name] = answer
        else:
            dropped.append((name, answer))
    return valid, dropped


def _check_answers(names: Sequence[str], answers: Any) -> dict[str, str]:
    """答えの辞書を確かめる。使えない答えがあれば、理由を書いて ValueError"""
    if answers is None:
        return {}
    if not isinstance(answers, Mapping):
        raise ValueError("bone_answers は 骨の名前 → 答え の辞書にしてください")
    for name, answer in answers.items():
        if not isinstance(name, str) or not isinstance(answer, str):
            raise ValueError(f"bone_answers の名前と答えは文字にしてください（{_describe(name)} → {_describe(answer)}）")
    raw = map_bones(names)
    name_set = frozenset(names)
    for name in sorted(answers):
        problem = _answer_problem(raw, name_set, name, answers[name], answers)
        if problem is not None:
            raise ValueError(problem)
    return {name: answers[name] for name in sorted(answers)}


def map_with_answers(names: Iterable[str], answers: Mapping[str, str] | None = None) -> BoneMapping:
    """骨の名前を、辞書（map_bones）とユーザーの答えで共通の名前に対応させる。答えが無ければ map_bones と同じ。

    answers：元の名前 → 共通の名前（字のまま）か "extra"（人型の骨ではない）。使えない答え（_answer_problem）は
    ValueError。答えた骨は unknown・ambiguous・取り合い（conflicts）から出て、共通の名前なら mapping に、
    "extra" なら known_extras に入る。答えで取り合いが解けた共通の名前は、答えた骨（無ければ残った1つ）を使い、
    ほかの骨は shadowed に入る。足りない骨の一覧（missing_*）も、答えを入れて作り直す。並びは map_bones と同じ決まり。
    """
    unique = list(dict.fromkeys(_names(names, "names")))
    checked = _check_answers(unique, answers)
    raw = map_bones(unique)
    if not checked:
        return raw
    order = {name: index for index, name in enumerate(canonical_names())}
    conflict_of = {name: canonical for canonical, claimants in raw.conflicts.items() for name in claimants}
    known = set(raw.known_extras)
    claims: dict[str, list[tuple[str, bool]]] = {}
    unknown: list[str] = []
    extras: list[str] = []
    for name in unique:
        answer = checked.get(name)
        if answer == EXTRA_ANSWER:
            extras.append(name)
            continue
        if answer is not None:
            claims.setdefault(answer, []).append((name, True))
            continue
        canonical = raw.mapping.get(name) or raw.shadowed.get(name) or conflict_of.get(name)
        if canonical is not None:
            claims.setdefault(canonical, []).append((name, False))
        elif name in known:
            extras.append(name)
        else:
            unknown.append(name)
    winners: dict[str, str] = {}
    shadowed_by: dict[str, str] = {}
    conflicts: dict[str, tuple[str, ...]] = {}
    for canonical in sorted(claims, key=order.__getitem__):
        claimants = claims[canonical]
        answered = [name for name, by_user in claimants if by_user]
        everyone = [name for name, _ in claimants]
        if len(answered) == 1:
            winner = answered[0]
        elif answered:
            conflicts[canonical] = tuple(everyone)
            continue
        elif len(everyone) == 1:
            winner = everyone[0]
        else:
            # map_bones と同じ決まり：共通の名前そのものの形の名前が1つだけなら、それを使う
            exact = [name for name in everyone if normalize(name) == normalize(canonical)]
            if len(exact) != 1:
                conflicts[canonical] = tuple(everyone)
                continue
            winner = exact[0]
        winners[winner] = canonical
        shadowed_by.update((name, canonical) for name in everyone if name != winner)
    present = set(winners.values()) | set(conflicts)
    bones = canonical_bones()
    return replace(
        raw,
        mapping={name: winners[name] for name in unique if name in winners},
        unknown=tuple(unknown),
        known_extras=tuple(extras),
        conflicts=conflicts,
        missing_standard=tuple(b.name for b in bones if b.standard and b.name not in present),
        missing_semi_standard=tuple(b.name for b in bones if not b.standard and not b.helper and b.name not in present),
        ambiguous={name: candidates for name, candidates in raw.ambiguous.items() if name not in checked},
        shadowed={name: shadowed_by[name] for name in unique if name in shadowed_by},
    )


def bone_mapping_to_json(mapping: BoneMapping) -> dict[str, Any]:
    """map_bones の結果を JSON の形にする（registry の bones.map_names と同じ形）"""
    if not isinstance(mapping, BoneMapping):
        raise ValueError(f"mapping は BoneMapping にしてください（いまは {_describe(mapping)}）")
    return {
        "mapping": dict(mapping.mapping),
        "unknown": list(mapping.unknown),
        "known_extras": list(mapping.known_extras),
        "conflicts": {canonical: list(names) for canonical, names in mapping.conflicts.items()},
        "ambiguous": {name: list(candidates) for name, candidates in mapping.ambiguous.items()},
        "shadowed": dict(mapping.shadowed),
        "missing_standard": list(mapping.missing_standard),
        "missing_semi_standard": list(mapping.missing_semi_standard),
        "semi_standard_compliant": mapping.is_semi_standard_compliant,
    }


# MARK: - 体を測る


def _entry(value: float | None, unit: str, method: str, reason: str, bones: Sequence[str]) -> dict[str, Any]:
    """測った値1つの形。値が無いときは、理由を必ず書く"""
    if value is None and not reason:
        raise ValueError("測れなかった値には、理由を書いてください")
    return {"value": None if value is None else float(value), "unit": unit, "method": method, "reason": reason,
            "bones": list(bones)}


def _side_names(names: Sequence[str], side: str) -> tuple[str, ...]:
    return tuple(names) if side == "L" else tuple(mirror_name(name) for name in names)


def _missing_text(names: Sequence[str]) -> str:
    return "・".join(names)


def _world_length(linear: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """アーマチュアの中の2点の、世界の距離（置き方の回転と大きさを通す。平行移動は距離に効かない）"""
    return float(np.linalg.norm(linear @ (np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64))))


def _leg_length(joints: Mapping[str, np.ndarray], side: str, linear: np.ndarray) -> dict[str, Any]:
    standard = _side_names(_LEG_BONES, side)
    d_chain = _side_names(_LEG_D_BONES, side)
    for chain, method, note in ((standard, LEG_METHOD, ""),
                                (d_chain, LEG_D_METHOD, f"{_missing_text([n for n in standard if n not in joints])} "
                                                        f"が無いので、D の骨で測りました")):
        if all(name in joints for name in chain):
            points = [joints[name] for name in chain]
            length = sum(_world_length(linear, a, b) for a, b in zip(points, points[1:]))
            return _entry(length, "m", method, note, chain)
    missing = [name for name in standard + d_chain if name not in joints]
    return _entry(None, "m", LEG_METHOD, f"脚の骨（{_missing_text(missing)}）が無いので測れません", standard)


def _arm_down_deg(upper: Any, lower: Any) -> float | None:
    """convert_pmx.rest_arm_angle と同じ計算：腕の根元からひじの根元への向きの、水平からの下がり角（度。0.1 度に丸める）。

    upper・lower はアーマチュアの中の座標（Blender の bone.head_local）。rest_arm_angle は mathutils の Vector
    （float32）で引き算するので、ここも float32 で引き算し、その後は Python の float（float64）で同じ式にする：
        direction = lower.head_local - upper.head_local
        horizontal = (direction.x ** 2 + direction.y ** 2) ** 0.5
        round(float(np.degrees(np.arctan2(-direction.z, horizontal))), 1)
    向きの長さが 0 のときは None（角度が決まらない。rest_arm_angle は 0.0 を返すが、ここでは測れないとする）。
    """
    direction = np.asarray(lower, dtype=np.float32) - np.asarray(upper, dtype=np.float32)
    dx, dy, dz = (float(component) for component in direction)
    horizontal = (dx ** 2 + dy ** 2) ** 0.5
    if horizontal == 0.0 and dz == 0.0:
        return None
    return round(float(np.degrees(np.arctan2(-dz, horizontal))), APP_ANGLE_DIGITS)


def _arm_rest_down(joints: Mapping[str, np.ndarray], side: str) -> dict[str, Any]:
    upper, elbow = _side_names((_UPPER_ARM, _ELBOW), side)
    missing = [name for name in (upper, elbow) if name not in joints]
    if missing:
        return _entry(None, "deg", ARM_METHOD, f"腕の骨（{_missing_text(missing)}）が無いので測れません", (upper, elbow))
    angle = _arm_down_deg(joints[upper], joints[elbow])
    if angle is None:
        return _entry(None, "deg", ARM_METHOD, f"{upper} と {elbow} の根元が同じ位置なので、向きが決まりません",
                      (upper, elbow))
    return _entry(angle, "deg", ARM_METHOD, "", (upper, elbow))


def _shoulder_width(joints: Mapping[str, np.ndarray], linear: np.ndarray) -> dict[str, Any]:
    left, right = _UPPER_ARM, mirror_name(_UPPER_ARM)
    missing = [name for name in (left, right) if name not in joints]
    if missing:
        return _entry(None, "m", SHOULDER_METHOD, f"腕の骨（{_missing_text(missing)}）が無いので測れません", (left, right))
    return _entry(_world_length(linear, joints[left], joints[right]), "m", SHOULDER_METHOD, "", (left, right))


def _range(values: np.ndarray) -> tuple[float, float] | None:
    return None if values.size == 0 else (float(values.min()), float(values.max()))


def _up_ranges(vertices: np.ndarray, mesh: MeshArrays | None,
               up: np.ndarray) -> tuple[tuple[float, float] | None, dict[str, tuple[float, float] | None] | None]:
    """メッシュの、世界の上向きでの高さの範囲（全体と、材質が2つ以上なら材質ごと）。

    高さは up（世界の +Z をアーマチュアの中の向きにした物。placement の3行目）との内積：
    世界の z − アーマチュアの原点の世界の z（メートル）。材質ごとの範囲は、その材質の三角形の頂点で作る。
    材質の無い三角形（-1）があれば、キー MATERIAL_NONE に入れる。
    """
    heights = vertices @ up
    whole = _range(heights)
    if mesh is None or len(mesh.material_names) < 2:
        return whole, None
    per_material: dict[str, tuple[float, float] | None] = {}
    for index, name in enumerate(mesh.material_names):
        used = np.unique(mesh.triangles[mesh.triangle_materials == index])
        per_material[name] = _range(heights[used])
    without = mesh.triangle_materials < 0
    if without.any():
        per_material[MATERIAL_NONE] = _range(heights[np.unique(mesh.triangles[without])])
    return whole, per_material


def _body_range(mesh_ranges: Mapping[str, Sequence[float] | None],
                material_ranges: Mapping[str, Mapping[str, Sequence[float] | None]],
                kinds: Sequence[tuple[str, str]],
                material_kinds: Mapping[str, Mapping[str, str]],
                ) -> tuple[list[str], str, dict[str, list[str]], tuple[float, float] | None]:
    """高さを測るメッシュ・その選び方・外した材質・(いちばん下, いちばん上)。頂点が1つも無ければ範囲は None。

    材質ごとの種類がある部品は、体ではないと決まった種類（NON_BODY_KINDS）の材質の三角形を外す
    （"unknown" の材質と、材質の無い三角形は外さない）。外す材質が無ければ、前と同じく部品の全部の頂点で測る。
    """
    body = [name for name, kind in kinds if kind == "body"]
    source = BODY_SOURCE_KIND if body else BODY_SOURCE_ALL
    used = body if body else [name for name, _ in kinds]
    excluded: dict[str, list[str]] = {}
    ranges: list[Sequence[float]] = []
    for name in used:
        materials = material_ranges.get(name)
        chosen = material_kinds.get(name, {})
        dropped = [material for material, kind in chosen.items() if kind in NON_BODY_KINDS]
        if materials is not None and dropped:
            excluded[name] = dropped
            ranges.extend(r for material, r in materials.items() if material not in dropped and r is not None)
        elif mesh_ranges.get(name) is not None:
            ranges.append(mesh_ranges[name])
    if not ranges:
        return used, source, excluded, None
    return used, source, excluded, (min(float(r[0]) for r in ranges), max(float(r[1]) for r in ranges))


def _plain_range(value: Sequence[float] | None) -> list[float] | None:
    return None if value is None else [float(c) for c in value]


def _measure(joints: Mapping[str, np.ndarray], mesh_ranges: Mapping[str, Sequence[float] | None],
             material_ranges: Mapping[str, Mapping[str, Sequence[float] | None]],
             kinds: Sequence[tuple[str, str]], material_kinds: Mapping[str, Mapping[str, str]],
             placement: np.ndarray) -> dict[str, Any]:
    """関節の位置（アーマチュアの中）・高さの範囲・部品と材質の種類・置き方から、measurements（JSON の形）を作る"""
    linear = placement[:3, :3]
    up = placement[2, :3]
    used, source, excluded, extent = _body_range(mesh_ranges, material_ranges, kinds, material_kinds)
    all_note = ("体の部品が決まっていないので、全部のメッシュで測りました（髪・帽子・服などを含むことがあります）"
                if source == BODY_SOURCE_ALL else "")
    values: dict[str, dict[str, Any]] = {}
    if extent is None:
        values[HEIGHT] = _entry(None, "m", HEIGHT_METHOD, "高さを測るメッシュ（体の材質）に、頂点がありません", ())
    else:
        values[HEIGHT] = _entry(extent[1] - extent[0], "m", HEIGHT_METHOD, all_note, ())
    for side in ("L", "R"):
        values[LEG_LENGTH[side]] = _leg_length(joints, side, linear)
    for side in ("L", "R"):
        values[ARM_REST_DOWN[side]] = _arm_rest_down(joints, side)
    values[SHOULDER_WIDTH] = _shoulder_width(joints, linear)
    if extent is None:
        values[HEAD_RATIO] = _entry(None, "ratio", HEAD_RATIO_METHOD, "背の高さを測れないので測れません", (_HEAD,))
    elif _HEAD not in joints:
        values[HEAD_RATIO] = _entry(None, "ratio", HEAD_RATIO_METHOD, f"骨 {_HEAD} が無いので測れません", (_HEAD,))
    else:
        head_length = extent[1] - float(np.dot(up, joints[_HEAD]))
        if head_length <= 0.0:
            values[HEAD_RATIO] = _entry(None, "ratio", HEAD_RATIO_METHOD,
                                        f"骨 {_HEAD} の根元が、体のいちばん上より下にありません", (_HEAD,))
        else:
            values[HEAD_RATIO] = _entry((extent[1] - extent[0]) / head_length, "ratio", HEAD_RATIO_METHOD, all_note,
                                        (_HEAD,))
    order = canonical_names()
    return {
        "space": "armature",
        "joints": {name: [float(c) for c in joints[name]] for name in order if name in joints},
        "missing_joints": [name for name in order if name not in joints],
        "mesh_up_range": {name: _plain_range(mesh_ranges.get(name)) for name, _ in kinds},
        "material_up_range": {name: {material: _plain_range(r) for material, r in ranges.items()}
                              for name, ranges in material_ranges.items()},
        "excluded_materials": excluded,
        "body_meshes": list(used),
        "body_meshes_source": source,
        "values": values,
    }


def _joints_of(names: Sequence[str], heads: Any, mapping: Mapping[str, str]) -> dict[str, np.ndarray]:
    """共通の名前 → 骨の根元の位置（対応が決まった骨だけ）。mapping は 元の名前 → 共通の名前"""
    index = {name: i for i, name in enumerate(names)}
    points = np.asarray(heads, dtype=np.float64)
    return {canonical: points[index[original]] for original, canonical in mapping.items()}


def _check_material_kinds(material_kinds: Any, material_ranges: Mapping[str, Mapping[str, Any]],
                          material_order: Mapping[str, Sequence[str]]) -> dict[str, dict[str, str]]:
    """材質の種類の辞書を確かめ、材質ごとの範囲がある部品の全部の材質を、材質の順にそろえる（無い物は "unknown"）"""
    if material_kinds is None:
        material_kinds = {}
    if not isinstance(material_kinds, Mapping):
        raise ValueError("material_kinds は メッシュの名前 → {材質の名前: 種類} の辞書にしてください")
    for mesh_name, chosen in material_kinds.items():
        if mesh_name not in material_ranges or not isinstance(chosen, Mapping):
            raise ValueError(f"material_kinds の {mesh_name!r} は、材質が2つ以上あるメッシュ（MeshArrays）の、"
                             f"材質の名前 → 種類 の辞書にしてください")
        unknown = [name for name in chosen if name not in material_order[mesh_name]]
        if unknown:
            raise ValueError(f"material_kinds の {mesh_name!r} に、メッシュに無い材質があります：{unknown}")
        for kind in chosen.values():
            _check_kind(kind)
    return {mesh_name: {name: material_kinds.get(mesh_name, {}).get(name, "unknown")
                        for name in material_order[mesh_name]}
            for mesh_name in material_ranges}


def measure_body(bone_names: Sequence[str], heads: Any, meshes: Mapping[str, Any],
                 part_kinds: Mapping[str, str], *, placement: Any = None,
                 material_kinds: Mapping[str, Mapping[str, str]] | None = None,
                 bone_answers: Mapping[str, str] | None = None) -> dict[str, Any]:
    """体を測る。返す物は measurements（JSON の形の辞書）。

    bone_names・heads: 骨の名前と根元の位置 (B, 3)（アーマチュアの中の座標・休みの姿勢）。名前は map_bones
        （bone_answers があれば、その答えも。map_with_answers）で共通の名前にする
    meshes: メッシュの名前 → 頂点 (N, 3)（同じ座標）か MeshArrays。並びの順を覚えておく。
        材質が2つ以上ある MeshArrays は、材質ごとの高さの範囲も作る
    part_kinds: メッシュの名前 → 部品の種類。無いメッシュは "unknown" とみなす
    placement: アーマチュアの置き方（matrix_world の (4, 4)）。None は単位行列（アーマチュアの中 = 世界）
    material_kinds: メッシュの名前 → {材質の名前: 種類}。無い材質は "unknown"
    bone_answers: 辞書に無い骨への答え（map_with_answers）

    measurements の中身：
        space: "armature"（関節の位置の空間）
        joints: 共通の名前 → 根元の位置 [x, y, z]（アーマチュアの中。対応が決まった骨だけ）
        missing_joints: 位置が分からない共通の名前（辞書の順。骨が無い・取り合いで決まらない）
        mesh_up_range: メッシュの名前 → [いちばん下, いちばん上]（世界の上向きの高さ。アーマチュアの原点から。
            頂点が無ければ None）
        material_up_range: 材質が2つ以上あるメッシュの名前 → {材質の名前: [下, 上] | None}（材質の無い三角形は ""）
        excluded_materials: 高さを測るときに外した材質（メッシュの名前 → 材質の名前。外さなかったメッシュは入れない）
        body_meshes: 高さを測ったメッシュ。body_meshes_source: "kind"（体の部品）か "all_meshes"（体の部品が無い）
        values: 名前（MEASUREMENT_NAMES）→ {"value", "unit", "method", "reason", "bones"}。測れないときは value が None で
            reason に理由がある。値はあるが気を付けること（全部のメッシュで測った など）も reason に書く
    測る値：
        height_m: 体の部品（体ではない材質の三角形を外す）の、世界の上向きでの いちばん上 − いちばん下
        leg_length_L_m・leg_length_R_m: |足 − ひざ| + |ひざ − 足首|（根元どうし。世界の長さ）。無ければ D の骨で
        arm_rest_down_L_deg・arm_rest_down_R_deg: 腕 → ひじ の向きの下がり角（アーマチュアの中で、
            convert_pmx.rest_arm_angle と同じ計算・丸め）
        shoulder_width_m: |左腕 − 右腕|（根元どうし。世界の長さ）
        head_ratio: 背の高さ ÷ (体のいちばん上 − 頭 の根元の高さ)。目安（HEAD_RATIO_METHOD）
    """
    names = _names(bone_names, "bone_names")
    if len(set(names)) != len(names):
        raise ValueError("bone_names に、同じ名前が2つあります")
    points = as_points(heads, "heads")
    if len(points) != len(names):
        raise ValueError(f"heads の数（{len(points)}）が、骨の名前の数（{len(names)}）と違います")
    if not isinstance(meshes, Mapping) or not isinstance(part_kinds, Mapping):
        raise ValueError("meshes と part_kinds は、メッシュの名前をキーにした辞書にしてください")
    unknown = [name for name in part_kinds if name not in meshes]
    if unknown:
        raise ValueError(f"part_kinds に、meshes に無いメッシュがあります：{unknown}")
    matrix = _placement_array(placement)
    up = matrix[2, :3]
    mesh_ranges: dict[str, tuple[float, float] | None] = {}
    material_ranges: dict[str, dict[str, tuple[float, float] | None]] = {}
    material_order: dict[str, tuple[str, ...]] = {}
    kinds: list[tuple[str, str]] = []
    for name, value in meshes.items():
        if not isinstance(name, str):
            raise ValueError(f"メッシュの名前は文字にしてください（いまは {_describe(name)}）")
        mesh = value if isinstance(value, MeshArrays) else None
        vertices = mesh.vertices if mesh is not None else as_points(value, f"{name} の頂点")
        mesh_ranges[name], per_material = _up_ranges(vertices, mesh, up)
        if per_material is not None:
            material_ranges[name] = per_material
            material_order[name] = mesh.material_names
        kinds.append((name, _check_kind(part_kinds.get(name, "unknown"))))
    chosen = _check_material_kinds(material_kinds, material_ranges, material_order)
    mapping = map_with_answers(names, bone_answers)
    return _measure(_joints_of(names, points, mapping.mapping), mesh_ranges, material_ranges, kinds, chosen, matrix)


def _remeasure_card(card: ModelCard, parts: Sequence[Part], bone_mapping: Mapping[str, Any] | None = None,
                    ) -> dict[str, Any]:
    """カードに入っている骨の位置・高さの範囲・置き方から、部品と材質の種類（と骨の対応）だけを変えて測り直す
    （形は読み直さない）"""
    mapping = (card.bone_mapping if bone_mapping is None else bone_mapping)["mapping"]
    skeleton = card.skeleton
    joints = _joints_of(skeleton["names"], skeleton["heads"], mapping)
    material_kinds = {part.object_name: {item.name: item.kind for item in part.materials}
                      for part in parts if part.materials}
    return _measure(joints, card.measurements["mesh_up_range"], card.measurements["material_up_range"],
                    [(part.object_name, part.kind) for part in parts], material_kinds, card.placement_matrix())


# MARK: - 置き場所


def placement_change(stored: Any, current: Any) -> str:
    """2つの置き方（アーマチュアの matrix_world）の違い：PLACEMENT_SAME・PLACEMENT_MOVED・PLACEMENT_CHANGED。

    - same：どの成分も再現性の幅の中（平行移動は REPRO_TOLERANCE_LENGTH、回転と大きさは REPRO_TOLERANCE_RATIO
      × 成分のいちばん大きい絶対値）。
    - moved：違うが、世界の長さと世界の上向きの高さが変わらない（平行移動と Z まわりの回転だけ）。
      見るのは、上向きの行（3行目）・長さの決まり（L^T L）・裏返しかどうか（行列式の符号）。カードの値はそのまま使える。
    - changed：大きさ・傾き・裏返しが変わった。世界の長さか高さが変わるので、測り直しが要る。
    """
    a = _placement_array(stored)
    b = _placement_array(current)
    la, lb = a[:3, :3], b[:3, :3]
    tiny = np.finfo(np.float64).tiny
    tolerance = thresholds.REPRO_TOLERANCE_RATIO * max(float(np.abs(la).max()), float(np.abs(lb).max()), tiny)
    if float(np.abs(la - lb).max()) <= tolerance and \
            float(np.abs(a[:3, 3] - b[:3, 3]).max()) <= thresholds.REPRO_TOLERANCE_LENGTH:
        return PLACEMENT_SAME
    metric_a, metric_b = la.T @ la, lb.T @ lb
    metric_tolerance = thresholds.REPRO_TOLERANCE_RATIO * max(float(np.abs(metric_a).max()),
                                                              float(np.abs(metric_b).max()), tiny)
    same_up = float(np.abs(la[2] - lb[2]).max()) <= tolerance
    same_metric = float(np.abs(metric_a - metric_b).max()) <= metric_tolerance
    same_hand = bool(np.sign(np.linalg.det(la)) == np.sign(np.linalg.det(lb)))
    return PLACEMENT_MOVED if same_up and same_metric and same_hand else PLACEMENT_CHANGED


# MARK: - カード


def _plain_json(value: Any, what: str) -> Any:
    """JSON にできる形の写し（JSON にして読み戻した物）。NaN・無限大・JSON にできない型は ValueError"""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{what} を JSON にできません：{error}") from error


def _check_string_list(value: Any, what: str) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{what} は文字のリストにしてください")


def _check_string_dict(value: Any, what: str, *, list_values: bool = False) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{what} は {{}} の形にしてください")
    for key, item in value.items():
        if list_values:
            _check_string_list(item, f"{what} の {key}")
        elif not isinstance(item, str):
            raise ValueError(f"{what} の {key} の値は文字にしてください")


def _check_bone_mapping(data: dict[str, Any]) -> None:
    if set(data) != set(BONE_MAPPING_KEYS):
        raise ValueError(f"bone_mapping の項目は {'・'.join(BONE_MAPPING_KEYS)} にしてください")
    _check_string_dict(data["mapping"], "bone_mapping.mapping")
    _check_string_dict(data["shadowed"], "bone_mapping.shadowed")
    _check_string_dict(data["conflicts"], "bone_mapping.conflicts", list_values=True)
    _check_string_dict(data["ambiguous"], "bone_mapping.ambiguous", list_values=True)
    for key in ("unknown", "known_extras", "missing_standard", "missing_semi_standard"):
        _check_string_list(data[key], f"bone_mapping.{key}")
    if not isinstance(data["semi_standard_compliant"], bool):
        raise ValueError("bone_mapping.semi_standard_compliant は true か false にしてください")


def _check_point(value: Any, length: int, what: str) -> None:
    if not isinstance(value, list) or len(value) != length or not all(_is_number(c) for c in value):
        raise ValueError(f"{what} は {length} つの有限の数のリストにしてください")


def _check_ranges(ranges: Any, names: Iterable[str], what: str, *, optional: Iterable[str] = ()) -> None:
    """名前 → [下, 上] | None の辞書。キーは names を全部と、optional のうちのいくつか"""
    required, allowed = set(names), set(names) | set(optional)
    if not isinstance(ranges, dict) or not required <= set(ranges) <= allowed:
        raise ValueError(f"{what} は、名前ごとの [下, 上] の辞書にしてください（名前が合いません）")
    for name, value in ranges.items():
        if value is not None:
            _check_point(value, 2, f"{what} の {name}")


def _check_measurements(data: dict[str, Any], parts: Sequence[Part], version: int) -> None:
    keys = _MEASUREMENT_KEYS[version]
    part_names = [part.object_name for part in parts]
    if set(data) != set(keys):
        raise ValueError(f"measurements の項目は {'・'.join(keys)} にしてください")
    if data["space"] != _MEASUREMENT_SPACE[version]:
        raise ValueError(f"版 {version} のカードの measurements.space は \"{_MEASUREMENT_SPACE[version]}\" にしてください")
    if not isinstance(data["joints"], dict):
        raise ValueError("measurements.joints は {} の形にしてください")
    for name, position in data["joints"].items():
        _check_point(position, 3, f"measurements.joints の {name}")
    _check_string_list(data["missing_joints"], "measurements.missing_joints")
    if version == LEGACY_SCHEMA_VERSION:
        _check_ranges(data["mesh_z_range"], part_names, "measurements.mesh_z_range")
    else:
        _check_ranges(data["mesh_up_range"], part_names, "measurements.mesh_up_range")
        material_ranges = data["material_up_range"]
        with_materials = {part.object_name: part for part in parts if part.materials}
        if not isinstance(material_ranges, dict) or set(material_ranges) != set(with_materials):
            raise ValueError("measurements.material_up_range は、材質ごとの種類がある部品だけの辞書にしてください")
        for name, ranges in material_ranges.items():
            _check_ranges(ranges, with_materials[name].material_names, f"measurements.material_up_range の {name}",
                          optional=(MATERIAL_NONE,))
        excluded = data["excluded_materials"]
        _check_string_dict(excluded, "measurements.excluded_materials", list_values=True)
        if not set(excluded) <= set(with_materials):
            raise ValueError("measurements.excluded_materials に、材質ごとの種類が無い部品があります")
    _check_string_list(data["body_meshes"], "measurements.body_meshes")
    if data["body_meshes_source"] not in (BODY_SOURCE_KIND, BODY_SOURCE_ALL):
        raise ValueError("measurements.body_meshes_source は \"kind\" か \"all_meshes\" にしてください")
    values = data["values"]
    if not isinstance(values, dict) or set(values) != set(MEASUREMENT_NAMES):
        raise ValueError(f"measurements.values の項目は {'・'.join(MEASUREMENT_NAMES)} にしてください")
    for name, entry in values.items():
        if not isinstance(entry, dict) or set(entry) != set(_VALUE_KEYS):
            raise ValueError(f"measurements.values.{name} の項目は {'・'.join(_VALUE_KEYS)} にしてください")
        if entry["value"] is not None and not _is_number(entry["value"]):
            raise ValueError(f"measurements.values.{name}.value は有限の数か null にしてください")
        if not all(isinstance(entry[key], str) for key in ("unit", "method", "reason")):
            raise ValueError(f"measurements.values.{name} の unit・method・reason は文字にしてください")
        if entry["value"] is None and not entry["reason"]:
            raise ValueError(f"measurements.values.{name}：測れなかった値には理由を書いてください")
        _check_string_list(entry["bones"], f"measurements.values.{name}.bones")


def _check_fingerprints(data: dict[str, Any], part_names: list[str]) -> None:
    if set(data) != {"shapes", "bones"}:
        raise ValueError("fingerprints の項目は shapes・bones にしてください")
    shapes = data["shapes"]
    if not isinstance(shapes, dict) or set(shapes) != set(part_names):
        raise ValueError("fingerprints.shapes は、部品ごとの指紋の辞書にしてください（部品の名前と合いません）")
    for name, value in [*shapes.items(), ("bones", data["bones"])]:
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"指紋（{name}）は16進の小文字 64 文字にしてください")


def _check_source(data: Any) -> dict[str, str]:
    if not isinstance(data, Mapping) or set(data) != set(SOURCE_FIELDS):
        raise ValueError(f"source の項目は {'・'.join(SOURCE_FIELDS)} にしてください")
    for key in SOURCE_FIELDS:
        if not isinstance(data[key], str):
            raise ValueError(f"source.{key} は文字にしてください（いまは {_describe(data[key])}）")
    return {key: data[key] for key in SOURCE_FIELDS}


def _check_time(value: Any) -> None:
    if not isinstance(value, str):
        raise ValueError(f"created_at は ISO-8601 の時刻の文字にしてください（いまは {_describe(value)}）")
    try:
        moment = datetime.datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"created_at が ISO-8601 の時刻ではありません（いまは {value!r}）") from None
    if moment.tzinfo is None:
        raise ValueError(f"created_at に時差（Z など）が付いていません（いまは {value!r}）")


def _check_skeleton(data: Any) -> dict[str, Any]:
    """skeleton（骨の名前と根元の位置。アーマチュアの中）を確かめる"""
    if not isinstance(data, dict) or set(data) != set(_SKELETON_KEYS):
        raise ValueError(f"skeleton の項目は {'・'.join(_SKELETON_KEYS)} にしてください")
    _check_string_list(data["names"], "skeleton.names")
    if len(set(data["names"])) != len(data["names"]):
        raise ValueError("skeleton.names に、同じ名前が2つあります")
    heads = data["heads"]
    if not isinstance(heads, list) or len(heads) != len(data["names"]):
        raise ValueError("skeleton.heads は、骨の名前と同じ数の [x, y, z] のリストにしてください")
    for name, head in zip(data["names"], heads):
        _check_point(head, 3, f"skeleton.heads の {name}")
    return data


def _check_answer_values(answers: Any, names: Sequence[str]) -> dict[str, str]:
    """bone_answers の形を確かめる（骨はカードにある名前、答えは "extra" か共通の名前の字のまま）。
    辞書で決まる骨かどうかは、ここでは見ない（辞書が新しくなっても、カードは読めるように）"""
    if not isinstance(answers, dict):
        raise ValueError("bone_answers は {} の形にしてください")
    known = set(names)
    for name, answer in answers.items():
        if name not in known:
            raise ValueError(f"bone_answers の骨 {name!r} は、カードの骨（skeleton）にありません")
        bone = canonical_bone(answer) if isinstance(answer, str) and answer != EXTRA_ANSWER else None
        if answer != EXTRA_ANSWER and (bone is None or bone.name != answer):
            raise ValueError(f"bone_answers の {name!r} の答えは、{EXTRA_ANSWER!r} か共通の名前の字のままにしてください"
                             f"（いまは {_describe(answer)}）")
    return {name: answers[name] for name in sorted(answers)}


def _check_mapping_names(mapping: dict[str, Any], names: Sequence[str]) -> None:
    """bone_mapping に出てくる元の名前が、どれもカードの骨（skeleton）にあるか"""
    used = [*mapping["mapping"], *mapping["unknown"], *mapping["known_extras"], *mapping["ambiguous"],
            *mapping["shadowed"], *(name for claimants in mapping["conflicts"].values() for name in claimants)]
    unknown = sorted(set(used) - set(names))
    if unknown:
        raise ValueError(f"bone_mapping に、カードの骨（skeleton）に無い名前があります：{unknown}")


def empty_source() -> dict[str, str]:
    """まだ何も書いていない、素材の出所と利用条件"""
    return dict.fromkeys(SOURCE_FIELDS, "")


@dataclass(frozen=True, kw_only=True)
class ModelCard:
    """モデルの登録カード。

    schema_version: カードの形の版（SCHEMA_VERSION。LEGACY_SCHEMA_VERSION のカードは読めるだけ）
    blend_name: 測った .blend の名前（保存していないファイルは ""）
    armature_name: アーマチュアのオブジェクトの名前
    parts: 部品（メッシュのオブジェクト）の並び。名前は重ならない
    bone_mapping: 骨の名前の対応（bone_mapping_to_json の形。辞書と bone_answers から作った物）
    measurements: 測った値（measure_body の形）
    fingerprints: {"shapes": {メッシュの名前: 指紋}, "bones": 指紋}（版 2 はアーマチュアの中の座標で作った物）
    source: 素材の出所と利用条件 {"origin", "author", "url", "terms"}（ユーザーが書いた文字のまま）
    created_at: 作った時刻（ISO-8601。時差付き）
    tool_version: 拡張機能の版
    blender_version: Blender の版
    placement: アーマチュアの置き方（matrix_world。4×4 の数のリスト）。版 1 は None
    skeleton: {"names": 骨の名前（armature.data.bones の順）, "heads": 根元の位置（アーマチュアの中）}。
        骨の答えを変えたときに、関節の位置を作り直すのに使う。版 1 は None
    bone_answers: 辞書に無い骨へのユーザーの答え（元の名前 → 共通の名前か "extra"）。版 1 は空

    作るときに形を確かめ、辞書は JSON にできる写しにする（あとで外から書き換えられないように）。
    変えるときは set_part_kind・set_material_kind・set_bone_answer・set_source などで、新しいカードを作る
    （版 1 のカードは変えられない）。
    """

    schema_version: int = SCHEMA_VERSION
    blend_name: str
    armature_name: str
    parts: list[Part]
    bone_mapping: dict[str, Any]
    measurements: dict[str, Any]
    fingerprints: dict[str, Any]
    source: dict[str, str]
    created_at: str
    tool_version: str
    blender_version: str
    placement: list[list[float]] | None = None
    skeleton: dict[str, Any] | None = None
    bone_answers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        version = self.schema_version
        if type(version) is not int or version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"このカードの形は schema_version {'・'.join(map(str, SUPPORTED_SCHEMA_VERSIONS))} だけを"
                             f"扱えます（いまは {_describe(version)}）")
        for name in ("blend_name", "armature_name", "tool_version", "blender_version"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} は文字にしてください（いまは {_describe(getattr(self, name))}）")
        if not self.armature_name:
            raise ValueError("armature_name が空です")
        if isinstance(self.parts, (str, bytes, Mapping)) or not isinstance(self.parts, Iterable):
            raise ValueError("parts は Part の並びにしてください")
        parts = list(self.parts)
        if not all(isinstance(part, Part) for part in parts):
            raise ValueError("parts の中は Part にしてください")
        part_names = [part.object_name for part in parts]
        if len(set(part_names)) != len(part_names):
            raise ValueError("parts に、同じ名前の部品が2つあります")
        bone_mapping = _plain_json(self.bone_mapping, "bone_mapping")
        measurements = _plain_json(self.measurements, "measurements")
        fingerprints = _plain_json(self.fingerprints, "fingerprints")
        for name, value in (("bone_mapping", bone_mapping), ("measurements", measurements),
                            ("fingerprints", fingerprints)):
            if not isinstance(value, dict):
                raise ValueError(f"{name} は {{}} の形にしてください")
        _check_bone_mapping(bone_mapping)
        _check_measurements(measurements, parts, version)
        _check_fingerprints(fingerprints, part_names)
        _check_time(self.created_at)
        if version == LEGACY_SCHEMA_VERSION:
            if self.placement is not None or self.skeleton is not None or self.bone_answers:
                raise ValueError("版 1 のカードには、placement・skeleton・bone_answers を入れられません")
            if any(part.materials for part in parts):
                raise ValueError("版 1 のカードには、材質ごとの種類を入れられません")
            placement, skeleton, answers = None, None, {}
        else:
            if self.placement is None:
                raise ValueError(f"版 {version} のカードには placement（アーマチュアの置き方）が要ります")
            placement = _placement_array(_plain_json(self.placement, "placement")).tolist()
            skeleton = _check_skeleton(_plain_json(self.skeleton, "skeleton"))
            answers = _check_answer_values(_plain_json(self.bone_answers, "bone_answers"), skeleton["names"])
            _check_mapping_names(bone_mapping, skeleton["names"])
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "bone_mapping", bone_mapping)
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "fingerprints", fingerprints)
        object.__setattr__(self, "source", _check_source(self.source))
        object.__setattr__(self, "placement", placement)
        object.__setattr__(self, "skeleton", skeleton)
        object.__setattr__(self, "bone_answers", answers)

    @property
    def is_legacy(self) -> bool:
        """古い形（版 1。世界の座標）のカードか。読めるだけで、変えられない（一度測り直す）"""
        return self.schema_version != SCHEMA_VERSION

    def part(self, object_name: str) -> Part:
        """名前で部品を引く。無い名前は ValueError"""
        for part in self.parts:
            if part.object_name == object_name:
                return part
        raise ValueError(f"部品 {object_name!r} はカードにありません")

    def value(self, name: str) -> float | None:
        """測った値（MEASUREMENT_NAMES のどれか）。測れなかったときは None"""
        if name not in MEASUREMENT_NAMES:
            raise ValueError(f"測った値の名前は {'・'.join(MEASUREMENT_NAMES)} のどれかにしてください（いまは {name!r}）")
        return self.measurements["values"][name]["value"]

    def placement_matrix(self) -> np.ndarray | None:
        """置き方（アーマチュアの matrix_world）の (4, 4) の float64。版 1 は None"""
        return None if self.placement is None else np.array(self.placement, dtype=np.float64)

    def to_data(self) -> dict[str, Any]:
        """JSON にする形（辞書は写し。カードの版の形）"""
        data = {
            "schema_version": self.schema_version,
            "blend_name": self.blend_name,
            "armature_name": self.armature_name,
            "parts": [part.to_data(self.schema_version) for part in self.parts],
            "bone_mapping": copy.deepcopy(self.bone_mapping),
            "measurements": copy.deepcopy(self.measurements),
            "fingerprints": copy.deepcopy(self.fingerprints),
            "source": dict(self.source),
            "created_at": self.created_at,
            "tool_version": self.tool_version,
            "blender_version": self.blender_version,
        }
        if not self.is_legacy:
            data["placement"] = copy.deepcopy(self.placement)
            data["skeleton"] = copy.deepcopy(self.skeleton)
            data["bone_answers"] = dict(self.bone_answers)
        return data

    def to_json(self) -> str:
        """決まった形の JSON（キーの順をそろえ、字下げ 2、日本語はそのまま、NaN は書かない）。
        同じカードからは、いつも同じ文字になる"""
        return json.dumps(self.to_data(), sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> ModelCard:
        """to_json() の文字からカードを作る。from_json(card.to_json()) == card、to_json(from_json(x)) == x になる。
        版 1 のカードも読める（is_legacy が True。変えられない）。

        JSON が読めない・同じキーが2回ある・NaN が書いてある・項目が違う・形が違う・知らない版のときは ValueError。
        入れ子が深すぎる JSON（.blend の中のテキストは、だれでも書ける）も ValueError にする（RecursionError にしない）。
        """
        if not isinstance(text, str):
            raise ValueError(f"カードの JSON は文字で渡してください（いまは {_describe(text)}）")
        try:
            return cls._from_text(text)
        except RecursionError:
            # 読むとき（json.loads）にも、読んだ後に写すとき（_plain_json の json.dumps）にも起きる
            raise ValueError("カードの JSON の入れ子が深すぎて読めません") from None

    @classmethod
    def _from_text(cls, text: str) -> ModelCard:
        try:
            data = json.loads(text, object_pairs_hook=_without_duplicates, parse_constant=_reject_constant)
        except json.JSONDecodeError as error:
            raise ValueError(f"カードの JSON が読めません：{error}") from error
        if not isinstance(data, dict):
            raise ValueError("カードの JSON は、いちばん外を {} にしてください")
        version = data.get("schema_version")
        if type(version) is not int or version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"このカードの形は schema_version {'・'.join(map(str, SUPPORTED_SCHEMA_VERSIONS))} だけを"
                             f"扱えます（いまは {_describe(version)}）")
        keys = _CARD_KEYS[version]
        if set(data) != set(keys):
            missing = sorted(set(keys) - set(data))
            unknown = sorted(set(data) - set(keys))
            raise ValueError(f"カードの項目が違います（足りない項目：{missing}、知らない項目：{unknown}）")
        if not isinstance(data["parts"], list):
            raise ValueError("カードの parts は [] の形にしてください")
        return cls(**{**data, "parts": [Part.from_data(item, version) for item in data["parts"]]})


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"カードの JSON に同じキー {key!r} が2回あります")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise ValueError(f"カードの JSON に {name} は書けません（測れなかった値は null にする）")


# MARK: - カードを作る・変える


def make_card(*, blend_name: str, armature_name: str, bones: BoneArrays, meshes: Sequence[MeshArrays],
              created_at: str, tool_version: str, blender_version: str,
              source: Mapping[str, str] | None = None, placement: Any = None,
              bone_answers: Mapping[str, str] | None = None) -> ModelCard:
    """並び（アーマチュアの中の座標・休みの姿勢）と置き方（アーマチュアの matrix_world。None は単位行列）から
    カードを作る。部品と材質の種類は名前からの推測（kind_source "guess"）、服の種類は None（ユーザーが選ぶ）。
    source を渡さなければ、素材の出所と利用条件は空。bone_answers は使えない答えがあれば ValueError"""
    if not isinstance(bones, BoneArrays):
        raise ValueError(f"bones は BoneArrays にしてください（いまは {_describe(bones)}）")
    items = _check_meshes(meshes)
    matrix = _placement_array(placement)
    answers = _check_answers(bones.names, bone_answers)
    parts = [_guess_part(mesh) for mesh in items]
    measurements = measure_body(bones.names, bones.heads, {mesh.name: mesh for mesh in items},
                                {part.object_name: part.kind for part in parts}, placement=matrix,
                                material_kinds={part.object_name: {item.name: item.kind for item in part.materials}
                                                for part in parts if part.materials},
                                bone_answers=answers)
    return ModelCard(
        blend_name=blend_name,
        armature_name=armature_name,
        parts=parts,
        bone_mapping=bone_mapping_to_json(map_with_answers(bones.names, answers)),
        measurements=measurements,
        fingerprints=fingerprints_of(bones, items),
        source=empty_source() if source is None else dict(source),
        created_at=created_at,
        tool_version=tool_version,
        blender_version=blender_version,
        placement=matrix.tolist(),
        skeleton={"names": list(bones.names), "heads": bones.heads.tolist()},
        bone_answers=answers,
    )


def _editable(card: Any) -> ModelCard:
    """変えてよいカードか（ModelCard で、版 1 ではない）"""
    if not isinstance(card, ModelCard):
        raise ValueError(f"card は ModelCard にしてください（いまは {_describe(card)}）")
    if card.is_legacy:
        raise ValueError(f"版 {card.schema_version} の古いカードは変えられません。先に測り直してください（build_card）")
    return card


def set_part_kind(card: ModelCard, object_name: str, kind: str, cloth_type: str | None = None) -> ModelCard:
    """ユーザーが部品の種類（と服の種類）を選んだ、新しいカード（kind_source "user"）。元のカードは変えない。

    kind は PART_KINDS、cloth_type は CLOTH_TYPE_IDS のどれか（服でない部品には付けられない）。
    体の部品が変わると背の高さ・頭身も変わるので、カードの中の骨の位置と高さの範囲から測り直す
    （形は読み直さない。形が変わったかどうかは指紋で見る）。
    """
    _editable(card)
    _check_kind(kind)
    _check_cloth_type(kind, cloth_type)
    target = card.part(object_name)
    parts = [replace(part, kind=kind, kind_source="user", cloth_type=cloth_type) if part is target else part
             for part in card.parts]
    return replace(card, parts=parts, measurements=_remeasure_card(card, parts))


def set_material_kind(card: ModelCard, object_name: str, material_name: str, kind: str,
                      cloth_type: str | None = None) -> ModelCard:
    """ユーザーが部品の中の材質1つの種類（と服の種類）を選んだ、新しいカード（kind_source "user"）。

    材質ごとの種類があるのは、材質が2つ以上ある部品だけ（1つ以下の部品は set_part_kind で選ぶ）。
    体ではない種類（NON_BODY_KINDS）の材質は、背の高さ・頭身を測るときに外すので、測り直す。
    """
    _editable(card)
    _check_kind(kind)
    _check_cloth_type(kind, cloth_type)
    target = card.part(object_name)
    material = target.material(material_name)
    materials = tuple(replace(item, kind=kind, kind_source="user", cloth_type=cloth_type) if item is material else item
                      for item in target.materials)
    parts = [replace(part, materials=materials) if part is target else part for part in card.parts]
    return replace(card, parts=parts, measurements=_remeasure_card(card, parts))


def set_bone_answer(card: ModelCard, original_name: str, answer: str | None) -> ModelCard:
    """辞書に無い骨（answerable_bones）へのユーザーの答えを入れた、新しいカード。元のカードは変えない。

    answer：共通の名前（MMD 準標準。全角・半角などの違いは問わず、字のままの形にして持つ）か "extra"
    （人型の骨ではない）。None は答えを消す。
    答えを入れると bone_mapping を作り直し（map_with_answers）、関節の位置と測った値も測り直す。
    辞書で決まる骨・共通の名前でない答え・ほかの骨が使っている共通の名前は ValueError。
    """
    _editable(card)
    if not isinstance(original_name, str):
        raise ValueError(f"original_name は文字にしてください（いまは {_describe(original_name)}）")
    names = card.skeleton["names"]
    answers = {name: value for name, value in card.bone_answers.items() if name != original_name}
    if answer is not None:
        if not isinstance(answer, str):
            raise ValueError(f"answer は共通の名前か {EXTRA_ANSWER!r} か None にしてください（いまは {_describe(answer)}）")
        bone = canonical_bone(answer) if answer != EXTRA_ANSWER else None
        value = EXTRA_ANSWER if answer == EXTRA_ANSWER else (bone.name if bone is not None else answer)
        problem = _answer_problem(map_bones(names), frozenset(names), original_name, value, answers)
        if problem is not None:
            raise ValueError(problem)
        answers[original_name] = value
    elif original_name not in names:
        raise ValueError(f"骨 {original_name!r} は、このモデルにありません")
    mapping = bone_mapping_to_json(map_with_answers(names, answers))
    return replace(card, bone_answers=answers, bone_mapping=mapping,
                   measurements=_remeasure_card(card, card.parts, mapping))


def set_source(card: ModelCard, **fields: str) -> ModelCard:
    """素材の出所と利用条件を書いた、新しいカード。渡した項目（origin・author・url・terms）だけを置きかえる。
    文字はそのまま残す（前後の空白も消さない）"""
    _editable(card)
    unknown = sorted(set(fields) - set(SOURCE_FIELDS))
    if unknown:
        raise ValueError(f"素材の項目は {'・'.join(SOURCE_FIELDS)} のどれかにしてください（知らない項目：{unknown}）")
    return replace(card, source={**card.source, **fields})


def _has_user_choices(card: ModelCard) -> bool:
    """ユーザーが決めた物（部品・材質の種類、骨の答え、素材の出所と利用条件）があるか"""
    if any(value.strip() for value in card.source.values()) or card.bone_answers:
        return True
    return any(part.kind_source == "user" or any(item.kind_source == "user" for item in part.materials)
               for part in card.parts)


def same_model(previous: ModelCard, fresh: ModelCard, *, legacy_bones_fingerprint: str | None = None) -> bool:
    """前のカード（previous）が、いま測ったカード（fresh）と同じモデルの物か：アーマチュアの名前と骨の指紋が同じ。

    版 1 の前のカードは、指紋が世界の座標で作ってあるので、いまのモデルの世界の座標の骨の指紋
    （legacy_bones_fingerprint。card_bpy が作る）と比べる。渡さなければ、同じとは言えない（False）。
    """
    for name, card in (("previous", previous), ("fresh", fresh)):
        if not isinstance(card, ModelCard):
            raise ValueError(f"{name} は ModelCard にしてください（いまは {_describe(card)}）")
    if previous.armature_name != fresh.armature_name:
        return False
    if previous.is_legacy:
        return legacy_bones_fingerprint is not None and previous.fingerprints["bones"] == legacy_bones_fingerprint
    return previous.fingerprints["bones"] == fresh.fingerprints["bones"]


def carry_user_choices(previous: ModelCard, fresh: ModelCard, *,
                       legacy_bones_fingerprint: str | None = None) -> tuple[ModelCard, list[str]]:
    """測り直した新しいカード（fresh）に、前のカードでユーザーが決めた物を写す。返す物は (カード, 知らせの文)。

    写すのは、前のカードが同じモデルの物（same_model：アーマチュアの名前と骨の指紋が同じ）のときだけ。
    違うときは何も写さず、そう知らせる（別のモデルに、部品の種類・骨の答え・利用条件を写さないため）。
    写す物：
    - 素材の出所と利用条件
    - 同じ名前の部品で、ユーザーが選んだ種類・服の種類（kind_source "user" の物だけ。推測は新しい推測のまま）
    - 同じ部品の同じ名前の材質で、ユーザーが選んだ種類・服の種類
    - 骨の答え（いまの辞書で使えない答えは外し、そう知らせる）
    なくなった部品・材質の選んだ種類は写せないので、そう知らせる。種類が変わると背の高さ・頭身も変わるので、測り直す。
    """
    if not isinstance(fresh, ModelCard) or fresh.is_legacy:
        raise ValueError(f"fresh は、いまの版の ModelCard にしてください（いまは {_describe(fresh)}）")
    notes: list[str] = []
    if not same_model(previous, fresh, legacy_bones_fingerprint=legacy_bones_fingerprint):
        if _has_user_choices(previous):
            notes.append(card_text("card.choices_not_carried", armature=previous.armature_name))
        return fresh, notes
    previous_parts = {part.object_name: part for part in previous.parts}
    fresh_names = {part.object_name for part in fresh.parts}
    parts = []
    for part in fresh.parts:
        old = previous_parts.get(part.object_name)
        if old is None:
            parts.append(part)
            continue
        if old.kind_source == "user":
            part = replace(part, kind=old.kind, kind_source="user", cloth_type=old.cloth_type)
        chosen = {item.name: item for item in old.materials if item.kind_source == "user"}
        if chosen:
            current = {item.name for item in part.materials}
            for name in chosen:
                if name not in current:
                    notes.append(card_text("card.material_choice_dropped", name=part.object_name, material=name))
            part = replace(part, materials=tuple(
                replace(item, kind=chosen[item.name].kind, kind_source="user", cloth_type=chosen[item.name].cloth_type)
                if item.name in chosen else item for item in part.materials))
        parts.append(part)
    for old in previous.parts:
        if old.object_name not in fresh_names and (
                old.kind_source == "user" or any(item.kind_source == "user" for item in old.materials)):
            notes.append(card_text("card.part_choice_dropped", name=old.object_name))
    names = fresh.skeleton["names"]
    answers, dropped = _split_answers(names, previous.bone_answers)
    for name, answer in dropped:
        notes.append(card_text("card.answer_dropped", bone=name, answer=answer))
    mapping = bone_mapping_to_json(map_with_answers(names, answers))
    card = replace(fresh, parts=parts, source=dict(previous.source), bone_answers=answers, bone_mapping=mapping)
    return replace(card, measurements=_remeasure_card(card, parts)), notes


def keep_user_choices(previous: ModelCard, fresh: ModelCard, *,
                      legacy_bones_fingerprint: str | None = None) -> ModelCard:
    """carry_user_choices のカードだけを返す（知らせの文は捨てる。知らせが要るときは carry_user_choices を使う）"""
    return carry_user_choices(previous, fresh, legacy_bones_fingerprint=legacy_bones_fingerprint)[0]


# MARK: - 測り直しが要るか


def remeasure_reasons(card: ModelCard, current_shape_fingerprints: Mapping[str, str],
                      current_bones_fingerprint: str) -> list[str]:
    """カードの指紋と、いまの指紋を比べ、測り直しが要る理由（いまの言語の文）を返す。空なら、カードはそのまま使える。

    並び：骨、カードにあるメッシュ（名前の順。形が変わった・なくなった）、増えたメッシュ（名前の順）。
    版 1 のカードは指紋の空間が違うので比べず、「古い形なので測り直す」の理由1つだけを返す。
    """
    if not isinstance(card, ModelCard):
        raise ValueError(f"card は ModelCard にしてください（いまは {_describe(card)}）")
    if not isinstance(current_shape_fingerprints, Mapping):
        raise ValueError("current_shape_fingerprints は、メッシュの名前 → 指紋 の辞書にしてください")
    if not isinstance(current_bones_fingerprint, str):
        raise ValueError("current_bones_fingerprint は文字にしてください")
    if card.is_legacy:
        return [card_text("card.legacy_card", version=str(card.schema_version))]
    reasons = []
    if current_bones_fingerprint != card.fingerprints["bones"]:
        reasons.append(card_text("card.bones_changed"))
    stored = card.fingerprints["shapes"]
    for name in sorted(stored):
        if name not in current_shape_fingerprints:
            reasons.append(card_text("card.mesh_removed", name=name))
        elif current_shape_fingerprints[name] != stored[name]:
            reasons.append(card_text("card.shape_changed", name=name))
    for name in sorted(set(current_shape_fingerprints) - set(stored)):
        reasons.append(card_text("card.mesh_added", name=name))
    return reasons


def check_against(card: ModelCard, *, armature_name: str, shapes: Mapping[str, str], bones: str,
                  placement: Any) -> tuple[list[str], list[str]]:
    """カードと、いまのモデル（アーマチュアの名前・メッシュの指紋・骨の指紋・置き方）を比べる。
    返す物は (測り直しが要る理由, 知らせ)。どちらも、いまの言語の文。

    - 理由：アーマチュアの名前が変わった・remeasure_reasons・置き方の大きさか傾きが変わった（placement_change が
      changed）。版 1 のカードは「古い形なので測り直す」の理由1つだけ。
    - 知らせ：置き場所だけが変わった（moved：平行移動と Z まわりの回転。形も値もそのまま）。
    """
    if not isinstance(armature_name, str):
        raise ValueError("armature_name は文字にしてください")
    reasons = remeasure_reasons(card, shapes, bones)
    if card.is_legacy:
        return reasons, []
    if armature_name != card.armature_name:
        reasons.insert(0, card_text("card.armature_changed", before=card.armature_name, after=armature_name))
    notes = []
    change = placement_change(card.placement_matrix(), placement)
    if change == PLACEMENT_CHANGED:
        reasons.append(card_text("card.placement_resized"))
    elif change == PLACEMENT_MOVED:
        notes.append(card_text("card.placement_moved"))
    return reasons, notes


# MARK: - 信号にまとめる

SEMI_STANDARD_NAME = "semi_standard_bones"
UNKNOWN_BONES_NAME = "unknown_bones"
UNCONFIRMED_PARTS_NAME = "unconfirmed_parts"
BODY_MEASUREMENTS_NAME = "body_measurements"
SOURCE_TERMS_NAME = "source_terms"
CARD_CURRENT_NAME = "card_current"


def _counted(name: str, count: int, bad: Signal, reason: str, details: dict[str, Any]) -> Measurement:
    if count:
        return Measurement(name, float(count), "count", bad, reason, details)
    return Measurement(name, 0.0, "count", Signal.GREEN, "", details)


def summarize_card(card: ModelCard, reasons: Sequence[str] | None = None) -> list[Measurement]:
    """カードを信号にまとめる。単位はどれも "count"。理由の文は、いまの言語。

    - "semi_standard_bones"：足りない MMD 標準・準標準の骨（補助の骨を除く）と、取り合いで決まらない骨の数。
      1つでもあれば黄。CLAUDE.md 3章：段階1では骨を足さず（M2）、足りない骨をここに一覧で出し、
      その骨が要る確かめのほうを灰にする。カードそのものは正しく測れているので、赤・灰にはしない。
      ユーザーの答えで対応した骨は、足りない骨に数えない。
    - "unknown_bones"：辞書に無い・1つに決まらない骨（答えた骨は除く）と、取り合いの数。1つでもあれば黄（ユーザーに聞く）。
    - "unconfirmed_parts"：ユーザーがまだ決めていない部品と材質の数（種類が推測のまま・"unknown"・服の種類が無い）。
      1つでもあれば黄。部品が1つも無ければ灰。
    - "body_measurements"：測れなかった値の数。1つでもあれば灰。全部測れたが、体の部品が無く全部のメッシュで
      背の高さを測ったときは黄。
    - "source_terms"：素材の出所と利用条件（REQUIRED_SOURCE_FIELDS）のうち、空の物の数。1つでもあれば黄。
    - "card_current"（reasons を渡したときだけ）：測り直しが要る理由の数。1つでもあれば灰（カードの値は、
      いまのモデルを測った値ではない）。
    """
    if not isinstance(card, ModelCard):
        raise ValueError(f"card は ModelCard にしてください（いまは {_describe(card)}）")
    separator = card_text("card.list_separator")
    mapping = card.bone_mapping
    missing = list(mapping["missing_standard"]) + list(mapping["missing_semi_standard"])
    conflicts = mapping["conflicts"]
    results = [
        _counted(SEMI_STANDARD_NAME, len(missing) + len(conflicts), Signal.YELLOW, card_text("card.semi_standard_missing"),
                 {"missing_standard": list(mapping["missing_standard"]),
                  "missing_semi_standard": list(mapping["missing_semi_standard"]),
                  "conflicts": copy.deepcopy(conflicts)}),
        _counted(UNKNOWN_BONES_NAME, len(mapping["unknown"]) + len(conflicts), Signal.YELLOW, card_text("card.unknown_bones"),
                 {"unknown": list(mapping["unknown"]), "ambiguous": copy.deepcopy(mapping["ambiguous"]),
                  "conflicts": copy.deepcopy(conflicts), "answers": dict(card.bone_answers)}),
    ]
    unconfirmed = [part.object_name for part in card.parts if not part.is_confirmed]
    materials = {part.object_name: list(part.unconfirmed_materials) for part in card.parts if part.unconfirmed_materials}
    if not card.parts:
        results.append(Measurement.unmeasured(UNCONFIRMED_PARTS_NAME, "count", card_text("card.no_parts")))
    else:
        count = len(unconfirmed) + sum(len(names) for names in materials.values())
        results.append(_counted(UNCONFIRMED_PARTS_NAME, count, Signal.YELLOW, card_text("card.unconfirmed_parts"),
                                {"parts": unconfirmed, "materials": materials}))
    values = card.measurements["values"]
    unmeasured = {name: values[name]["reason"] for name in MEASUREMENT_NAMES if values[name]["value"] is None}
    details = {"unmeasured": unmeasured, "body_meshes": list(card.measurements["body_meshes"]),
               "body_meshes_source": card.measurements["body_meshes_source"]}
    if unmeasured:
        results.append(Measurement(BODY_MEASUREMENTS_NAME, float(len(unmeasured)), "count", Signal.GRAY,
                                   card_text("card.unmeasured_values", names=separator.join(unmeasured)), details))
    elif card.measurements["body_meshes_source"] == BODY_SOURCE_ALL:
        results.append(Measurement(BODY_MEASUREMENTS_NAME, 0.0, "count", Signal.YELLOW,
                                   card_text("card.all_meshes_height"), details))
    else:
        results.append(Measurement(BODY_MEASUREMENTS_NAME, 0.0, "count", Signal.GREEN, "", details))
    empty = [key for key in REQUIRED_SOURCE_FIELDS if not card.source[key].strip()]
    labels = separator.join(card_text(f"card.source_{key}") for key in empty)
    results.append(_counted(SOURCE_TERMS_NAME, len(empty), Signal.YELLOW, card_text("card.source_missing", fields=labels),
                            {"empty": empty}))
    if reasons is not None:
        reason_list = list(_names(reasons, "reasons"))
        if reason_list:
            results.append(Measurement(CARD_CURRENT_NAME, float(len(reason_list)), "count", Signal.GRAY,
                                       card_text("card.stale", reasons=card_text("card.sentence_separator").join(reason_list)),
                                       {"reasons": reason_list}))
        else:
            results.append(Measurement(CARD_CURRENT_NAME, 0.0, "count", Signal.GREEN, "", {"reasons": []}))
    return results
