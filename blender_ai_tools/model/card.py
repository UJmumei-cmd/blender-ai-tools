"""A1 モデルの登録カード（bpy なし・numpy だけ）。

モデルごとに1回作ってファイルの中に保存し、あとの確かめ（A4・A5・E1・B5 など）が使い回す。カードに入れる物：

- 部品の種類（体・服・髪・アクセサリー・羽・その他・分からない）。名前から推測した物（kind_source "guess"）は
  ただの案で、ユーザーが一覧から選んで決める（"user"）。服の種類もユーザーが一覧（CLOTH_TYPES）から選ぶ。
  名前（オブジェクト・材質）は「データ」として決まった言葉と比べるだけで、指示としては読まない。
- 骨の名前の対応（A3 の map_bones の結果）。準標準の骨が足りないときは、ここに一覧で出す（CLAUDE.md 3章）。
- 測った値（背の高さ・脚の長さ・腕の下がり角・肩幅・頭身の目安）と関節の位置。
  位置は世界の座標（Blender のまま。Z が上・メートル）、休みの姿勢・モディファイアを通さない形で測る。
  測れない値は None にして理由を書く。数を作り出さない。
- 形と骨の指紋（sha256）。ユーザーが手で直すと指紋が変わり、「測り直しが必要」と分かる（remeasure_reasons）。
- 素材の出所と利用条件（ユーザーが書く。書いた文字はそのまま残す）。

計算はここ（numpy の並びを受け取る）、Blender から並びを取り出すのは card_bpy.py。
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
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .. import thresholds
from ..bones import BoneMapping, canonical_names, map_bones, mirror_name
from ..measure.geometry import as_array, as_points, as_triangles
from ..measure.results import Measurement, Signal

__all__ = [
    "ARM_REST_DOWN", "BODY_MEASUREMENTS_NAME", "BODY_SOURCE_ALL", "BODY_SOURCE_KIND", "BONE_MAPPING_KEYS",
    "BoneArrays", "CARD_CURRENT_NAME", "CLOTH_TYPES", "CLOTH_TYPE_IDS", "CLOTH_TYPE_LABELS", "HEAD_RATIO",
    "HEAD_RATIO_METHOD", "HEIGHT", "KIND_SOURCES", "LEG_LENGTH", "MEASUREMENT_NAMES", "MeshArrays", "ModelCard",
    "PART_KINDS", "PART_KIND_LABELS", "Part", "REQUIRED_SOURCE_FIELDS", "SCHEMA_VERSION", "SEMI_STANDARD_NAME",
    "SHOULDER_WIDTH", "SOURCE_FIELDS", "SOURCE_LABELS", "SOURCE_TERMS_NAME", "UNCONFIRMED_PARTS_NAME",
    "UNKNOWN_BONES_NAME", "bone_mapping_to_json", "bones_fingerprint", "empty_source", "fingerprints_of",
    "guess_part_kind", "keep_user_choices", "make_card", "measure_body", "remeasure_reasons", "set_part_kind",
    "set_source", "shape_fingerprint", "summarize_card",
]

# MARK: - 決まり

# カードの形の版。項目の意味を変えたら上げる
SCHEMA_VERSION = 1

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

# 測る値の名前（measurements["values"] のキー）。長さはメートル、角度は度
HEIGHT = "height_m"
LEG_LENGTH = {"L": "leg_length_L_m", "R": "leg_length_R_m"}
ARM_REST_DOWN = {"L": "arm_rest_down_L_deg", "R": "arm_rest_down_R_deg"}
SHOULDER_WIDTH = "shoulder_width_m"
HEAD_RATIO = "head_ratio"
MEASUREMENT_NAMES = (HEIGHT, LEG_LENGTH["L"], LEG_LENGTH["R"], ARM_REST_DOWN["L"], ARM_REST_DOWN["R"],
                     SHOULDER_WIDTH, HEAD_RATIO)

HEIGHT_METHOD = "体の部品のメッシュの、いちばん上 − いちばん下（休みの姿勢・モディファイアを通さない形）"
LEG_METHOD = "足 → ひざ → 足首 の根元をつないだ長さ"
LEG_D_METHOD = "足D → ひざD → 足首D の根元をつないだ長さ（足・ひざ・足首 が無いので D の骨で測った）"
ARM_METHOD = "腕 の根元から ひじ の根元への向きの、水平からの下がり角（convert_pmx.py の rest_arm_angle と同じ式。0.1 度に丸める）"
SHOULDER_METHOD = "左腕 の根元と 右腕 の根元の距離"
HEAD_RATIO_METHOD = "頭の骨の根元から頭のてっぺんまで（目安。A2 の目印で直す）"

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
_MEASUREMENT_KEYS = ("space", "joints", "missing_joints", "mesh_z_range", "body_meshes", "body_meshes_source",
                     "values")
_VALUE_KEYS = ("value", "unit", "method", "reason", "bones")
_PART_KEYS = ("object_name", "kind", "kind_source", "cloth_type", "vertex_count", "triangle_count",
              "material_names", "matched_keyword")
_CARD_KEYS = ("schema_version", "blend_name", "armature_name", "parts", "bone_mapping", "measurements",
              "fingerprints", "source", "created_at", "tool_version", "blender_version")

# 指紋の形の印（中身の作り方を変えたら v の数を上げる）
_SHAPE_TAG = "blender_ai_tools.card.shape.v1"
_BONES_TAG = "blender_ai_tools.card.bones.v1"
# 指紋の位置をそろえる幅（再現性の幅。これより小さい差は、変化として数えない）
_QUANTUM = thresholds.REPRO_TOLERANCE_LENGTH
# 整数にした座標の上限（int64 に入る大きさより十分小さい）
_MAX_QUANTIZED = float(2 ** 62)
_SHA256 = re.compile(r"[0-9a-f]{64}")


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


# MARK: - 部品


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
        raise ValueError(f"服の種類（{cloth_type}）を付けられるのは、種類が服（cloth）の部品だけです（いまの種類：{kind}）")
    return cloth_type


@dataclass(frozen=True)
class Part:
    """モデルの部品（メッシュのオブジェクト1つ）。

    object_name: オブジェクトの名前
    kind: 種類（PART_KINDS のどれか）
    kind_source: "guess"（名前からの推測。ただの案）か "user"（ユーザーが選んだ）
    cloth_type: 服の種類（CLOTH_TYPE_IDS のどれか）。ユーザーが選ぶまでは None。服でない部品はいつも None
    vertex_count・triangle_count: 頂点と三角形の数
    material_names: 材質の名前（材質の枠の順。同じ名前は1つ）
    matched_keyword: 推測で当たった言葉（推測の記録。当たらなかったときは None）
    """

    object_name: str
    kind: str
    kind_source: str
    cloth_type: str | None
    vertex_count: int
    triangle_count: int
    material_names: tuple[str, ...] = ()
    matched_keyword: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.object_name, str) or not self.object_name:
            raise ValueError(f"部品の object_name は空でない文字にしてください（いまは {_describe(self.object_name)}）")
        _check_kind(self.kind)
        if self.kind_source not in KIND_SOURCES:
            raise ValueError(f"{self.object_name}：kind_source は \"guess\" か \"user\" にしてください"
                             f"（いまは {_describe(self.kind_source)}）")
        _check_cloth_type(self.kind, self.cloth_type)
        for name in ("vertex_count", "triangle_count"):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{self.object_name}：{name} は 0 以上の整数にしてください（いまは {_describe(value)}）")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "material_names", _names(self.material_names, f"{self.object_name} の material_names"))
        if self.matched_keyword is not None and not isinstance(self.matched_keyword, str):
            raise ValueError(f"{self.object_name}：matched_keyword は文字か None にしてください")

    @property
    def is_confirmed(self) -> bool:
        """ユーザーが決め終わったか：種類を選び（"unknown" ではない）、服なら服の種類も選んだ"""
        if self.kind_source != "user" or self.kind == "unknown":
            return False
        return self.kind != "cloth" or self.cloth_type is not None

    def to_data(self) -> dict[str, Any]:
        """JSON にする形"""
        return {
            "object_name": self.object_name,
            "kind": self.kind,
            "kind_source": self.kind_source,
            "cloth_type": self.cloth_type,
            "vertex_count": self.vertex_count,
            "triangle_count": self.triangle_count,
            "material_names": list(self.material_names),
            "matched_keyword": self.matched_keyword,
        }

    @classmethod
    def from_data(cls, data: Any) -> Part:
        """to_data() の形から作る。項目が違うときは ValueError"""
        if not isinstance(data, dict) or set(data) != set(_PART_KEYS):
            raise ValueError(f"部品の項目は {'・'.join(_PART_KEYS)} にしてください")
        if not isinstance(data["material_names"], list):
            raise ValueError("部品の material_names は [] の形にしてください")
        return cls(**data)


# MARK: - 入力の並び


def _readonly(array: np.ndarray) -> np.ndarray:
    array = np.array(array, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class BoneArrays:
    """骨の休みの形の並び（model.extract_bpy.BoneRest と同じ中身。並びは armature.data.bones の順）。

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
    material_names: 材質の名前
    """

    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    material_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"メッシュの名前は空でない文字にしてください（いまは {_describe(self.name)}）")
        vertices = as_points(self.vertices, f"{self.name} の vertices")
        triangles = as_triangles(self.triangles, len(vertices), f"{self.name} の triangles")
        object.__setattr__(self, "vertices", _readonly(vertices))
        object.__setattr__(self, "triangles", _readonly(triangles))
        object.__setattr__(self, "material_names", _names(self.material_names, f"{self.name} の material_names"))


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
    同じ入力からは、いつも同じ指紋になる。
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
    骨の並びを変えるだけでも変わる（ウェイトの骨の番号が変わるため）。
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


def _leg_length(joints: Mapping[str, np.ndarray], side: str) -> dict[str, Any]:
    standard = _side_names(_LEG_BONES, side)
    d_chain = _side_names(_LEG_D_BONES, side)
    for chain, method, note in ((standard, LEG_METHOD, ""),
                                (d_chain, LEG_D_METHOD, f"{_missing_text([n for n in standard if n not in joints])} "
                                                        f"が無いので、D の骨で測りました")):
        if all(name in joints for name in chain):
            points = [joints[name] for name in chain]
            length = sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:]))
            return _entry(length, "m", method, note, chain)
    missing = [name for name in standard + d_chain if name not in joints]
    return _entry(None, "m", LEG_METHOD, f"脚の骨（{_missing_text(missing)}）が無いので測れません", standard)


def _arm_down_deg(upper: np.ndarray, lower: np.ndarray) -> float | None:
    """convert_pmx.rest_arm_angle と同じ式：腕の根元からひじの根元への向きの、水平からの下がり角（度。0.1 度に丸める）。
    向きの長さが 0 のときは None（角度が決まらない）"""
    direction = lower - upper
    horizontal = (float(direction[0]) ** 2 + float(direction[1]) ** 2) ** 0.5
    if horizontal == 0.0 and float(direction[2]) == 0.0:
        return None
    return round(float(np.degrees(np.arctan2(-float(direction[2]), horizontal))), APP_ANGLE_DIGITS)


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


def _shoulder_width(joints: Mapping[str, np.ndarray]) -> dict[str, Any]:
    left, right = _UPPER_ARM, mirror_name(_UPPER_ARM)
    missing = [name for name in (left, right) if name not in joints]
    if missing:
        return _entry(None, "m", SHOULDER_METHOD, f"腕の骨（{_missing_text(missing)}）が無いので測れません", (left, right))
    return _entry(float(np.linalg.norm(joints[left] - joints[right])), "m", SHOULDER_METHOD, "", (left, right))


def _body_range(z_ranges: Mapping[str, Sequence[float] | None],
                kinds: Sequence[tuple[str, str]]) -> tuple[list[str], str, tuple[float, float] | None]:
    """高さを測るメッシュ・その選び方・(いちばん下, いちばん上)。頂点が1つも無ければ範囲は None"""
    body = [name for name, kind in kinds if kind == "body"]
    source = BODY_SOURCE_KIND if body else BODY_SOURCE_ALL
    used = body if body else [name for name, _ in kinds]
    ranges = [z_ranges[name] for name in used if z_ranges.get(name) is not None]
    if not ranges:
        return used, source, None
    return used, source, (min(float(r[0]) for r in ranges), max(float(r[1]) for r in ranges))


def _measure(joints: Mapping[str, np.ndarray], z_ranges: Mapping[str, Sequence[float] | None],
             kinds: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """関節の位置・メッシュの高さの範囲・部品の種類から、measurements（JSON の形）を作る"""
    used, source, extent = _body_range(z_ranges, kinds)
    all_note = ("体の部品が決まっていないので、全部のメッシュで測りました（髪・帽子・服などを含むことがあります）"
                if source == BODY_SOURCE_ALL else "")
    values: dict[str, dict[str, Any]] = {}
    if extent is None:
        values[HEIGHT] = _entry(None, "m", HEIGHT_METHOD, "高さを測るメッシュに、頂点がありません", ())
    else:
        values[HEIGHT] = _entry(extent[1] - extent[0], "m", HEIGHT_METHOD, all_note, ())
    for side in ("L", "R"):
        values[LEG_LENGTH[side]] = _leg_length(joints, side)
    for side in ("L", "R"):
        values[ARM_REST_DOWN[side]] = _arm_rest_down(joints, side)
    values[SHOULDER_WIDTH] = _shoulder_width(joints)
    if extent is None:
        values[HEAD_RATIO] = _entry(None, "ratio", HEAD_RATIO_METHOD, "背の高さを測れないので測れません", (_HEAD,))
    elif _HEAD not in joints:
        values[HEAD_RATIO] = _entry(None, "ratio", HEAD_RATIO_METHOD, f"骨 {_HEAD} が無いので測れません", (_HEAD,))
    else:
        head_length = extent[1] - float(joints[_HEAD][2])
        if head_length <= 0.0:
            values[HEAD_RATIO] = _entry(None, "ratio", HEAD_RATIO_METHOD,
                                        f"骨 {_HEAD} の根元が、体のいちばん上より下にありません", (_HEAD,))
        else:
            values[HEAD_RATIO] = _entry((extent[1] - extent[0]) / head_length, "ratio", HEAD_RATIO_METHOD, all_note,
                                        (_HEAD,))
    order = canonical_names()
    return {
        "space": "world",
        "joints": {name: [float(c) for c in joints[name]] for name in order if name in joints},
        "missing_joints": [name for name in order if name not in joints],
        "mesh_z_range": {name: None if z_ranges.get(name) is None else [float(z) for z in z_ranges[name]]
                         for name, _ in kinds},
        "body_meshes": list(used),
        "body_meshes_source": source,
        "values": values,
    }


def _joints_of(names: Sequence[str], heads: np.ndarray, mapping: BoneMapping) -> dict[str, np.ndarray]:
    """共通の名前 → 骨の根元の位置（対応が決まった骨だけ）"""
    index = {name: i for i, name in enumerate(names)}
    return {canonical: heads[index[original]] for canonical, original in mapping.by_canonical.items()}


def measure_body(bone_names: Sequence[str], heads: Any, meshes: Mapping[str, Any],
                 part_kinds: Mapping[str, str]) -> dict[str, Any]:
    """体を測る。返す物は measurements（JSON の形の辞書）。

    bone_names・heads: 骨の名前と根元の位置 (B, 3)（世界の座標・休みの姿勢）。名前は map_bones で共通の名前にする
    meshes: メッシュの名前 → 頂点 (N, 3)（同じ座標）。並びの順を覚えておく
    part_kinds: メッシュの名前 → 部品の種類。無いメッシュは "unknown" とみなす

    measurements の中身：
        space: "world"
        joints: 共通の名前 → 根元の位置 [x, y, z]（対応が決まった骨だけ）
        missing_joints: 位置が分からない共通の名前（辞書の順。骨が無い・取り合いで決まらない）
        mesh_z_range: メッシュの名前 → [いちばん下の z, いちばん上の z]（頂点が無ければ None）
        body_meshes: 高さを測ったメッシュ。body_meshes_source: "kind"（体の部品）か "all_meshes"（体の部品が無い）
        values: 名前（MEASUREMENT_NAMES）→ {"value", "unit", "method", "reason", "bones"}。測れないときは value が None で
            reason に理由がある。値はあるが気を付けること（全部のメッシュで測った など）も reason に書く
    測る値：
        height_m: 体の部品のいちばん上 − いちばん下（床 = 体の部品のいちばん下）
        leg_length_L_m・leg_length_R_m: |足 − ひざ| + |ひざ − 足首|（根元どうし）。無ければ D の骨で
        arm_rest_down_L_deg・arm_rest_down_R_deg: 腕 → ひじ の向きの下がり角（convert_pmx.rest_arm_angle と同じ式・丸め）
        shoulder_width_m: |左腕 − 右腕|（根元どうし）
        head_ratio: 背の高さ ÷ (体のいちばん上 − 頭 の根元の z)。目安（HEAD_RATIO_METHOD）
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
    z_ranges: dict[str, tuple[float, float] | None] = {}
    kinds: list[tuple[str, str]] = []
    for name, vertices in meshes.items():
        if not isinstance(name, str):
            raise ValueError(f"メッシュの名前は文字にしてください（いまは {_describe(name)}）")
        v = as_points(vertices, f"{name} の頂点")
        z_ranges[name] = None if len(v) == 0 else (float(v[:, 2].min()), float(v[:, 2].max()))
        kinds.append((name, _check_kind(part_kinds.get(name, "unknown"))))
    return _measure(_joints_of(names, points, map_bones(names)), z_ranges, kinds)


def _remeasure_card(card: ModelCard, parts: Sequence[Part]) -> dict[str, Any]:
    """カードに入っている関節の位置と高さの範囲から、部品の種類だけを変えて測り直す（形は読み直さない）"""
    joints = {name: np.array(position, dtype=np.float64) for name, position in card.measurements["joints"].items()}
    return _measure(joints, card.measurements["mesh_z_range"], [(part.object_name, part.kind) for part in parts])


# MARK: - 骨の名前の対応


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


def _check_measurements(data: dict[str, Any], part_names: list[str]) -> None:
    if set(data) != set(_MEASUREMENT_KEYS):
        raise ValueError(f"measurements の項目は {'・'.join(_MEASUREMENT_KEYS)} にしてください")
    if data["space"] != "world":
        raise ValueError("measurements.space は \"world\" だけを扱えます")
    if not isinstance(data["joints"], dict):
        raise ValueError("measurements.joints は {} の形にしてください")
    for name, position in data["joints"].items():
        _check_point(position, 3, f"measurements.joints の {name}")
    _check_string_list(data["missing_joints"], "measurements.missing_joints")
    ranges = data["mesh_z_range"]
    if not isinstance(ranges, dict) or set(ranges) != set(part_names):
        raise ValueError("measurements.mesh_z_range は、部品ごとの [下, 上] の辞書にしてください（部品の名前と合いません）")
    for name, z_range in ranges.items():
        if z_range is not None:
            _check_point(z_range, 2, f"measurements.mesh_z_range の {name}")
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


def empty_source() -> dict[str, str]:
    """まだ何も書いていない、素材の出所と利用条件"""
    return dict.fromkeys(SOURCE_FIELDS, "")


@dataclass(frozen=True, kw_only=True)
class ModelCard:
    """モデルの登録カード。

    schema_version: カードの形の版（SCHEMA_VERSION）
    blend_name: 測った .blend の名前（保存していないファイルは ""）
    armature_name: アーマチュアのオブジェクトの名前
    parts: 部品（メッシュのオブジェクト）の並び。名前は重ならない
    bone_mapping: 骨の名前の対応（bone_mapping_to_json の形）
    measurements: 測った値（measure_body の形）
    fingerprints: {"shapes": {メッシュの名前: 指紋}, "bones": 指紋}
    source: 素材の出所と利用条件 {"origin", "author", "url", "terms"}（ユーザーが書いた文字のまま）
    created_at: 作った時刻（ISO-8601。時差付き）
    tool_version: 拡張機能の版
    blender_version: Blender の版

    作るときに形を確かめ、辞書は JSON にできる写しにする（あとで外から書き換えられないように）。
    変えるときは set_part_kind・set_source などで、新しいカードを作る。
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

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"このカードの形は schema_version {SCHEMA_VERSION} だけを扱えます"
                             f"（いまは {_describe(self.schema_version)}）")
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
        _check_measurements(measurements, part_names)
        _check_fingerprints(fingerprints, part_names)
        _check_time(self.created_at)
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "bone_mapping", bone_mapping)
        object.__setattr__(self, "measurements", measurements)
        object.__setattr__(self, "fingerprints", fingerprints)
        object.__setattr__(self, "source", _check_source(self.source))

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

    def to_data(self) -> dict[str, Any]:
        """JSON にする形（辞書は写し）"""
        return {
            "schema_version": self.schema_version,
            "blend_name": self.blend_name,
            "armature_name": self.armature_name,
            "parts": [part.to_data() for part in self.parts],
            "bone_mapping": copy.deepcopy(self.bone_mapping),
            "measurements": copy.deepcopy(self.measurements),
            "fingerprints": copy.deepcopy(self.fingerprints),
            "source": dict(self.source),
            "created_at": self.created_at,
            "tool_version": self.tool_version,
            "blender_version": self.blender_version,
        }

    def to_json(self) -> str:
        """決まった形の JSON（キーの順をそろえ、字下げ 2、日本語はそのまま、NaN は書かない）。
        同じカードからは、いつも同じ文字になる"""
        return json.dumps(self.to_data(), sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> ModelCard:
        """to_json() の文字からカードを作る。from_json(card.to_json()) == card、to_json(from_json(x)) == x になる。

        JSON が読めない・同じキーが2回ある・NaN が書いてある・項目が違う・形が違うときは ValueError。
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
        if set(data) != set(_CARD_KEYS):
            missing = sorted(set(_CARD_KEYS) - set(data))
            unknown = sorted(set(data) - set(_CARD_KEYS))
            raise ValueError(f"カードの項目が違います（足りない項目：{missing}、知らない項目：{unknown}）")
        if not isinstance(data["parts"], list):
            raise ValueError("カードの parts は [] の形にしてください")
        return cls(**{**data, "parts": [Part.from_data(item) for item in data["parts"]]})


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
              source: Mapping[str, str] | None = None) -> ModelCard:
    """並び（世界の座標・休みの姿勢）からカードを作る。部品の種類は名前からの推測（kind_source "guess"）、
    服の種類は None（ユーザーが選ぶ）。source を渡さなければ、素材の出所と利用条件は空"""
    if not isinstance(bones, BoneArrays):
        raise ValueError(f"bones は BoneArrays にしてください（いまは {_describe(bones)}）")
    items = _check_meshes(meshes)
    parts = []
    for mesh in items:
        kind, keyword = guess_part_kind(mesh.name, mesh.material_names)
        parts.append(Part(object_name=mesh.name, kind=kind, kind_source="guess", cloth_type=None,
                          vertex_count=len(mesh.vertices), triangle_count=len(mesh.triangles),
                          material_names=mesh.material_names, matched_keyword=keyword))
    measurements = measure_body(bones.names, bones.heads, {mesh.name: mesh.vertices for mesh in items},
                                {part.object_name: part.kind for part in parts})
    return ModelCard(
        blend_name=blend_name,
        armature_name=armature_name,
        parts=parts,
        bone_mapping=bone_mapping_to_json(map_bones(bones.names)),
        measurements=measurements,
        fingerprints=fingerprints_of(bones, items),
        source=empty_source() if source is None else dict(source),
        created_at=created_at,
        tool_version=tool_version,
        blender_version=blender_version,
    )


def set_part_kind(card: ModelCard, object_name: str, kind: str, cloth_type: str | None = None) -> ModelCard:
    """ユーザーが部品の種類（と服の種類）を選んだ、新しいカード（kind_source "user"）。元のカードは変えない。

    kind は PART_KINDS、cloth_type は CLOTH_TYPE_IDS のどれか（服でない部品には付けられない）。
    体の部品が変わると背の高さ・頭身も変わるので、カードの中の関節の位置と高さの範囲から測り直す
    （形は読み直さない。形が変わったかどうかは指紋で見る）。
    """
    if not isinstance(card, ModelCard):
        raise ValueError(f"card は ModelCard にしてください（いまは {_describe(card)}）")
    _check_kind(kind)
    _check_cloth_type(kind, cloth_type)
    target = card.part(object_name)
    parts = [replace(part, kind=kind, kind_source="user", cloth_type=cloth_type) if part is target else part
             for part in card.parts]
    return replace(card, parts=parts, measurements=_remeasure_card(card, parts))


def set_source(card: ModelCard, **fields: str) -> ModelCard:
    """素材の出所と利用条件を書いた、新しいカード。渡した項目（origin・author・url・terms）だけを置きかえる。
    文字はそのまま残す（前後の空白も消さない）"""
    if not isinstance(card, ModelCard):
        raise ValueError(f"card は ModelCard にしてください（いまは {_describe(card)}）")
    unknown = sorted(set(fields) - set(SOURCE_FIELDS))
    if unknown:
        raise ValueError(f"素材の項目は {'・'.join(SOURCE_FIELDS)} のどれかにしてください（知らない項目：{unknown}）")
    return replace(card, source={**card.source, **fields})


def keep_user_choices(previous: ModelCard, fresh: ModelCard) -> ModelCard:
    """測り直した新しいカード（fresh）に、前のカードでユーザーが決めた物を写す。

    - 素材の出所と利用条件
    - 同じ名前の部品で、ユーザーが選んだ種類・服の種類（kind_source "user" の物だけ。推測は新しい推測のまま）
    部品の種類が変わると背の高さ・頭身も変わるので、測り直す。
    """
    for name, card in (("previous", previous), ("fresh", fresh)):
        if not isinstance(card, ModelCard):
            raise ValueError(f"{name} は ModelCard にしてください（いまは {_describe(card)}）")
    chosen = {part.object_name: part for part in previous.parts if part.kind_source == "user"}
    parts = []
    for part in fresh.parts:
        old = chosen.get(part.object_name)
        parts.append(part if old is None else replace(part, kind=old.kind, kind_source="user",
                                                      cloth_type=old.cloth_type))
    return replace(fresh, parts=parts, source=dict(previous.source), measurements=_remeasure_card(fresh, parts))


# MARK: - 測り直しが要るか


def remeasure_reasons(card: ModelCard, current_shape_fingerprints: Mapping[str, str],
                      current_bones_fingerprint: str) -> list[str]:
    """カードの指紋と、いまの指紋を比べ、測り直しが要る理由（日本語）を返す。空なら、カードはそのまま使える。

    並び：骨、カードにあるメッシュ（名前の順。形が変わった・なくなった）、増えたメッシュ（名前の順）。
    """
    if not isinstance(card, ModelCard):
        raise ValueError(f"card は ModelCard にしてください（いまは {_describe(card)}）")
    if not isinstance(current_shape_fingerprints, Mapping):
        raise ValueError("current_shape_fingerprints は、メッシュの名前 → 指紋 の辞書にしてください")
    if not isinstance(current_bones_fingerprint, str):
        raise ValueError("current_bones_fingerprint は文字にしてください")
    reasons = []
    if current_bones_fingerprint != card.fingerprints["bones"]:
        reasons.append("骨（名前・並び・親子・位置・ねじれ・形を動かすかどうか）が変わりました")
    stored = card.fingerprints["shapes"]
    for name in sorted(stored):
        if name not in current_shape_fingerprints:
            reasons.append(f"{name} がなくなりました")
        elif current_shape_fingerprints[name] != stored[name]:
            reasons.append(f"{name} の形が変わりました")
    for name in sorted(set(current_shape_fingerprints) - set(stored)):
        reasons.append(f"{name} が増えました")
    return reasons


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
    """カードを信号にまとめる。単位はどれも "count"。

    - "semi_standard_bones"：足りない MMD 標準・準標準の骨（補助の骨を除く）と、取り合いで決まらない骨の数。
      1つでもあれば黄。CLAUDE.md 3章：段階1では骨を足さず（M2）、足りない骨をここに一覧で出し、
      その骨が要る確かめのほうを灰にする。カードそのものは正しく測れているので、赤・灰にはしない。
    - "unknown_bones"：辞書に無い・1つに決まらない骨の名前と、取り合いの数。1つでもあれば黄（ユーザーに聞く）。
    - "unconfirmed_parts"：ユーザーがまだ決めていない部品の数（種類が推測のまま・"unknown"・服の種類が無い）。
      1つでもあれば黄。部品が1つも無ければ灰。
    - "body_measurements"：測れなかった値の数。1つでもあれば灰。全部測れたが、体の部品が無く全部のメッシュで
      背の高さを測ったときは黄。
    - "source_terms"：素材の出所と利用条件（REQUIRED_SOURCE_FIELDS）のうち、空の物の数。1つでもあれば黄。
    - "card_current"（reasons を渡したときだけ）：測り直しが要る理由の数。1つでもあれば灰（カードの値は、
      いまのモデルを測った値ではない）。
    """
    if not isinstance(card, ModelCard):
        raise ValueError(f"card は ModelCard にしてください（いまは {_describe(card)}）")
    mapping = card.bone_mapping
    missing = list(mapping["missing_standard"]) + list(mapping["missing_semi_standard"])
    conflicts = mapping["conflicts"]
    results = [
        _counted(SEMI_STANDARD_NAME, len(missing) + len(conflicts), Signal.YELLOW,
                 "MMD 準標準の骨が足りないか、決まらない骨があります（その骨が要る確かめは灰になります）",
                 {"missing_standard": list(mapping["missing_standard"]),
                  "missing_semi_standard": list(mapping["missing_semi_standard"]),
                  "conflicts": copy.deepcopy(conflicts)}),
        _counted(UNKNOWN_BONES_NAME, len(mapping["unknown"]) + len(conflicts), Signal.YELLOW,
                 "辞書に無い骨・1つに決まらない骨があります（ユーザーに聞きます）",
                 {"unknown": list(mapping["unknown"]), "ambiguous": copy.deepcopy(mapping["ambiguous"]),
                  "conflicts": copy.deepcopy(conflicts)}),
    ]
    unconfirmed = [part.object_name for part in card.parts if not part.is_confirmed]
    if not card.parts:
        results.append(Measurement.unmeasured(UNCONFIRMED_PARTS_NAME, "count", "部品（メッシュ）が1つもありません"))
    else:
        results.append(_counted(UNCONFIRMED_PARTS_NAME, len(unconfirmed), Signal.YELLOW,
                                "種類（服なら服の種類も）をまだ決めていない部品があります", {"parts": unconfirmed}))
    values = card.measurements["values"]
    unmeasured = {name: values[name]["reason"] for name in MEASUREMENT_NAMES if values[name]["value"] is None}
    details = {"unmeasured": unmeasured, "body_meshes": list(card.measurements["body_meshes"]),
               "body_meshes_source": card.measurements["body_meshes_source"]}
    if unmeasured:
        results.append(Measurement(BODY_MEASUREMENTS_NAME, float(len(unmeasured)), "count", Signal.GRAY,
                                   "測れなかった値があります：" + "、".join(unmeasured), details))
    elif card.measurements["body_meshes_source"] == BODY_SOURCE_ALL:
        results.append(Measurement(BODY_MEASUREMENTS_NAME, 0.0, "count", Signal.YELLOW,
                                   "体の部品が決まっていないので、背の高さを全部のメッシュで測りました", details))
    else:
        results.append(Measurement(BODY_MEASUREMENTS_NAME, 0.0, "count", Signal.GREEN, "", details))
    empty = [key for key in REQUIRED_SOURCE_FIELDS if not card.source[key].strip()]
    results.append(_counted(SOURCE_TERMS_NAME, len(empty), Signal.YELLOW,
                            "素材の出所と利用条件が書かれていません：" + "、".join(SOURCE_LABELS[key] for key in empty),
                            {"empty": empty}))
    if reasons is not None:
        reason_list = list(_names(reasons, "reasons"))
        if reason_list:
            results.append(Measurement(CARD_CURRENT_NAME, float(len(reason_list)), "count", Signal.GRAY,
                                       "測り直しが必要です：" + "。".join(reason_list), {"reasons": reason_list}))
        else:
            results.append(Measurement(CARD_CURRENT_NAME, 0.0, "count", Signal.GREEN, "", {"reasons": []}))
    return results
