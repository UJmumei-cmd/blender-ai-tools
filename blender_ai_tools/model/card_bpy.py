"""A1 モデルの登録カードを、いまのシーンから作る・ファイルの中に置く・確かめる所（Blender の中だけ）。

計算は card.py（bpy なし）。ここは model.extract_bpy で並びを取り出して渡すだけにする。

- build_card・build_card_with_notes：書き出しの対象（convert_blend.py と同じ決まりの find_model）の、
  骨の休みの形と、メッシュの元の形（モディファイアを通さない・シェイプキーは基本の形）を、アーマチュアの中の座標で
  読み、アーマチュアの置き方（matrix_world）と三角形ごとの材質も添えてカードを作る。
  前のカードを渡すと、同じモデルのときだけユーザーが決めた物を写す（card.carry_user_choices）。
- save_card・load_card：カードの JSON を、.blend の中のテキスト（名前 TEXT_NAME）に置く・読む。
  .blend のファイルは保存しない（保存するのはユーザー。元のファイルを上書きしない）。
- check_card：置いたカードと、いまのシーンを比べ、測り直しが要る理由と知らせを返す（CardCheck）。

どれも、フレーム・モード・アクティブなオブジェクト・選択・ポーズを変えない（読むだけ）。
指紋と位置はアーマチュアの中の座標で作る。モデル全体を動かす・Z まわりに回すだけなら測り直しにならず、
「置き場所が変わりました（形はそのまま）」の知らせだけになる。大きさ・傾きを変えたときは、世界の長さと高さが
変わるので測り直しになる。メッシュだけをアーマチュアに対して動かしたときは、形が変わったことになる。

休みの姿勢で測る：骨を親にしたメッシュ（parent_type が BONE。髪飾りなど）は、ポーズで世界の位置が動く。
そのメッシュは、親の骨が休みの姿勢にあるときの位置（extract_bpy.mesh_rest）で読む。ポーズを付けた・動きを
読み込んだだけでは、値も指紋も変わらない。ただし、オブジェクトの拘束（コンストレイント）が効いているものは
式で位置を出せないので、いまの位置のまま読む（拘束で骨に付けたメッシュは、ポーズで指紋が変わることがある）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy
import numpy as np

from .. import provenance
from . import extract_bpy
from .card import (
    BoneArrays, MeshArrays, ModelCard, bones_fingerprint, card_text, carry_user_choices, check_against, fingerprints_of,
    make_card,
)

# カードを置くテキストの名前
TEXT_NAME = "ai_tools_card.json"


# MARK: - シーンを読む


@dataclass(frozen=True)
class _ModelData:
    """カードを作る・確かめるのに要る物（どちらも、ここで読んだ同じ物から作る）"""

    armature: Any
    bones: BoneArrays
    meshes: tuple[MeshArrays, ...]
    placement: np.ndarray


def _check_scene(scene: Any) -> None:
    """位置は、いまのビューレイヤーで計算した物を読む。ほかのシーンは読めない"""
    if scene is not None and scene != bpy.context.scene:
        raise ValueError("登録カードは、いまのシーンのモデルだけを読めます（ほかのシーンは、いまのシーンにしてから読んでください）")


def _materials(obj: Any, triangle_count: int) -> tuple[tuple[str, ...], np.ndarray]:
    """(材質の名前（枠の順。空の枠は飛ばし、同じ名前は1つ）, 三角形ごとの材質の番号（その名前の番号。無いときは -1）)。

    空の枠の三角形と、枠の数以上の番号の三角形は「材質の無い三角形」（-1）にする。
    """
    indices, slot_names = extract_bpy.triangle_materials(obj)
    if len(indices) != triangle_count:
        raise ValueError(f"{obj.name}：三角形の数が、形と材質で違います（{triangle_count} と {len(indices)}）")
    names = tuple(dict.fromkeys(name for name in slot_names if name))
    lookup = np.array([names.index(name) if name else -1 for name in slot_names] + [-1], dtype=np.int64)
    inside = (indices >= 0) & (indices < len(slot_names))
    return names, lookup[np.where(inside, indices, len(slot_names))]


def _read_model(scene: Any) -> _ModelData:
    _check_scene(scene)
    model = extract_bpy.find_model(scene)
    armature = model.armature
    rest = extract_bpy.bone_rest(armature, space="armature")
    bones = BoneArrays(names=rest.names, parents=rest.parents, heads=rest.heads, tails=rest.tails,
                       deform=rest.deform, matrices=rest.matrices)
    meshes = []
    for obj in model.meshes:
        # 骨を親にしたメッシュも、mesh_rest が休みの姿勢の位置に置く（メッシュの中の座標から作るので、ポーズで指紋が揺れない）
        vertices, triangles = extract_bpy.mesh_rest(obj, space="armature", armature=armature)
        names, materials = _materials(obj, len(triangles))
        meshes.append(MeshArrays(name=obj.name, vertices=vertices, triangles=triangles, material_names=names,
                                 triangle_materials=materials))
    return _ModelData(armature=armature, bones=bones, meshes=tuple(meshes),
                      placement=extract_bpy.world_matrix(armature))


def _legacy_bones_fingerprint(armature: Any) -> str:
    """版 1 のカードと同じ作り方の骨の指紋（世界の座標）。版 1 の前のカードと同じモデルかを確かめるのに使う"""
    rest = extract_bpy.bone_rest(armature, space="world")
    return bones_fingerprint(rest.names, rest.parents, rest.heads, rest.tails, rest.deform, rest.matrices)


# MARK: - 作る・置く・読む


def build_card_with_notes(scene: Any = None, *, previous: ModelCard | None = None) -> tuple[ModelCard, list[str]]:
    """いまのシーンのモデルの登録カードを作る（置かない。置くのは save_card）。返す物は (カード, 知らせの文)。

    部品と材質の種類は名前からの推測、服の種類は None（ユーザーが選ぶ）。previous（前のカード）を渡すと、
    前のカードが同じモデルの物（アーマチュアの名前と骨の指紋が同じ）のときだけ、ユーザーが決めた部品と材質の種類・
    服の種類・骨の答え・素材の出所と利用条件を写す。違うモデルの物なら何も写さず、知らせの文でそう言う
    （card.carry_user_choices）。版 1 の前のカードは、世界の座標の骨の指紋で同じモデルかを確かめる。
    モデルが見つからない・読めない物があるときは ValueError（extract_bpy の日本語の説明のまま）。
    """
    data = _read_model(scene)
    path = bpy.data.filepath
    card = make_card(
        blend_name=Path(path).name if path else "",
        armature_name=data.armature.name,
        bones=data.bones,
        meshes=data.meshes,
        created_at=provenance.now_iso(),
        tool_version=provenance.addon_version(),
        blender_version=bpy.app.version_string,
        placement=data.placement,
    )
    if previous is None:
        return card, []
    legacy = _legacy_bones_fingerprint(data.armature) if previous.is_legacy else None
    return carry_user_choices(previous, card, legacy_bones_fingerprint=legacy)


def build_card(scene: Any = None, *, previous: ModelCard | None = None) -> ModelCard:
    """build_card_with_notes のカードだけを返す（知らせの文が要るときは build_card_with_notes を使う）"""
    return build_card_with_notes(scene, previous=previous)[0]


def save_card(card: ModelCard) -> str:
    """カードの JSON を、.blend の中のテキスト TEXT_NAME に置く（あれば中身を置きかえる）。テキストの名前を返す。

    .blend のファイルは保存しない。テキストには「偽のユーザー」を付け、使われていなくても保存のときに消えないようにする。
    ほかのファイルから読み込んだ（リンクした）テキストは書きかえられないので ValueError。
    置く前に、JSON から読み戻せるかを確かめる。カードの parts・辞書は、作った後でも書きかえられてしまうので、
    書きかえて形がくずれたカード（同じ部品が2つ など）は置かずに ValueError（置いてあったカードはそのまま）。
    """
    if not isinstance(card, ModelCard):
        raise ValueError(f"保存できるのは ModelCard だけです（いまは {type(card).__name__}）")
    # 先に JSON にして読み戻す（できないときは、何も変えない）
    try:
        text_json = card.to_json()
        ModelCard.from_json(text_json)
    except (TypeError, ValueError) as error:
        raise ValueError(f"このカードは読み戻せない形になっているので置けません：{error}") from error
    text = bpy.data.texts.get(TEXT_NAME)
    if text is not None and text.library is not None:
        raise ValueError(f"テキスト {TEXT_NAME} は、ほかのファイルから読み込んだ物なので書きかえられません")
    if text is None:
        text = bpy.data.texts.new(TEXT_NAME)
    text.clear()
    text.write(text_json + "\n")
    text.use_fake_user = True
    return text.name


def load_card() -> ModelCard | None:
    """置いたカードを読む。テキストが無ければ None。中身が読めない（形が違う）ときは ValueError。
    版 1 のカードも読める（is_legacy。変えられない）"""
    text = bpy.data.texts.get(TEXT_NAME)
    if text is None:
        return None
    return ModelCard.from_json(text.as_string())


# MARK: - 確かめる


class CardCheck(tuple):
    """check_card の結果。(カード, 測り直しが要る理由) の2つの組として使える（card, reasons = check_card()）。

    card: 置いたカード（無い・読めないときは None）
    reasons: 測り直しが要る理由（いまの言語の文）。空なら、カードはそのまま使える
    notes: 測り直しは要らないが、知らせること（置き場所が変わった など。いまの言語の文）
    組として比べるときは (card, reasons) だけを比べる（notes は入らない）。
    """

    notes: list[str]

    def __new__(cls, card: ModelCard | None, reasons: list[str], notes: list[str] | None = None) -> CardCheck:
        self = super().__new__(cls, (card, list(reasons)))
        self.notes = list(notes or [])
        return self

    @property
    def card(self) -> ModelCard | None:
        return self[0]

    @property
    def reasons(self) -> list[str]:
        return self[1]

    def __repr__(self) -> str:
        return f"CardCheck(card={self.card!r}, reasons={self.reasons!r}, notes={self.notes!r})"


def current_fingerprints(scene: Any = None) -> tuple[str, dict[str, str], str]:
    """いまのシーンの (アーマチュアの名前, メッシュの指紋, 骨の指紋)。build_card と同じ読み方・作り方
    （アーマチュアの中の座標）"""
    data = _read_model(scene)
    prints = fingerprints_of(data.bones, data.meshes)
    return data.armature.name, prints["shapes"], prints["bones"]


def check_card(scene: Any = None) -> CardCheck:
    """置いたカードが、いまのモデルにまだ合うかを確かめる。返す物は CardCheck（カード, 理由, 知らせ）。

    理由が空なら、カードはそのまま使える。
    - カードが無い：(None, ["登録カードがありません…"])
    - カードが読めない：(None, ["登録カードが読めません：…"])
    - 版 1 の古いカード：(カード, ["古い形（版 1）の登録カードです…"])（一度測り直す）
    - いまのモデルが読めない：(カード, ["いまのモデルを読めません：…"])
    - アーマチュアの名前が変わった・指紋が違う・大きさか傾きが変わった：(カード, 理由の並び)
    - 置き場所だけが変わった（平行移動・Z まわりの回転）：理由は空のまま、知らせ notes に
      「置き場所が変わりました（形はそのまま）」
    """
    try:
        card = load_card()
    except ValueError as error:
        return CardCheck(None, [card_text("card.unreadable_card", error=str(error))])
    if card is None:
        return CardCheck(None, [card_text("card.no_card")])
    if card.is_legacy:
        reasons, notes = check_against(card, armature_name=card.armature_name, shapes={},
                                       bones=card.fingerprints["bones"], placement=None)
        return CardCheck(card, reasons, notes)
    try:
        data = _read_model(scene)
    except ValueError as error:
        return CardCheck(card, [card_text("card.unreadable_model", error=str(error))])
    prints = fingerprints_of(data.bones, data.meshes)
    reasons, notes = check_against(card, armature_name=data.armature.name, shapes=prints["shapes"],
                                   bones=prints["bones"], placement=data.placement)
    return CardCheck(card, reasons, notes)
