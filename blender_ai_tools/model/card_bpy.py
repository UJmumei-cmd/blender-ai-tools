"""A1 モデルの登録カードを、いまのシーンから作る・ファイルの中に置く・確かめる所（Blender の中だけ）。

計算は card.py（bpy なし）。ここは model.extract_bpy で並びを取り出して渡すだけにする。

- build_card：書き出しの対象（convert_blend.py と同じ決まりの find_model）の、骨の休みの形（世界の座標）と、
  メッシュの元の形（モディファイアを通さない・シェイプキーは基本の形・世界の座標）からカードを作る。
- save_card・load_card：カードの JSON を、.blend の中のテキスト（名前 TEXT_NAME）に置く・読む。
  .blend のファイルは保存しない（保存するのはユーザー。元のファイルを上書きしない）。
- check_card：置いたカードの指紋と、いまのシーンの指紋を比べ、測り直しが要る理由を返す。

どれも、フレーム・モード・アクティブなオブジェクト・選択・ポーズを変えない（読むだけ）。
指紋は世界の座標で作る。アーマチュアやメッシュを動かしただけでも「測り直しが必要」になる
（カードの関節の位置は世界の座標なので、動かすと値が古くなるため）。

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

from .. import provenance
from . import extract_bpy
from .card import BoneArrays, MeshArrays, ModelCard, fingerprints_of, keep_user_choices, make_card, remeasure_reasons

# カードを置くテキストの名前
TEXT_NAME = "ai_tools_card.json"


# MARK: - シーンを読む


@dataclass(frozen=True)
class _ModelData:
    """カードを作る・確かめるのに要る物（どちらも、ここで読んだ同じ物から作る）"""

    armature_name: str
    bones: BoneArrays
    meshes: tuple[MeshArrays, ...]


def _check_scene(scene: Any) -> None:
    """位置は、いまのビューレイヤーで計算した物を読む。ほかのシーンは読めない"""
    if scene is not None and scene != bpy.context.scene:
        raise ValueError("登録カードは、いまのシーンのモデルだけを読めます（ほかのシーンは、いまのシーンにしてから読んでください）")


def _material_names(obj: Any) -> tuple[str, ...]:
    """オブジェクトの材質の名前（材質の枠の順。空の枠は飛ばし、同じ名前は1つ）"""
    names: list[str] = []
    for slot in obj.material_slots:
        material = slot.material
        if material is not None and material.name not in names:
            names.append(material.name)
    return tuple(names)


def _read_model(scene: Any) -> _ModelData:
    _check_scene(scene)
    model = extract_bpy.find_model(scene)
    rest = extract_bpy.bone_rest(model.armature, space="world")
    bones = BoneArrays(names=rest.names, parents=rest.parents, heads=rest.heads, tails=rest.tails,
                       deform=rest.deform, matrices=rest.matrices)
    meshes = []
    for obj in model.meshes:
        # 骨を親にしたメッシュも、mesh_rest が休みの姿勢の位置に置く（メッシュの中の座標から作るので、ポーズで指紋が揺れない）
        vertices, triangles = extract_bpy.mesh_rest(obj, space="world")
        meshes.append(MeshArrays(name=obj.name, vertices=vertices, triangles=triangles,
                                 material_names=_material_names(obj)))
    return _ModelData(armature_name=model.armature.name, bones=bones, meshes=tuple(meshes))


# MARK: - 作る・置く・読む


def build_card(scene: Any = None, *, previous: ModelCard | None = None) -> ModelCard:
    """いまのシーンのモデルの登録カードを作る（置かない。置くのは save_card）。

    部品の種類は名前からの推測、服の種類は None（ユーザーが選ぶ）。previous（前のカード）を渡すと、
    ユーザーが決めた部品の種類・服の種類と、素材の出所と利用条件を写す（keep_user_choices）。
    モデルが見つからない・読めない物があるときは ValueError（extract_bpy の日本語の説明のまま）。
    """
    data = _read_model(scene)
    path = bpy.data.filepath
    card = make_card(
        blend_name=Path(path).name if path else "",
        armature_name=data.armature_name,
        bones=data.bones,
        meshes=data.meshes,
        created_at=provenance.now_iso(),
        tool_version=provenance.addon_version(),
        blender_version=bpy.app.version_string,
    )
    return card if previous is None else keep_user_choices(previous, card)


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
    """置いたカードを読む。テキストが無ければ None。中身が読めない（形が違う）ときは ValueError"""
    text = bpy.data.texts.get(TEXT_NAME)
    if text is None:
        return None
    return ModelCard.from_json(text.as_string())


# MARK: - 確かめる


def current_fingerprints(scene: Any = None) -> tuple[str, dict[str, str], str]:
    """いまのシーンの (アーマチュアの名前, メッシュの指紋, 骨の指紋)。build_card と同じ読み方・作り方"""
    data = _read_model(scene)
    prints = fingerprints_of(data.bones, data.meshes)
    return data.armature_name, prints["shapes"], prints["bones"]


def check_card(scene: Any = None) -> tuple[ModelCard | None, list[str]]:
    """置いたカードが、いまのモデルにまだ合うかを確かめる。返す物は (カード, 測り直しが要る理由)。

    理由が空なら、カードはそのまま使える。
    - カードが無い：(None, ["登録カードがありません…"])
    - カードが読めない：(None, ["登録カードが読めません：…"])
    - いまのモデルが読めない：(カード, ["いまのモデルを読めません：…"])
    - アーマチュアの名前が変わった・指紋が違う：(カード, 理由の並び)
    """
    try:
        card = load_card()
    except ValueError as error:
        return None, [f"登録カードが読めません：{error}"]
    if card is None:
        return None, ["登録カードがありません。先にモデルを登録してください"]
    try:
        armature_name, shapes, bones = current_fingerprints(scene)
    except ValueError as error:
        return card, [f"いまのモデルを読めません：{error}"]
    reasons = []
    if armature_name != card.armature_name:
        reasons.append(f"アーマチュアが変わりました（{card.armature_name} → {armature_name}）")
    reasons.extend(remeasure_reasons(card, shapes, bones))
    return card, reasons
