"""A6 2つの版の記録（snapshot.py）の違いを、変わった所の一覧にする（bpy なし・numpy だけ）。

- diff_snapshots：前の版と新しい版の記録を比べて、変わった所（Change）の並びを返す。
- summarize_changes：「変えてよい所」の外の変化を、1つの信号（"unexpected_changes"）にまとめる。
  外の変化が1つでもあれば赤（ロードマップ段階1 基準5：変えてよい所の外の変化が0）。
- change_lines：変わった所を、決まった文の形（texts/versions.py）に数字を入れた文にする（CLAUDE.md 2章3）。

比べ方：
- 再現性の幅より小さい差は、変化として数えない（CLAUDE.md 7章・ロードマップ段階1 基準5）。
  長さ（頂点・シェイプキー・骨・オブジェクトの位置）は tolerance（ふつう thresholds.REPRO_TOLERANCE_LENGTH）より
  大きく動いた物だけ（ちょうどは含まない。measure.deformation.changed_vertices と同じ数え方）。
  ウェイトは weight_tolerance（ふつう thresholds.REPRO_TOLERANCE_WEIGHT）より大きく違う物だけ。
  オブジェクトの回転・大きさ（行列の 3×3 の成分）は ratio_tolerance（ふつう thresholds.REPRO_TOLERANCE_RATIO）。
- 名前で対応させる：オブジェクト・シェイプキー・頂点グループ・骨・モディファイア・ドライバー（持ち主・場所・番号）。
  名前を変えた物は「無くなった」と「増えた」の2つになる。
- 頂点は番号で対応させる。頂点の数が変わったメッシュは、頂点ごとの比べ（頂点・シェイプキーの形・ウェイト）を
  できないので「形のつながりが変わった」だけを出す（名前の増減は出す）。
- 形のつながりは、面（角の頂点の並び）で比べる（四角形を三角形に分ける向きは、頂点が動くだけで変わることがあるため）。
- シェイプキーの形は、基本の形（vertices）からのずれで比べる。基本の形の頂点を動かして、ほかのキーも一緒に動いた
  （Blender の編集モードで Basis を動かしたとき）なら、キーの形は変わっていない。
- 増えた・無くなったオブジェクト、種類の変わったオブジェクトは、それだけを出す（中身は比べない）。

「変えてよい所」（AllowedChanges）：オブジェクトの名前 → 変えてよい種類（CATEGORIES の物の集まり、か "*" で全部）。
書いていないオブジェクト・種類の変化は、どれも「変えてよい所の外」（outside_allowed が True）。
allowed を渡さないときは、どこも変えてはいけない（measure.deformation.changed_vertices の allowed=None と同じ）。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .. import texts, thresholds
from ..measure.deformation import CHANGES_NAME, changed_vertices, displacement
from ..measure.geometry import as_nonnegative_float
from ..measure.results import Measurement, Signal
from .snapshot import ArmatureState, Driver, MeshState, ObjectState, Snapshot

# MARK: - 決まり

# 変わった所の種類（この順に並べる）
CATEGORIES = ("object", "vertices", "topology", "shape_keys", "weights", "bones", "drivers", "materials", "modifiers")
# AllowedChanges で「どの種類も変えてよい」を表す印
ALL_CATEGORIES = "*"

# MARK: - 計算・表示のための数字（判定の基準ではない）

# 1行に書く名前の数。これより多いときは「ほか N 個」にする
MAX_NAMES_IN_LINE = 5
# ウェイトを比べるとき、一度に作る (頂点 × 頂点グループ) の表の大きさの上限（1つ 8 バイト。約 32 MB）
WEIGHT_CHUNK_ELEMENTS = 1 << 22

_TEXT_PREFIX = "versions."


# MARK: - 変わった所

def _check_value(key: str, value: Any) -> Any:
    """文に入れる値：文字・整数・有限の小数・None・文字の並び（tuple）だけ"""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"文に入れる値 {key} に True・False は使えません")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"文に入れる値 {key} が NaN か無限大です")
        return number
    if isinstance(value, (tuple, list)) and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError(f"文に入れる値 {key} は、文字・数・None・文字の並びにしてください（いまの値：{value!r}）")


@dataclass(frozen=True)
class Change:
    """変わった所の1つ。

    category: 種類（CATEGORIES のどれか）
    object: オブジェクトの名前
    text_key: 文の形のキー（texts/versions.py の "versions.*"）
    values: 文に入れる値（キー → 文字・数・None・文字の並び）。文の形の {name} とそろえる
    count: 変わった物の数（頂点の数・骨の数・キーの数など。オブジェクトの置き方などは 1）
    outside_allowed: 「変えてよい所」の外の変化か
    """

    category: str
    object: str
    text_key: str
    values: Mapping[str, Any]
    count: int
    outside_allowed: bool

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"category は {CATEGORIES} のどれかにしてください（いまの値：{self.category!r}）")
        if not isinstance(self.object, str) or not self.object:
            raise ValueError(f"object はオブジェクトの名前にしてください（いまの値：{self.object!r}）")
        if not isinstance(self.text_key, str) or not self.text_key.startswith(_TEXT_PREFIX) \
                or self.text_key not in texts.TEXTS:
            raise ValueError(f"text_key は texts/versions.py の文のキーにしてください（いまの値：{self.text_key!r}）")
        if not isinstance(self.values, Mapping):
            raise ValueError("values は {名前: 値} にしてください")
        values = {key: _check_value(key, value) for key, value in self.values.items()}
        needed = texts.placeholders(texts.TEXTS[self.text_key][texts.FALLBACK_LANGUAGE])
        if needed != set(values):
            raise ValueError(f"{self.text_key}: 入れる値が合いません（いる：{sorted(needed)}、渡した：{sorted(values)}）")
        if isinstance(self.count, (bool, np.bool_)) or not isinstance(self.count, (int, np.integer)) or self.count < 0:
            raise ValueError(f"count は 0 以上の整数にしてください（いまの値：{self.count!r}）")
        if not isinstance(self.outside_allowed, (bool, np.bool_)):
            raise ValueError(f"outside_allowed は True か False にしてください（いまの値：{self.outside_allowed!r}）")
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "count", int(self.count))
        object.__setattr__(self, "outside_allowed", bool(self.outside_allowed))

    def text(self, language: str | None = None) -> str:
        """この変化の文（「変えてよい所の外」の印は付けない）"""
        return texts.text(self.text_key, language=language, **_display_values(self.values, language))


@dataclass(frozen=True)
class AllowedChanges:
    """「変えてよい所」：オブジェクトの名前 → 変えてよい種類（CATEGORIES の物の集まり）か "*"（どの種類も）。

    例：AllowedChanges({"Skirt": {"vertices", "weights"}, "Hair": "*"})
    """

    objects: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.objects, Mapping):
            raise ValueError("objects は {オブジェクトの名前: 種類の集まりか \"*\"} にしてください")
        normalized: dict[str, Any] = {}
        for name, categories in self.objects.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"変えてよい所のオブジェクトの名前は文字にしてください（いまの値：{name!r}）")
            if categories == ALL_CATEGORIES:
                normalized[name] = ALL_CATEGORIES
                continue
            if isinstance(categories, (str, bytes)) or not isinstance(categories, Iterable):
                raise ValueError(f"{name} の変えてよい種類は、種類の集まりか \"*\" にしてください（いまの値：{categories!r}）")
            found = frozenset(categories)
            unknown = sorted(str(c) for c in found if c not in CATEGORIES)
            if unknown:
                raise ValueError(f"{name} の変えてよい種類に、知らない種類があります：{unknown}（使えるのは {CATEGORIES}）")
            normalized[name] = found
        object.__setattr__(self, "objects", MappingProxyType(normalized))

    def allows(self, object_name: str, category: str) -> bool:
        """object_name の category の変化を、変えてよいか"""
        if category not in CATEGORIES:
            raise ValueError(f"category は {CATEGORIES} のどれかにしてください（いまの値：{category!r}）")
        allowed = self.objects.get(object_name)
        if allowed is None:
            return False
        return allowed == ALL_CATEGORIES or category in allowed


_NOTHING_ALLOWED = AllowedChanges({})


# MARK: - 比べる

class _Collector:
    """変わった所を集める（変えてよいかを、ここで決める）"""

    def __init__(self, allowed: AllowedChanges) -> None:
        self.allowed = allowed
        self.changes: list[Change] = []

    def add(self, category: str, object_name: str, key: str, items: int, /, **values: Any) -> None:
        """変化を1つ足す。items は Change.count（変わった物の数）。values は文に入れる値（"object" は自動で入れる）"""
        self.changes.append(Change(category=category, object=object_name, text_key=_TEXT_PREFIX + key,
                                   values={"object": object_name, **values}, count=items,
                                   outside_allowed=not self.allowed.allows(object_name, category)))


def _common_in_order(names: Sequence[str], others: Iterable[str]) -> list[str]:
    other = set(others)
    return [name for name in names if name in other]


def _added_removed(old: Sequence[str], new: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(new にだけある名前（new の順）, old にだけある名前（old の順）)"""
    old_set, new_set = set(old), set(new)
    return tuple(n for n in new if n not in old_set), tuple(n for n in old if n not in new_set)


def _names_changes(out: _Collector, category: str, name: str, old: Sequence[str], new: Sequence[str],
                   added_key: str, removed_key: str) -> None:
    added, removed = _added_removed(old, new)
    if added:
        out.add(category, name, added_key, len(added), count=len(added), names=added)
    if removed:
        out.add(category, name, removed_key, len(removed), count=len(removed), names=removed)


def _parent_label(parent: str | None, bone: str) -> str | None:
    if parent is None:
        return None
    return f"{parent} / {bone}" if bone else parent


def _diff_object(out: _Collector, name: str, a: ObjectState, b: ObjectState, tolerance: float,
                 ratio_tolerance: float) -> None:
    old_parent, new_parent = _parent_label(a.parent, a.parent_bone), _parent_label(b.parent, b.parent_bone)
    if old_parent != new_parent:
        out.add("object", name, "object_parent_changed", 1, old=old_parent, new=new_parent)
    distance = float(np.linalg.norm(b.matrix[:3, 3] - a.matrix[:3, 3]))
    linear = float(np.abs(b.matrix[:3, :3] - a.matrix[:3, :3]).max())
    projective = float(np.abs(b.matrix[3] - a.matrix[3]).max())
    if distance > tolerance or linear > ratio_tolerance or projective > ratio_tolerance:
        out.add("object", name, "object_moved", 1, distance_mm=distance * 1000.0, linear=max(linear, projective))


def _diff_modifiers(out: _Collector, name: str, a: ObjectState, b: ObjectState) -> None:
    old = {m[0]: m for m in a.modifiers}
    new = {m[0]: m for m in b.modifiers}
    _names_changes(out, "modifiers", name, list(old), list(new), "modifiers_added", "modifiers_removed")
    common = _common_in_order(list(new), old)
    retyped = tuple(m for m in common if old[m][1] != new[m][1])
    if retyped:
        out.add("modifiers", name, "modifiers_type_changed", len(retyped), count=len(retyped), names=retyped)
    render = tuple(m for m in common if old[m][2] != new[m][2])
    if render:
        out.add("modifiers", name, "modifiers_render_changed", len(render), count=len(render), names=render)
    if _common_in_order(list(old), new) != common:
        out.add("modifiers", name, "modifier_order_changed", 1, names=tuple(common))


def _face_materials(mesh: MeshState, slots: Sequence[str], lookup: Mapping[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """面ごとの材質：(枠の中の番号か, 値)。枠の中なら値は材質の名前の番号（lookup）、枠の外なら番号そのまま。
    枠の並びを入れ替えただけ（面の材質の名前は同じ）なら、同じ値になる"""
    index = mesh.face_materials
    inside = (index >= 0) & (index < len(slots))
    names = np.array([lookup[slot] for slot in slots] + [-1], dtype=np.int64)
    return inside, np.where(inside, names[np.where(inside, index, len(slots))], index)


def _diff_materials(out: _Collector, name: str, a: ObjectState, b: ObjectState, old_mesh: MeshState | None,
                    new_mesh: MeshState | None, same_faces: bool) -> None:
    if a.materials != b.materials:
        out.add("materials", name, "material_slots_changed", 1, old=a.materials, new=b.materials)
    if old_mesh is not None and new_mesh is not None and same_faces:
        lookup = {slot: i for i, slot in enumerate(dict.fromkeys(a.materials + b.materials))}
        inside_a, value_a = _face_materials(old_mesh, a.materials, lookup)
        inside_b, value_b = _face_materials(new_mesh, b.materials, lookup)
        changed = int(((inside_a != inside_b) | (value_a != value_b)).sum())
        if changed:
            out.add("materials", name, "face_materials_changed", changed, count=changed)


def _same_faces(a: MeshState, b: MeshState) -> bool:
    return (a.vertex_count == b.vertex_count and np.array_equal(a.face_sizes, b.face_sizes)
            and np.array_equal(a.face_vertices, b.face_vertices))


def _weight_differences(a: MeshState, b: MeshState) -> np.ndarray:
    """頂点ごとの、頂点グループの重みの差のいちばん大きい値 (N,)（グループは名前で対応させる。無いグループは重み 0）"""
    lookup = {g: i for i, g in enumerate(dict.fromkeys(a.group_names + b.group_names))}
    maps = [np.array([lookup[g] for g in mesh.group_names], dtype=np.int64) for mesh in (a, b)]
    width = max(len(lookup), 1)
    rows_per_chunk = max(1, WEIGHT_CHUNK_ELEMENTS // width)
    count = a.vertex_count
    result = np.zeros(count, dtype=np.float64)
    for start in range(0, count, rows_per_chunk):
        stop = min(start + rows_per_chunk, count)
        dense = []
        for mesh, mapping in zip((a, b), maps):
            ids = mesh.weights.bone_ids[start:stop]
            used = ids >= 0
            rows = np.broadcast_to(np.arange(stop - start)[:, None], ids.shape)[used]
            flat = rows * width + mapping[ids[used]]
            dense.append(np.bincount(flat, weights=mesh.weights.weights[start:stop][used],
                                     minlength=(stop - start) * width).reshape(stop - start, width))
        result[start:stop] = np.abs(dense[0] - dense[1]).max(axis=1)
    return result


def _diff_mesh(out: _Collector, name: str, a: MeshState, b: MeshState, tolerance: float,
               weight_tolerance: float) -> bool:
    """メッシュの違いを集める。形のつながり（面）が同じなら True"""
    same_count = a.vertex_count == b.vertex_count
    if same_count:
        moved = changed_vertices(a.vertices, b.vertices, tolerance=tolerance)
        if len(moved):
            distance = float(displacement(a.vertices, b.vertices)[moved].max())
            out.add("vertices", name, "vertices_moved", len(moved), count=len(moved), distance_mm=distance * 1000.0)
    same_faces = _same_faces(a, b)
    if not same_faces:
        out.add("topology", name, "topology_changed", 1, old_vertices=a.vertex_count, new_vertices=b.vertex_count,
                old_faces=a.face_count, new_faces=b.face_count)

    old_keys, new_keys = list(a.shape_keys), list(b.shape_keys)
    _names_changes(out, "shape_keys", name, old_keys, new_keys, "shape_keys_added", "shape_keys_removed")
    common = _common_in_order(new_keys, old_keys)
    if same_count:
        for key in common:
            old_offset = a.shape_keys[key] - a.vertices
            new_offset = b.shape_keys[key] - b.vertices
            moved = changed_vertices(old_offset, new_offset, tolerance=tolerance)
            if len(moved):
                distance = float(displacement(old_offset, new_offset)[moved].max())
                out.add("shape_keys", name, "shape_key_changed", len(moved), name=key, count=len(moved),
                        distance_mm=distance * 1000.0)
    if _common_in_order(old_keys, new_keys) != common:
        out.add("shape_keys", name, "shape_key_order_changed", 1, names=tuple(common))

    _names_changes(out, "weights", name, a.group_names, b.group_names, "groups_added", "groups_removed")
    if same_count and a.vertex_count:
        differences = _weight_differences(a, b)
        changed = differences > weight_tolerance
        if changed.any():
            count = int(changed.sum())
            out.add("weights", name, "weights_changed", count, count=count, difference=float(differences.max()))
    return same_faces


def _roll_points(arm: ArmatureState) -> np.ndarray | None:
    """骨のねじれ（ロール）を表す点：根元 + 骨の Z 軸の向き × 骨の長さ（model/card.py の指紋と同じ考え方）"""
    if arm.matrices is None:
        return None
    z_axes = arm.matrices[:, :3, 2]
    norms = np.linalg.norm(z_axes, axis=1)
    unit = np.zeros_like(z_axes)
    good = norms > 0.0
    unit[good] = z_axes[good] / norms[good, None]
    lengths = np.linalg.norm(arm.tails - arm.heads, axis=1)
    return arm.heads + unit * lengths[:, None]


def _parent_names(arm: ArmatureState) -> list[str | None]:
    return [None if p < 0 else arm.names[p] for p in arm.parents]


def _diff_armature(out: _Collector, name: str, a: ArmatureState, b: ArmatureState, tolerance: float) -> None:
    _names_changes(out, "bones", name, a.names, b.names, "bones_added", "bones_removed")
    common = _common_in_order(b.names, a.names)
    if common:
        ia = np.array([a.names.index(n) for n in common], dtype=np.int64)
        ib = np.array([b.names.index(n) for n in common], dtype=np.int64)
        distances = np.maximum(np.linalg.norm(b.heads[ib] - a.heads[ia], axis=1),
                               np.linalg.norm(b.tails[ib] - a.tails[ia], axis=1))
        rolls_a, rolls_b = _roll_points(a), _roll_points(b)
        if rolls_a is not None and rolls_b is not None:
            distances = np.maximum(distances, np.linalg.norm(rolls_b[ib] - rolls_a[ia], axis=1))
        moved = distances > tolerance
        if moved.any():
            names = tuple(n for n, m in zip(common, moved) if m)
            out.add("bones", name, "bones_moved", len(names), count=len(names), names=names,
                    distance_mm=float(distances[moved].max()) * 1000.0)
        parents_a, parents_b = _parent_names(a), _parent_names(b)
        reparented = tuple(n for n, i, j in zip(common, ia, ib) if parents_a[i] != parents_b[j])
        if reparented:
            out.add("bones", name, "bone_parents_changed", len(reparented), count=len(reparented), names=reparented)
        deform = tuple(n for n, i, j in zip(common, ia, ib) if bool(a.deform[i]) != bool(b.deform[j]))
        if deform:
            out.add("bones", name, "bone_deform_changed", len(deform), count=len(deform), names=deform)
    if _common_in_order(a.names, b.names) != common:
        out.add("bones", name, "bone_order_changed", 1)


def _drivers_by_owner(drivers: Iterable[Driver]) -> dict[str, dict[tuple[str, str, int], tuple[Driver, ...]]]:
    grouped: dict[str, dict[tuple[str, str, int], tuple[Driver, ...]]] = {}
    for driver in drivers:
        table = grouped.setdefault(driver.owner, {})
        table[driver.key] = table.get(driver.key, ()) + (driver,)
    return grouped


def _driver_label(key: tuple[str, str, int]) -> str:
    _, path, index = key
    return f"{path}[{index}]" if index else path


def _diff_drivers(out: _Collector, name: str, old: Mapping[tuple[str, str, int], tuple[Driver, ...]],
                  new: Mapping[tuple[str, str, int], tuple[Driver, ...]]) -> None:
    added = tuple(_driver_label(k) for k in new if k not in old)
    removed = tuple(_driver_label(k) for k in old if k not in new)
    changed = tuple(_driver_label(k) for k in new if k in old and old[k] != new[k])
    if added:
        out.add("drivers", name, "drivers_added", len(added), count=len(added), names=added)
    if removed:
        out.add("drivers", name, "drivers_removed", len(removed), count=len(removed), names=removed)
    if changed:
        out.add("drivers", name, "drivers_changed", len(changed), count=len(changed), names=changed)


def _as_allowed(allowed: Any) -> AllowedChanges:
    if allowed is None:
        return _NOTHING_ALLOWED
    if isinstance(allowed, AllowedChanges):
        return allowed
    if isinstance(allowed, Mapping):
        return AllowedChanges(allowed)
    raise ValueError(f"allowed は AllowedChanges か {{オブジェクトの名前: 種類の集まり}} にしてください"
                     f"（いまの型：{type(allowed).__name__}）")


def diff_snapshots(old: Snapshot, new: Snapshot, *, allowed: AllowedChanges | Mapping[str, Any] | None = None,
                   tolerance: float = thresholds.REPRO_TOLERANCE_LENGTH,
                   weight_tolerance: float = thresholds.REPRO_TOLERANCE_WEIGHT,
                   ratio_tolerance: float = thresholds.REPRO_TOLERANCE_RATIO) -> list[Change]:
    """前の版（old）と新しい版（new）の記録の違い。並びは、オブジェクトの名前の順、その中は CATEGORIES の順。

    allowed：変えてよい所（AllowedChanges か、それに渡す辞書）。None なら、どこも変えてはいけない。
    tolerance（m）・weight_tolerance・ratio_tolerance：これより大きい差だけを変化として数える（ちょうどは含まない）。
    """
    if not isinstance(old, Snapshot) or not isinstance(new, Snapshot):
        raise ValueError("diff_snapshots には Snapshot を2つ渡してください")
    limit = as_nonnegative_float(tolerance, "tolerance")
    weight_limit = as_nonnegative_float(weight_tolerance, "weight_tolerance")
    ratio_limit = as_nonnegative_float(ratio_tolerance, "ratio_tolerance")
    out = _Collector(_as_allowed(allowed))
    old_drivers, new_drivers = _drivers_by_owner(old.drivers), _drivers_by_owner(new.drivers)
    for name in sorted(set(old.objects) | set(new.objects)):
        a, b = old.objects.get(name), new.objects.get(name)
        if a is None:
            out.add("object", name, "object_added", 1, type=b.type)
            continue
        if b is None:
            out.add("object", name, "object_removed", 1, type=a.type)
            continue
        if a.type != b.type:
            out.add("object", name, "object_type_changed", 1, old=a.type, new=b.type)
            continue
        _diff_object(out, name, a, b, limit, ratio_limit)
        old_mesh, new_mesh = old.meshes.get(name), new.meshes.get(name)
        same_faces = False
        if old_mesh is not None and new_mesh is not None:
            same_faces = _diff_mesh(out, name, old_mesh, new_mesh, limit, weight_limit)
        if name in old.armatures and name in new.armatures:
            _diff_armature(out, name, old.armatures[name], new.armatures[name], limit)
        _diff_drivers(out, name, old_drivers.get(name, {}), new_drivers.get(name, {}))
        _diff_materials(out, name, a, b, old_mesh, new_mesh, same_faces)
        _diff_modifiers(out, name, a, b)
    order = {category: i for i, category in enumerate(CATEGORIES)}
    return sorted(out.changes, key=lambda change: (change.object, order[change.category]))


# MARK: - まとめと文

def _as_changes(changes: Any) -> list[Change]:
    if isinstance(changes, (str, bytes)) or not isinstance(changes, Iterable):
        raise ValueError("changes は Change の並びにしてください")
    result = list(changes)
    for change in result:
        if not isinstance(change, Change):
            raise ValueError(f"changes の中身は Change にしてください（いまの型：{type(change).__name__}）")
    return result


def summarize_changes(changes: Iterable[Change], *, language: str | None = None) -> Measurement:
    """変わった所を1つの信号にまとめる（名前 "unexpected_changes"、単位 "count"）。

    値は「変えてよい所」の外の変化の件数（Change の数）。1件でもあれば赤、無ければ緑（理由に、変えてよい所の中の件数）。
    details：outside（外の Change）・allowed（中の Change）・outside_items（外の変化の count の合計）。
    理由の文は texts の文（language が None なら、いまの言語）。
    """
    found = _as_changes(changes)
    outside = tuple(c for c in found if c.outside_allowed)
    inside = tuple(c for c in found if not c.outside_allowed)
    details = {"outside": outside, "allowed": inside, "outside_items": sum(c.count for c in outside),
               "change_count": len(found)}
    if outside:
        reason = texts.text("versions.summary_outside", language=language, count=len(outside), allowed=len(inside))
        return Measurement(CHANGES_NAME, float(len(outside)), "count", Signal.RED, reason, details)
    if inside:
        reason = texts.text("versions.summary_allowed", language=language, allowed=len(inside))
    else:
        reason = texts.text("versions.summary_none", language=language)
    return Measurement(CHANGES_NAME, 0.0, "count", Signal.GREEN, reason, details)


def _join_names(names: Sequence[str], language: str | None) -> str:
    if not names:
        return texts.text("versions.none_name", language=language)
    shown = [name if name else texts.text("versions.none_name", language=language) for name in names]
    separator = texts.text("versions.list_separator", language=language)
    joined = separator.join(shown[:MAX_NAMES_IN_LINE])
    if len(shown) > MAX_NAMES_IN_LINE:
        return texts.text("versions.more_names", language=language, names=joined,
                          count=len(shown) - MAX_NAMES_IN_LINE)
    return joined


def _display_values(values: Mapping[str, Any], language: str | None) -> dict[str, Any]:
    """文に入れる形にする：None は「（なし）」、名前の並びは区切りでつなぐ。数はそのまま（文の形が桁を決める）"""
    shown: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            shown[key] = texts.text("versions.none_name", language=language)
        elif isinstance(value, tuple):
            shown[key] = _join_names(value, language)
        else:
            shown[key] = value
    return shown


def change_lines(changes: Iterable[Change], language: str | None = None, *, mark_outside: bool = True) -> list[str]:
    """変わった所を、1件1行の文にする（language が None なら、いまの言語）。

    mark_outside が True なら、変えてよい所の外の変化に「【変えてよい所の外】」の印を付ける。
    変わった所が無いときは「前の版から変わった所はありません」の1行にする。
    """
    found = _as_changes(changes)
    if not found:
        return [texts.text("versions.no_changes", language=language)]
    lines = []
    for change in found:
        line = change.text(language)
        if mark_outside and change.outside_allowed:
            line = texts.text("versions.outside_mark", language=language, line=line)
        lines.append(line)
    return lines
