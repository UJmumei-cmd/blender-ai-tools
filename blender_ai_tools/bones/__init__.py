"""A3 骨の名前の対応（辞書）。共通の名前は MMD 準標準の日本語の名前。

- 共通の名前の一覧は data/mmd_semi_standard.json、ほかの名前（VRM・VRoid・Mixamo・MMD の英語名）からの
  対応は data/aliases_*.json にある。名前の字（全角・半角）は、mmd_semi_standard.json の形を正とする。
- 名前を比べるときは normalize() で「比べるための形」にしてから比べる。全角と半角（NFKC）、英字の大文字と小文字、
  区切り（. _ 空白 -）、名前の前の「名前空間:」（mixamorig: など）の違いは無視する。
- Blender（mmd_tools）の左右の書き方（腕.L・足ＩＫ_L・肩P.R・腰キャンセル.L）は、共通の名前から作る。
  左右の字を外した残り（腕・足ＩＫ）が、共通の名前と「ぴったり」同じときだけ対応させる。
  そのため Leg_Belt・ribbon.L・スカート_L のような名前は、L/R の字があっても対応しない。
- 1つの名前に、表によって違う意味があるもの（VRM 0.x と 1.0 の leftThumbProximal、mmd_tools と Mixamo の
  LeftLeg）は、lookup() では決めない（None）。map_bones() が、同じモデルのほかの骨の名前から決める。
  ただし、名前空間が書き方を示す名前（"mixamorig:LeftLeg"・"mixamorig12:LeftLeg"。表の namespaces）は、
  その表の意味で読む（Mixamo の 左ひざ）。
- bpy を使わない（Blender の外でもテストできる）。
"""

from __future__ import annotations

import functools
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CANONICAL_FILE = "mmd_semi_standard.json"
ALIAS_FILES = ("aliases_mmd.json", "aliases_vrm.json", "aliases_mixamo.json")

SIDES = ("L", "R")
GROUPS = ("trunk", "eye", "arm", "finger", "leg")
# 名前の中の左右の字（キャラから見た左右）
_SIDE_CHARS = {"L": "左", "R": "右"}
# 共通の名前そのものの表と、そこから作る左右の書き方の表の名前（aliases_*.json の表の名前と重ねない）
_TABLE_CANONICAL = "mmd"
_TABLE_CANONICAL_LR = "mmd_lr"

# MARK: - 名前の正規化


def _check_name(name: object) -> str:
    """骨の名前が文字列かを確かめる"""
    if not isinstance(name, str):
        raise ValueError(f"骨の名前は文字列で渡してください（受け取った型：{type(name).__name__}）")
    return name


_ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_SEPARATORS = re.compile(r"[._\-\s]+")


def _base_form(name: str) -> str:
    """NFKC にし、名前空間（最後の ':' まで）を外し、英字を小文字にした形"""
    text = unicodedata.normalize("NFKC", _check_name(name))
    text = text.rsplit(":", 1)[-1]
    return text.translate(_ASCII_LOWER)


def normalize(name: str) -> str:
    """名前を比べるための形にする。

    - 全角と半角をそろえる（NFKC。ＩＫ→IK、０→0、Ｄ→D、ｾﾝﾀｰ→センター）
    - 名前空間を外す（最後の ':' までを外す。"mixamorig12:LeftArm" → "leftarm"）
    - 英字（A〜Z）を小文字にする
    - 区切り（. _ 空白 -）を消す（"leg IK_L"・"LegIK.L" → "legikl"、"左足ＩＫ" → "左足ik"）
    文字列でないときは ValueError
    """
    return _SEPARATORS.sub("", _base_form(name))


def _loose_form(name: str) -> str:
    """前置き（J_Sec_ など）で比べるための形。区切りは消さずに '_' にそろえる。
    消すと "JSecret" が "J_Sec_" で始まるように見えてしまうため"""
    return _SEPARATORS.sub("_", _base_form(name))


def _namespace_key(name: str) -> str:
    """名前空間の、比べるための形。名前のすぐ前の1つだけ（"Character:mixamorig12:Hips" → "mixamorig12"）。
    名前空間が無いときは ""。normalize() は名前空間を外すので、どの書き方の名前かの印はここから取る"""
    parts = unicodedata.normalize("NFKC", _check_name(name)).split(":")
    return "" if len(parts) < 2 else _SEPARATORS.sub("", parts[-2].translate(_ASCII_LOWER))


def _lr_key(name: str) -> str | None:
    """Blender（mmd_tools）の左右の書き方の、比べるための形。

    名前に 左 か 右 が1つだけあるとき、その字を外した残りの後ろに l/r を付ける
    （左腕 → "腕l"、腰キャンセル右 → "腰キャンセルr"）。"腕.L"・"腕_L"・"腕 l" を normalize() した形と同じになる。
    左右の字が無い・2つ以上あるときは None
    """
    found = [(side, char) for side, char in _SIDE_CHARS.items() if char in name]
    if len(found) != 1 or name.count(found[0][1]) != 1:
        return None
    side, char = found[0]
    base = normalize(name.replace(char, ""))
    return base + side.lower() if base else None


# MARK: - 辞書の読み込み


@dataclass(frozen=True)
class CanonicalBone:
    """共通の名前（MMD 準標準）の骨1本。

    name: 名前（mmd_semi_standard.json の字のまま）
    side: キャラから見た左右（"L"・"R"）。体の中心の骨は None
    group: "trunk"（体の中心）・"eye"（目）・"arm"（腕）・"finger"（指）・"leg"（脚）
    standard: True は MMD の標準の骨、False は準標準で足された骨
    helper: True は隠れた補助の骨（肩C・腕捩1〜3・手捩1〜3）。準標準に合っているかの確かめでは、無くてもよい
    """

    name: str
    side: str | None
    group: str
    standard: bool
    helper: bool


@dataclass(frozen=True)
class _Dictionary:
    """読み込んだ辞書（中だけで使う）"""

    bones: tuple[CanonicalBone, ...]
    by_name: dict[str, CanonicalBone]
    order: dict[str, int]  # 共通の名前 → 並びの順
    canonical_by_key: dict[str, str]  # 共通の名前の比べる形 → 共通の名前
    mirrors: dict[str, str]  # 左右のある共通の名前 → 反対側の共通の名前
    entries: dict[str, tuple[tuple[str, str], ...]]  # 比べる形 → ((表の名前, 共通の名前), ...)
    extra_keys: frozenset[str]
    extra_prefixes: tuple[str, ...]  # _loose_form() の形
    namespace_tables: tuple[tuple[str, str], ...]  # (名前空間の比べる形, 表の名前)。例 ("mixamorig", "mixamo")
    chains: tuple[tuple[str, tuple[str, ...]], ...]


def _read_json(file_name: str) -> dict:
    data = json.loads((DATA_DIR / file_name).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{file_name}: いちばん外は {{}} にしてください")
    return data


def _load_bones(data: dict) -> tuple[CanonicalBone, ...]:
    bones = []
    for item in data.get("bones", []):
        if not isinstance(item, dict):
            raise ValueError(f"{CANONICAL_FILE}: bones の中身は {{}} にしてください：{item}")
        name, side, group = item.get("name"), item.get("side"), item.get("group")
        standard, helper = item.get("standard"), item.get("helper")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{CANONICAL_FILE}: 骨の名前がありません：{item}")
        if side is not None and side not in SIDES:
            raise ValueError(f"{CANONICAL_FILE}: {name} の side は \"L\"・\"R\"・null のどれかにしてください")
        if group not in GROUPS:
            raise ValueError(f"{CANONICAL_FILE}: {name} の group が決まりにありません：{group}")
        if not isinstance(standard, bool) or not isinstance(helper, bool):
            raise ValueError(f"{CANONICAL_FILE}: {name} の standard・helper は true か false にしてください")
        if standard and helper:
            raise ValueError(f"{CANONICAL_FILE}: {name} は標準の骨なので、補助の骨（helper）にはできません")
        # 左右の字と side が合っているか（中心の骨には 左・右 の字が無い）
        chars = [s for s, c in _SIDE_CHARS.items() if c in name]
        if (side is None and chars) or (side is not None and (chars != [side] or name.count(_SIDE_CHARS[side]) != 1)):
            raise ValueError(f"{CANONICAL_FILE}: {name} の名前の左右の字と side（{side}）が合いません")
        bones.append(CanonicalBone(name=name, side=side, group=group, standard=standard, helper=helper))
    if not bones:
        raise ValueError(f"{CANONICAL_FILE}: 骨が1つもありません")
    return tuple(bones)


def _name_list(value: object, source: str, table: str, field_name: str) -> list[str]:
    """表の中の名前の並び（known_extras・known_extra_prefixes・namespaces）を確かめる。
    文字列のリストで、どの名前も比べる形が空でないこと。
    リストでない "J_Sec_" は1字ずつの前置きになり、空の前置きはどの名前にも当たる。
    どちらも、知らない骨を「聞かなくてよい骨」として隠してしまうので、読み込みのときに止める"""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{source}: 表 {table} の {field_name} は、名前（文字列）のリスト [] にしてください")
    for item in value:
        if not normalize(item):
            raise ValueError(f"{source}: 表 {table} の {field_name} に、比べる形が空になる名前があります：{item!r}")
    return value


def _mirror_of(name: str, side: str) -> str:
    other = "R" if side == "L" else "L"
    return name.replace(_SIDE_CHARS[side], _SIDE_CHARS[other])


@functools.lru_cache(maxsize=None)
def _dictionary() -> _Dictionary:
    """data/ の JSON を読み、確かめてから、引くための表を作る（1回だけ）"""
    canonical_data = _read_json(CANONICAL_FILE)
    bones = _load_bones(canonical_data)
    by_name: dict[str, CanonicalBone] = {}
    canonical_by_key: dict[str, str] = {}
    for bone in bones:
        key = normalize(bone.name)
        if bone.name in by_name or key in canonical_by_key:
            raise ValueError(f"{CANONICAL_FILE}: {bone.name} が2回あります（全角・半角などの違いだけの名前を含む）")
        by_name[bone.name] = bone
        canonical_by_key[key] = bone.name
    order = {bone.name: i for i, bone in enumerate(bones)}

    # 左右の組：左の骨には右の骨が、右の骨には左の骨が、同じ種類で必ずある
    mirrors: dict[str, str] = {}
    for bone in bones:
        if bone.side is None:
            continue
        other = by_name.get(_mirror_of(bone.name, bone.side))
        if other is None or other.side == bone.side or (other.group, other.standard, other.helper) != (
                bone.group, bone.standard, bone.helper):
            raise ValueError(f"{CANONICAL_FILE}: {bone.name} の反対側の骨がありません（または種類が違います）")
        mirrors[bone.name] = other.name

    # 比べる形 → (表の名前, 共通の名前)。同じ表の中で、1つの形が2つの骨を指してはいけない
    entries: dict[str, dict[str, str]] = {}

    def add(key: str | None, table: str, canonical: str, source: str) -> None:
        if not key:
            raise ValueError(f"{source}: 比べる形が空になる名前があります（{canonical}）")
        if canonical not in by_name:
            raise ValueError(f"{source}: 対応先 {canonical} は共通の名前にありません")
        previous = entries.setdefault(key, {}).setdefault(table, canonical)
        if previous != canonical:
            raise ValueError(f"{source}: 表 {table} の中で、同じ形 {key} が {previous} と {canonical} の両方を指しています")

    for bone in bones:
        add(normalize(bone.name), _TABLE_CANONICAL, bone.name, CANONICAL_FILE)
        if bone.side is not None:
            add(_lr_key(bone.name), _TABLE_CANONICAL_LR, bone.name, CANONICAL_FILE)

    extra_keys: set[str] = set()
    extra_prefixes: list[str] = []
    namespace_tables: list[tuple[str, str]] = []
    table_names = {_TABLE_CANONICAL, _TABLE_CANONICAL_LR}
    for file_name in ALIAS_FILES:
        tables = _read_json(file_name).get("tables")
        if not isinstance(tables, dict) or not tables:
            raise ValueError(f"{file_name}: tables がありません")
        for table, spec in tables.items():
            if not isinstance(spec, dict) or not isinstance(spec.get("aliases", {}), dict):
                raise ValueError(f"{file_name}: 表 {table} は {{}} で書き、その中の aliases も {{}} にしてください")
            if table in table_names:
                raise ValueError(f"{file_name}: 表の名前 {table} が、ほかの表と重なっています")
            table_names.add(table)
            derive_lr = spec.get("derive_lr_suffix", False)
            for alias, canonical in spec.get("aliases", {}).items():
                add(normalize(alias), table, canonical, file_name)
                if derive_lr and _lr_key(alias):
                    add(_lr_key(alias), table, canonical, file_name)
            for extra in _name_list(spec.get("known_extras", []), file_name, table, "known_extras"):
                extra_keys.add(normalize(extra))
                if derive_lr and _lr_key(extra):
                    extra_keys.add(_lr_key(extra))
            prefixes = _name_list(spec.get("known_extra_prefixes", []), file_name, table, "known_extra_prefixes")
            extra_prefixes.extend(_loose_form(prefix) for prefix in prefixes)
            namespaces = _name_list(spec.get("namespaces", []), file_name, table, "namespaces")
            namespace_tables.extend((normalize(namespace), table) for namespace in namespaces)

    # 共通の名前そのものの形は、ほかの表でも同じ骨だけを指す（共通の名前が迷うことはない）
    for key, canonical in canonical_by_key.items():
        if set(entries[key].values()) != {canonical}:
            raise ValueError(f"共通の名前 {canonical} と同じ形の名前が、ほかの骨を指しています")
    # 知っている骨（対応しない）と、対応する名前が重なってはいけない
    overlap = sorted(extra_keys & entries.keys())
    if overlap:
        raise ValueError(f"known_extras と aliases の両方にある名前があります：{overlap}")

    chain_data = canonical_data.get("chains", {})
    chains: list[tuple[str, tuple[str, ...]]] = []
    for chain_name, names in chain_data.items():
        names = tuple(names)
        for name in names:
            if name not in by_name:
                raise ValueError(f"{CANONICAL_FILE}: 骨の並び {chain_name} の {name} は共通の名前にありません")
        chains.append((chain_name, names))
        if chain_name.endswith("_L"):
            if not all(by_name[name].side == "L" for name in names):
                raise ValueError(f"{CANONICAL_FILE}: 骨の並び {chain_name} には、左の骨だけを書いてください")
            chains.append((chain_name[:-2] + "_R", tuple(mirrors[name] for name in names)))
    chain_names = [name for name, _ in chains]
    if len(set(chain_names)) != len(chain_names):
        raise ValueError(f"{CANONICAL_FILE}: 骨の並びの名前が重なっています")

    return _Dictionary(
        bones=bones,
        by_name=by_name,
        order=order,
        canonical_by_key=canonical_by_key,
        mirrors=mirrors,
        entries={key: tuple(sorted(tables.items())) for key, tables in entries.items()},
        extra_keys=frozenset(extra_keys),
        extra_prefixes=tuple(extra_prefixes),
        namespace_tables=tuple(namespace_tables),
        chains=tuple(chains),
    )


# MARK: - 共通の名前


def canonical_bones() -> tuple[CanonicalBone, ...]:
    """共通の名前の骨の一覧（mmd_semi_standard.json の順。これを「共通の名前の順」とする）"""
    return _dictionary().bones


def canonical_names() -> tuple[str, ...]:
    """共通の名前の一覧（共通の名前の順）"""
    return tuple(bone.name for bone in _dictionary().bones)


def canonical_bone(name: str) -> CanonicalBone | None:
    """共通の名前の骨。name は共通の名前（全角・半角などの違いは問わない）。共通の名前でなければ None"""
    d = _dictionary()
    canonical = d.canonical_by_key.get(normalize(name))
    return None if canonical is None else d.by_name[canonical]


def mirror_name(canonical: str) -> str | None:
    """左右を入れかえた共通の名前（左腕 → 右腕、腰キャンセル左 → 腰キャンセル右）。体の中心の骨は None。

    canonical は共通の名前（全角・半角などの違いは問わない）。共通の名前でないときは ValueError
    """
    bone = canonical_bone(canonical)
    if bone is None:
        raise ValueError(f"{canonical} は共通の名前（MMD 準標準）にありません。先に lookup() で共通の名前にしてください")
    return _dictionary().mirrors.get(bone.name)


def chains() -> dict[str, tuple[str, ...]]:
    """長さを測るための骨の並び（親子の順ではない）。

    spine（下半身→頭）、arm_L/arm_R（肩→手首）、leg_L/leg_R（足→つま先）、leg_d_L/leg_d_R（足D→足先EX）、
    thumb・index・middle・ring・little の _L/_R（指の付け根→先）。呼ぶたびに新しい dict を返す
    """
    return dict(_dictionary().chains)


# MARK: - 名前を引く


_DIGITS = re.compile(r"[0-9]*")


def _namespace_table(d: _Dictionary, name: str) -> str | None:
    """名前空間が示す表の名前（"mixamorig:"・"mixamorig12:" → "mixamo"）。示さないときは None。
    名前空間は、表の namespaces の名前そのものか、その後ろに数字だけが付いたもの"""
    namespace = _namespace_key(name)
    for prefix, table in d.namespace_tables:
        if namespace.startswith(prefix) and _DIGITS.fullmatch(namespace[len(prefix):]):
            return table
    return None


def _candidate_tables(d: _Dictionary, name: str) -> dict[str, frozenset[str]]:
    """name が指しうる共通の名前 → その名前を載せている表の名前。

    名前空間が表を示し（mixamorig: など）、その表に name が載っているときは、その表の意味だけを残す
    （"mixamorig:LeftLeg" は Mixamo の 左ひざ。mmd_tools の英語名の 左足 ではない）
    """
    found: dict[str, set[str]] = {}
    for table, canonical in d.entries.get(normalize(name), ()):
        found.setdefault(canonical, set()).add(table)
    hint = _namespace_table(d, name)
    if hint is not None and any(hint in tables for tables in found.values()):
        found = {canonical: {hint} for canonical, tables in found.items() if hint in tables}
    return {canonical: frozenset(tables) for canonical, tables in found.items()}


def lookup_candidates(name: str) -> tuple[str, ...]:
    """name が指しうる共通の名前の全部（共通の名前の順）。

    ふつうは0個か1個。表によって意味が違う名前（leftThumbProximal・名前空間の無い LeftLeg）だけ2個になる。
    知っている骨（known extra）と、辞書に無い名前は空
    """
    d = _dictionary()
    return tuple(sorted(_candidate_tables(d, name), key=d.order.__getitem__))


def lookup(name: str) -> str | None:
    """name に対応する共通の名前。辞書に無い名前・知っている骨（known extra）・1つに決まらない名前は None"""
    found = lookup_candidates(name)
    return found[0] if len(found) == 1 else None


def is_known_extra(name: str) -> bool:
    """知っている骨だが、共通の骨には対応しない名前か（upperChest・J_Sec_*・HeadTop_End・上半身3 など）。

    ユーザーに聞かなくてよい名前。共通の名前に対応する名前は False
    """
    d = _dictionary()
    key = normalize(name)
    if key in d.entries:
        return False
    if key in d.extra_keys:
        return True
    loose = _loose_form(name)
    return any(loose.startswith(prefix) for prefix in d.extra_prefixes)


def is_ambiguous(name: str) -> bool:
    """表によって意味が違い、名前1つだけでは共通の名前が決まらないか"""
    return len(lookup_candidates(name)) > 1


# MARK: - モデルの骨の名前をまとめて対応させる


@dataclass(frozen=True)
class BoneMapping:
    """map_bones() の結果。入力の名前は、次のどれか1つだけに入る：
    mapping のキー・unknown・known_extras・conflicts の値・shadowed のキー。

    mapping: 元の名前 → 共通の名前（入力の順）
    unknown: 辞書に無い名前と、1つに決まらない名前（入力の順）。ユーザーに聞く名前
    known_extras: 知っている骨だが、共通の骨には対応しない名前（入力の順）
    conflicts: 共通の名前 → それを取り合った元の名前（共通の名前の順。値は入力の順）。どれも mapping に入れない
    missing_standard: 対応する名前が無い MMD 標準の骨（共通の名前の順）
    missing_semi_standard: 対応する名前が無い、準標準で足された骨（補助の骨を除く。共通の名前の順）
    ambiguous: 1つに決まらなかった名前 → 候補の共通の名前（入力の順。候補は共通の名前の順）。この名前は unknown にも入る
    shadowed: 共通の名前そのものの名前があったので使わなかった名前 → 共通の名前（入力の順）。
        例：「左腕」と「腕.L」の両方があると、「腕.L」がここに入る

    conflicts に出た骨は、骨はある（どれか決まらない）ので missing に入れない。
    """

    mapping: dict[str, str]
    unknown: tuple[str, ...]
    known_extras: tuple[str, ...]
    conflicts: dict[str, tuple[str, ...]]
    missing_standard: tuple[str, ...]
    missing_semi_standard: tuple[str, ...]
    ambiguous: dict[str, tuple[str, ...]] = field(default_factory=dict)
    shadowed: dict[str, str] = field(default_factory=dict)

    @property
    def is_semi_standard_compliant(self) -> bool:
        """準標準に合っているか：標準の骨と、準標準の骨（補助の骨を除く）が全部あり、取り合いが無い"""
        return not self.missing_standard and not self.missing_semi_standard and not self.conflicts

    @property
    def by_canonical(self) -> dict[str, str]:
        """共通の名前 → 元の名前（共通の名前の順）"""
        order = _dictionary().order
        return {canonical: original
                for original, canonical in sorted(self.mapping.items(), key=lambda item: order[item[1]])}


def _unique_names(names: Iterable[str]) -> list[str]:
    """名前を確かめ、同じ名前（字がまったく同じもの）を1つにする（入力の順）"""
    if isinstance(names, (str, bytes, bytearray)):
        raise ValueError("骨の名前の並び（リストなど）を渡してください。1つの文字列は渡せません")
    try:
        iterator = iter(names)
    except TypeError:
        raise ValueError(f"骨の名前の並び（リストなど）を渡してください（受け取った型：{type(names).__name__}）") from None
    unique: dict[str, None] = {}
    for name in iterator:
        unique.setdefault(_check_name(name), None)
    return list(unique)


def _resolve(groups: dict[str, frozenset[str]], name_tables: dict[str, frozenset[str]], name: str) -> str | None:
    """表によって意味が違う名前を、ほかの名前から決める。

    groups: 候補の共通の名前 → それを指す表の名前
    name_tables: 1つに決まる名前 → その名前が載っている表の名前
    候補ごとに「その候補の表にだけ載っていて、ほかの候補の表には載っていない名前」の数を数え、
    数が1以上の候補がちょうど1つのときだけ、それに決める（例：leftThumbMetacarpal は vrm1 の表にだけあるので、
    leftThumbProximal を VRM 1.0 の意味＝親指１ に決める）
    """
    all_tables = frozenset().union(*groups.values())
    evidence = dict.fromkeys(groups, 0)
    for other, tables in name_tables.items():
        if other == name:
            continue
        hit = tables & all_tables
        if not hit:
            continue
        for canonical, own in groups.items():
            if hit <= own:
                evidence[canonical] += 1
    chosen = [canonical for canonical, count in evidence.items() if count > 0]
    return chosen[0] if len(chosen) == 1 else None


def map_bones(names: Iterable[str]) -> BoneMapping:
    """モデルの骨の名前の並びを、共通の名前に対応させる。結果は入力の順によらず同じ中身になる（並びの決まりは BoneMapping）。

    - 字がまったく同じ名前は1つとして数える。
    - 表によって意味が違う名前（leftThumbProximal・LeftLeg）は、ほかの名前から決める（_resolve）。
      決まらないときは unknown と ambiguous に入れる。名前空間 mixamorig の付いた名前は、はじめから Mixamo の意味で読み、
      ほかの名前を決めるときの Mixamo の印にもなる。
    - 2つ以上の名前が同じ共通の名前を指したとき：比べる形が共通の名前そのものと同じ名前が1つだけなら、それを使う
      （ほかは shadowed）。そうでなければ conflicts に入れ、どれも使わない。
    """
    d = _dictionary()
    unique = _unique_names(names)
    keys = {name: normalize(name) for name in unique}

    # 名前ごとに：指しうる共通の名前 → それを載せている表（名前空間が示す表があれば、その表だけ）
    groups_by_name = {name: _candidate_tables(d, name) for name in unique}
    name_tables = {name: frozenset().union(*groups.values())
                   for name, groups in groups_by_name.items() if len(groups) == 1}

    claims: dict[str, list[str]] = {}
    unknown: list[str] = []
    extras: list[str] = []
    ambiguous: dict[str, tuple[str, ...]] = {}
    for name in unique:
        groups = groups_by_name[name]
        if not groups:
            (extras if is_known_extra(name) else unknown).append(name)
            continue
        if len(groups) == 1:
            canonical = next(iter(groups))
        else:
            canonical = _resolve(groups, name_tables, name)
            if canonical is None:
                unknown.append(name)
                ambiguous[name] = tuple(sorted(groups, key=d.order.__getitem__))
                continue
        claims.setdefault(canonical, []).append(name)

    winners: dict[str, str] = {}
    shadowed_by: dict[str, str] = {}
    conflicts: dict[str, tuple[str, ...]] = {}
    for canonical in sorted(claims, key=d.order.__getitem__):
        claimants = claims[canonical]
        if len(claimants) == 1:
            winners[claimants[0]] = canonical
            continue
        exact = [name for name in claimants if keys[name] == normalize(canonical)]
        if len(exact) == 1:
            winners[exact[0]] = canonical
            shadowed_by.update((name, canonical) for name in claimants if name != exact[0])
        else:
            conflicts[canonical] = tuple(claimants)

    mapping = {name: winners[name] for name in unique if name in winners}
    shadowed = {name: shadowed_by[name] for name in unique if name in shadowed_by}
    present = set(mapping.values()) | set(conflicts)
    missing_standard = tuple(b.name for b in d.bones if b.standard and b.name not in present)
    missing_semi = tuple(b.name for b in d.bones if not b.standard and not b.helper and b.name not in present)
    return BoneMapping(
        mapping=mapping,
        unknown=tuple(unknown),
        known_extras=tuple(extras),
        conflicts=conflicts,
        missing_standard=missing_standard,
        missing_semi_standard=missing_semi,
        ambiguous=ambiguous,
        shadowed=shadowed,
    )
