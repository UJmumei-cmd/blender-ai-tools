"""E2 往復の確認：書き出した USD（USDZ・USDC・USDA）を読んで、numpy の並びにする（pxr を使う。Blender の中だけ）。

読むだけで、ファイルには何も書かない（Usd.Stage を開いて読み、保存しない）。
読んだ物は roundtrip.ExportedModel（bpy なしの形）にする。比べる計算は roundtrip.py。

決まり（roundtrip.py の説明も見る）：
- 行列は Gf の行ベクトルの形（平行移動は下の行）を転置して、列ベクトルの形（平行移動は右の列）にする。
- 骨で動く形（UsdSkel で骨格に結び付いた Mesh）の骨・重み・geomBindTransform・シェイプキーは、
  UsdSkel.Cache の SkinningQuery で読む（先祖の部品から受け継いだ値も、USD の決まりどおりに読む）。
- 点は、メッシュの中の座標のまま（USD の points）。骨格の空間へは geomBindTransform で移す。
- Blender の名前は、Blender が書く userProperties:blender:object_name（オブジェクト）と
  userProperties:blender:data_name（メッシュなどのデータ）から読む。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Sdf, Tf, Usd, UsdGeom, UsdSkel

from .roundtrip import (EXPECTED_PRIM_TYPES, PHYSICS_PREFIX, ExportedBlendShape, ExportedMesh, ExportedModel,
                        fan_triangulate, gf_to_column)

__all__ = ["ExportedBlendShape", "ExportedMesh", "ExportedModel", "OBJECT_NAME_ATTR", "DATA_NAME_ATTR", "read_usdz"]

# Blender が USD に書く、元の名前
OBJECT_NAME_ATTR = "userProperties:blender:object_name"
DATA_NAME_ATTR = "userProperties:blender:data_name"

_LEFT_HANDED = "leftHanded"
_CONSTANT = "constant"


# MARK: - 小さな読み取り


def _column(matrix: Any) -> np.ndarray:
    """Gf.Matrix4d（行ベクトルの形）を、列ベクトルの形の (4, 4) にする"""
    return gf_to_column(np.array(matrix, dtype=np.float64).reshape(4, 4))


def _columns(matrices: Any, count: int, name: str) -> np.ndarray:
    """Vt.Matrix4dArray を (count, 4, 4) の列ベクトルの形にする"""
    array = np.array(matrices if matrices is not None else [], dtype=np.float64)
    if array.size == 0:
        array = array.reshape(0, 4, 4)
    if array.shape != (count, 4, 4):
        raise ValueError(f"{name} の数が骨の数（{count}）と違います（いまの形：{array.shape}）")
    return gf_to_column(array)


def _string_attr(prim: Usd.Prim, name: str) -> str | None:
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.HasAuthoredValue():
        return None
    value = attribute.Get()
    return value if isinstance(value, str) and value else None


def _points(prim: Usd.Prim) -> np.ndarray:
    value = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    array = np.array(value if value is not None else [], dtype=np.float64)
    return array.reshape(-1, 3) if array.size else np.zeros((0, 3))


def _ints(value: Any) -> np.ndarray:
    return np.array(value if value is not None else [], dtype=np.int64).reshape(-1)


def _source_name(prim: Usd.Prim) -> str:
    """Blender の元のオブジェクトの名前：自分の object_name → 親の object_name → 自分の data_name → 部品の名前"""
    own = _string_attr(prim, OBJECT_NAME_ATTR)
    if own:
        return own
    parent = prim.GetParent()
    if parent and parent.IsValid():
        inherited = _string_attr(parent, OBJECT_NAME_ATTR)
        if inherited:
            return inherited
    return _string_attr(prim, DATA_NAME_ATTR) or prim.GetName()


# MARK: - 形


def _blend_shapes(stage: Usd.Stage, query: Any, point_count: int) -> tuple[ExportedBlendShape, ...]:
    """結び付いたシェイプキー（skel:blendShapes の名前と skel:blendShapeTargets の BlendShape）"""
    if not query.HasBlendShapes():
        return ()
    names = list(query.GetBlendShapeOrder() or [])
    targets = list(query.GetBlendShapeTargetsRel().GetTargets()) if query.GetBlendShapeTargetsRel() else []
    if len(names) != len(targets):
        raise ValueError(f"{query.GetPrim().GetPath()}：skel:blendShapes（{len(names)}）と skel:blendShapeTargets"
                         f"（{len(targets)}）の数が違います")
    shapes = []
    for name, target in zip(names, targets):
        shape = UsdSkel.BlendShape(stage.GetPrimAtPath(target))
        if not shape:
            raise ValueError(f"シェイプキー {name} の BlendShape（{target}）がありません")
        offsets = np.array(shape.GetOffsetsAttr().Get() or [], dtype=np.float64).reshape(-1, 3)
        indices_attr = shape.GetPointIndicesAttr()
        indices = _ints(indices_attr.Get()) if indices_attr and indices_attr.HasAuthoredValue() else None
        if indices is not None and len(indices) == 0 and len(offsets) == point_count:
            indices = None  # 空の pointIndices は、全部の点とみなす
        shapes.append(ExportedBlendShape(name=str(name), offsets=offsets, point_indices=indices))
    return tuple(shapes)


def _influences(query: Any, point_count: int) -> tuple[np.ndarray, np.ndarray]:
    """頂点ごとの骨の番号と重み (N, K)。constant（形全体で同じ）のときは、全部の頂点に同じ物を並べる"""
    if not query.HasJointInfluences():
        return np.zeros((point_count, 0), dtype=np.int64), np.zeros((point_count, 0))
    result = query.ComputeJointInfluences()
    if not result:
        raise ValueError(f"{query.GetPrim().GetPath()}：骨の番号と重みを読めません（primvars:skel:jointIndices・jointWeights）")
    indices, weights = result
    k = int(query.GetNumInfluencesPerComponent())
    ids = _ints(indices)
    w = np.array(weights if weights is not None else [], dtype=np.float64).reshape(-1)
    if k <= 0 or len(ids) != len(w):
        raise ValueError(f"{query.GetPrim().GetPath()}：jointIndices と jointWeights の数が合いません")
    if str(query.GetInterpolation()) == _CONSTANT:
        if len(ids) != k:
            raise ValueError(f"{query.GetPrim().GetPath()}：constant の重みの数（{len(ids)}）が elementSize（{k}）と違います")
        return np.tile(ids, (point_count, 1)), np.tile(w, (point_count, 1))
    if len(ids) != point_count * k:
        raise ValueError(f"{query.GetPrim().GetPath()}：重みの数（{len(ids)}）が 頂点の数 × elementSize"
                         f"（{point_count} × {k}）と違います")
    return ids.reshape(point_count, k), w.reshape(point_count, k)


def _mesh(stage: Usd.Stage, prim: Usd.Prim, query: Any, skeleton_paths: list[str], skeleton_inverse: np.ndarray,
          xform_cache: UsdGeom.XformCache) -> ExportedMesh:
    mesh = UsdGeom.Mesh(prim)
    points = _points(prim)
    left = str(mesh.GetOrientationAttr().Get() or "") == _LEFT_HANDED
    triangles = fan_triangulate(_ints(mesh.GetFaceVertexCountsAttr().Get()),
                                _ints(mesh.GetFaceVertexIndicesAttr().Get()), len(points), left_handed=left)
    local_to_skeleton = skeleton_inverse @ _column(xform_cache.GetLocalToWorldTransform(prim))
    common = {"path": str(prim.GetPath()), "name": prim.GetName(), "points": points, "triangles": triangles,
              "local_to_skeleton": local_to_skeleton, "source_name": _source_name(prim)}
    if query is None or not query.HasJointInfluences():
        shapes = _blend_shapes(stage, query, len(points)) if query is not None else ()
        return ExportedMesh(skinned=False, blend_shapes=shapes, **common)
    ids, weights = _influences(query, len(points))
    order = query.GetJointOrder()
    if order is None:
        joint_names = tuple(path.split("/")[-1] for path in skeleton_paths)
        joint_map = None
    else:
        lookup = {path: index for index, path in enumerate(skeleton_paths)}
        joint_names = tuple(str(path).split("/")[-1] for path in order)
        joint_map = np.array([lookup.get(str(path), -1) for path in order], dtype=np.int64)
    return ExportedMesh(geom_bind=_column(query.GetGeomBindTransform()), joint_indices=ids, joint_weights=weights,
                        joint_names=joint_names, joint_map=joint_map, skinned=True,
                        blend_shapes=_blend_shapes(stage, query, len(points)), **common)


# MARK: - 読む


def _open(path: Path) -> Usd.Stage:
    try:
        stage = Usd.Stage.Open(str(path))
    except Tf.ErrorException as error:
        raise ValueError(f"USD のファイルを開けませんでした：{path.name}（{error}）") from None
    if stage is None:
        raise ValueError(f"USD のファイルを開けませんでした：{path.name}")
    return stage


def read_usdz(path: str | os.PathLike[str]) -> ExportedModel:
    """書き出した USD のファイル（.usdz・.usdc・.usda）を読む。ファイルには何も書かない。

    骨格（Skeleton）がちょうど1つでないとき、ファイルが無い・開けない・形がこわれているときは、日本語の ValueError。
    骨で動く形の骨の番号・重み・geomBindTransform・シェイプキーは、UsdSkel の決まりどおりに読む（ExportedMesh）。
    """
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError(f"path はファイルの場所にしてください（いまの型：{type(path).__name__}）")
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"USD のファイルがありません：{file}")
    stage = _open(file)
    time = Usd.TimeCode.Default()
    xform_cache = UsdGeom.XformCache(time)

    prims: list[tuple[str, str]] = []
    other: list[tuple[str, str]] = []
    schemas: list[tuple[str, str]] = []
    object_names: dict[str, str] = {}
    skeletons: list[Usd.Prim] = []
    roots: list[Usd.Prim] = []
    mesh_prims: list[Usd.Prim] = []
    for prim in stage.Traverse():
        kind = str(prim.GetTypeName())
        path_text = str(prim.GetPath())
        prims.append((path_text, kind))
        if kind not in EXPECTED_PRIM_TYPES:
            other.append((path_text, kind))
        schemas.extend((path_text, str(s)) for s in prim.GetAppliedSchemas() if str(s).startswith(PHYSICS_PREFIX))
        name = _string_attr(prim, OBJECT_NAME_ATTR)
        if name:
            object_names[path_text] = name
        if prim.IsA(UsdSkel.Skeleton):
            skeletons.append(prim)
        elif prim.IsA(UsdSkel.Root):
            roots.append(prim)
        elif prim.IsA(UsdGeom.Mesh):
            mesh_prims.append(prim)
    if len(skeletons) != 1:
        raise ValueError(f"骨格（Skeleton）がちょうど1つではありません（{len(skeletons)} 個：{[str(p.GetPath()) for p in skeletons]}）")
    skeleton_prim = skeletons[0]
    skeleton = UsdSkel.Skeleton(skeleton_prim)
    joint_paths = [str(token) for token in (skeleton.GetJointsAttr().Get() or [])]
    count = len(joint_paths)
    bind = _columns(skeleton.GetBindTransformsAttr().Get(), count, "bindTransforms")
    rest_value = skeleton.GetRestTransformsAttr().Get()
    rest = None if rest_value is None or len(rest_value) == 0 else _columns(rest_value, count, "restTransforms")
    skeleton_world = _column(xform_cache.GetLocalToWorldTransform(skeleton_prim))
    try:
        skeleton_inverse = np.linalg.inv(skeleton_world)
    except np.linalg.LinAlgError:
        raise ValueError("骨格の位置の行列がつぶれています（逆行列がありません）") from None

    # 骨で動く形：骨格の SkelRoot ごとに、結び付きを UsdSkel の決まりで計算する
    cache = UsdSkel.Cache()
    queries: dict[Sdf.Path, Any] = {}
    predicate = Usd.PrimDefaultPredicate
    for root_prim in roots:
        root = UsdSkel.Root(root_prim)
        cache.Populate(root, predicate)
        for binding in cache.ComputeSkelBindings(root, predicate):
            if binding.GetSkeleton().GetPrim() != skeleton_prim:
                continue
            for query in binding.GetSkinningTargets():
                queries[query.GetPrim().GetPath()] = query

    meshes = tuple(_mesh(stage, prim, queries.get(prim.GetPath()), joint_paths, skeleton_inverse, xform_cache)
                   for prim in mesh_prims)
    default = stage.GetDefaultPrim()
    root_matrix = _column(xform_cache.GetLocalToWorldTransform(default)) if default and default.IsValid() else None
    return ExportedModel(
        joint_paths=tuple(joint_paths),
        bind=bind,
        rest_local=rest,
        meshes=meshes,
        path=str(file),
        up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
        root_matrix=root_matrix,
        skeleton_path=str(skeleton_prim.GetPath()),
        skeleton_world=skeleton_world,
        prims=tuple(prims),
        other_prims=tuple(other),
        physics_schemas=tuple(schemas),
        object_names=object_names,
    )
