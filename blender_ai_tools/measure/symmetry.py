"""A4 左右のずれ（bpy なし・numpy だけ）。

向きは Blender のまま：キャラの左は +X、左右を分ける面は X = 0（CLAUDE.md 7章）。
頂点を左右に映し（X の符号を変え）、映した点から、元の頂点のいちばん近い物までの距離を「ずれ」とする。

注意：頂点どうしを比べるので、左右で面の分け方（頂点の置き方）が違うと、形が対称でもずれが出る。
わざと左右が違う所（横に結んだ髪など）は、exclude で除く。

Blender の中では、あとで mathutils の KDTree を使う速い版を作る。ここは、それと比べるための正しい答え（全部の組を調べる）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import thresholds
from .geometry import as_mask, as_points, as_positive_int, as_values, worst_indices
from .results import Measurement

# 点の組を、一度にいくつまで計算するか。1組あたり約 50 バイトなので、約 13MB になる
NEAREST_CHUNK_PAIRS = 1 << 18

ASYMMETRY_NAME = "asymmetry"


def mirror_x(points: Any) -> np.ndarray:
    """点を左右に映す（X の符号を変える）。元の並びは変えず、新しい並びを返す"""
    result = as_points(points, "points").copy()
    result[:, 0] = -result[:, 0]
    return result


def nearest_distances(query: Any, reference: Any, *,
                      chunk_size: int = NEAREST_CHUNK_PAIRS) -> tuple[np.ndarray, np.ndarray]:
    """query の点ごとに、reference の中でいちばん近い点までの距離 (Q,) と、その番号 (Q,) int64。

    全部の組を調べる正しい答え。小分けにして、メモリを chunk_size 組ぶんに抑える。
    同じ距離の点が2つ以上あるときは、いちばん小さい番号を選ぶ。
    query が空なら空の答え。reference が空なのに query があるときは ValueError。
    """
    q = as_points(query, "query")
    r = as_points(reference, "reference")
    chunk = as_positive_int(chunk_size, "chunk_size")
    if len(q) == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    if len(r) == 0:
        raise ValueError("reference に点がありません（いちばん近い点を決められません）")

    best_d2 = np.zeros(len(q))
    best_index = np.zeros(len(q), dtype=np.int64)
    ref_block = min(len(r), chunk)
    query_block = max(1, chunk // ref_block)
    # reference は小さい番号の小分けから順に調べ、「より近い」ときだけ置き換える（同じ距離なら先の番号が残る）
    for r0 in range(0, len(r), ref_block):
        rb = r[r0:r0 + ref_block]
        for q0 in range(0, len(q), query_block):
            qb = q[q0:q0 + query_block]
            # 足す順を決めて計算する（小分けの仕方で答えが変わらないように）
            dx = qb[:, None, 0] - rb[None, :, 0]
            d2 = dx * dx
            dy = qb[:, None, 1] - rb[None, :, 1]
            d2 += dy * dy
            dz = qb[:, None, 2] - rb[None, :, 2]
            d2 += dz * dz
            column = np.argmin(d2, axis=1)  # 同じ値なら最初（小さい番号）
            candidate = d2[np.arange(len(qb)), column]
            better = np.ones(len(qb), dtype=bool) if r0 == 0 else candidate < best_d2[q0:q0 + len(qb)]
            target = np.arange(q0, q0 + len(qb))[better]
            best_d2[target] = candidate[better]
            best_index[target] = column[better] + r0
    return np.sqrt(best_d2), best_index


def asymmetry(vertices: Any, *, exclude: Any = None, chunk_size: int = NEAREST_CHUNK_PAIRS) -> np.ndarray:
    """頂点ごとの左右のずれ (N,)（メートル）：映した頂点から、元の頂点のいちばん近い物までの距離。

    X = 0 の面の上の頂点は、映しても同じ場所なので 0 になる。
    exclude（真偽の並び (N,)）で除いた頂点は NaN にし、ほかの頂点の相手としても使わない。
    """
    v = as_points(vertices, "vertices")
    excluded = np.zeros(len(v), dtype=bool) if exclude is None else as_mask(exclude, len(v), "exclude")
    result = np.full(len(v), np.nan)
    keep = np.flatnonzero(~excluded)
    if keep.size == 0:
        return result
    distances, _ = nearest_distances(mirror_x(v[keep]), v[keep], chunk_size=chunk_size)
    result[keep] = distances
    return result


def summarize_asymmetry(distances: Any) -> Measurement:
    """左右のずれを、1つの信号にまとめる（名前 "asymmetry"、単位 "m"）。

    値は NaN（除いた頂点）を除いた中で、いちばん大きいずれ。基準は thresholds.ASYMMETRY_YELLOW・ASYMMETRY_RED。
    点が無い、全部 NaN のときは灰。負の値・無限大は ValueError。
    """
    d = as_values(distances, "distances", allow_nan=True)
    if (d < 0.0).any():
        raise ValueError("distances に負の値があります（距離は 0 以上）")
    if len(d) == 0:
        return Measurement.unmeasured(ASYMMETRY_NAME, "m", "点がありません")
    measured = ~np.isnan(d)
    safe = np.where(measured, d, 0.0)
    details: dict[str, Any] = {
        "point_count": len(d),
        "excluded_count": int((~measured).sum()),
        "over_yellow_count": int((measured & (safe >= thresholds.ASYMMETRY_YELLOW)).sum()),
        "over_red_count": int((measured & (safe >= thresholds.ASYMMETRY_RED)).sum()),
        "worst_indices": worst_indices(safe, measured & (safe > 0.0)),
    }
    if not measured.any():
        return Measurement.unmeasured(ASYMMETRY_NAME, "m", "比べられる点がありません（全部除かれています）", details)
    return Measurement.from_value(ASYMMETRY_NAME, float(safe[measured].max()), "m", red=thresholds.ASYMMETRY_RED,
                                  yellow=thresholds.ASYMMETRY_YELLOW, details=details)
