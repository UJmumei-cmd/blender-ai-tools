"""テスト用の形。どれも面は外向き（外から見て反時計回り）。長さはメートル。"""

import numpy as np


def box(size=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """閉じた直方体（頂点 8、三角形 12）"""
    hx, hy, hz = (np.asarray(size, dtype=np.float64) / 2.0)
    vertices = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ]) + np.asarray(center, dtype=np.float64)
    triangles = np.array([
        [0, 2, 1], [0, 3, 2],  # 下（-Z）
        [4, 5, 6], [4, 6, 7],  # 上（+Z）
        [0, 1, 5], [0, 5, 4],  # 前（-Y）
        [2, 3, 7], [2, 7, 6],  # 後ろ（+Y）
        [1, 2, 6], [1, 6, 5],  # 右（+X）
        [3, 0, 4], [3, 4, 7],  # 左（-X）
    ], dtype=np.int64)
    return vertices, triangles


def icosphere(radius: float = 1.0, subdivisions: int = 2, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """閉じた球（正二十面体を分けて作る）"""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    vertices = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    vertices = [list(np.asarray(v, dtype=np.float64) / np.linalg.norm(v)) for v in vertices]
    for _ in range(subdivisions):
        midpoints: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in midpoints:
                m = (np.asarray(vertices[a]) + np.asarray(vertices[b])) / 2.0
                vertices.append(list(m / np.linalg.norm(m)))
                midpoints[key] = len(vertices) - 1
            return midpoints[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    points = np.asarray(vertices, dtype=np.float64) * radius + np.asarray(center, dtype=np.float64)
    return points, np.asarray(faces, dtype=np.int64)


def grid(nx: int = 4, ny: int = 4, size=(1.0, 1.0), z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """開いた平らな格子（+Z 向き）。頂点 (nx+1)*(ny+1)、三角形 2*nx*ny"""
    xs = np.linspace(-size[0] / 2.0, size[0] / 2.0, nx + 1)
    ys = np.linspace(-size[1] / 2.0, size[1] / 2.0, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    vertices = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=1)
    triangles = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b, c, d = a + 1, a + nx + 2, a + nx + 1
            triangles += [[a, b, c], [a, c, d]]
    return vertices, np.asarray(triangles, dtype=np.int64)


def signed_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """閉じた形の体積（外向きなら正）"""
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
