"""テスト用の形そのものが正しいか（外向き・閉じている・体積）を確かめる"""

import unittest

import numpy as np

from tests import shapes


def edge_use(triangles: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for k in range(3):
            edge = (int(tri[k]), int(tri[(k + 1) % 3]))
            counts[edge] = counts.get(edge, 0) + 1
    return counts


class ShapesTest(unittest.TestCase):
    def assert_closed_and_outward(self, vertices, triangles):
        counts = edge_use(triangles)
        # 閉じていて向きがそろっていれば、どの辺も逆向きでちょうど1回ずつ使われる
        for (a, b), n in counts.items():
            self.assertEqual(n, 1)
            self.assertEqual(counts.get((b, a)), 1)
        center = vertices.mean(axis=0)
        a, b, c = (vertices[triangles[:, i]] for i in range(3))
        normals = np.cross(b - a, c - a)
        outward = np.einsum("ij,ij->i", normals, (a + b + c) / 3.0 - center)
        self.assertTrue(np.all(outward > 0))

    def test_box(self):
        vertices, triangles = shapes.box(size=(2.0, 3.0, 4.0), center=(1.0, 0.0, 0.0))
        self.assert_closed_and_outward(vertices, triangles)
        self.assertAlmostEqual(shapes.signed_volume(vertices, triangles), 24.0)

    def test_icosphere(self):
        vertices, triangles = shapes.icosphere(radius=0.5, subdivisions=3)
        self.assert_closed_and_outward(vertices, triangles)
        self.assertTrue(np.allclose(np.linalg.norm(vertices, axis=1), 0.5))
        self.assertAlmostEqual(shapes.signed_volume(vertices, triangles), 4.0 / 3.0 * np.pi * 0.125, delta=0.01)

    def test_grid_faces_up(self):
        vertices, triangles = shapes.grid(3, 2, size=(3.0, 2.0))
        self.assertEqual(len(vertices), 12)
        self.assertEqual(len(triangles), 12)
        a, b, c = (vertices[triangles[:, i]] for i in range(3))
        self.assertTrue(np.all(np.cross(b - a, c - a)[:, 2] > 0))


if __name__ == "__main__":
    unittest.main()
