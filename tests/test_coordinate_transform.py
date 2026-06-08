"""
坐标转换模块单元测试

验证世界坐标与像素坐标的双向转换正确性。
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from coordinate_transform import CoordinateTransformer, world_to_minimap


class TestCoordinateTransformer(unittest.TestCase):
    """测试坐标转换器。"""

    def setUp(self):
        self.transformer = CoordinateTransformer()

    def test_dust2_known_spawns(self):
        """
        测试de_dust2已知出生点的坐标转换。

        T出生点约在 (-700, 1800)，应在地图中下方区域。
        CT出生点约在 (1200, 2600)，应在地图右上方区域。
        """
        t_px, t_py = self.transformer.world_to_minimap(-700, 1800, "de_dust2")
        self.assertGreater(t_px, 300)
        self.assertLess(t_px, 500)
        self.assertGreater(t_py, 250)
        self.assertLess(t_py, 400)

        ct_px, ct_py = self.transformer.world_to_minimap(1200, 2600, "de_dust2")
        self.assertGreater(ct_px, 800)
        self.assertLess(ct_px, 900)
        self.assertGreater(ct_py, 100)
        self.assertLess(ct_py, 200)

    def test_dust2_bomb_sites(self):
        """
        测试de_dust2 A/B包点的坐标转换。

        A点约在 (1350, 2600) → 像素应接近 (869, 145)
        B点约在 (-1650, 800) → 像素应接近 (187, 554)
        """
        a_px, a_py = self.transformer.world_to_minimap(1350, 2600, "de_dust2")
        self.assertAlmostEqual(a_px, 869, delta=5)
        self.assertAlmostEqual(a_py, 145, delta=5)

        b_px, b_py = self.transformer.world_to_minimap(-1650, 800, "de_dust2")
        self.assertAlmostEqual(b_px, 187, delta=5)
        self.assertAlmostEqual(b_py, 554, delta=5)

    def test_clamp_to_bounds(self):
        """测试坐标超出范围时会被截断到 [0, 1024]。"""
        px, py = self.transformer.world_to_minimap(99999, -99999, "de_dust2")
        self.assertEqual(px, 1024)
        self.assertEqual(py, 1024)

        px2, py2 = self.transformer.world_to_minimap(-99999, 99999, "de_dust2")
        self.assertEqual(px2, 0)
        self.assertEqual(py2, 0)

    def test_roundtrip(self):
        """测试世界坐标→像素坐标→世界坐标的往返一致性。"""
        original_x, original_y = 500.0, 1500.0
        px, py = self.transformer.world_to_minimap(original_x, original_y, "de_dust2")
        back_x, back_y = self.transformer.minimap_to_world(px, py, "de_dust2")

        # 由于像素坐标是整数，往返后会有不超过scale的误差
        scale = self.transformer.map_config["de_dust2"]["scale"]
        self.assertAlmostEqual(original_x, back_x, delta=scale)
        self.assertAlmostEqual(original_y, back_y, delta=scale)

    def test_convenience_function(self):
        """测试便捷函数world_to_minimap。"""
        px, py = world_to_minimap(0, 0, "de_dust2")
        self.assertGreaterEqual(px, 0)
        self.assertLessEqual(px, 1024)
        self.assertGreaterEqual(py, 0)
        self.assertLessEqual(py, 1024)

    def test_unknown_map_raises(self):
        """测试未知地图会抛出KeyError。"""
        with self.assertRaises(KeyError):
            self.transformer.world_to_minimap(0, 0, "de_unknown")


if __name__ == "__main__":
    unittest.main()
