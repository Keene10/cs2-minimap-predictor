"""
坐标转换模块

将CS2游戏世界坐标转换为小地图像素坐标。
支持基于地图overview参数的精确转换。
"""

import json
from pathlib import Path
from typing import Tuple


# 默认配置文件路径
DEFAULT_MAP_CONFIG = Path(__file__).parent.parent / "map_overview.json"


def load_map_config(config_path: str = None) -> dict:
    """
    加载地图overview配置文件。

    Args:
        config_path: 配置文件路径，默认使用项目根目录的map_overview.json

    Returns:
        包含所有地图配置的字典
    """
    if config_path is None:
        config_path = DEFAULT_MAP_CONFIG
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


class CoordinateTransformer:
    """
    CS2世界坐标与小地图像素坐标的转换器。

    转换公式（与awpy保持一致，显式写出）：
        px = (world_x - pos_x) / scale
        py = (pos_y - world_y) / scale   # 注意Y轴翻转

    其中pos_x, pos_y是雷达图左上角对应的游戏世界坐标，scale是缩放比例。
    """

    def __init__(self, config_path: str = None):
        """
        初始化转换器。

        Args:
            config_path: 地图配置文件路径
        """
        self.map_config = load_map_config(config_path)

    def world_to_minimap(
        self, world_x: float, world_y: float, map_name: str
    ) -> Tuple[int, int]:
        """
        将游戏世界坐标转换为小地图像素坐标 (px, py)。

        Args:
            world_x: 游戏世界X坐标
            world_y: 游戏世界Y坐标
            map_name: 地图名称，如 "de_dust2"

        Returns:
            小地图上的整数像素坐标 (px, py)，范围通常落在 [0, 1024]

        Raises:
            KeyError: 当指定地图不在配置中时抛出
        """
        if map_name not in self.map_config:
            raise KeyError(f"地图 '{map_name}' 的配置不存在，请检查map_overview.json")

        cfg = self.map_config[map_name]
        pos_x = cfg["pos_x"]
        pos_y = cfg["pos_y"]
        scale = cfg["scale"]

        # 显式坐标转换公式
        px = (world_x - pos_x) / scale
        py = (pos_y - world_y) / scale

        # 截断到 [0, 1024] 范围（标准雷达图尺寸）
        px = max(0, min(1024, px))
        py = max(0, min(1024, py))

        return int(px), int(py)

    def minimap_to_world(
        self, px: float, py: float, map_name: str
    ) -> Tuple[float, float]:
        """
        将小地图像素坐标转换回游戏世界坐标。

        Args:
            px: 小地图像素X坐标
            py: 小地图像素Y坐标
            map_name: 地图名称

        Returns:
            游戏世界坐标 (world_x, world_y)
        """
        if map_name not in self.map_config:
            raise KeyError(f"地图 '{map_name}' 的配置不存在")

        cfg = self.map_config[map_name]
        pos_x = cfg["pos_x"]
        pos_y = cfg["pos_y"]
        scale = cfg["scale"]

        world_x = px * scale + pos_x
        world_y = pos_y - py * scale

        return world_x, world_y

    def is_on_lower_level(
        self, world_x: float, world_y: float, world_z: float, map_name: str
    ) -> bool:
        """
        判断给定世界坐标是否位于地图的下层（如de_nuke的下层）。

        Args:
            world_x: 世界X坐标
            world_y: 世界Y坐标
            world_z: 世界Z坐标（高度）
            map_name: 地图名称

        Returns:
            若位于下层则返回True，否则返回False
        """
        if map_name not in self.map_config:
            return False
        cfg = self.map_config[map_name]
        lower_threshold = cfg.get("lower_level_max_units", -1000000)
        return world_z <= lower_threshold


def world_to_minimap(
    world_x: float, world_y: float, map_name: str, config_path: str = None
) -> Tuple[int, int]:
    """
    便捷函数：将游戏世界坐标转换为小地图像素坐标。

    Args:
        world_x: 游戏世界X坐标
        world_y: 游戏世界Y坐标
        map_name: 地图名称
        config_path: 可选的自定义配置文件路径

    Returns:
        小地图像素坐标 (px, py)
    """
    transformer = CoordinateTransformer(config_path)
    return transformer.world_to_minimap(world_x, world_y, map_name)


if __name__ == "__main__":
    # 简单自测：用de_dust2的已知出生点验证
    transformer = CoordinateTransformer()

    # T方出生点大约在 (-700, 1800)
    t_spawn_px, t_spawn_py = transformer.world_to_minimap(-700, 1800, "de_dust2")
    print(f"de_dust2 T出生点像素坐标: ({t_spawn_px}, {t_spawn_py})")

    # CT方出生点大约在 (1200, 2600)
    ct_spawn_px, ct_spawn_py = transformer.world_to_minimap(1200, 2600, "de_dust2")
    print(f"de_dust2 CT出生点像素坐标: ({ct_spawn_px}, {ct_spawn_py})")

    # A点大约 (1350, 2600)
    a_px, a_py = transformer.world_to_minimap(1350, 2600, "de_dust2")
    print(f"de_dust2 A点像素坐标: ({a_px}, {a_py})")

    # B点大约 (-1650, 800)
    b_px, b_py = transformer.world_to_minimap(-1650, 800, "de_dust2")
    print(f"de_dust2 B点像素坐标: ({b_px}, {b_py})")
