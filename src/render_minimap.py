"""
小地图重绘引擎

基于Pillow库，将单回合CSV中的玩家坐标绘制到地图雷达图上。
支持炸弹携带、掉落、安装状态的完整可视化。
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import pandas as pd

from coordinate_transform import CoordinateTransformer


# 颜色配置
COLOR_T = "#CD6A70"
COLOR_CT = "#5B8FF9"
COLOR_BOMB_CARRIER_BORDER = "#E8B87D"
COLOR_BOMB_DROP = "#FFD700"
COLOR_BOMB_PLANTED = "#FF0000"
COLOR_BOMB_PLANTED_GLOW = "#FF4444"
COLOR_DEAD = "#999999"
COLOR_BOMB_SITE_FLASH = "#FFD700"

# 渲染参数
MAP_SIZE = 1024
PLAYER_DOT_MAX_RADIUS = 10
PLAYER_DOT_MIN_RADIUS = 4
DIRECTION_LINE_LENGTH = 8
BOMB_CARRIER_BORDER_WIDTH = 3
BOMB_CROSS_SIZE = 8
BOMB_PLANTED_RADIUS = 14
BOMB_PLANTED_GLOW_RADIUS = 22
DEAD_ALPHA = 128


class MinimapRenderer:
    """小地图渲染器。"""

    def __init__(self, map_name: str, map_config_path: str = None):
        self.map_name = map_name
        self.transformer = CoordinateTransformer(map_config_path)

        config = self.transformer.map_config.get(map_name)
        if not config:
            raise ValueError(f"地图 '{map_name}' 配置不存在")

        base_path = Path(map_config_path).parent if map_config_path else Path(".")
        map_img_path = base_path / config["image"]
        if not map_img_path.exists():
            map_img_path = Path(config["image"])
        if not map_img_path.exists():
            raise FileNotFoundError(f"地图图片不存在: {map_img_path}")

        self.map_img = Image.open(map_img_path).convert("RGBA")
        if self.map_img.size != (MAP_SIZE, MAP_SIZE):
            self.map_img = self.map_img.resize((MAP_SIZE, MAP_SIZE), Image.LANCZOS)

        bomb_path = base_path / "bomb_sites.json"
        self.bomb_sites = {}
        if bomb_path.exists():
            with open(bomb_path, "r", encoding="utf-8") as f:
                self.bomb_sites = json.load(f).get(map_name, {})

    def _prepare_background(self) -> Image.Image:
        """
        准备地图背景。
        
        策略：
        - 地图可到达区域：轻度暗化（亮度0.75），保持可见
        - 地图外区域：浅灰色背景，与地图形成对比
        """
        bg = self.map_img.copy()
        
        # 轻度暗化底图
        enhancer = ImageEnhance.Brightness(bg)
        bg = enhancer.enhance(0.75)
        
        # 创建浅灰背景
        gray_bg = Image.new("RGBA", (MAP_SIZE, MAP_SIZE), (220, 220, 220, 255))
        
        # 将地图叠加到浅灰背景上（利用alpha通道，地图外区域是透明的）
        # 先合并：浅灰背景 + 暗化后的地图
        composite = Image.alpha_composite(gray_bg, bg)
        return composite

    def _world_to_px(self, world_x: float, world_y: float) -> Tuple[int, int]:
        return self.transformer.world_to_minimap(world_x, world_y, self.map_name)

    def _get_dot_radius(self, health: int) -> int:
        if health <= 0:
            return PLAYER_DOT_MIN_RADIUS
        ratio = health / 100.0
        return int(PLAYER_DOT_MIN_RADIUS + ratio * (PLAYER_DOT_MAX_RADIUS - PLAYER_DOT_MIN_RADIUS))

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _draw_direction_line(self, draw: ImageDraw.Draw, cx: int, cy: int, yaw: float, length: int, color: str):
        """绘制玩家朝向线。"""
        rad = math.radians(-yaw + 90)
        end_x = cx + length * math.cos(rad)
        end_y = cy - length * math.sin(rad)
        draw.line([(cx, cy), (end_x, end_y)], fill=color, width=2)

    def _draw_bomb_drop(self, img: Image.Image, px: int, py: int) -> Image.Image:
        """绘制炸弹掉落叉号：带发光效果的金色X。"""
        overlay = Image.new("RGBA", (MAP_SIZE, MAP_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        size = BOMB_CROSS_SIZE
        # 外发光（橙色半透明）
        glow_size = size + 6
        draw.line([(px - glow_size, py - glow_size), (px + glow_size, py + glow_size)], 
                  fill=(255, 140, 0, 120), width=7)
        draw.line([(px - glow_size, py + glow_size), (px + glow_size, py - glow_size)], 
                  fill=(255, 140, 0, 120), width=7)
        
        # 金色叉号
        draw.line([(px - size, py - size), (px + size, py + size)], fill=COLOR_BOMB_DROP, width=4)
        draw.line([(px - size, py + size), (px + size, py - size)], fill=COLOR_BOMB_DROP, width=4)
        
        # 中心白点
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 255, 255, 255))
        
        return Image.alpha_composite(img, overlay)

    def _draw_bomb_planted(self, img: Image.Image, px: int, py: int) -> Image.Image:
        """绘制炸弹安装状态：红色发光圆环。"""
        overlay = Image.new("RGBA", (MAP_SIZE, MAP_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 外发光（半透明红色）
        draw.ellipse(
            [px - BOMB_PLANTED_GLOW_RADIUS, py - BOMB_PLANTED_GLOW_RADIUS,
             px + BOMB_PLANTED_GLOW_RADIUS, py + BOMB_PLANTED_GLOW_RADIUS],
            fill=(255, 68, 68, 60)
        )
        
        # 中间圆环
        draw.ellipse(
            [px - BOMB_PLANTED_RADIUS, py - BOMB_PLANTED_RADIUS,
             px + BOMB_PLANTED_RADIUS, py + BOMB_PLANTED_RADIUS],
            outline=COLOR_BOMB_PLANTED,
            fill=(255, 0, 0, 180),
            width=3
        )
        
        # 中心白点
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 255, 255, 255))
        
        return Image.alpha_composite(img, overlay)

    def render_frame(
        self,
        frame_data: pd.DataFrame,
        bomb_state: Optional[Dict] = None,
        flash_bomb_site: Optional[str] = None,
    ) -> Image.Image:
        """
        渲染单帧画面。

        Args:
            frame_data: 玩家状态DataFrame
            bomb_state: 炸弹状态字典，包含:
                - status: "carried" | "dropped" | "planted" | None
                - pos_x, pos_y: 炸弹世界坐标（dropped/planted时）
            flash_bomb_site: 高亮炸弹点 "A" 或 "B"

        Returns:
            渲染好的RGBA图像
        """
        img = self._prepare_background()
        draw = ImageDraw.Draw(img)

        # 高亮炸弹点区域
        if flash_bomb_site and flash_bomb_site in self.bomb_sites:
            site = self.bomb_sites[flash_bomb_site]
            cx, cy = site["center"]
            radius = site["radius"]
            overlay = Image.new("RGBA", (MAP_SIZE, MAP_SIZE), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            rgb = self._hex_to_rgb(COLOR_BOMB_SITE_FLASH)
            overlay_draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=(*rgb, 60),
                outline=COLOR_BOMB_SITE_FLASH,
                width=2
            )
            img = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img)

        # 绘制炸弹状态（掉落或安装）
        if bomb_state:
            status = bomb_state.get("status")
            if status in ("dropped", "planted"):
                bx, by = self._world_to_px(bomb_state["pos_x"], bomb_state["pos_y"])
                if status == "dropped":
                    img = self._draw_bomb_drop(img, bx, by)
                    draw = ImageDraw.Draw(img)
                elif status == "planted":
                    img = self._draw_bomb_planted(img, bx, by)
                    draw = ImageDraw.Draw(img)

        # 绘制每个玩家
        for _, row in frame_data.iterrows():
            team = str(row.get("team_side", "")).upper()
            health_raw = row.get("health", 100)
            health = int(health_raw) if pd.notna(health_raw) else 0
            alive = str(row.get("is_alive", "true")).lower() == "true"
            has_bomb = str(row.get("has_bomb", "false")).lower() == "true"
            yaw = float(row.get("yaw_angle", 0))

            px, py = self._world_to_px(row["pos_x"], row["pos_y"])

            if not alive:
                radius = self._get_dot_radius(0)
                overlay = Image.new("RGBA", (MAP_SIZE, MAP_SIZE), (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                rgb = self._hex_to_rgb(COLOR_DEAD)
                overlay_draw.ellipse(
                    [px - radius, py - radius, px + radius, py + radius],
                    fill=(*rgb, DEAD_ALPHA)
                )
                img = Image.alpha_composite(img, overlay)
                draw = ImageDraw.Draw(img)
                continue

            color = COLOR_T if team == "T" else COLOR_CT
            radius = self._get_dot_radius(health)

            draw.ellipse(
                [px - radius, py - radius, px + radius, py + radius],
                fill=color
            )

            if has_bomb:
                draw.ellipse(
                    [px - radius - BOMB_CARRIER_BORDER_WIDTH,
                     py - radius - BOMB_CARRIER_BORDER_WIDTH,
                     px + radius + BOMB_CARRIER_BORDER_WIDTH,
                     py + radius + BOMB_CARRIER_BORDER_WIDTH],
                    outline=COLOR_BOMB_CARRIER_BORDER,
                    width=BOMB_CARRIER_BORDER_WIDTH
                )

            self._draw_direction_line(draw, px, py, yaw, DIRECTION_LINE_LENGTH, color)

        return img

    def render_round(
        self,
        csv_path: str,
        bomb_events_path: Optional[str],
        output_dir: str,
        timestamps: List[float] = None,
    ):
        """
        渲染单个回合的多个关键帧。

        Args:
            csv_path: 回合CSV文件路径
            bomb_events_path: 炸弹事件CSV路径（可选）
            output_dir: 输出图片目录
            timestamps: 相对于回合开始的时间偏移列表（秒）
        """
        csv_path = Path(csv_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(str(csv_path))
        if len(df) == 0:
            print(f"空数据: {csv_path}")
            return []

        round_num = df["round_number"].iloc[0]
        round_start = df["frame_time"].min()
        round_end = df["frame_time"].max()

        # 读取炸弹事件，构建时间段状态列表
        bomb_state_intervals = []
        if bomb_events_path and Path(bomb_events_path).exists():
            bomb_events = pd.read_csv(str(bomb_events_path))
            round_bomb = bomb_events[bomb_events["round_num"] == round_num].sort_values("tick")
            
            tick_rate = 128
            current_status = None
            current_pos = None
            last_time = round_start
            
            for _, evt in round_bomb.iterrows():
                event_type = evt["event"]
                evt_time = evt.get("tick", 0) / tick_rate if "tick" in evt else None
                if evt_time is None:
                    continue
                    
                # 记录上一个状态持续的时间段
                if current_status is not None:
                    bomb_state_intervals.append({
                        "start": last_time,
                        "end": evt_time,
                        "status": current_status,
                        "pos_x": current_pos[0] if current_pos else None,
                        "pos_y": current_pos[1] if current_pos else None,
                    })
                
                # 更新状态
                if event_type == "pickup":
                    current_status = "carried"
                    current_pos = None
                elif event_type == "drop":
                    current_status = "dropped"
                    current_pos = (evt["X"], evt["Y"])
                elif event_type == "plant":
                    current_status = "planted"
                    current_pos = (evt["X"], evt["Y"])
                elif event_type in ("defuse", "exploded"):
                    current_status = None
                    current_pos = None
                    
                last_time = evt_time
            
            # 记录最后一个状态到回合结束
            if current_status is not None:
                bomb_state_intervals.append({
                    "start": last_time,
                    "end": round_end + 10,
                    "status": current_status,
                    "pos_x": current_pos[0] if current_pos else None,
                    "pos_y": current_pos[1] if current_pos else None,
                })

        if timestamps is None:
            timestamps = [0, 30, 60]
            if round_end - round_start > 90:
                timestamps.append(90)
            timestamps.append(round_end - round_start)

        rendered_files = []
        for offset in timestamps:
            target_time = round_start + offset
            df["time_diff"] = (df["frame_time"] - target_time).abs()
            closest_time = df.loc[df["time_diff"].idxmin(), "frame_time"]
            frame_data = df[df["frame_time"] == closest_time]

            # 获取该帧的炸弹状态（查找所在时间段）
            bomb_state = None
            for interval in bomb_state_intervals:
                if interval["start"] <= closest_time <= interval["end"]:
                    bomb_state = {
                        "status": interval["status"],
                        "pos_x": interval["pos_x"],
                        "pos_y": interval["pos_y"],
                    }
                    break
            # carried状态由玩家身上的金色边框表示，不单独渲染
            if bomb_state and bomb_state.get("status") == "carried":
                bomb_state = None

            img = self.render_frame(frame_data, bomb_state=bomb_state)
            out_path = output_dir / f"{csv_path.stem}_offset{offset:.0f}s.png"
            img.save(str(out_path))
            rendered_files.append(str(out_path))
            print(f"已渲染: {out_path} ({len(frame_data)} 玩家, 炸弹{bomb_state['status'] if bomb_state else '无'})")

        return rendered_files


def parse_timestamps(ts_str: str) -> Optional[List[float]]:
    if ts_str.lower() in ("auto", "default", ""):
        return None
    return [float(x.strip()) for x in ts_str.split(",")]


def main():
    parser = argparse.ArgumentParser(description="小地图关键帧渲染脚本")
    parser.add_argument("--csv", required=True, help="单回合CSV文件路径")
    parser.add_argument("--map", required=True, help="地图名称")
    parser.add_argument("--bomb-events", default=None, help="炸弹事件CSV路径")
    parser.add_argument("--timestamps", default="auto", help='时间偏移（秒），逗号分隔')
    parser.add_argument("--output", required=True, help="输出图片目录")
    args = parser.parse_args()

    renderer = MinimapRenderer(args.map)
    timestamps = parse_timestamps(args.timestamps)
    renderer.render_round(args.csv, args.bomb_events, args.output, timestamps)


if __name__ == "__main__":
    main()
