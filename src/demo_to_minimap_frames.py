"""
端到端 Demo → 小地图帧序列渲染脚本

输入: .dem 文件 + 地图名 + 回合号
输出: 该回合的小地图 PNG 帧序列

用法:
    python src/demo_to_minimap_frames.py \
        --input data/demos/xxx.dem \
        --map de_dust2 \
        --round 5 \
        --output outputs/minimap_frames
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from awpy import Demo
from loguru import logger
from PIL import Image, ImageDraw, ImageEnhance

from coordinate_transform import CoordinateTransformer


# ============ 渲染配置 ============
COLOR_T = "#CD6A70"
COLOR_CT = "#5B8FF9"
COLOR_BOMB_CARRIER_BORDER = "#E8B87D"
COLOR_BOMB_DROP = "#FFD700"
COLOR_BOMB_PLANTED = "#FF0000"
COLOR_BOMB_PLANTED_GLOW = "#FF4444"
COLOR_DEAD = "#999999"
COLOR_BOMB_SITE_FLASH = "#FFD700"

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
    """小地图渲染器（内联，避免外部依赖）。"""

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
        bg = self.map_img.copy()
        enhancer = ImageEnhance.Brightness(bg)
        bg = enhancer.enhance(0.75)
        gray_bg = Image.new("RGBA", (MAP_SIZE, MAP_SIZE), (220, 220, 220, 255))
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
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def _draw_direction_line(self, draw: ImageDraw.Draw, cx: int, cy: int, yaw: float, length: int, color: str):
        rad = math.radians(-yaw + 90)
        end_x = cx + length * math.cos(rad)
        end_y = cy - length * math.sin(rad)
        draw.line([(cx, cy), (end_x, end_y)], fill=color, width=2)

    def _draw_bomb_drop(self, img: Image.Image, px: int, py: int) -> Image.Image:
        overlay = Image.new("RGBA", (MAP_SIZE, MAP_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        size = BOMB_CROSS_SIZE
        glow_size = size + 6
        draw.line([(px - glow_size, py - glow_size), (px + glow_size, py + glow_size)],
                  fill=(255, 140, 0, 120), width=7)
        draw.line([(px - glow_size, py + glow_size), (px + glow_size, py - glow_size)],
                  fill=(255, 140, 0, 120), width=7)
        draw.line([(px - size, py - size), (px + size, py + size)], fill=COLOR_BOMB_DROP, width=4)
        draw.line([(px - size, py + size), (px + size, py - size)], fill=COLOR_BOMB_DROP, width=4)
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 255, 255, 255))
        return Image.alpha_composite(img, overlay)

    def _draw_bomb_planted(self, img: Image.Image, px: int, py: int) -> Image.Image:
        overlay = Image.new("RGBA", (MAP_SIZE, MAP_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.ellipse(
            [px - BOMB_PLANTED_GLOW_RADIUS, py - BOMB_PLANTED_GLOW_RADIUS,
             px + BOMB_PLANTED_GLOW_RADIUS, py + BOMB_PLANTED_GLOW_RADIUS],
            fill=(255, 68, 68, 60)
        )
        draw.ellipse(
            [px - BOMB_PLANTED_RADIUS, py - BOMB_PLANTED_RADIUS,
             px + BOMB_PLANTED_RADIUS, py + BOMB_PLANTED_RADIUS],
            outline=COLOR_BOMB_PLANTED,
            fill=(255, 0, 0, 180),
            width=3
        )
        draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 255, 255, 255))
        return Image.alpha_composite(img, overlay)

    def render_frame(
        self,
        frame_data: pd.DataFrame,
        bomb_state: Optional[Dict] = None,
        flash_bomb_site: Optional[str] = None,
    ) -> Image.Image:
        img = self._prepare_background()
        draw = ImageDraw.Draw(img)

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


# ============ Demo 解析 ============

def parse_demo_round(
    dem_path: str,
    target_round: int,
    parse_rate: int = 128,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    解析 demo 文件，提取指定回合的 ticks 和炸弹事件。

    Returns:
        (round_ticks_df, bomb_events_df, map_name)
        如果回合不存在则返回 (None, None, map_name)
    """
    dem_path = Path(dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"Demo 文件不存在: {dem_path}")

    logger.info(f"开始解析 demo: {dem_path}")
    demo = Demo(str(dem_path), verbose=False)

    demo.parse(
        player_props=[
            "CCSPlayerPawn.m_angEyeAngles",
            "CCSPlayerPawn.m_ArmorValue",
            "CCSPlayerController.m_bPawnIsAlive",
        ]
    )

    map_name = demo.header.get("map_name", "unknown")
    tick_rate = demo.header.get("tickRate", 128)
    logger.info(f"地图: {map_name}, 总回合数: {len(demo.rounds)}, 总 ticks: {len(demo.ticks)}")

    if len(demo.ticks) == 0:
        logger.warning("未解析到有效 ticks 数据")
        return None, None, map_name

    # --- 炸弹事件 ---
    bomb_df = demo.bomb.clone()
    if len(bomb_df) > 0:
        bomb_df = bomb_df.sort(["round_num", "tick"])

    # --- 处理 ticks ---
    ticks = demo.ticks.clone()
    ticks = ticks.rename({
        "CCSPlayerPawn.m_angEyeAngles": "eye_angles",
        "CCSPlayerPawn.m_ArmorValue": "armor",
        "CCSPlayerController.m_bPawnIsAlive": "is_alive",
    })
    ticks = ticks.with_columns(ticks["eye_angles"].list.get(1).alias("yaw"))
    ticks = ticks.with_columns(ticks["side"].str.to_uppercase().alias("team_side"))
    ticks = ticks.with_columns((ticks["tick"] / tick_rate).alias("frame_time"))

    # --- 合并炸弹持有状态 ---
    if len(bomb_df) > 0:
        bomb_states = []
        for round_num in bomb_df["round_num"].unique().to_list():
            round_bomb_events = bomb_df.filter(bomb_df["round_num"] == round_num).sort("tick")
            round_ticks = ticks.filter(ticks["round_num"] == round_num)
            if len(round_ticks) == 0:
                continue

            tick_list = sorted(round_ticks["tick"].unique().to_list())
            carrier = ""
            tick_to_carrier = {}
            event_ticks = round_bomb_events.select(["tick", "event", "name"]).to_dicts()
            event_idx = 0

            for t in tick_list:
                while event_idx < len(event_ticks) and event_ticks[event_idx]["tick"] <= t:
                    evt = event_ticks[event_idx]["event"]
                    pname = event_ticks[event_idx]["name"]
                    if evt == "pickup":
                        carrier = pname if pname else ""
                    elif evt in ("drop", "plant", "defuse", "exploded"):
                        carrier = ""
                    event_idx += 1
                tick_to_carrier[t] = carrier

            if tick_to_carrier:
                carrier_df = pd.DataFrame({
                    "tick": list(tick_to_carrier.keys()),
                    "bomb_carrier": list(tick_to_carrier.values()),
                    "round_num": [round_num] * len(tick_to_carrier)
                })
                bomb_states.append(carrier_df)

        if bomb_states:
            all_bomb_states = pd.concat(bomb_states, ignore_index=True)
            ticks = ticks.to_pandas()
            ticks = ticks.merge(all_bomb_states, on=["round_num", "tick"], how="left")
            ticks["bomb_carrier"] = ticks["bomb_carrier"].fillna("")
            ticks["has_bomb"] = ticks["name"] == ticks["bomb_carrier"]
        else:
            ticks = ticks.to_pandas()
            ticks["has_bomb"] = False
    else:
        ticks = ticks.to_pandas()
        ticks["has_bomb"] = False

    # 提取目标回合
    round_ticks = ticks[ticks["round_num"] == target_round]
    if len(round_ticks) == 0:
        logger.warning(f"回合 {target_round} 不存在于 demo 中")
        return None, None, map_name

    # 采样（默认每秒一帧）
    unique_ticks = sorted(round_ticks["frame_time"].unique())
    sampled = unique_ticks[::parse_rate] if len(unique_ticks) > parse_rate else unique_ticks
    round_ticks = round_ticks[round_ticks["frame_time"].isin(sampled)]

    # 标准化列名
    round_ticks = round_ticks.rename(columns={
        "round_num": "round_number",
        "name": "player_name",
        "steamid": "player_id",
        "X": "pos_x",
        "Y": "pos_y",
        "Z": "pos_z",
        "yaw": "yaw_angle",
        "health": "health",
    })

    round_bomb_df = None
    if len(bomb_df) > 0:
        round_bomb_df = bomb_df.filter(bomb_df["round_num"] == target_round).to_pandas()

    logger.success(f"回合 {target_round}: 提取 {len(round_ticks)} 行, {len(sampled)} 帧")
    return round_ticks, round_bomb_df, map_name


def build_bomb_intervals(bomb_df: Optional[pd.DataFrame], round_num: int,
                         round_start: float, round_end: float,
                         tick_rate: int = 128) -> List[Dict]:
    """从炸弹事件构建状态时间段列表。"""
    intervals = []
    if bomb_df is None or len(bomb_df) == 0:
        return intervals

    round_bomb = bomb_df[bomb_df["round_num"] == round_num].sort_values("tick")
    if len(round_bomb) == 0:
        return intervals

    current_status = None
    current_pos = None
    last_time = round_start

    for _, evt in round_bomb.iterrows():
        event_type = evt["event"]
        evt_time = evt.get("tick", 0) / tick_rate if pd.notna(evt.get("tick")) else None
        if evt_time is None:
            continue

        if current_status is not None:
            intervals.append({
                "start": last_time,
                "end": evt_time,
                "status": current_status,
                "pos_x": current_pos[0] if current_pos else None,
                "pos_y": current_pos[1] if current_pos else None,
            })

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

    if current_status is not None:
        intervals.append({
            "start": last_time,
            "end": round_end + 10,
            "status": current_status,
            "pos_x": current_pos[0] if current_pos else None,
            "pos_y": current_pos[1] if current_pos else None,
        })

    return intervals


def get_bomb_state_at_time(intervals: List[Dict], t: float) -> Optional[Dict]:
    for iv in intervals:
        if iv["start"] <= t <= iv["end"]:
            return {
                "status": iv["status"],
                "pos_x": iv["pos_x"],
                "pos_y": iv["pos_y"],
            }
    return None


# ============ 主流程 ============

def frames_to_video(
    frame_paths: List[str],
    output_video_path: str,
    fps: float = 10.0,
) -> str:
    """将 PNG 帧序列合成为 MP4 视频。"""
    if not frame_paths:
        logger.warning("没有帧可合成视频")
        return ""

    # 读取第一帧获取尺寸
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"无法读取第一帧: {frame_paths[0]}")

    height, width = first_frame.shape[:2]
    # 使用 avc1 (H.264) 编码，兼容性优于 mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {output_video_path}")

    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            logger.warning(f"跳过无法读取的帧: {path}")
            continue
        writer.write(frame)

    writer.release()
    logger.success(f"视频已保存: {output_video_path} ({len(frame_paths)} 帧, {fps} fps)")
    return output_video_path


def render_minimap_frames(
    dem_path: str,
    map_name: str,
    round_number: int,
    output_dir: str,
    parse_rate: int = 128,
    map_config_path: str = "map_overview.json",
    video_fps: float = 10.0,
) -> Dict:
    """
    从 demo 文件渲染指定回合的小地图帧序列。

    Args:
        dem_path: .dem 文件路径
        map_name: 地图名称（如 de_dust2）
        round_number: 回合号（从1开始）
        output_dir: 输出帧序列的根目录
        parse_rate: tick 采样间隔，默认 128（1fps @ 128 tick）
        map_config_path: 地图配置文件路径

    Returns:
        结果字典，包含输出路径和统计信息
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 解析 demo
    round_ticks, bomb_df, detected_map = parse_demo_round(dem_path, round_number, parse_rate)
    if round_ticks is None:
        return {"status": "error", "reason": f"回合 {round_number} 不存在"}

    # 如果用户未指定地图名，尝试使用检测到的
    if not map_name or map_name == "auto":
        map_name = detected_map
        logger.info(f"自动检测地图: {map_name}")

    # 2. 初始化渲染器
    renderer = MinimapRenderer(map_name, map_config_path)

    # 3. 构建炸弹状态时间线
    round_start = round_ticks["frame_time"].min()
    round_end = round_ticks["frame_time"].max()
    bomb_intervals = build_bomb_intervals(bomb_df, round_number, round_start, round_end)

    # 4. 逐帧渲染
    frame_times = sorted(round_ticks["frame_time"].unique())
    rendered_files = []

    for i, frame_time in enumerate(frame_times):
        frame_data = round_ticks[round_ticks["frame_time"] == frame_time]
        if len(frame_data) == 0:
            continue

        # 炸弹状态（carried 不单独渲染，由玩家边框表示）
        bomb_state = get_bomb_state_at_time(bomb_intervals, frame_time)
        if bomb_state and bomb_state.get("status") == "carried":
            bomb_state = None

        img = renderer.render_frame(frame_data, bomb_state=bomb_state)

        # RGBA → RGB（去掉透明通道）
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (240, 240, 240))
            bg.paste(img, mask=img.split()[3])
            img = bg

        filename = f"frame_{i:04d}_t{frame_time:.1f}.png"
        out_path = output_dir / filename
        img.save(str(out_path))
        rendered_files.append(str(out_path))

    result = {
        "status": "success",
        "dem_path": dem_path,
        "map_name": map_name,
        "round_number": round_number,
        "output_dir": str(output_dir),
        "total_frames": len(rendered_files),
        "duration_seconds": round_end - round_start,
        "sample_rate": parse_rate,
        "files": rendered_files,
    }

    logger.success(f"渲染完成: {len(rendered_files)} 帧 → {output_dir}")

    # 交互式询问是否合成视频
    print("\n" + "=" * 50)
    print(f"帧序列已生成: {len(rendered_files)} 张 PNG")
    print("=" * 50)
    print("请选择下一步操作:")
    print("  [1] 合成为 MP4 视频")
    print("  [2] 直接退出（保留 PNG 帧序列）")
    print("=" * 50)

    try:
        choice = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "2"
        print()

    if choice == "1":
        video_path = str(output_dir / f"round_{round_number:02d}_minimap.mp4")
        try:
            frames_to_video(rendered_files, video_path, fps=video_fps)
            result["video_path"] = video_path
            result["video_fps"] = video_fps
        except Exception as e:
            logger.error(f"视频合成失败: {e}")
            result["video_error"] = str(e)
    else:
        print("已退出，帧序列保留在输出目录中。")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="端到端 Demo → 小地图帧序列渲染"
    )
    parser.add_argument("--input", required=True, help="输入 .dem 文件路径")
    parser.add_argument("--map", default="auto", help="地图名称（如 de_dust2），默认 auto 从 demo 检测")
    parser.add_argument("--round", type=int, required=True, help="回合号（从1开始）")
    parser.add_argument("--output", required=True, help="输出帧序列目录")
    parser.add_argument("--parse-rate", type=int, default=128,
                        help="tick 采样间隔，默认 128（约 1fps @ 128 tick）")
    parser.add_argument("--map-config", default="map_overview.json",
                        help="地图配置文件路径")
    parser.add_argument("--video-fps", type=float, default=2.0,
                        help="输出视频帧率，默认 2 fps（每帧展示 0.5 秒）")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        sys.exit(1)

    result = render_minimap_frames(
        dem_path=args.input,
        map_name=args.map,
        round_number=args.round,
        output_dir=args.output,
        parse_rate=args.parse_rate,
        map_config_path=args.map_config,
        video_fps=args.video_fps,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
