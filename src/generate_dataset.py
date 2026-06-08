"""
批量数据集生成流水线

按比赛组织数据集，支持1秒间隔采样。
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image
from loguru import logger

from render_minimap import MinimapRenderer


def extract_match_name(demo_name: str) -> str:
    """从demo文件名提取比赛名（支持新旧两种命名格式）。
    
    新格式(推荐): '赛事名-队伍A-vs-队伍B-mN-地图名'
      例如: 'BLAST-Open-Rotterdam-2026-falcons-vs-furia-m1-mirage'
            'PGL-Astana-2026-furia-vs-falcons-m1-mirage'
      → 返回 'BLAST-Open-Rotterdam-2026-falcons-vs-furia'
      → 返回 'PGL-Astana-2026-furia-vs-falcons'
    
    旧格式(兼容): '队伍A-vs-队伍B-mN-地图名'
      例如: 'falcons-vs-furia-m1-mirage'
      → 返回 'falcons-vs-furia-m1-mirage' (完整demo名，避免合并)
    
    这样不同赛事的同一队伍组合会被分到不同文件夹。
    """
    import re
    # 找到末尾的 -m数字-地图名
    m = re.search(r'-m\d+-[\w]+$', demo_name)
    if not m:
        return demo_name
    
    # 去掉末尾的 -mN-地图名
    base = demo_name[:m.start()]
    
    # 判断是新格式还是旧格式:
    # 新格式: base 包含赛事前缀，即 base 中 -vs- 前面还有赛事名部分
    # 旧格式: base 只有 "队伍A-vs-队伍B"
    # 简单判断: 如果 base 符合 "^([\w-]+)-vs-([\w-]+)$" 且队伍名不含额外连字符，则是旧格式
    # 更可靠: 检测已知赛事关键词
    tournament_keywords = [
        'blast', 'pgl', 'esl', 'iem', 'major', 'rivals', 'pro-league',
        'championship', 'tournament', 'open', 'grand',
    ]
    base_lower = base.lower()
    has_tournament_prefix = any(kw in base_lower for kw in tournament_keywords)
    
    if has_tournament_prefix:
        # 新格式: 返回 赛事名-队伍A-vs-队伍B
        return base
    else:
        # 旧格式: 保留完整demo名，避免不同赛事的同一队伍组合被合并
        return demo_name


def build_bomb_intervals(bomb_events_path: Path, round_num: int,
                         round_start: float, round_end: float,
                         tick_rate: int = 128) -> List[Dict]:
    """从炸弹事件构建状态时间段列表。"""
    intervals = []
    if not bomb_events_path.exists():
        return intervals

    bomb_df = pd.read_csv(str(bomb_events_path))
    round_bomb = bomb_df[bomb_df["round_num"] == round_num].sort_values("tick")

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


def process_round(
    csv_path: Path,
    map_name: str,
    direction: str,
    winner: str,
    output_dir: Path,
    bomb_events_path: Optional[Path] = None,
    sample_interval: float = 1.0,
    last_seconds: int = 30,
    img_format: str = "png",
    map_config_path: str = "map_overview.json",
) -> List[Dict]:
    """处理单个回合，生成采样帧图片和label记录。"""
    df = pd.read_csv(str(csv_path))
    if len(df) == 0:
        return []

    round_num = int(df["round_number"].iloc[0])
    round_start = df["frame_time"].min()
    round_end = df["frame_time"].max()

    # freeze time 结束点 (11秒)
    freeze_end_time = round_start + 11.0
    
    # 从 freeze time 结束后开始采样，直到回合结束
    all_times = sorted(df["frame_time"].unique())
    
    # 只保留 freeze time 之后的数据用于采样
    valid_times = [t for t in all_times if t >= freeze_end_time - 0.5]
    if len(valid_times) == 0:
        logger.warning(f"回合 {round_num} 在 freeze time 后无数据")
        return []

    # 固定间隔采样 (从 freeze_end_time 开始)
    sampled_times = []
    target = freeze_end_time
    while target <= round_end + 0.001:
        closest = min(all_times, key=lambda t: abs(t - target))
        # 确保 closest 严格在 freeze time 之后
        if closest < freeze_end_time - 0.001:
            target += sample_interval
            continue
        if not sampled_times or abs(closest - sampled_times[-1]) > 0.5:
            sampled_times.append(closest)
        target += sample_interval

    if not sampled_times:
        logger.warning(f"回合 {round_num} 采样后无帧")
        return []

    # 输出目录: {output_dir}/{direction}/
    class_dir = output_dir / direction
    class_dir.mkdir(parents=True, exist_ok=True)

    renderer = MinimapRenderer(map_name, map_config_path)

    bomb_intervals = build_bomb_intervals(
        bomb_events_path or Path("/dev/null"), round_num, round_start, round_end
    )

    records = []
    skipped = 0
    rendered = 0
    for frame_time in sampled_times:
        frame_data = df[df["frame_time"] == frame_time]
        if len(frame_data) == 0:
            continue

        rel_time = frame_time - round_start
        time_ms = int(round(rel_time * 10))
        interval_tag = f"{sample_interval:g}s".replace(".", "p")
        filename = f"{map_name}_r{round_num:02d}_t{time_ms:04d}_i{interval_tag}.{img_format}"
        img_path = class_dir / filename

        # 计算 record 信息（无论是否跳过渲染都需要）
        alive_t = int(frame_data[frame_data["team_side"] == "T"]["is_alive"].fillna(0).sum())
        alive_ct = int(frame_data[frame_data["team_side"] == "CT"]["is_alive"].fillna(0).sum())

        record = {
            "image_path": str(img_path.relative_to(output_dir)),
            "map_name": map_name,
            "round_number": round_num,
            "direction": direction,
            "winner": winner,
            "frame_time": float(frame_time),
            "round_relative_time": float(rel_time),
            "sample_interval": sample_interval,
            "num_players_alive_t": alive_t,
            "num_players_alive_ct": alive_ct,
        }
        records.append(record)

        # 文件存在检测：已存在则跳过渲染
        if img_path.exists():
            skipped += 1
            continue

        # 渲染新图片
        bomb_state = get_bomb_state_at_time(bomb_intervals, frame_time)
        if bomb_state and bomb_state.get("status") == "carried":
            bomb_state = None

        img = renderer.render_frame(frame_data, bomb_state=bomb_state)

        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (240, 240, 240))
            bg.paste(img, mask=img.split()[3])
            img = bg

        img.save(str(img_path))
        rendered += 1

    logger.info(f"回合 {round_num} ({map_name}) → {direction}: {len(records)} 张图 "
                f"(渲染 {rendered}, 跳过 {skipped})")

    return records


def generate_dataset(
    parsed_demo_dir: str,
    output_base_dir: str,
    sample_interval: float = 1.0,
    last_seconds: int = 30,
    img_format: str = "png",
    map_config_path: str = "map_overview.json",
) -> Dict:
    """为一个demo生成数据集，输出到对应比赛目录下。"""
    parsed_demo_dir = Path(parsed_demo_dir)
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    summary_path = parsed_demo_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"未找到summary.json: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    demo_name = summary["demo_name"]
    match_name = extract_match_name(demo_name)
    map_name = summary["map_name"]
    
    # 输出到: dataset_by_match/{match_name}/{map_name}/
    output_dir = output_base_dir / match_name / map_name
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = parsed_demo_dir / "csv"
    bomb_events_path = parsed_demo_dir / "bomb_events.csv"
    round_directions = {r["round_num"]: r.get("direction", "none") for r in summary.get("rounds", [])}
    round_winners = {r["round_num"]: r.get("winner", "") for r in summary.get("rounds", [])}

    csv_files = sorted(csv_dir.glob("*_round_*.csv"))
    if not csv_files:
        logger.warning(f"未找到CSV文件: {csv_dir}")
        return {"total": 0}

    all_records = []
    for csv_path in csv_files:
        try:
            round_num = int(csv_path.stem.split("_round_")[-1])
        except ValueError:
            continue
        direction = round_directions.get(round_num, "none")
        winner = round_winners.get(round_num, "")
        records = process_round(
            csv_path, map_name, direction, winner, output_dir,
            bomb_events_path if bomb_events_path.exists() else None,
            sample_interval, last_seconds, img_format, map_config_path
        )
        all_records.extend(records)

    # 保存该比赛-地图组合的metadata
    # 直接覆盖写入，避免重复追加
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_info": {
                "match_name": match_name,
                "demo_name": demo_name,
                "map_name": map_name,
                "sample_interval": sample_interval,
                "freeze_time": 11.0,
                "total_samples": len(all_records),
            },
            "samples": all_records,
        }, f, indent=2, ensure_ascii=False)

    logger.success(f"数据集生成完成: {len(all_records)} 张图 → {output_dir}")
    return {
        "total": len(all_records),
        "match_name": match_name,
        "map_name": map_name,
        "output_dir": str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="批量数据集生成流水线")
    parser.add_argument("--parsed-dir", required=True, help="已解析的demo目录")
    parser.add_argument("--output", required=True, help="数据集输出根目录 (如 data/dataset_by_match)")
    parser.add_argument("--sample-interval", type=float, default=1.0,
                        help="采样间隔(秒)，默认1秒一帧")
    parser.add_argument("--last-seconds", type=int, default=30,
                        help="截取回合最后N秒")
    parser.add_argument("--img-format", default="png", help="图片格式")
    parser.add_argument("--map-config", default="map_overview.json",
                        help="地图配置文件路径")
    args = parser.parse_args()

    result = generate_dataset(
        args.parsed_dir,
        args.output,
        args.sample_interval,
        args.last_seconds,
        args.img_format,
        args.map_config,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
