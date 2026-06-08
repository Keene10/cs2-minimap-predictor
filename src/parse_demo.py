"""
Demo解析主脚本

基于awpy库解析CS2 demo文件，提取每回合的玩家状态数据和炸弹事件。
每个demo的结果单独存放在以demo名称命名的子文件夹中。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import polars as pl
from awpy import Demo
from loguru import logger


def classify_round(bomb_site: str) -> str:
    """根据bomb_site字段分类回合label。"""
    if bomb_site == "bombsite_a":
        return "A"
    elif bomb_site == "bombsite_b":
        return "B"
    else:
        return "defense"


def parse_demo_to_csv(input_path: str, output_base_dir: str, parse_rate: int = 128) -> dict:
    input_path = Path(input_path)
    output_base_dir = Path(output_base_dir)

    demo_name = input_path.stem
    demo_dir = output_base_dir / demo_name
    csv_dir = demo_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始解析: {input_path} → {demo_dir}")
    demo = Demo(str(input_path), verbose=False)

    demo.parse(
        player_props=[
            "CCSPlayerPawn.m_angEyeAngles",
            "CCSPlayerPawn.m_ArmorValue",
            "CCSPlayerController.m_bPawnIsAlive",
        ]
    )

    map_name = demo.header.get("map_name", "unknown")
    match_id = demo.header.get("client_name", demo_name)
    tick_rate = demo.header.get("tickRate", 128)
    logger.info(f"地图: {map_name}, 回合数: {len(demo.rounds)}, ticks: {len(demo.ticks)}")

    if len(demo.ticks) == 0:
        logger.warning("未解析到有效ticks数据")
        return {"status": "empty", "map": map_name, "rounds": 0, "demo_dir": str(demo_dir)}

    # --- 处理炸弹事件 ---
    bomb_df = demo.bomb.clone()
    if len(bomb_df) > 0:
        bomb_df = bomb_df.sort(["round_num", "tick"])
        # 保存炸弹事件CSV，供渲染使用
        bomb_events_path = demo_dir / "bomb_events.csv"
        bomb_df.write_csv(str(bomb_events_path))
        logger.info(f"炸弹事件已保存: {bomb_events_path} ({len(bomb_df)} 条)")
    else:
        logger.warning("未解析到炸弹事件")

    # --- 处理ticks数据 ---
    ticks = demo.ticks.clone()
    ticks = ticks.rename({
        "CCSPlayerPawn.m_angEyeAngles": "eye_angles",
        "CCSPlayerPawn.m_ArmorValue": "armor",
        "CCSPlayerController.m_bPawnIsAlive": "is_alive",
    })

    ticks = ticks.with_columns(pl.col("eye_angles").list.get(1).alias("yaw"))
    ticks = ticks.with_columns(pl.col("side").str.to_uppercase().alias("team_side"))
    ticks = ticks.with_columns((pl.col("tick") / tick_rate).alias("frame_time"))

    # 合并炸弹持有状态
    if len(bomb_df) > 0:
        bomb_states = []
        for round_num in bomb_df["round_num"].unique().to_list():
            round_bomb_events = bomb_df.filter(pl.col("round_num") == round_num).sort("tick")
            round_ticks = ticks.filter(pl.col("round_num") == round_num)
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
                carrier_df = pl.DataFrame({
                    "tick": list(tick_to_carrier.keys()),
                    "bomb_carrier": list(tick_to_carrier.values()),
                    "round_num": [round_num] * len(tick_to_carrier)
                })
                bomb_states.append(carrier_df)

        if bomb_states:
            all_bomb_states = pl.concat(bomb_states)
            ticks = ticks.join(all_bomb_states, on=["round_num", "tick"], how="left")
            ticks = ticks.with_columns(pl.col("bomb_carrier").fill_null("").alias("bomb_carrier"))
            ticks = ticks.with_columns((pl.col("name") == pl.col("bomb_carrier")).alias("has_bomb"))
        else:
            ticks = ticks.with_columns(pl.lit(False).alias("has_bomb"))
    else:
        ticks = ticks.with_columns(pl.lit(False).alias("has_bomb"))

    round_nums = ticks["round_num"].unique().sort().to_list()
    saved_files = []
    rounds_info = []
    rounds_df = demo.rounds.to_pandas()

    for rnum in round_nums:
        round_ticks = ticks.filter(pl.col("round_num") == rnum)
        if len(round_ticks) == 0:
            continue

        output_df = round_ticks.select([
            pl.col("round_num").alias("round_number"),
            pl.col("frame_time"),
            pl.col("name").alias("player_name"),
            pl.col("steamid").alias("player_id"),
            pl.col("team_side"),
            pl.col("X").alias("pos_x"),
            pl.col("Y").alias("pos_y"),
            pl.col("Z").alias("pos_z"),
            pl.col("yaw").alias("yaw_angle"),
            pl.col("is_alive"),
            pl.col("has_bomb"),
            pl.col("health"),
            pl.col("armor"),
        ]).sort(["frame_time", "player_name"])

        unique_ticks = output_df["frame_time"].unique().sort().to_list()
        sampled_frames = unique_ticks[::parse_rate] if len(unique_ticks) > parse_rate else unique_ticks
        output_df = output_df.filter(pl.col("frame_time").is_in(sampled_frames))

        out_file = csv_dir / f"{demo_name}_round_{rnum:02d}.csv"
        output_df.write_csv(str(out_file))
        saved_files.append(str(out_file))

        round_meta = rounds_df[rounds_df["round_num"] == rnum]
        if not round_meta.empty:
            bomb_site = str(round_meta.iloc[0].get("bomb_site", "not_planted"))
            winner = str(round_meta.iloc[0].get("winner", "unknown"))
            reason = str(round_meta.iloc[0].get("reason", "unknown"))
        else:
            bomb_site = "not_planted"
            winner = "unknown"
            reason = "unknown"

        rounds_info.append({
            "round_num": rnum,
            "bomb_site": bomb_site,
            "winner": winner,
            "reason": reason,
            "label": classify_round(bomb_site),
            "csv_file": str(out_file),
            "num_frames": len(sampled_frames),
        })

        logger.info(f"回合 {rnum} [{classify_round(bomb_site)}] → {out_file.name} ({len(output_df)} 行)")

    summary = {
        "demo_file": str(input_path),
        "demo_name": demo_name,
        "demo_dir": str(demo_dir),
        "map_name": map_name,
        "match_id": match_id,
        "tick_rate": tick_rate,
        "total_rounds": len(round_nums),
        "total_ticks_original": len(demo.ticks),
        "parse_rate": parse_rate,
        "csv_files": saved_files,
        "rounds": rounds_info,
        "bomb_events": len(bomb_df) if len(bomb_df) > 0 else 0,
    }
    summary_path = demo_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.success(f"解析完成: {len(saved_files)} 个回合 → {demo_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="CS2 Demo解析脚本")
    parser.add_argument("--input", required=True, help="输入.dem文件路径")
    parser.add_argument("--output", required=True, help="输出根目录")
    parser.add_argument("--parse-rate", type=int, default=128, help="tick采样间隔")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        sys.exit(1)

    result = parse_demo_to_csv(args.input, args.output, args.parse_rate)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
