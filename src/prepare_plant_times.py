#!/usr/bin/env python3
"""
预处理：提取所有 demo 每个 round 的 plant 时间。
输出 data/plant_times.json，供 dataset.py 使用。
"""
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PARSED_DIR = PROJECT_ROOT / "data/parsed"
OUTPUT_PATH = PROJECT_ROOT / "data/plant_times.json"


def main():
    plant_times = {}
    
    for parsed_dir in sorted(PARSED_DIR.iterdir()):
        if not parsed_dir.is_dir():
            continue
        
        demo_name = parsed_dir.name
        bomb_path = parsed_dir / "bomb_events.csv"
        summary_path = parsed_dir / "summary.json"
        
        if not bomb_path.exists() or not summary_path.exists():
            continue
        
        # 读取 summary
        with open(summary_path) as f:
            summary = json.load(f)
        
        # 读取 bomb events
        bomb_df = pd.read_csv(str(bomb_path))
        
        # round_num -> {plant_tick, round_start, direction}
        round_info = {}
        for r in summary.get("rounds", []):
            rnum = r["round_num"]
            round_info[rnum] = {
                "direction": r.get("direction", "none"),
                "winner": r.get("winner", ""),
            }
        
        # 找 plant 事件
        plants = bomb_df[bomb_df["event"] == "plant"]
        for _, row in plants.iterrows():
            rnum = int(row["round_num"])
            tick = int(row["tick"])
            if rnum in round_info:
                round_info[rnum]["plant_tick"] = tick
        
        # 找每个 round 的 CSV 文件获取 round_start
        csv_dir = parsed_dir / "csv"
        for csv_file in sorted(csv_dir.glob("*_round_*.csv")):
            try:
                rnum = int(csv_file.stem.split("_round_")[-1])
            except ValueError:
                continue
            
            if rnum not in round_info:
                continue
            
            df = pd.read_csv(str(csv_file))
            if len(df) == 0:
                continue
            
            round_start = float(df["frame_time"].min())
            tick_rate = 128  # awpy 默认 tick rate
            
            info = round_info[rnum]
            
            # 计算 plant 相对时间（秒）
            if "plant_tick" in info:
                plant_time = info["plant_tick"] / tick_rate
                plant_rel = round(plant_time - round_start, 2)
                info["plant_relative_time"] = plant_rel
            else:
                info["plant_relative_time"] = None
            
            info["round_duration"] = round(float(df["frame_time"].max()) - round_start, 2)
        
        plant_times[demo_name] = round_info
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(plant_times, f, indent=2)
    
    # 统计
    total_rounds = sum(len(v) for v in plant_times.values())
    planted_rounds = sum(
        1 for demo in plant_times.values() 
        for r in demo.values() 
        if r.get("plant_relative_time") is not None
    )
    print(f"Total demos: {len(plant_times)}")
    print(f"Total rounds: {total_rounds}")
    print(f"Planted rounds: {planted_rounds}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
