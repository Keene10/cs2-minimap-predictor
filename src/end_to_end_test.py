"""
端到端验证脚本

验证完整流程: .dem → CSV → 渲染帧 → 数据集分类
输出验证报告。
"""

import json
import os
import tempfile
from pathlib import Path

from loguru import logger

from parse_demo import parse_demo_to_csv
from render_minimap import MinimapRenderer
from generate_dataset import generate_dataset


def run_e2e(demo_path: str, output_base: str) -> dict:
    """
    运行端到端测试。

    Args:
        demo_path: .dem文件路径
        output_base: 输出根目录

    Returns:
        验证报告字典
    """
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    report = {
        "demo_file": demo_path,
        "stages": {},
        "checks": {},
    }

    # === Stage 1: 解析Demo ===
    logger.info("=== Stage 1: 解析Demo ===")
    csv_dir = output_base / "parsed_csv"
    summary = parse_demo_to_csv(demo_path, str(csv_dir), parse_rate=128)
    report["stages"]["parse"] = {
        "map": summary["map_name"],
        "rounds": summary["total_rounds"],
        "csv_files": len(summary["csv_files"]),
    }

    # === Stage 2: 坐标转换抽查 ===
    logger.info("=== Stage 2: 坐标转换抽查 ===")
    from coordinate_transform import CoordinateTransformer
    transformer = CoordinateTransformer()

    # 读取第一个CSV，抽查几个坐标
    first_csv = Path(summary["csv_files"][0])
    df = __import__("pandas").read_csv(str(first_csv))
    sample = df.iloc[0]
    px, py = transformer.world_to_minimap(sample["pos_x"], sample["pos_y"], summary["map_name"])

    report["checks"]["coordinate_transform"] = {
        "sample_world": (sample["pos_x"], sample["pos_y"]),
        "sample_pixel": (px, py),
        "in_bounds": 0 <= px <= 1024 and 0 <= py <= 1024,
    }

    # === Stage 3: 渲染关键帧 ===
    logger.info("=== Stage 3: 渲染关键帧 ===")
    frames_dir = output_base / "frames"
    renderer = MinimapRenderer(summary["map_name"])

    # 渲染3个关键时刻
    rendered = renderer.render_round(str(first_csv), str(frames_dir), timestamps=[0, 30, 60])
    report["stages"]["render"] = {
        "frames_rendered": len(rendered),
        "frame_files": rendered,
    }

    # 检查图片尺寸
    from PIL import Image
    img = Image.open(rendered[0])
    report["checks"]["image_size"] = {
        "size": img.size,
        "mode": img.mode,
        "correct": img.size == (1024, 1024),
    }

    # === Stage 4: 数据集生成 ===
    logger.info("=== Stage 4: 数据集生成 ===")
    summary_json = csv_dir / f"{Path(demo_path).stem}_summary.json"
    dataset_dir = output_base / "dataset"
    ds_result = generate_dataset(
        str(csv_dir), str(summary_json), str(dataset_dir),
        train_ratio=0.8, fps=5, last_seconds=30
    )
    report["stages"]["dataset"] = ds_result

    # 标签正确性抽查
    with open(dataset_dir / "dataset_metadata.json") as f:
        meta = json.load(f)

    labels = [s["round_label"] for s in meta["samples"]]
    label_dist = {"A": labels.count("A"), "B": labels.count("B"), "defense": labels.count("defense")}
    report["checks"]["label_distribution"] = label_dist

    # === 汇总 ===
    report["overall_pass"] = all([
        report["checks"]["coordinate_transform"]["in_bounds"],
        report["checks"]["image_size"]["correct"],
        report["stages"]["parse"]["rounds"] > 0,
        report["stages"]["render"]["frames_rendered"] == 3,
        report["stages"]["dataset"]["total"] > 0,
    ])

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="端到端验证")
    parser.add_argument("--demo", required=True, help="测试用demo文件")
    parser.add_argument("--output", default="demo_end_to_end/test_run", help="输出目录")
    args = parser.parse_args()

    report = run_e2e(args.demo, args.output)
    print("\n" + "="*60)
    print("端到端验证报告")
    print("="*60)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if report["overall_pass"]:
        print("\n✅ 所有检查项通过！")
    else:
        print("\n❌ 存在失败的检查项，请查看详情。")


if __name__ == "__main__":
    main()
