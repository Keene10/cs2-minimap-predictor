#!/usr/bin/env python3
"""批量为 data/parsed/ 下所有已解析的 demo 生成数据集。"""
import sys
from pathlib import Path
from generate_dataset import generate_dataset
from loguru import logger

# 项目根目录 = src/ 的父目录
PROJECT_ROOT = Path(__file__).parent.parent

def main():
    parsed_base = PROJECT_ROOT / "data/parsed"
    output_base = PROJECT_ROOT / "data/dataset_by_match"
    
    parsed_dirs = sorted([d for d in parsed_base.iterdir() if d.is_dir()])
    total = len(parsed_dirs)
    ok = 0
    failed = []
    total_samples = 0
    
    for i, parsed_dir in enumerate(parsed_dirs, 1):
        logger.info(f"[{i}/{total}] 生成数据集: {parsed_dir.name}...")
        try:
            result = generate_dataset(
                str(parsed_dir),
                str(output_base),
                sample_interval=1.0,
                last_seconds=30,
            )
            n = result.get("total", 0)
            total_samples += n
            ok += 1
            logger.success(f"[{i}/{total}] OK {parsed_dir.name} → {n} 张图")
        except Exception as e:
            logger.error(f"[{i}/{total}] FAIL {parsed_dir.name}: {e}")
            failed.append((parsed_dir.name, str(e)))
    
    print(f"\n{'='*60}")
    print(f"数据集生成完成: 成功 {ok}/{total}, 失败 {len(failed)}")
    print(f"总样本数: {total_samples} 张")
    if failed:
        print("失败列表:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
