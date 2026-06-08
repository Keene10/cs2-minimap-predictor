#!/usr/bin/env python3
"""批量解析 data/demos/ 下所有未解析的 .dem 文件。"""
import sys
from pathlib import Path
from parse_demo import parse_demo_to_csv
from loguru import logger

# 项目根目录 = src/ 的父目录
PROJECT_ROOT = Path(__file__).parent.parent

def main():
    demos_dir = PROJECT_ROOT / "data/demos"
    parsed_base = PROJECT_ROOT / "data/parsed"
    
    demos = sorted(demos_dir.glob("*.dem"))
    total = len(demos)
    parsed_count = 0
    skipped = 0
    failed = []
    
    for i, demo_path in enumerate(demos, 1):
        demo_name = demo_path.stem
        output_dir = parsed_base / demo_name
        
        # 检查是否已解析（有 summary.json 就算完成）
        # parse_demo_to_csv 会在 parsed_base 下自动创建 demo_name/ 子目录
        if (output_dir / "summary.json").exists():
            skipped += 1
            continue
        
        logger.info(f"[{i}/{total}] 解析 {demo_name}...")
        try:
            parse_demo_to_csv(str(demo_path), str(parsed_base))
            parsed_count += 1
            logger.success(f"[{i}/{total}] OK {demo_name}")
        except Exception as e:
            logger.error(f"[{i}/{total}] FAIL {demo_name}: {e}")
            failed.append((demo_name, str(e)))
    
    print(f"\n{'='*60}")
    print(f"解析完成: 成功 {parsed_count}, 跳过 {skipped}, 失败 {len(failed)}")
    if failed:
        print("失败列表:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
