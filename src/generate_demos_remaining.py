"""Generate remaining demo videos (T wins)."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEMOS = [
    ("iem-rio-2026-natus-vincere-vs-aurora", "de_inferno", 4, "T"),
    ("blast-open-rotterdam-2026-falcons-vs-furia", "de_mirage", 7, "T"),
]

def extract_round_frames(match_name, map_name, round_num, dataset_base):
    map_dir = Path(dataset_base) / match_name / map_name
    meta_path = map_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    round_frames = [s for s in meta.get("samples", []) if s.get("round_number") == round_num]
    round_frames.sort(key=lambda s: s.get("frame_time", 0))
    temp_dir = tempfile.mkdtemp(prefix=f"demo_{match_name}_{map_name}_r{round_num:02d}_")
    temp_path = Path(temp_dir)
    for i, sample in enumerate(round_frames):
        img_rel = sample["image_path"]
        src = map_dir / img_rel
        if not src.exists():
            print(f"  WARNING: missing {src}")
            continue
        dst = temp_path / f"{i:04d}_{src.name}"
        shutil.copy2(src, dst)
    print(f"  Extracted {len(list(temp_path.glob('*.png')))} frames to {temp_dir}")
    return temp_dir

def run_inference(frame_dir, output_path, model_dir, sample_fps=2, output_fps=2):
    cmd = [
        sys.executable,
        "src/infer_video_sequence.py",
        "--input", str(frame_dir),
        "--output", str(output_path),
        "--model-dir", str(model_dir),
        "--sample-fps", str(sample_fps),
        "--output-fps", str(output_fps),
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-base", type=str, default="data/dataset_by_match")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--output-dir", type=str, default="demo_videos")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for match_name, map_name, round_num, true_label in DEMOS:
        print(f"\n{'='*60}")
        print(f"Processing: {match_name} / {map_name} / round_{round_num:02d} (true={true_label})")
        print(f"{'='*60}")
        try:
            frame_dir = extract_round_frames(match_name, map_name, round_num, args.dataset_base)
            out_name = f"{match_name}_{map_name}_r{round_num:02d}_{true_label}win_seq.mp4"
            output_path = output_dir / out_name
            run_inference(frame_dir, output_path, args.model_dir)
            shutil.rmtree(frame_dir, ignore_errors=True)
            print(f"  Done: {output_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
