"""
Generate demo videos from validation-set rounds.
Extracts frames for specific rounds, runs sequence inference, renders videos.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Each demo: (match_name, map_name, round_number, label)
# All from validation set, seed=42, split_ratio=0.7
DEMOS = [
    # CT wins
    ("blast-open-rotterdam-2026-falcons-vs-natus-vincere", "de_inferno", 10, "CT"),
    ("blast-open-rotterdam-2026-falcons-vs-natus-vincere", "de_mirage", 4, "CT"),
    # T wins
    ("iem-rio-2026-natus-vincere-vs-aurora", "de_inferno", 4, "T"),
    ("blast-open-rotterdam-2026-falcons-vs-furia", "de_mirage", 7, "T"),
]


def extract_round_frames(match_name, map_name, round_num, dataset_base):
    """
    Read metadata.json, find all frames for the given round,
    copy them to a temp dir sorted by time, return the temp dir path.
    """
    map_dir = Path(dataset_base) / match_name / map_name
    meta_path = map_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    # Filter frames for this round
    round_frames = []
    for sample in meta.get("samples", []):
        if sample.get("round_number") == round_num:
            round_frames.append(sample)

    if not round_frames:
        raise ValueError(f"No frames found for round {round_num} in {match_name}/{map_name}")

    # Sort by frame_time (or by filename time component)
    round_frames.sort(key=lambda s: s.get("frame_time", 0))

    # Create temp dir
    temp_dir = tempfile.mkdtemp(prefix=f"demo_{match_name}_{map_name}_r{round_num:02d}_")
    temp_path = Path(temp_dir)

    # Copy frames
    for i, sample in enumerate(round_frames):
        img_rel = sample["image_path"]  # e.g. "A/de_inferno_r10_t0580_i1s.png"
        src = map_dir / img_rel
        if not src.exists():
            print(f"  WARNING: missing {src}")
            continue
        # Rename to ensure sorted order: 0000_xxx.png
        dst = temp_path / f"{i:04d}_{src.name}"
        shutil.copy2(src, dst)

    print(f"  Extracted {len(list(temp_path.glob('*.png')))} frames to {temp_dir}")
    return temp_dir


def run_inference(frame_dir, output_path, model_dir, sample_fps=2, output_fps=2):
    """Run infer_video_sequence.py on the frame directory."""
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
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--output-fps", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for match_name, map_name, round_num, true_label in DEMOS:
        print(f"\n{'='*60}")
        print(f"Processing: {match_name} / {map_name} / round_{round_num:02d} (true={true_label})")
        print(f"{'='*60}")

        try:
            frame_dir = extract_round_frames(
                match_name, map_name, round_num, args.dataset_base
            )

            out_name = f"{match_name}_{map_name}_r{round_num:02d}_{true_label}win_seq.mp4"
            output_path = output_dir / out_name

            run_inference(
                frame_dir, output_path,
                args.model_dir, args.sample_fps, args.output_fps
            )

            # Cleanup temp dir
            shutil.rmtree(frame_dir, ignore_errors=True)
            print(f"  Done: {output_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
