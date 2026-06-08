"""
CS2 Video Prediction Pipeline — Sequence Model
Uses causal temporal model with sliding window (up to 20 past frames).
Slower but much more accurate than single-frame.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from models import CS2SequencePredictor

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = 224
MAX_FRAMES = 20
DIRECTION_NAMES = ["A点", "B点", "无"]
DIRECTION_COLORS = {"A点": (0, 200, 0), "B点": (200, 100, 0), "无": (128, 128, 128)}
WINNER_NAMES = ["CT", "T"]
WINNER_COLORS = {"CT": (255, 100, 0), "T": (0, 50, 255)}


def get_transform():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_seq_model(checkpoint_path, num_classes):
    model = CS2SequencePredictor(
        num_classes=num_classes,
        pretrained=False,
        img_size=IMG_SIZE,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3,
        model_name="swin_tiny_patch4_window7_224",
    )
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def build_sequence(pil_frames, transform):
    """
    Build a variable-length sequence tensor from a list of PIL images.
    No padding: real frames only (1~20).
    Returns: seq_tensor [1, n, 3, H, W], valid_mask [1, n]
    """
    tensors = [transform(pil) for pil in pil_frames]
    seq = torch.stack(tensors).unsqueeze(0).to(DEVICE)  # [1, n, 3, H, W]
    n = len(tensors)
    valid_mask = torch.ones(1, n, dtype=torch.bool, device=DEVICE)
    return seq, valid_mask


def infer_sequence(seq_tensor, valid_mask, direction_model, winner_model):
    with torch.no_grad():
        dir_logits = direction_model(seq_tensor, valid_mask=valid_mask)
        dir_probs = F.softmax(dir_logits, dim=1).cpu().numpy()[0]
        dir_pred = int(dir_logits.argmax(dim=1).item())
        dir_conf = float(dir_probs[dir_pred])

        win_logits = winner_model(seq_tensor, valid_mask=valid_mask)
        win_probs = F.softmax(win_logits, dim=1).cpu().numpy()[0]
        win_pred = int(win_logits.argmax(dim=1).item())
        ct_prob = float(win_probs[0])
        t_prob = float(win_probs[1])

    return {
        "direction": DIRECTION_NAMES[dir_pred],
        "direction_conf": dir_conf,
        "direction_probs": dir_probs.tolist(),
        "winner": WINNER_NAMES[win_pred],
        "ct_prob": ct_prob,
        "t_prob": t_prob,
    }


def draw_prediction(frame_bgr, result, frame_idx, fps, seq_len):
    h, w = frame_bgr.shape[:2]
    bar_h = 60
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    alpha = 0.6
    frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

    ct_width = int(w * result["ct_prob"])
    cv2.rectangle(frame_bgr, (0, 0), (ct_width, bar_h), WINNER_COLORS["CT"], -1)
    cv2.rectangle(frame_bgr, (ct_width, 0), (w, bar_h), WINNER_COLORS["T"], -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    ct_text = f"CT {result['ct_prob']*100:.1f}%"
    t_text = f"T {result['t_prob']*100:.1f}%"
    (tw_ct, th_ct), _ = cv2.getTextSize(ct_text, font, font_scale, thickness)
    (tw_t, th_t), _ = cv2.getTextSize(t_text, font, font_scale, thickness)
    cv2.putText(frame_bgr, ct_text, (10, bar_h // 2 + th_ct // 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame_bgr, t_text, (w - tw_t - 10, bar_h // 2 + th_t // 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Side panel with PIL for Chinese
    panel_x = 10
    panel_y = bar_h + 20
    line_h = 30

    texts = [
        f"帧: #{frame_idx} 时间: {frame_idx/fps:.1f}s 历史: {seq_len}帧",
        f"可能获胜: {result['winner']} (置信度: {max(result['ct_prob'], result['t_prob'])*100:.1f}%)",
        f"可能进攻: {result['direction']} (置信度: {result['direction_conf']*100:.1f}%)",
    ]

    pil_frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_frame)
    try:
        font_pil = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 22)
    except:
        try:
            font_pil = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 22)
        except:
            font_pil = ImageFont.load_default()

    for i, text in enumerate(texts):
        y = panel_y + i * line_h
        draw.text((panel_x + 1, y + 1), text, font=font_pil, fill=(0, 0, 0))
        draw.text((panel_x, y), text, font=font_pil, fill=(255, 255, 255))

    frame_bgr = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

    # Direction circle
    dir_color = DIRECTION_COLORS[result["direction"]]
    center = (w - 40, bar_h + 40)
    radius = 25
    cv2.circle(frame_bgr, center, radius, dir_color, -1)
    cv2.circle(frame_bgr, center, radius, (255, 255, 255), 2)
    label = result["direction"].replace("点", "")
    if label == "无":
        label = "×"
    (tl, tt), _ = cv2.getTextSize(label, font, 0.5, 1)
    cv2.putText(frame_bgr, label, (center[0] - tl // 2, center[1] + tt // 2),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame_bgr


def process_video_sequence(input_path, output_path, direction_model, winner_model,
                           sample_fps=2, output_fps=2):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_interval = max(1, int(input_fps / sample_fps))

    print(f"Input: {input_path}")
    print(f"  Resolution: {width}x{height}, FPS: {input_fps:.1f}, Total: {total_frames}")
    print(f"  Sampling every {sample_interval} frames → {sample_fps} FPS")
    print(f"  Output: {output_path} at {output_fps} FPS")
    print(f"  Device: {DEVICE} (sequence model, max_history={MAX_FRAMES})")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

    transform = get_transform()

    # Step 1: Collect all sampled PIL frames
    print("\nStep 1: Loading frames...")
    all_pil_frames = []
    all_bgr_frames = []
    frame_idx = 0
    sampled_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_pil_frames.append(Image.fromarray(rgb))
            all_bgr_frames.append(frame)
            sampled_idx += 1
        frame_idx += 1

    cap.release()
    total_sampled = len(all_pil_frames)
    print(f"  Total sampled frames: {total_sampled}")

    # Step 2: Sequence inference for each frame
    print("\nStep 2: Sequence inference (this may take a while)...")
    results_log = []
    start_time = time.time()

    for i in range(total_sampled):
        # Build sliding window: up to MAX_FRAMES past frames
        window_start = max(0, i - MAX_FRAMES + 1)
        window_pil = all_pil_frames[window_start:i + 1]
        seq_len = len(window_pil)

        seq_tensor, valid_mask = build_sequence(window_pil, transform)
        result = infer_sequence(seq_tensor, valid_mask, direction_model, winner_model)
        result["frame_idx"] = i
        result["timestamp"] = i / sample_fps
        result["history_frames"] = seq_len
        results_log.append(result)

        # Render
        rendered = draw_prediction(all_bgr_frames[i], result, i, sample_fps, seq_len)
        out.write(rendered)

        if i % 5 == 0 or i == total_sampled - 1:
            elapsed = time.time() - start_time
            fps_inf = (i + 1) / elapsed
            eta = (total_sampled - i - 1) / fps_inf if fps_inf > 0 else 0
            print(f"  [{i+1}/{total_sampled}] dir={result['direction']} win={result['winner']} "
                  f"| {fps_inf:.1f} fps | ETA {eta:.0f}s")

    out.release()

    # Save JSON
    json_path = str(output_path).replace(".mp4", "_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - start_time
    print(f"\nDone! Total time: {total_elapsed:.1f}s ({total_sampled/total_elapsed:.1f} seq/sec)")
    print(f"Output video: {output_path}")
    print(f"Results JSON: {json_path}")


def frames_to_video(frame_dir, output_path, fps=2):
    """Convert a directory of PNG frames to MP4 video."""
    frame_paths = sorted(Path(frame_dir).glob("*.png"))
    if not frame_paths:
        raise ValueError(f"No frames found in {frame_dir}")

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is not None:
            out.write(img)
    out.release()
    print(f"Created {output_path} from {len(frame_paths)} frames at {fps} fps")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="CS2 Sequence Video Prediction")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input video path OR directory of frames")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output video path")
    parser.add_argument("--model-dir", type=str, default="../models",
                        help="Directory containing model checkpoints")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--output-fps", type=float, default=2.0)
    args = parser.parse_args()

    input_path = Path(args.input)

    # If input is a directory, convert frames to video first
    if input_path.is_dir():
        temp_video = input_path.parent / f"{input_path.name}_temp.mp4"
        frames_to_video(input_path, temp_video, fps=args.sample_fps)
        input_path = temp_video

    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_seq_predicted.mp4"
    else:
        output_path = Path(args.output)

    model_dir = Path(args.model_dir)
    dir_ckpt = model_dir / "direction_sequence_best.pth"
    win_ckpt = model_dir / "winner_sequence_best.pth"

    print("Loading sequence models...")
    direction_model = load_seq_model(dir_ckpt, num_classes=3)
    winner_model = load_seq_model(win_ckpt, num_classes=2)
    print(f"  Direction: {dir_ckpt}")
    print(f"  Winner: {win_ckpt}")

    process_video_sequence(input_path, output_path, direction_model, winner_model,
                           sample_fps=args.sample_fps, output_fps=args.output_fps)

    # Clean up temp video if created from frames
    if str(input_path).endswith("_temp.mp4"):
        input_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
