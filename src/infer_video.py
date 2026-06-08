"""
CS2 Video Prediction Pipeline
Input: minimap video (e.g., from demo renderer)
Output: annotated video with win probability bars and direction predictions
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from models import CS2Predictor

# ---------- Config ----------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = 224
DIRECTION_NAMES = ["A点", "B点", "无"]
DIRECTION_COLORS = {"A点": (0, 200, 0), "B点": (200, 100, 0), "无": (128, 128, 128)}
WINNER_NAMES = ["CT", "T"]
WINNER_COLORS = {"CT": (255, 100, 0), "T": (0, 50, 255)}  # BGR: CT=blue, T=red


def get_transform():
    """Same transform as training (val mode, no augmentation)."""
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint_path, num_classes):
    """Load a CS2Predictor from checkpoint."""
    model = CS2Predictor(
        num_classes=num_classes,
        pretrained=False,
        img_size=IMG_SIZE,
        dropout=0.3,
        model_name="swin_tiny_patch4_window7_224",
    )
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def infer_frame(pil_img, direction_model, winner_model, transform):
    """Run inference on a single PIL image."""
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Direction
        dir_logits = direction_model(tensor)
        dir_probs = F.softmax(dir_logits, dim=1).cpu().numpy()[0]
        dir_pred = int(dir_logits.argmax(dim=1).item())
        dir_conf = float(dir_probs[dir_pred])

        # Winner
        win_logits = winner_model(tensor)
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


def draw_prediction(frame_bgr, result, frame_idx, fps):
    """Render prediction results on a BGR frame."""
    h, w = frame_bgr.shape[:2]
    bar_h = 60  # height of top probability bar

    # Create overlay
    overlay = frame_bgr.copy()

    # ---- Top bar background (semi-transparent black) ----
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    alpha = 0.6
    frame_bgr = cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)

    # ---- Win probability bar ----
    ct_width = int(w * result["ct_prob"])
    t_width = w - ct_width

    # CT bar (left, blue)
    cv2.rectangle(frame_bgr, (0, 0), (ct_width, bar_h), WINNER_COLORS["CT"], -1)
    # T bar (right, red)
    cv2.rectangle(frame_bgr, (ct_width, 0), (w, bar_h), WINNER_COLORS["T"], -1)

    # Bar text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    ct_text = f"CT {result['ct_prob']*100:.1f}%"
    t_text = f"T {result['t_prob']*100:.1f}%"
    (tw_ct, th_ct), _ = cv2.getTextSize(ct_text, font, font_scale, thickness)
    (tw_t, th_t), _ = cv2.getTextSize(t_text, font, font_scale, thickness)

    # CT text (left side of bar)
    cv2.putText(frame_bgr, ct_text, (10, bar_h // 2 + th_ct // 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    # T text (right side of bar)
    cv2.putText(frame_bgr, t_text, (w - tw_t - 10, bar_h // 2 + th_t // 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # ---- Side panel: direction & winner info (PIL for Chinese) ----
    panel_x = 10
    panel_y = bar_h + 20
    line_h = 30

    texts = [
        f"帧: #{frame_idx}  时间: {frame_idx/fps:.1f}s",
        f"可能获胜: {result['winner']}  (置信度: {max(result['ct_prob'], result['t_prob'])*100:.1f}%)",
        f"可能进攻: {result['direction']}  (置信度: {result['direction_conf']*100:.1f}%)",
    ]

    # Convert to PIL for Chinese text rendering
    pil_frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_frame)
    font_pil = None
    for font_path in [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/PingFang.ttc",
    ]:
        try:
            font_pil = ImageFont.truetype(font_path, 22)
            break
        except:
            continue
    if font_pil is None:
        font_pil = ImageFont.load_default()

    for i, text in enumerate(texts):
        y = panel_y + i * line_h
        # Shadow
        draw.text((panel_x + 1, y + 1), text, font=font_pil, fill=(0, 0, 0))
        # Text
        draw.text((panel_x, y), text, font=font_pil, fill=(255, 255, 255))

    frame_bgr = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

    # ---- Direction indicator circle (top-right) ----
    dir_color = DIRECTION_COLORS[result["direction"]]
    center = (w - 40, bar_h + 40)
    radius = 25
    cv2.circle(frame_bgr, center, radius, dir_color, -1)
    cv2.circle(frame_bgr, center, radius, (255, 255, 255), 2)
    label = result["direction"].replace("点", "")
    (tl, tt), _ = cv2.getTextSize(label, font, 0.5, 1)
    cv2.putText(frame_bgr, label, (center[0] - tl // 2, center[1] + tt // 2),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame_bgr


def process_video(input_path, output_path, direction_model, winner_model,
                  sample_fps=2, output_fps=2):
    """
    Main pipeline:
      1. Read video, sample frames at `sample_fps`
      2. Run inference on each sampled frame
      3. Render predictions
      4. Write output video at `output_fps`
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame interval for sampling
    sample_interval = int(input_fps / sample_fps)
    if sample_interval < 1:
        sample_interval = 1

    print(f"Input: {input_path}")
    print(f"  Resolution: {width}x{height}, FPS: {input_fps:.1f}, Total frames: {total_frames}")
    print(f"  Sampling every {sample_interval} frames → {sample_fps} FPS")
    print(f"  Output: {output_path} at {output_fps} FPS")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))

    transform = get_transform()
    frame_idx = 0
    sampled_idx = 0
    results_log = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            # Convert BGR to RGB PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # Inference
            result = infer_frame(pil_img, direction_model, winner_model, transform)
            result["frame_idx"] = sampled_idx
            result["timestamp"] = sampled_idx / sample_fps
            results_log.append(result)

            # Render
            rendered = draw_prediction(frame, result, sampled_idx, sample_fps)
            out.write(rendered)

            if sampled_idx % 10 == 0:
                print(f"  Processed frame {sampled_idx} | "
                      f"Direction: {result['direction']} | "
                      f"Winner: T={result['t_prob']:.2f} CT={result['ct_prob']:.2f}")

            sampled_idx += 1

        frame_idx += 1

    cap.release()
    out.release()

    # Save results JSON
    json_path = str(output_path).replace(".mp4", "_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Output video: {output_path}")
    print(f"Results JSON: {json_path}")
    print(f"Total sampled frames: {sampled_idx}")


def main():
    parser = argparse.ArgumentParser(description="CS2 Minimap Video Prediction")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input minimap video")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to output video (default: input_predicted.mp4)")
    parser.add_argument("--model-dir", type=str, default="../models",
                        help="Directory containing model checkpoints")
    parser.add_argument("--sample-fps", type=float, default=2.0,
                        help="Frame sampling rate from input video")
    parser.add_argument("--output-fps", type=float, default=2.0,
                        help="Output video FPS")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_predicted.mp4"
    else:
        output_path = Path(args.output)

    model_dir = Path(args.model_dir)
    dir_ckpt = model_dir / "direction_single_best.pth"
    win_ckpt = model_dir / "winner_single_best.pth"

    print("Loading models...")
    direction_model = load_model(dir_ckpt, num_classes=3)
    winner_model = load_model(win_ckpt, num_classes=2)
    print(f"  Direction model: {dir_ckpt}")
    print(f"  Winner model: {win_ckpt}")
    print(f"  Device: {DEVICE}")

    process_video(input_path, output_path, direction_model, winner_model,
                  sample_fps=args.sample_fps, output_fps=args.output_fps)


if __name__ == "__main__":
    main()
