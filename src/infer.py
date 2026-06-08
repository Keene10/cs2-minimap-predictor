"""
CS2预测系统推理脚本
支持单帧和序列推理，输出direction和winner预测结果
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Union, Tuple
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import CS2Predictor, CS2SequencePredictor, CS2WinProbPredictor
from dataset import default_transform


# ---------------------------------------------------------------------------
# 标签映射
# ---------------------------------------------------------------------------
DIRECTION_LABELS = {0: "A", 1: "B", 2: "none"}
WINNER_LABELS = {0: "CT", 1: "T"}


# ---------------------------------------------------------------------------
# 推理引擎
# ---------------------------------------------------------------------------
class CS2InferenceEngine:
    """
    加载并运行4个PyTorch模型，提供单帧/序列推理接口。
    """

    def __init__(
        self,
        model_dir: str = "/root/autodl-tmp/cs2_minimap/outputs",
        device: str = None,
        use_sequence: bool = False,
        img_size: int = 224,
        dropout: float = 0.3,
    ):
        """
        Args:
            model_dir: 存放4个模型子目录的根目录
            device: torch device，None则自动选择
            use_sequence: 是否加载序列模型（否则加载单帧模型）
            img_size: 输入图像尺寸
            dropout: 模型dropout率（需与训练时一致）
        """
        self.device = self._auto_device() if device is None else torch.device(device)
        self.use_sequence = use_sequence
        self.img_size = img_size
        self.dropout = dropout
        self.model_dir = Path(model_dir)

        # 推理用的transform（无数据增强）
        self.transform = default_transform(img_size, is_train=False)

        self.direction_model = None
        self.winner_model = None
        self._load_models()

    @staticmethod
    def _auto_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_models(self):
        """加载direction和winner模型到self.device"""
        suffix = "sequence" if self.use_sequence else "single"

        # ---------- direction ----------
        dir_ckpt = self.model_dir / f"direction_{suffix}_swin_tiny_224" / "best_model.pth"
        if dir_ckpt.exists():
            if self.use_sequence:
                self.direction_model = CS2SequencePredictor(
                    num_classes=3, model_name="swin_tiny_patch4_window7_224",
                    img_size=self.img_size, dropout=self.dropout,
                )
            else:
                self.direction_model = CS2Predictor(
                    num_classes=3, model_name="swin_tiny_patch4_window7_224",
                    img_size=self.img_size, dropout=self.dropout,
                )
            ckpt = torch.load(dir_ckpt, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.direction_model.load_state_dict(state, strict=False)
            self.direction_model.to(self.device)
            self.direction_model.eval()
            print(f"[INFO] Loaded direction_{suffix} from {dir_ckpt}")
        else:
            warnings.warn(f"Direction model not found: {dir_ckpt}")

        # ---------- winner ----------
        win_ckpt = self.model_dir / f"winner_{suffix}_swin_tiny_224" / "best_model.pth"
        if win_ckpt.exists():
            if self.use_sequence:
                # winner序列模型复用CS2SequencePredictor(num_classes=2)
                self.winner_model = CS2SequencePredictor(
                    num_classes=2, model_name="swin_tiny_patch4_window7_224",
                    img_size=self.img_size, dropout=self.dropout,
                )
            else:
                self.winner_model = CS2WinProbPredictor(
                    model_name="swin_tiny_patch4_window7_224",
                    img_size=self.img_size, dropout=self.dropout,
                )
            ckpt = torch.load(win_ckpt, map_location=self.device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            self.winner_model.load_state_dict(state, strict=False)
            self.winner_model.to(self.device)
            self.winner_model.eval()
            print(f"[INFO] Loaded winner_{suffix} from {win_ckpt}")
        else:
            warnings.warn(f"Winner model not found: {win_ckpt}")

    # -----------------------------------------------------------------------
    # 单帧推理
    # -----------------------------------------------------------------------
    def infer_single(self, image_path: Union[str, Path]) -> Dict:
        """
        单张图片推理。
        返回:
            {
                "direction": "A/B/none",
                "direction_conf": float,
                "winner": "T/CT",
                "ct_prob": float,
                "t_prob": float,
            }
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)  # [1,3,H,W]

        result = {
            "direction": "none",
            "direction_conf": 0.0,
            "winner": "CT",
            "ct_prob": 0.5,
            "t_prob": 0.5,
        }

        # direction
        if self.direction_model is not None:
            with torch.no_grad():
                logits = self.direction_model(tensor)
                probs = F.softmax(logits, dim=-1)
                conf, pred = probs.max(dim=-1)
            result["direction"] = DIRECTION_LABELS[pred.item()]
            result["direction_conf"] = round(conf.item(), 4)

        # winner
        if self.winner_model is not None:
            with torch.no_grad():
                if self.use_sequence:
                    # 序列winner模型：扩展一维时间轴
                    seq = tensor.unsqueeze(1)  # [1,1,3,H,W]
                    valid_mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
                    logits = self.winner_model(seq, valid_mask=valid_mask)
                    probs = F.softmax(logits, dim=-1)
                else:
                    logits, probs = self.winner_model(tensor)
                ct_p, t_p = probs[0, 0].item(), probs[0, 1].item()
            result["ct_prob"] = round(ct_p, 4)
            result["t_prob"] = round(t_p, 4)
            result["winner"] = "CT" if ct_p > t_p else "T"

        return result

    # -----------------------------------------------------------------------
    # 序列推理（视频）
    # -----------------------------------------------------------------------
    def infer_sequence(
        self,
        video_path: Union[str, Path],
        window_size: int = 20,
        stride: int = 5,
        fps_target: int = None,
    ) -> List[Dict]:
        """
        视频文件推理。
        步骤：拆帧 -> 滑动窗口(窗口20帧, 步长stride) -> 每窗口时序预测。
        因果约束：窗口只包含过去帧（按时间顺序取帧自然满足）。

        Args:
            video_path: 视频文件路径
            window_size: 滑动窗口长度（帧数）
            stride: 滑动步长（帧数）
            fps_target: 若指定，则按此fps抽帧；否则取原视频fps

        返回:
            每窗口一个dict，包含timestamp_sec、direction、winner概率
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # ---------- 1. 拆帧 ----------
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = fps_target or video_fps
        frame_interval = max(1, int(round(video_fps / fps))) if fps_target else 1

        frames = []
        timestamps = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                tensor = self.transform(pil_img)
                frames.append(tensor)
                timestamps.append(frame_idx / video_fps)
            frame_idx += 1
        cap.release()

        if len(frames) == 0:
            raise ValueError("No frames extracted from video")

        print(f"[INFO] Extracted {len(frames)} frames (fps={fps:.1f})")

        # ---------- 2. 滑动窗口推理 ----------
        results = []
        n_frames = len(frames)

        for start in range(0, n_frames, stride):
            end = start + window_size
            window_frames = frames[start:end]
            window_ts = timestamps[start:end]
            actual_len = len(window_frames)

            # 填充：不足window_size时复制最后一帧
            if actual_len < window_size:
                last = window_frames[-1]
                while len(window_frames) < window_size:
                    window_frames.append(last.clone())

            # stack -> [window_size, 3, H, W] -> [1, window_size, 3, H, W]
            seq = torch.stack(window_frames).unsqueeze(0).to(self.device)
            valid_mask = torch.zeros(1, window_size, dtype=torch.bool, device=self.device)
            valid_mask[0, :actual_len] = True

            mid_ts = window_ts[actual_len // 2] if actual_len > 0 else window_ts[0]

            rec = {
                "timestamp_sec": round(mid_ts, 3),
                "start_frame": start,
                "end_frame": min(end, n_frames),
                "direction": "none",
                "direction_conf": 0.0,
                "winner": "CT",
                "ct_prob": 0.5,
                "t_prob": 0.5,
            }

            # direction序列推理
            if self.direction_model is not None:
                with torch.no_grad():
                    logits = self.direction_model(seq, valid_mask=valid_mask)
                    probs = F.softmax(logits, dim=-1)
                    conf, pred = probs.max(dim=-1)
                rec["direction"] = DIRECTION_LABELS[pred.item()]
                rec["direction_conf"] = round(conf.item(), 4)

            # winner序列推理
            if self.winner_model is not None:
                with torch.no_grad():
                    logits = self.winner_model(seq, valid_mask=valid_mask)
                    probs = F.softmax(logits, dim=-1)
                    ct_p, t_p = probs[0, 0].item(), probs[0, 1].item()
                rec["ct_prob"] = round(ct_p, 4)
                rec["t_prob"] = round(t_p, 4)
                rec["winner"] = "CT" if ct_p > t_p else "T"

            results.append(rec)

            if end >= n_frames:
                break

        return results


# ---------------------------------------------------------------------------
# 可视化辅助函数
# ---------------------------------------------------------------------------
def draw_result_on_image(
    img: Image.Image,
    direction: str,
    winner_probs: Tuple[float, float],
    bar_height: int = 60,
    alpha: int = 180,
) -> Image.Image:
    """
    在PIL图像顶端绘制半透明黑色背景 + direction文字 + winner概率条形图。

    Args:
        img: 原始PIL图像 (RGB)
        direction: "A"/"B"/"none"
        winner_probs: (ct_prob, t_prob)，和应为1
        bar_height: 顶部信息条高度
        alpha: 背景透明度 (0-255)

    Returns:
        绘制后的PIL图像
    """
    ct_prob, t_prob = winner_probs
    ct_prob = max(0.0, min(1.0, ct_prob))
    t_prob = max(0.0, min(1.0, t_prob))
    total = ct_prob + t_prob
    if total > 0:
        ct_prob /= total
        t_prob /= total

    W, H = img.size
    overlay = Image.new("RGBA", (W, bar_height), (0, 0, 0, alpha))
    img_rgba = img.convert("RGBA")

    # 把overlay贴到顶部
    img_rgba.paste(overlay, (0, 0), overlay)
    draw = ImageDraw.Draw(img_rgba)

    # 尝试加载字体，失败则使用默认
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

    # 文字内容
    dir_text = f"进攻地点：{direction}点" if direction != "none" else "进攻地点：暂无"
    win_text = f"CT {ct_prob*100:.1f}%  vs  T {t_prob*100:.1f}%"

    margin = 10
    text_y = margin

    # 左侧：direction文字
    draw.text((margin, text_y), dir_text, fill=(255, 255, 255, 255), font=font)

    # 条形图参数
    bar_y = text_y + 22
    bar_w = W - 2 * margin
    bar_h = 16
    bar_x = margin

    # 背景条（深灰）
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(50, 50, 50, 255))

    # CT蓝色左半部分
    ct_w = int(bar_w * ct_prob)
    if ct_w > 0:
        draw.rectangle([bar_x, bar_y, bar_x + ct_w, bar_y + bar_h], fill=(59, 130, 246, 255))

    # T红色右半部分
    t_w = int(bar_w * t_prob)
    if t_w > 0:
        draw.rectangle([bar_x + bar_w - t_w, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(239, 68, 68, 255))

    # 条形图上的百分比文字
    pct_text = f"CT {ct_prob*100:.0f}% | T {t_prob*100:.0f}%"
    bbox = draw.textbbox((0, 0), pct_text, font=font_small)
    tw = bbox[2] - bbox[0]
    draw.text((bar_x + (bar_w - tw) // 2, bar_y - 1), pct_text, fill=(255, 255, 255, 255), font=font_small)

    return img_rgba.convert("RGB")


def plot_sequence_results(results: List[Dict], save_path: str = None):
    """
    为序列推理结果绘制时间轴图表（matplotlib）。
    返回PIL Image。
    """
    if not results:
        return None

    ts = [r["timestamp_sec"] for r in results]
    ct_probs = [r["ct_prob"] for r in results]
    t_probs = [r["t_prob"] for r in results]

    # direction编码：A=0, B=1, none=2
    dir_map = {"A": 0, "B": 1, "none": 2}
    dirs = [dir_map.get(r["direction"], 2) for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # ---- 上图：winner概率曲线 ----
    ax = axes[0]
    ax.fill_between(ts, 0, ct_probs, color="#3b82f6", alpha=0.5, label="CT")
    ax.fill_between(ts, 0, t_probs, color="#ef4444", alpha=0.5, label="T")
    ax.plot(ts, ct_probs, color="#1d4ed8", linewidth=1.5)
    ax.plot(ts, t_probs, color="#b91c1c", linewidth=1.5)
    ax.set_ylabel("Win Probability")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.set_title("Winner Probability over Time")
    ax.grid(True, alpha=0.3)

    # ---- 下图：direction离散点 ----
    ax = axes[1]
    colors = {0: "#22c55e", 1: "#a855f7", 2: "#9ca3af"}  # A=绿, B=紫, none=灰
    labels = {0: "A", 1: "B", 2: "none"}
    for val in [0, 1, 2]:
        xs = [t for t, d in zip(ts, dirs) if d == val]
        ys = [d for d in dirs if d == val]
        ax.scatter(xs, ys, c=colors[val], label=labels[val], s=50, zorder=3)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["A", "B", "none"])
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Direction")
    ax.set_title("Predicted Bomb Site over Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存或转PIL
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved plot to {save_path}")

    # 转PIL（兼容方式：先保存到内存再读取）
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# CLI入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CS2预测系统推理")
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/cs2_minimap/outputs",
                        help="模型根目录")
    parser.add_argument("--mode", choices=["single", "sequence"], default="single",
                        help="推理模式：single(单张图片) / sequence(视频)")
    parser.add_argument("--input", type=str, required=True,
                        help="输入文件路径（图片或视频）")
    parser.add_argument("--use_sequence", action="store_true",
                        help="使用序列模型而非单帧模型")
    parser.add_argument("--window_size", type=int, default=20,
                        help="序列推理窗口大小（帧）")
    parser.add_argument("--stride", type=int, default=5,
                        help="序列推理滑动步长（帧）")
    parser.add_argument("--output", type=str, default="infer_result.json",
                        help="输出JSON路径")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备：cuda / mps / cpu")
    parser.add_argument("--visualize", action="store_true",
                        help="同时保存可视化结果图")
    args = parser.parse_args()

    engine = CS2InferenceEngine(
        model_dir=args.model_dir,
        device=args.device,
        use_sequence=args.use_sequence,
    )

    if args.mode == "single":
        result = engine.infer_single(args.input)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if args.visualize:
            img = Image.open(args.input).convert("RGB")
            vis = draw_result_on_image(img, result["direction"], (result["ct_prob"], result["t_prob"]))
            vis_path = str(Path(args.output).with_suffix(".png"))
            vis.save(vis_path)
            print(f"[INFO] Visualization saved to {vis_path}")

    else:
        results = engine.infer_sequence(
            args.input,
            window_size=args.window_size,
            stride=args.stride,
        )
        print(f"[INFO] Generated {len(results)} window predictions")
        print(json.dumps(results[:3], indent=2, ensure_ascii=False))

        if args.visualize:
            plot_path = str(Path(args.output).with_suffix(".png"))
            plot_sequence_results(results, save_path=plot_path)

    # 保存JSON
    with open(args.output, "w", encoding="utf-8") as f:
        if args.mode == "single":
            json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Result saved to {args.output}")


if __name__ == "__main__":
    main()
