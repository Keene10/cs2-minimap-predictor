"""
CS2预测系统演示UI（Gradio）
支持图片单帧推理和视频序列推理可视化。
"""
import argparse
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from infer import CS2InferenceEngine, draw_result_on_image, plot_sequence_results


class CS2DemoUI:
    def __init__(self, engine: CS2InferenceEngine):
        self.engine = engine

    def predict_image(self, image: np.ndarray):
        """
        Gradio图片推理回调。
        image: numpy array [H,W,3] RGB
        返回: PIL Image（绘制了结果）
        """
        if image is None:
            return None
        pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        # 临时保存以复用infer_single接口（或者直接用engine接口）
        # 但infer_single接收路径，我们直接调用engine的方法
        # 这里需要把numpy转tensor...不如直接临时存一张图
        tmp_path = "/tmp/cs2_demo_tmp.png"
        pil_img.save(tmp_path)
        result = self.engine.infer_single(tmp_path)

        vis = draw_result_on_image(
            pil_img,
            result["direction"],
            (result["ct_prob"], result["t_prob"]),
        )
        return vis

    def predict_video(self, video_path: str):
        """
        Gradio视频推理回调。
        返回: (matplotlib图表PIL, JSON结果字符串)
        """
        if video_path is None or video_path == "":
            return None, "{}"
        results = self.engine.infer_sequence(video_path, window_size=20, stride=5)
        plot_img = plot_sequence_results(results)
        # 简化JSON输出（只保留前5条和后5条，避免过长）
        display = {
            "total_windows": len(results),
            "first_3": results[:3],
            "last_3": results[-3:] if len(results) >= 3 else [],
        }
        return plot_img, display

    def build(self):
        with gr.Blocks(title="CS2 战术预测演示") as demo:
            gr.Markdown(
                """
                # 🎯 CS2 战术预测演示
                上传小地图截图（图片）或回合录像（视频），模型将预测：
                - **进攻地点**：A点 / B点 / 暂无
                - **回合胜率**：CT阵营 vs T阵营
                """
            )

            with gr.Tab("🖼️ 图片推理（单帧）"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            label="上传小地图截图",
                            type="numpy",
                            height=400,
                        )
                        btn_img = gr.Button("运行推理", variant="primary")
                    with gr.Column(scale=1):
                        img_output = gr.Image(
                            label="推理结果",
                            type="pil",
                            height=400,
                        )
                        with gr.Row():
                            dir_text = gr.Textbox(
                                label="预测进攻地点",
                                interactive=False,
                            )
                            win_text = gr.Textbox(
                                label="预测获胜阵营",
                                interactive=False,
                            )

                def _on_image(img):
                    vis = self.predict_image(img)
                    # 同时也返回文字
                    tmp_path = "/tmp/cs2_demo_tmp.png"
                    if img is not None:
                        Image.fromarray(img.astype(np.uint8)).save(tmp_path)
                        res = self.engine.infer_single(tmp_path)
                        dtxt = f"{res['direction']}点" if res['direction'] != 'none' else "暂无"
                        wtxt = f"{res['winner']}阵营 (CT {res['ct_prob']*100:.1f}% / T {res['t_prob']*100:.1f}%)"
                        return vis, dtxt, wtxt
                    return None, "", ""

                btn_img.click(
                    _on_image,
                    inputs=[img_input],
                    outputs=[img_output, dir_text, win_text],
                )

            with gr.Tab("🎬 视频推理（序列）"):
                vid_input = gr.Video(
                    label="上传回合视频",
                    height=360,
                )
                btn_vid = gr.Button("运行推理", variant="primary")

                with gr.Row():
                    vid_plot = gr.Image(
                        label="时间轴图表（Direction + Winner Probability）",
                        type="pil",
                        height=400,
                    )
                    vid_json = gr.JSON(
                        label="推理结果摘要",
                    )

                btn_vid.click(
                    self.predict_video,
                    inputs=[vid_input],
                    outputs=[vid_plot, vid_json],
                )

            gr.Markdown(
                """
                ---
                **模型说明**：基于 Swin-Tiny 骨干网络，支持单帧/序列两种推理模式。
                序列模式采用因果时序建模（只依赖过去帧），滑动窗口大小20帧。
                """
            )

        return demo


def main():
    parser = argparse.ArgumentParser(description="CS2 Demo UI")
    parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/cs2_minimap/outputs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_sequence", action="store_true", help="默认加载序列模型")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成Gradio公网链接")
    args = parser.parse_args()

    print("[INFO] Loading models...")
    engine = CS2InferenceEngine(
        model_dir=args.model_dir,
        device=args.device,
        use_sequence=args.use_sequence,
    )

    demo_ui = CS2DemoUI(engine)
    demo = demo_ui.build()

    print(f"[INFO] Starting Gradio server at http://{args.host}:{args.port}")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
