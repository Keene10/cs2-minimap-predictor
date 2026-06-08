# CS2Predictor: Predicting Round Outcomes and Bomb Plant Locations via Temporal Minimap Analysis

Predicting tactical outcomes in **Counter-Strike 2** from the observer's minimap perspective using causal temporal modeling.

---

## Overview

This project addresses the problem of predicting round outcomes and bomb plant locations in CS2 using only minimap screenshots — the same information available to spectators and commentators. A causal temporal model based on unidirectional GRUs is proposed that strictly adheres to the real-time constraint: predictions may only attend to past frames.

### Key Results

| Task | Model | Accuracy |
|------|-------|----------|
| Direction (A/B/None) | Single-frame | 69.45% |
| Direction (A/B/None) | **Sequence** | **96.18%** |
| Winner (CT/T) | Single-frame | 75.09% |
| Winner (CT/T) | **Sequence** | **88.00%** |

---

## Repository Structure

```
.
├── src/                    # Core source code
│   ├── parse_demo.py       # Demo parser (.dem → structured data)
│   ├── render_minimap.py   # Minimap rendering engine
│   ├── coordinate_transform.py
│   ├── generate_dataset.py
│   ├── dataset.py
│   ├── models.py           # Swin-Tiny + GRU models
│   ├── train.py
│   ├── infer.py            # Single-round inference
│   ├── infer_video.py      # Video inference with visualization
│   ├── demo_ui.py          # Gradio demo interface
│   └── demo_to_minimap_frames.py
│
├── tests/                  # Unit tests
├── requirements.txt
├── bomb_sites.json
└── map_overview.json
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
awpy get maps
```

### 2. Parse a Demo

```bash
python src/parse_demo.py \
    --input match.dem \
    --output data/parsed_csv \
    --parse-rate 128
```

### 3. Render Minimap Frames

```bash
python src/render_minimap.py \
    --csv data/parsed_csv/match_round_01.csv \
    --map de_dust2 \
    --timestamps "0,10,20,30" \
    --output data/frames
```

### 4. Generate Dataset

```bash
python src/generate_all_datasets.py \
    --csv-dir data/parsed_csv \
    --output data/dataset_by_match \
    --fps 5
```

### 5. Train

```bash
python src/train.py \
    --task direction \
    --mode sequence \
    --dataset data/dataset_by_match \
    --output outputs/direction_sequence
```

### 6. Inference on Video

```bash
python src/infer_video_sequence.py \
    --video demo.mp4 \
    --checkpoint outputs/direction_sequence/best_model.pth \
    --output annotated.mp4
```

---

## Dataset

- **33 professional matches** from tier-1 tournaments
- **1,726 rounds** across 6 competitive maps
- **33,620 pre-plant frames** after truncation
- Anti-leakage design: match-level train/val split + pre-plant frame truncation

---

## Key Design Choices

- **Match-level splits**: Prevents memorizing team-specific tactics
- **Pre-plant truncation**: Model must infer intent, not observe planted bomb
- **Causal GRU**: Unidirectional temporal modeling for real-time prediction
