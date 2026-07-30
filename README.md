# 🚦 Smart Traffic Light — Adaptive Signal Control via Vehicle Detection

**Capstone Project — AI/ML Fundamentals Course (Individual Project Track)**
**Student:** [ВПИШИ ПОЛНОЕ ИМЯ]

## Problem Statement

Fixed-timing traffic lights keep a constant green-light duration regardless of actual road congestion, causing unnecessary queues during peak hours and wasted green time during low traffic. This project builds a prototype that detects vehicles at an intersection, estimates queue size and waiting time, and adaptively adjusts green-light duration — acting like a traffic officer directing congestion.

## Project Track

Individual Project Track (Track 1) — original idea, approved via Project Brief.

## Dataset Source

- Training/validation/test data: [Roboflow Universe dataset name + link — ВСТАВЬ]
- Format: YOLOv8 (images + bounding box annotations)
- Demo video: [YouTube URL used for the end-to-end demo — ВСТАВЬ]
- License/usage conditions: [ВСТАВЬ]

## Data Audit & Leakage Check

- **EDA:** class balance, dataset size per split — see notebook Step 2 (`eda_class_balance.png`)
- **Leakage check:** exact-duplicate image hashes checked across train/valid/test splits — see notebook Step 2b (`duplicate_and_group_check.csv`, auto-generated on run)
- **Key risk identified:** CV datasets built from video frames can leak the same scene across splits, inflating test metrics. Result: [ВСТАВЬ после запуска — найдены ли дубликаты, как это повлияло на интерпретацию метрик]
- **Split strategy:** used Roboflow's pre-made train/valid/test split; final metrics reported only once on the test split (Step 7); demo video is a fully independent, never-seen source used for qualitative validation
- **Known limitation:** no source-video group metadata available from the Roboflow export, so exact per-video grouping could not be verified beyond exact-duplicate hashing

## ML Task Type

Object Detection (vehicle localization + classification) combined with Multi-Object Tracking (ByteTrack) for temporal analysis (waiting time estimation).

## Project Pipeline / System Architecture

```
Video frame
   ↓
YOLOv8 (fine-tuned) — vehicle detection
   ↓
ByteTrack — multi-object tracking (assigns persistent IDs across frames)
   ↓
Queue analysis — count of stationary vehicles, average waiting time
   ↓
Rule-based signal logic — decides green-light extension/reduction
   ↓
Annotated video output + signal decision
```

## Models / Approaches Tested

| Approach | Description |
|---|---|
| Baseline | Background subtraction (MOG2), classical CV, no ML |
| YOLOv8 pretrained | Off-the-shelf COCO weights, no fine-tuning |
| YOLOv8 fine-tuned (final) | Fine-tuned on domain-specific vehicle dataset |

Multiple fine-tuning experiments were run varying epochs, image size, and learning rate — see `experiment_log.csv` and the notebook's "Experiment tracking" section.

## Final Model and Justification

Final model: YOLOv8n fine-tuned, experiment `exp2_more_epochs` (25 epochs, imgsz=640). Selected based on best mAP@50 vs. training time trade-off among tested experiments — see comparison table in the notebook (Step 4).

## Evaluation Metrics and Results

Evaluated on a held-out **test split** (unseen data, not used in training or validation):

| Model | mAP@50 | Precision | Recall |
|---|---|---|---|
| YOLOv8 pretrained | [ВСТАВЬ] | [ВСТАВЬ] | [ВСТАВЬ] |
| YOLOv8 fine-tuned (final) | [ВСТАВЬ] | [ВСТАВЬ] | [ВСТАВЬ] |

Full results: `model_comparison.csv`. Error analysis with example failure cases: see notebook Step 8 and `error_analysis_samples.png`.

## Installation Instructions

```bash
pip install -r requirements.txt
```

Or simply open `smart_traffic_light.ipynb` in Google Colab — all dependencies are installed in the first cell.

## Training / Fine-tuning Instructions

1. Open `smart_traffic_light.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Get a free Roboflow API key (roboflow.com → Settings → API Key) and paste it into Step 1
4. Run cells sequentially through Step 5 (training + experiment comparison)

## Demo and Inference Run Instructions (Colab-first)

1. In the notebook, set `YOUTUBE_URL` (Step 6) to any traffic-intersection video
2. Run Steps 6–12 sequentially
3. Step 12 loads the model fresh from the saved `artifacts/best_model.pt` file (not from training memory) — this is the reproducible inference demo
4. Output: `demo_output.mp4` (annotated video) + printed signal decision

## Example Input and Output

**Input:** short video clip of a traffic intersection (mp4)
**Output:**
- Annotated video with bounding boxes, vehicle count, and queue count overlaid
- Console output: e.g. `Машин в очереди: 6 → Решение: Средняя очередь -> немного продлеваем → Новое время зелёного: 35 сек`

## Known Limitations

- Queue length is approximated by vehicle count, not physical distance (no camera calibration)
- Signal control logic is a simple threshold rule, not an optimization algorithm (e.g., reinforcement learning)
- Trained on a limited dataset and epoch budget — see `experiment_log.csv`
- Does not account for multiple lanes/directions separately or multi-intersection coordination ("green wave")
- Prototype only — not validated for real-world deployment

## Responsible AI Considerations

- **Bias:** dataset may over-represent certain conditions (daytime, specific road types) and vehicle classes (cars over buses/trucks) — see EDA in notebook Step 2
- **Privacy:** traffic camera footage may capture license plates and faces; real deployment requires anonymization (blurring) before storage/processing
- **Safety:** detection errors could lead to suboptimal or unsafe signal timing; not suitable for direct control of real infrastructure without further testing and fail-safe mechanisms

## Repository Structure

```
.
├── README.md
├── PROJECT_STATUS.md
├── requirements.txt
├── smart_traffic_light.ipynb    # main notebook: EDA, leakage check, training, evaluation, demo
├── artifacts/
│   └── best_model.pt            # saved fine-tuned model weights (generated on run)
├── experiment_log.csv           # experiment tracking table (generated on run)
├── model_comparison.csv         # baseline vs pretrained vs fine-tuned comparison (generated on run)
├── duplicate_and_group_check.csv # leakage/duplicate check across splits (generated on run)
├── eda_class_balance.png
├── training_curves.png
├── error_analysis_samples.png
├── traffic_over_time.png
└── slides/
    └── presentation.pdf         # defense slides
```

## Dataset License / Acknowledgements

[ВСТАВЬ: license of the Roboflow dataset used, and attribution if required]
