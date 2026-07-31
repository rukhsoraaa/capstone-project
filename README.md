# Smart Traffic Light — Adaptive Signal Control via Vehicle Detection

Capstone Project — AI/ML Fundamentals Course (Individual Project Track)
Student: Ruxsora Ahmedova

## The problem

Regular traffic lights just run on a timer. Green stays on for a fixed number of seconds no matter how many cars are actually waiting. That means long queues during rush hour and wasted green time when the road is empty. This project is a prototype that watches a traffic camera, figures out how many cars are stuck at the light and for how long, and adjusts the green light time accordingly — basically doing what a traffic officer does by hand.

## Project track

Individual Project Track (Track 1).

## Dataset

- Data: Traffic Intersection Vehicle Detection, from Roboflow Universe — https://universe.roboflow.com/vai/traffic-intersection-vehicle-detection
- 4,526 images, classes: car, truck, bus, motorbike, person
- License: CC BY 4.0
- Format: YOLOv8 (images + bounding boxes)
- Demo video (used only for the live demo, not for training): [paste the YouTube URL you used]

## Data audit and leakage check

Before training, I checked whether the dataset could leak information between splits. This matters especially for traffic camera data, since images are often extracted from video, and near-identical frames could end up in both train and test — that would make the model's test score look better than it really is.

- Class balance and dataset size: notebook Step 2 (see `eda_class_balance.png`)
- Exact-duplicate check across train/valid/test: notebook Step 2b (`duplicate_and_group_check.csv`)
- Result: [fill in after running Step 2b - whether duplicates were found and how it affects the metrics]
- Split used: Roboflow's pre-made train/valid/test split. Final metrics are reported once on the test split (Step 7) and never used to pick the model beforehand. The demo video is a completely separate source the model never saw during training.
- Limitation: Roboflow doesn't expose which video each image came from, so I could only check for exact duplicates, not full video-level grouping.

## What kind of ML task this is

Object detection — the model finds vehicles in an image and draws a box around each one, with a class label (car/truck/bus/motorbike). On top of that, a tracker (ByteTrack) follows each vehicle across frames so I can tell how long it's been sitting still.

## How the pipeline works

1. A video frame comes in
2. YOLOv8 (fine-tuned) detects the vehicles in it
3. ByteTrack keeps track of each vehicle across frames (gives it an ID)
4. The code counts how many vehicles are stopped and for how long
5. A simple rule decides whether to extend or shorten the green light
6. Output: annotated video + the signal decision

## Models compared

| Approach | What it is |
|---|---|
| Baseline | Background subtraction (MOG2) — classic computer vision, no neural network |
| YOLOv8 pretrained | The stock COCO-trained model, no fine-tuning |
| YOLOv8 fine-tuned (final model) | Same architecture, retrained on the traffic dataset above |

I ran three fine-tuning experiments with different epoch counts / image size / learning rate — the full comparison is in `experiment_log.csv` and notebook Step 4.

## Which model I picked and why

Final model: YOLOv8n fine-tuned — experiment `[fill in the experiment name you chose]`. I picked it by comparing validation mAP@50 against training time across the three experiments; the full table is in `experiment_log.csv` and notebook Step 4.

## Results

Measured on a held-out split the model never saw during training or model selection:

| Model | mAP@50 | Precision | Recall |
|---|---|---|---|
| YOLOv8 pretrained (COCO class IDs, not aligned) | [fill in] | [fill in] | [fill in] |
| YOLOv8 fine-tuned (final) | [fill in] | [fill in] | [fill in] |

An important caveat about the first row: the pretrained model uses COCO's class numbering, where a car is class 2, while this dataset numbers its classes differently. When the pretrained model is scored against these labels the IDs do not line up, so its mAP is close to zero. That number reflects the label mismatch, not how well the pretrained model actually sees vehicles. It is reported for transparency, but the meaningful comparison for this project is against the background-subtraction baseline.

Baseline comparison (notebook Step 7b): background subtraction detects moving regions rather than vehicles. It cannot classify what it finds, cannot follow an individual vehicle between frames, and — most importantly for this task — vehicles that stop for a long time are absorbed into the background model and disappear. Since the project depends on counting stationary vehicles, that is the failure case that matters most.

Full numbers: `model_comparison.csv`. Failure cases: notebook Step 8, `error_analysis_samples.png`.

## How to run this

```bash
pip install -r requirements.txt
```

Or just open `smart_traffic_light.ipynb` in Google Colab — it installs everything it needs in the first cell.

## Training the model

1. Open the notebook in Colab
2. Turn on GPU: Runtime -> Change runtime type -> GPU
3. Get a free Roboflow API key (roboflow.com -> Settings -> API Key) and paste it into Step 1
4. Run the cells in order through Step 5

## Running the demo

1. Set `YOUTUBE_URL` in Step 6 to any traffic intersection video
2. Run Steps 6 through 12
3. Step 12 loads the model fresh from the saved file (`artifacts/best_model.pt`), not from memory — this proves the saved model actually works on its own, not just right after training
4. Output: `demo_output.mp4` and a printed signal decision

## Example output

Input: a short clip of a traffic intersection.

Output: an annotated video with boxes around each vehicle, a live count, and a line like:
```
Vehicles waiting: 6
Decision: Moderate queue - extend green slightly
Recommended green duration: 35 seconds
```

## Limitations

- Queue length is measured in number of cars, not meters — I didn't calibrate the camera for real-world distance
- The signal decision is a simple threshold rule, not an optimization algorithm
- Trained on a limited epoch budget because of the project timeline and free-tier GPU limits — see `experiment_log.csv`
- Waiting time is detected using fixed pixel and time thresholds (5 px of movement, 2 seconds), which are tuned to this camera's resolution and angle and would need re-tuning elsewhere
- Doesn't handle multiple lanes separately, or coordination between nearby intersections
- This is a prototype for a course project, not something ready for real deployment

## Responsible AI notes

- Bias: the dataset may skew toward certain conditions (time of day, road type) and vehicle types — cars likely outnumber buses/trucks by a lot, see the class balance chart
- Privacy: real traffic footage can show license plates and faces. A real deployment would need to blur these before storing or processing anything
- Safety: detection mistakes could lead to bad signal timing decisions. This is not meant to control a real traffic light without a lot more testing and safety checks

## Repository structure

```
.
├── README.md
├── PROJECT_STATUS.md
├── requirements.txt
├── smart_traffic_light.ipynb
├── artifacts/
│   └── best_model.pt
├── experiment_log.csv
├── model_comparison.csv
├── duplicate_and_group_check.csv
├── eda_class_balance.png
├── training_curves.png
├── error_analysis_samples.png
├── traffic_over_time.png
├── demo_frames.png
└── slides/
    └── presentation.pdf
```
