# Project Status

## Project

Smart Traffic Light - Adaptive Signal Control via Vehicle Detection (YOLOv8 fine-tuning + tracking)

## Current stage

Complete - all steps run, artifacts committed, README finalized.

## Completed

- Project scope defined
- Dataset downloaded: Traffic Intersection Vehicle Detection (Roboflow, VAI, CC BY 4.0)
- EDA completed: class balance chart, imbalance ratio 326.7x documented
- Data leakage check completed: no exact duplicates found across splits
- Baseline implemented: background subtraction (MOG2)
- Three fine-tuning experiments run: exp1_short, exp2_longer, exp3_lower_lr
- Final model selected: exp2_longer (mAP@50 = 0.882 on test split)
- Evaluation on unseen test data completed
- Error analysis completed: 8 test images reviewed, failure cases documented
- End-to-end demo completed: tracking, signal logic, reproducible inference from saved file
- All artifacts committed to GitHub
- README finalized with real metrics

## Current task

Submit project link to LMS before August 2, 14:00.

## Next

- Send Project Brief to mentor for Track 1 approval
- Optional: add class balancing experiment (tomorrow if time allows)

## Known problems / blockers

- Project Brief not yet submitted to mentor - send today