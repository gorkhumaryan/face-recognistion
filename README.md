rgba(170, 177, 255, 0.93) Recognition System (FaceNet + Cosine Similarity)
## Overview

This project implements a face recognition system using a pretrained FaceNet model.

## Pipeline
Image → 512D Embedding → Cosine Similarity vs Gallery → Identity or "Unknown"

The system supports open-set recognition, meaning it can detect unknown individuals.

## Features

 Pretrained FaceNet (VGGFace2 weights)

 Cosine similarity matching

 Threshold tuning using validation set

 Open-set evaluation (unknown detection)

 ROC curve and AUC analysis

## Installation

Create a virtual environment and install dependencies:

pip install -r requirements.txt

## Usage
### Split Dataset
python split_dataset.py
### Build Gallery
python build_gallery.py
### Test Recognition
python recognize.py
### Open-set Evaluation
python open_set_eval.py
## Results (Open-set Evaluation)
Validation

Known accuracy: 0.995

Unknown detection: 1.0

Test

Known accuracy: 0.991 (221 images)

Unknown detection: 0.944 (36 images)

ROC AUC ≈ 0.999

## Technical Details

Embedding size: 512

Similarity metric: Cosine similarity

Threshold selected via validation set

Open-set evaluation protocol applied

## Notes

Dataset images are assumed to be cropped faces.

Performance may decrease under real-world conditions (lighting, blur, occlusion).

Model weights are not included in the repository.