# Custom YOLO11 P2 Model

This repository contains a custom Ultralytics YOLO11 model configuration modified to detect on **P2, P3, and P4** feature maps instead of the default **P3, P4, and P5**.

The main purpose of this modification is to improve performance on **small-object detection** tasks by adding a higher-resolution **P2 detection head** and removing the lower-resolution **P5 head**.

---

## Overview

Standard YOLO11 detection heads operate on:

- P3/8
- P4/16
- P5/32

This custom version changes the detection pipeline to:

- P2/4
- P3/8
- P4/16

Because the P2 feature map has a higher spatial resolution, this architecture is better suited for datasets where the objects of interest are very small and may be lost at deeper, lower-resolution stages.

---

## What Was Changed

The original YOLO11 head detects on:

- P3
- P4
- P5

This custom model:

- adds an extra upsampling branch to create a **P2 head**
- removes the final **P5 detection output**
- uses detection outputs from:
  - **P2**
  - **P3**
  - **P4**

In short:

- default: `Detect(P3, P4, P5)`
- custom: `Detect(P2, P3, P4)`

---

## Why Use a P2 Head

A P2 head can help when:

- objects are very small in the image
- defects occupy only a tiny region
- higher-resolution features are needed for detection
- the default P5 head is less useful because large-object emphasis is not the priority

This is especially relevant for:

- defect detection
- microscopic or fine-detail inspection
- industrial vision tasks
- dense small-object datasets

---

## Tradeoffs

This modification is not automatically better in every case.

### Potential advantages
- better small-object sensitivity
- improved localization for tiny targets
- stronger high-resolution feature usage

### Potential disadvantages
- higher memory usage
- slower training/inference
- possible reduction in large-object performance
- more sensitivity to batch size and image resolution limits

If you use a P2 head with large input sizes such as 1280, GPU memory usage can increase significantly.

---

## File Purpose

This repository provides a custom model YAML for use with the Ultralytics framework.

It is intended for:

- custom training
- architecture experiments
- small-object detection research
- comparison against standard YOLO11 variants

---

## Model Structure

### Original YOLO11 detection outputs
- P3/8
- P4/16
- P5/32

### Custom YOLO11 detection outputs
- P2/4
- P3/8
- P4/16

This means the model emphasizes higher-resolution detection features and removes the coarsest detection scale.

---

## How to Use

You can train this model by passing the custom YAML file directly into Ultralytics.

### Python
```python
from ultralytics import YOLO

model = YOLO("custom_yolo11_p2.yaml")
model.train(data="data.yaml", epochs=100, imgsz=640)
