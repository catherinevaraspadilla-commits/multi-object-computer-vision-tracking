#!/usr/bin/env python
"""
SLEAP Bottom-Up training script for Bunya (UQ HPC).
Converts COCO dataset from Roboflow -> SLEAP labels -> trains model.

Usage:
    python train_sleap.py --epochs 20
    python train_sleap.py --epochs 50
"""

import argparse
import os
import sys
import subprocess
import json
import shutil
import numpy as np
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="SLEAP Bottom-Up training")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: results_<epochs>ep)")
    return parser.parse_args()


def check_gpu():
    print("=" * 60)
    print("Checking GPU availability...")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("WARNING: No GPU detected. Training will be very slow on CPU.")
    print("=" * 60)


def download_dataset():
    """Download dataset from Roboflow in COCO format."""
    from roboflow import Roboflow

    rf = Roboflow(api_key="kD254mWPx6WlP2phIeC4")
    project = rf.workspace("modelos-yolo").project("pruebasratslabs-c02t9")
    version = project.version(8)
    dataset_coco = version.download("coco")
    print(f"Dataset downloaded to: {dataset_coco.location}")
    return dataset_coco.location


def convert_coco_to_sleap(dataset_location):
    """Convert COCO annotations to SLEAP .slp format."""
    import sleap_io as sio
    import cv2

    keypoint_names = [
        "tail_tip", "tail_base", "tail_start",
        "mid_body", "nose", "right_ear", "left_ear"
    ]
    skeleton = sio.Skeleton(keypoint_names)

    train_dir = os.path.join(dataset_location, "train")
    json_path = os.path.join(train_dir, "_annotations.coco.json")

    with open(json_path) as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_size = {img["id"]: (img["height"], img["width"]) for img in coco["images"]}

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    labeled_frames = []

    for img_id, anns in anns_by_image.items():
        filename = id_to_file[img_id]
        img_path = os.path.join(train_dir, filename)
        if not os.path.exists(img_path):
            continue

        video = sio.Video(img_path)
        instances = []

        for ann in anns:
            kps = ann["keypoints"]
            points = np.array(
                [[kps[i * 3], kps[i * 3 + 1]] for i in range(7)], dtype=float
            )
            for i in range(7):
                if kps[i * 3 + 2] == 0:
                    points[i] = [np.nan, np.nan]
            instance = sio.Instance.from_numpy(points, skeleton=skeleton)
            instances.append(instance)

        lf = sio.LabeledFrame(video=video, frame_idx=0, instances=instances)
        labeled_frames.append(lf)

    labels = sio.Labels(labeled_frames)
    os.makedirs("data", exist_ok=True)
    labels.save("data/dataset_ratones.slp")
    print(f"Saved: {len(labeled_frames)} labeled frames to data/dataset_ratones.slp")

    # Add edges and symmetry to skeleton
    labels = sio.load_file("data/dataset_ratones.slp")
    skeleton = labels.skeletons[0]

    skeleton.add_edge("nose", "mid_body")
    skeleton.add_edge("mid_body", "tail_start")
    skeleton.add_edge("tail_start", "tail_base")
    skeleton.add_edge("tail_base", "tail_tip")
    skeleton.add_edge("nose", "right_ear")
    skeleton.add_edge("nose", "left_ear")
    skeleton.add_symmetry("right_ear", "left_ear")

    labels.save("data/dataset_ratones.slp")
    print(f"Edges: {skeleton.edges}")
    print("Skeleton saved with edges and symmetry.")

    return labels


def write_config(epochs, batch_size, lr, run_name):
    """Write SLEAP-NN training config YAML."""
    config_yaml = f"""
data_config:
  train_labels_path:
    - data/dataset_ratones.slp
  validation_fraction: 0.1
  augmentation_config:
    geometric:
      rotation_min: -180.0
      rotation_max: 180.0
      scale_min: 0.8
      scale_max: 1.2
      affine_p: 1.0

model_config:
  backbone_config:
    unet:
      kernel_size: 3
      filters: 64
      filters_rate: 2.0
      max_stride: 16
      middle_block: true
      up_interpolate: true
      stacks: 1
  head_configs:
    bottomup:
      confmaps:
        part_names:
          - tail_tip
          - tail_base
          - tail_start
          - mid_body
          - nose
          - right_ear
          - left_ear
        sigma: 2.0
        output_stride: 2
      pafs:
        sigma: 2.0
        output_stride: 2

trainer_config:
  max_epochs: {epochs}
  save_ckpt: true
  ckpt_dir: models/
  run_name: {run_name}
  train_data_loader:
    batch_size: {batch_size}
  val_data_loader:
    batch_size: {batch_size}
  optimizer_name: Adam
  optimizer:
    lr: {lr}
"""
    config_path = f"config_{run_name}.yaml"
    with open(config_path, "w") as f:
        f.write(config_yaml)
    print(f"Config written to {config_path}")
    return config_path


def train(config_path):
    """Run SLEAP-NN training."""
    print(f"Starting training with config: {config_path}")
    print("=" * 60)
    result = subprocess.run(
        ["sleap-nn", "train", config_path],
        check=True
    )
    print("=" * 60)
    print("Training complete.")


def collect_results(output_dir, run_name):
    """Copy model and logs to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    model_src = os.path.join("models", run_name)
    if os.path.exists(model_src):
        model_dst = os.path.join(output_dir, "model")
        if os.path.exists(model_dst):
            shutil.rmtree(model_dst)
        shutil.copytree(model_src, model_dst)
        print(f"Model copied to {model_dst}")

    # Copy config
    config_path = f"config_{run_name}.yaml"
    if os.path.exists(config_path):
        shutil.copy2(config_path, output_dir)

    print(f"Results saved to: {output_dir}")
    print(f"Contents: {os.listdir(output_dir)}")


def main():
    args = parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    run_name = f"bottomup_ratones_{epochs}ep"
    output_dir = args.output_dir or f"results_{epochs}ep"

    print(f"SLEAP Training - {epochs} epochs")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Run name: {run_name}")
    print(f"  Output dir: {output_dir}")
    print("=" * 60)

    check_gpu()

    # Download and convert dataset (skip if already exists)
    if not os.path.exists("data/dataset_ratones.slp"):
        print("Dataset not found, downloading and converting...")
        dataset_location = download_dataset()
        convert_coco_to_sleap(dataset_location)
    else:
        print("Dataset already exists at data/dataset_ratones.slp, skipping download.")

    # Write config and train
    config_path = write_config(epochs, batch_size, lr, run_name)
    train(config_path)

    # Collect results
    collect_results(output_dir, run_name)

    print("\nDone! Results are in:", output_dir)


if __name__ == "__main__":
    main()
