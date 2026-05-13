#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# 0. User configuration
# ============================================================

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


# Dataset / model selection ----------------------------------------------------
# Use "coco.yaml" for COCO-style pretraining, or set a local custom data.yaml path.
DATA_YAML = "coco.yaml"

# This script writes a valid YOLO11 P2 model YAML here unless WRITE_MODEL_YAML=False.
# If you already have your own custom YAML, set WRITE_MODEL_YAML=False and point MODEL_YAML to it.
MODEL_YAML = "/content/yolo11s-p2.yaml"
WRITE_MODEL_YAML = True

# Base pretrained weights used only for partial initialization.
# Keep this aligned with the model scale in MODEL_YAML.
BASE_WEIGHTS = "yolo11s.pt"

IMG_SIZE = 1280
EPOCHS = 200
BATCH = -1  # Auto-batch in many Ultralytics releases. If your version rejects this, set an integer like 16.
DEVICE = 0


# Google Drive / project layout ------------------------------------------------
USE_GOOGLE_DRIVE = True
GOOGLE_DRIVE_PROJECT_DIR = "/content/drive/MyDrive/yolo11s_p2_project"
LOCAL_PROJECT_DIR = "/content/yolo11s_p2_project"

RUN_NAME = "yolo11s_p2_run"
EXIST_OK = True
RESUME_IF_POSSIBLE = True
SAVE_PERIOD = 5  # Save an extra epoch checkpoint every N epochs for safer Colab recovery.


# Training hyperparameters -----------------------------------------------------
TRAIN_ARGS = {
    "epochs": EPOCHS,
    "imgsz": IMG_SIZE,
    "batch": BATCH,
    "device": DEVICE,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "close_mosaic": 30,
    "patience": 100,
    "cos_lr": True,
    "amp": True,
    "workers": 8,
    "cache": False,
}


# Validation / inference / export ---------------------------------------------
PREDICT_SOURCE = "https://ultralytics.com/images/bus.jpg"
EXPORT_ONNX = True
EXPORT_TENSORRT = False  # Requires compatible NVIDIA GPU + TensorRT support in the Colab runtime.
TENSORRT_WORKSPACE_GB = 4


# ============================================================
# 1. Install dependencies
# ============================================================

def run_cmd(cmd: list[str], check: bool = True) -> None:
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=check)


def install_dependencies() -> None:
    run_cmd([sys.executable, "-m", "pip", "install", "-U", "pip"])
    run_cmd([sys.executable, "-m", "pip", "install", "-U", "ultralytics"])


install_dependencies()


# ============================================================
# 2. Imports and environment checks
# ============================================================

import torch
import yaml
from ultralytics import YOLO, __version__ as ultralytics_version


def print_gpu_info() -> None:
    print("=" * 80)
    print("Ultralytics version :", ultralytics_version)
    print("PyTorch version     :", torch.__version__)
    print("CUDA available      :", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version        :", torch.version.cuda)
        print("GPU count           :", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total_mem_gb = props.total_memory / (1024 ** 3)
            print(
                f"GPU {idx}               : {props.name} | "
                f"Compute {props.major}.{props.minor} | "
                f"VRAM {total_mem_gb:.2f} GB"
            )
    else:
        print("WARNING: No CUDA GPU detected. Training will be very slow on CPU.")
    print("=" * 80)


def maybe_mount_google_drive() -> Path:
    if not USE_GOOGLE_DRIVE:
        project_dir = Path(LOCAL_PROJECT_DIR)
        project_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using local project directory: {project_dir}")
        return project_dir

    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
        project_dir = Path(GOOGLE_DRIVE_PROJECT_DIR)
        project_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using Google Drive project directory: {project_dir}")
        return project_dir
    except Exception as exc:
        print(f"Google Drive mount skipped or unavailable: {exc}")
        project_dir = Path(LOCAL_PROJECT_DIR)
        project_dir.mkdir(parents=True, exist_ok=True)
        print(f"Falling back to local project directory: {project_dir}")
        return project_dir


print_gpu_info()
PROJECT_DIR = maybe_mount_google_drive()
RUNS_DIR = PROJECT_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_YAML_PATH = Path(MODEL_YAML)
MODEL_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)

PARTIAL_WEIGHTS_OUT = PROJECT_DIR / "yolo11s_p2_partial_pretrained.pt"
TRANSFER_REPORT_OUT = PROJECT_DIR / "transfer_report.json"
TRAIN_CONFIG_OUT = PROJECT_DIR / "train_config_snapshot.json"


def make_run_paths() -> tuple[Path, Path, Path]:
    run_dir = RUNS_DIR / RUN_NAME
    weights_dir = run_dir / "weights"
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"
    return run_dir, best_ckpt, last_ckpt


def format_file_status(path: Path) -> str:
    if not path.exists():
        return f"{path} [missing]"
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    size_mb = stat.st_size / (1024 ** 2)
    return f"{path} [{size_mb:.2f} MB, modified {modified}]"


def print_checkpoint_status(run_dir: Path, best_ckpt: Path, last_ckpt: Path) -> None:
    print("=" * 80)
    print("Checkpoint status")
    print(f"Run directory : {run_dir}")
    print(f"Best checkpoint: {format_file_status(best_ckpt)}")
    print(f"Last checkpoint: {format_file_status(last_ckpt)}")
    if run_dir.exists():
        epoch_ckpts = sorted(run_dir.glob('weights/epoch*.pt'))
        if epoch_ckpts:
            print("Periodic checkpoints found:")
            for ckpt in epoch_ckpts[-5:]:
                print(f"  - {format_file_status(ckpt)}")
        else:
            print("Periodic checkpoints found: none yet")
    print("=" * 80)


RUN_DIR, BEST_CKPT, LAST_CKPT = make_run_paths()
print_checkpoint_status(RUN_DIR, BEST_CKPT, LAST_CKPT)


# ============================================================
# 3. Dataset setup
# ============================================================

def is_builtin_ultralytics_data(data_yaml: str) -> bool:
    builtin_prefixes = ("coco", "VOC", "Objects365", "xView", "SKU-110K", "Argoverse")
    data_name = Path(str(data_yaml)).name
    return data_name.endswith(".yaml") and any(data_name.startswith(prefix) for prefix in builtin_prefixes)


def resolve_data_yaml_path(data_yaml: str) -> str:
    if is_builtin_ultralytics_data(data_yaml):
        print(f"Using built-in Ultralytics dataset config: {data_yaml}")
        return data_yaml

    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(
            f"DATA_YAML does not exist: {data_path}\n"
            "Set DATA_YAML='coco.yaml' for COCO, or provide a valid local/custom data.yaml path."
        )
    print(f"Using custom dataset config: {data_path.resolve()}")
    return str(data_path.resolve())


def infer_num_classes(data_yaml: str) -> int:
    if is_builtin_ultralytics_data(data_yaml):
        if Path(data_yaml).name == "coco.yaml":
            return 80
        print("Built-in dataset detected. Falling back to 80 classes unless you override the model YAML.")
        return 80

    with open(data_yaml, "r", encoding="utf-8") as handle:
        data_cfg = yaml.safe_load(handle) or {}

    names = data_cfg.get("names")
    if isinstance(names, dict):
        return len(names)
    if isinstance(names, list):
        return len(names)
    if isinstance(data_cfg.get("nc"), int):
        return int(data_cfg["nc"])

    raise ValueError(
        "Could not infer the class count from your custom data.yaml.\n"
        "Add either 'nc: <int>' or a 'names:' list/dict to the file."
    )


RESOLVED_DATA_YAML = resolve_data_yaml_path(DATA_YAML)
MODEL_NC = infer_num_classes(RESOLVED_DATA_YAML)
print(f"Resolved class count for model YAML: nc={MODEL_NC}")


# ============================================================
# 4. Write the YOLO11 P2 model YAML
# ============================================================

YOLO11_P2_YAML = """# Ultralytics YOLO11 P2 detection model
# This model adds a high-resolution P2 detection branch for small-object detection.
# Detection outputs are:
#   P2/4  -> very small objects
#   P3/8  -> small objects
#   P4/16 -> medium objects
#
# This custom P2 variant removes the default coarse P5 detection output and
# predicts on P2, P3, and P4 instead. It is intended to be saved as
# yolo11s-p2.yaml so Ultralytics infers the 's' scale from the filename.

nc: {nc}  # number of classes; this script fills it from DATA_YAML when possible

scales:
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]          # 0  P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1  P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]  # 2  P2 features
  - [-1, 1, Conv, [256, 3, 2]]         # 3  P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]  # 4  P3 features
  - [-1, 1, Conv, [512, 3, 2]]         # 5  P4/16
  - [-1, 2, C3k2, [512, True]]         # 6  P4 features
  - [-1, 1, Conv, [1024, 3, 2]]        # 7  P5/32
  - [-1, 2, C3k2, [1024, True]]        # 8  P5 features
  - [-1, 1, SPPF, [1024, 5]]           # 9
  - [-1, 2, C2PSA, [1024]]             # 10

head:
  # P5 -> P4
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 11 upsample P5 to P4 resolution
  - [[-1, 6], 1, Concat, [1]]                   # 12 concat with backbone P4
  - [-1, 2, C3k2, [512, False]]                 # 13 P4 head features

  # P4 -> P3
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 14 upsample P4 head to P3 resolution
  - [[-1, 4], 1, Concat, [1]]                   # 15 concat with backbone P3
  - [-1, 2, C3k2, [256, False]]                 # 16 P3 head features

  # P3 -> P2
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 17 upsample P3 head to P2 resolution
  - [[-1, 2], 1, Concat, [1]]                   # 18 concat with backbone P2
  - [-1, 2, C3k2, [128, False]]                 # 19 P2 head features

  # P2 -> P3
  - [-1, 1, Conv, [256, 3, 2]]                  # 20 downsample back to P3 resolution
  - [[-1, 16], 1, Concat, [1]]                  # 21 concat with P3 head
  - [-1, 2, C3k2, [256, False]]                 # 22 P3 output branch

  # P3 -> P4
  - [-1, 1, Conv, [512, 3, 2]]                  # 23 downsample to P4 resolution
  - [[-1, 13], 1, Concat, [1]]                  # 24 concat with P4 head
  - [-1, 2, C3k2, [512, False]]                 # 25 P4 output branch

  # Detect on P2, P3, P4
  - [[19, 22, 25], 1, Detect, [nc]]
"""


def write_model_yaml(model_yaml_path: Path, nc: int) -> None:
    if not WRITE_MODEL_YAML:
        if not model_yaml_path.exists():
            raise FileNotFoundError(
                f"WRITE_MODEL_YAML=False but MODEL_YAML does not exist: {model_yaml_path}"
            )
        print(f"Using existing model YAML: {model_yaml_path}")
        return

    model_yaml_path.write_text(YOLO11_P2_YAML.format(nc=nc), encoding="utf-8")
    print(f"Wrote YOLO11 P2 model YAML to: {model_yaml_path}")


write_model_yaml(MODEL_YAML_PATH, MODEL_NC)
print(MODEL_YAML_PATH.read_text(encoding="utf-8"))


# ============================================================
# 5. Build the modified model and validate the YAML
# ============================================================

def build_model_from_yaml(model_yaml_path: Path) -> YOLO:
    try:
        model = YOLO(str(model_yaml_path))
        print(f"Successfully built model from YAML: {model_yaml_path}")
        return model
    except Exception as exc:
        raise RuntimeError(
            f"MODEL_YAML is invalid or incompatible with the installed Ultralytics version:\n"
            f"  {model_yaml_path}\n\n"
            f"Original error:\n{exc}\n\n"
            "If you swapped in your own YAML, re-check the layer indices and module names."
        ) from exc


def should_resume_from_last_checkpoint() -> bool:
    return RESUME_IF_POSSIBLE and LAST_CKPT.exists()


# ============================================================
# 6. Safe partial weight transfer
# ============================================================

def safe_partial_transfer(
    target_model: YOLO,
    source_weights: str,
    report_path: Path,
) -> dict[str, Any]:
    print(f"Loading base pretrained model for partial transfer: {source_weights}")
    source_model = YOLO(source_weights)

    target_state = target_model.model.state_dict()
    source_state = source_model.model.state_dict()

    transferred: dict[str, torch.Tensor] = {}
    skipped: list[dict[str, Any]] = []

    for key, value in source_state.items():
        if key not in target_state:
            skipped.append(
                {
                    "key": key,
                    "reason": "missing_in_target",
                    "source_shape": list(value.shape),
                    "target_shape": None,
                }
            )
            continue

        if tuple(value.shape) != tuple(target_state[key].shape):
            skipped.append(
                {
                    "key": key,
                    "reason": "shape_mismatch",
                    "source_shape": list(value.shape),
                    "target_shape": list(target_state[key].shape),
                }
            )
            continue

        transferred[key] = value.detach().clone()

    missing_from_source = [
        {
            "key": key,
            "reason": "missing_in_source",
            "source_shape": None,
            "target_shape": list(value.shape),
        }
        for key, value in target_state.items()
        if key not in source_state
    ]

    target_state.update(transferred)
    target_model.model.load_state_dict(target_state, strict=False)

    report = {
        "base_weights": source_weights,
        "model_yaml": str(MODEL_YAML_PATH),
        "transferred_layers": len(transferred),
        "skipped_layers": len(skipped),
        "target_only_layers": len(missing_from_source),
        "skipped_examples": skipped[:20],
        "target_only_examples": missing_from_source[:20],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Partial transfer report")
    print(f"Transferred layers/tensors : {report['transferred_layers']}")
    print(f"Skipped source tensors     : {report['skipped_layers']}")
    print(f"Target-only tensors        : {report['target_only_layers']}")
    if report["skipped_examples"]:
        print("Examples of skipped source tensors:")
        for item in report["skipped_examples"][:10]:
            print(
                f"  - {item['key']} | reason={item['reason']} | "
                f"source_shape={item['source_shape']} | target_shape={item['target_shape']}"
            )
    if report["target_only_examples"]:
        print("Examples of target-only tensors:")
        for item in report["target_only_examples"][:10]:
            print(
                f"  - {item['key']} | reason={item['reason']} | "
                f"source_shape={item['source_shape']} | target_shape={item['target_shape']}"
            )
    print(f"Full transfer report saved to: {report_path}")
    print("=" * 80)

    return report


def save_partial_checkpoint(model: YOLO, save_path: Path) -> None:
    model_copy = deepcopy(model.model).half().cpu()
    model_copy.args = getattr(model.model, "args", {})
    model_copy.pt_path = str(save_path)
    checkpoint = {
        "model": model_copy,
        "ema": model_copy,
        "train_args": {
            "task": "detect",
            "data": RESOLVED_DATA_YAML,
            "imgsz": IMG_SIZE,
        },
        "date": datetime.utcnow().isoformat() + "Z",
        "version": ultralytics_version,
        "license": "AGPL-3.0 (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }
    torch.save(checkpoint, save_path)
    print(f"Saved partial pretrained checkpoint: {save_path}")


if should_resume_from_last_checkpoint():
    print("Existing last.pt found. Skipping model rebuild and partial transfer for faster resume.")
    modified_model = None
else:
    modified_model = build_model_from_yaml(MODEL_YAML_PATH)
    transfer_report = safe_partial_transfer(
        target_model=modified_model,
        source_weights=BASE_WEIGHTS,
        report_path=TRANSFER_REPORT_OUT,
    )
    save_partial_checkpoint(modified_model, PARTIAL_WEIGHTS_OUT)


# ============================================================
# 7. Training and resume support
# ============================================================


def start_or_resume_training(model: YOLO | None) -> Any:
    if should_resume_from_last_checkpoint():
        print(f"Resuming training from: {LAST_CKPT}")
        resumed_model = YOLO(str(LAST_CKPT))
        return resumed_model.train(resume=True)

    if model is None:
        raise RuntimeError("No initialized model is available for a fresh training run.")

    train_kwargs = {
        "data": RESOLVED_DATA_YAML,
        "project": str(RUNS_DIR),
        "name": RUN_NAME,
        "exist_ok": EXIST_OK,
        "save": True,
        "save_period": SAVE_PERIOD,
        "plots": True,
        **TRAIN_ARGS,
    }

    TRAIN_CONFIG_OUT.write_text(json.dumps(train_kwargs, indent=2), encoding="utf-8")

    print("=" * 80)
    print("Starting training with the following arguments:")
    for key, value in train_kwargs.items():
        print(f"  {key}: {value}")
    print(f"Saved training config snapshot: {TRAIN_CONFIG_OUT}")
    print("=" * 80)

    return model.train(**train_kwargs)


train_results = start_or_resume_training(modified_model)

# Refresh run paths after training in case Ultralytics reused or resolved a directory internally.
RUN_DIR, BEST_CKPT, LAST_CKPT = make_run_paths()
if not BEST_CKPT.exists() and RUN_DIR.exists():
    candidate = RUN_DIR / "weights" / "best.pt"
    if candidate.exists():
        BEST_CKPT = candidate
if not LAST_CKPT.exists() and RUN_DIR.exists():
    candidate = RUN_DIR / "weights" / "last.pt"
    if candidate.exists():
        LAST_CKPT = candidate

print(f"Training run directory: {RUN_DIR}")
print(f"Best checkpoint       : {BEST_CKPT}")
print(f"Last checkpoint       : {LAST_CKPT}")
print_checkpoint_status(RUN_DIR, BEST_CKPT, LAST_CKPT)


# ============================================================
# 8. Validation / evaluation
# ============================================================

def scalar_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    return None


def summarize_val_metrics(val_results: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    if hasattr(val_results, "results_dict") and isinstance(val_results.results_dict, dict):
        summary["results_dict"] = val_results.results_dict

    if hasattr(val_results, "box"):
        box = val_results.box
        summary["precision"] = scalar_or_none(getattr(box, "mp", None))
        summary["recall"] = scalar_or_none(getattr(box, "mr", None))
        summary["mAP50"] = scalar_or_none(getattr(box, "map50", None))
        summary["mAP50_95"] = scalar_or_none(getattr(box, "map", None))
    else:
        results_dict = summary.get("results_dict", {})
        summary["precision"] = results_dict.get("metrics/precision(B)")
        summary["recall"] = results_dict.get("metrics/recall(B)")
        summary["mAP50"] = results_dict.get("metrics/mAP50(B)")
        summary["mAP50_95"] = results_dict.get("metrics/mAP50-95(B)")

    return summary


if not BEST_CKPT.exists():
    raise FileNotFoundError(
        f"Training finished but best checkpoint was not found at: {BEST_CKPT}\n"
        "Check the training logs above and verify that the run completed successfully."
    )

best_model = YOLO(str(BEST_CKPT))
val_batch = 1 if BATCH == -1 else BATCH

val_results = best_model.val(
    data=RESOLVED_DATA_YAML,
    imgsz=IMG_SIZE,
    batch=val_batch,
    device=DEVICE,
    split="val",
    project=str(RUN_DIR),
    name="val_after_train",
)

val_summary = summarize_val_metrics(val_results)
val_summary_path = RUN_DIR / "val_metrics_summary.json"
val_summary_path.write_text(json.dumps(val_summary, indent=2), encoding="utf-8")

print("=" * 80)
print("Validation summary")
print(f"Precision : {val_summary.get('precision')}")
print(f"Recall    : {val_summary.get('recall')}")
print(f"mAP50     : {val_summary.get('mAP50')}")
print(f"mAP50-95  : {val_summary.get('mAP50_95')}")
print(f"Saved to  : {val_summary_path}")
print("=" * 80)


# ============================================================
# 9. Inference test
# ============================================================

def run_inference(model: YOLO, source: str) -> Any:
    try:
        results = model.predict(
            source=source,
            imgsz=IMG_SIZE,
            device=DEVICE,
            save=True,
            project=str(RUN_DIR),
            name="predictions",
            exist_ok=True,
            conf=0.25,
        )
        print(f"Saved predictions under: {RUN_DIR / 'predictions'}")
        return results
    except Exception as exc:
        print(
            "Inference failed.\n"
            f"Source used: {source}\n"
            "If you are offline or the URL is blocked, set PREDICT_SOURCE to a local image or folder.\n"
            f"Original error: {exc}"
        )
        return None


_ = run_inference(best_model, PREDICT_SOURCE)


# ============================================================
# 10. Export
# ============================================================

if EXPORT_ONNX:
    print("Exporting best model to ONNX...")
    onnx_path = best_model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        dynamic=False,
        simplify=True,
    )
    print(f"ONNX export complete: {onnx_path}")

if EXPORT_TENSORRT:
    if not torch.cuda.is_available():
        print("TensorRT export skipped because CUDA is not available.")
    else:
        try:
            print("Exporting best model to TensorRT engine...")
            engine_path = best_model.export(
                format="engine",
                imgsz=IMG_SIZE,
                half=True,
                workspace=TENSORRT_WORKSPACE_GB,
                dynamic=False,
            )
            print(f"TensorRT export complete: {engine_path}")
        except Exception as exc:
            print(
                "TensorRT export failed.\n"
                "This is optional and often depends on the exact Colab runtime image.\n"
                f"Original error: {exc}"
            )


# ============================================================
# 11. Useful resume notes
# ============================================================

print("=" * 80)
print("Resume notes")
print(f"Project directory : {PROJECT_DIR}")
print(f"Run directory     : {RUN_DIR}")
print(f"Last checkpoint   : {LAST_CKPT}")
print("If Colab disconnects, reconnect and rerun the notebook.")
print("With RESUME_IF_POSSIBLE=True, this script will resume from last.pt automatically if it exists.")
print("=" * 80)


# ============================================================
# 12. Why partial transfer is required for a P2 model
# ============================================================

print(
    """
Why regular pretrained weights do not fully transfer into a P2 model:

1. A standard YOLO11 detection checkpoint is built for its original architecture, usually with P3/P4/P5 outputs.
2. After adding a P2 branch, the model graph changes: new layers are inserted, some later layer indices shift, and
   the Detect head now expects a different set of input feature maps.
3. Because of that, some parameter names no longer line up exactly, and some tensors that do line up by name still
   have different shapes.
4. Loading the entire checkpoint strictly would therefore fail or incorrectly map weights.

Why partial transfer is the correct practical approach:

1. The early backbone and many shared head layers still represent useful COCO-pretrained features.
2. Copying only the layers whose names and tensor shapes match preserves as much pretrained knowledge as possible.
3. New P2-specific layers and any mismatched head tensors remain randomly initialized and are learned during training.
4. This is usually the best compromise between stability and practicality when adapting YOLO to a modified topology.

Important practical note:

True pretraining from scratch on COCO for a custom architecture is expensive in both GPU time and cost. In practice,
most users initialize a modified model from a standard COCO-pretrained checkpoint using partial transfer, then train
or fine-tune on COCO or their target dataset from that partially initialized starting point.
""".strip()
)
