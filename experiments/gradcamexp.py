# experiments/gradcam.py
"""
Grad-CAM experiment runner (config-unified)

Uses config.yaml schema:
    data.test_dir
    training.num_classes
    preprocessing.output_size
    output.model_dir

Usage:
    python -m experiments.gradcam --model resnet50 --weights models/best_resnet50.pth
    python -m experiments.gradcam --model efficientnet_b0 --weights models/best_efficientnet_b0.pth
"""

import argparse
import yaml
import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader

from src.model import get_model
from src.dataset import SpermDataset
from src.transforms import get_val_test_transforms
from validation.gradcam import batch_gradcam_analysis


# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------
# Target layer detection
# -------------------------------------------------
def get_target_layer(model, backbone: str):
    backbone = backbone.lower()
    net = model.backbone if hasattr(model, "backbone") else model

    if "resnet" in backbone:
        return net.layer4
    elif "efficientnet" in backbone:
        return net.features[-1]
    elif "mobilenet" in backbone:
        return net.features[-1]
    else:
        raise ValueError(f"Unsupported backbone for GradCAM: {backbone}")


# -------------------------------------------------
# Checkpoint loading
# -------------------------------------------------
def load_weights(model, weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    return model


# -------------------------------------------------
# Main experiment
# -------------------------------------------------
def run_gradcam_experiment(args):

    set_seed(42)

    # -------------------------
    # Load config (UNIFIED)
    # -------------------------
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    num_classes = config["training"]["num_classes"]
    img_size = config["preprocessing"]["output_size"]
    if isinstance(img_size, list):
     img_size = img_size[0]
    test_dir = config["data"]["test_dir"]
    output_root = config["output"]["model_dir"]

    class_names = config.get(
        "class_names",
        ["normal", "abnormal", "non_sperm"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Output directory
    # -------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/results/gradcam/{args.model}/{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
# -------------------------
    # Model
    # -------------------------
    print(f"\nLoading model: {args.model}")

    model = get_model(
        model_type=args.model,   # ✅ correct param
        num_classes=num_classes,
        pretrained=False
    )

    model = load_weights(model, args.weights, device)
    model.to(device)
    model.eval()

    # -------------------------
    # Data
    # -------------------------
    transforms = get_val_test_transforms(img_size)

    dataset = SpermDataset(
        test_dir,
        transform=transforms
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # -------------------------
    # Grad-CAM target layer
    # -------------------------
    target_layer = get_target_layer(model, args.model)
    print(f"Grad-CAM target layer: {target_layer.__class__.__name__}")

    # -------------------------
    # Run Grad-CAM
    # -------------------------
    batch_gradcam_analysis(
        model=model,
        dataloader=loader,
        target_layer=target_layer, # type: ignore
        save_dir=str(exp_dir),
        n_samples=args.samples,
        class_names=class_names
    )

    # -------------------------
    # Save metadata
    # -------------------------
    meta = {
        "model": args.model,
        "weights": args.weights,
        "num_classes": num_classes,
        "img_size": img_size,
        "dataset": test_dir,
        "samples": args.samples,
        "batch_size": args.batch_size,
        "timestamp": timestamp,
        "device": str(device),
        "config_schema": "v1_training_preprocessing_data"
    }

    with open(exp_dir / "experiment.yaml", "w") as f:
        yaml.dump(meta, f)

    # Save config snapshot (paper reproducibility)
    with open(exp_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(config, f)

    print("\n✓ Grad-CAM experiment completed")
    print(f"Results saved to: {exp_dir}")


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grad-CAM interpretability experiment"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Backbone (resnet50, efficientnet_b0, mobilenet_v3, etc)"
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained weights (.pth)"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of Grad-CAM samples"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    args = parser.parse_args()

    run_gradcam_experiment(args)