"""
Investigate why TTA helps ResNet50 but hurts EfficientNet-B0
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataset import SpermDataset
from src.transforms import get_val_test_transforms, get_tta_transforms
from src.model import get_model
from src.evaluate import evaluate_model


def compute_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def analyze_tta(model, test_loader, device, model_name):
    print(f"\n{'='*60}")
    print(f"TTA Analysis: {model_name}")
    print(f"{'='*60}")

    # ---------- Single inference ----------
    y_true, y_pred, _ = evaluate_model(
        model, test_loader, device, use_tta=False
    )
    single_acc = compute_accuracy(y_true, y_pred)

    # ---------- TTA experiments ----------
    tta_results = {}

    tta_transforms = get_tta_transforms()

    for n_aug in [1, 3, 5, 7, 10]:
        selected_transforms = tta_transforms[:n_aug]

        y_true_tta, y_pred_tta, _ = evaluate_model(
            model,
            test_loader,
            device,
            use_tta=True,
            tta_transforms=selected_transforms
        )

        acc = compute_accuracy(y_true_tta, y_pred_tta)
        tta_results[n_aug] = acc

        delta = acc - single_acc
        print(f"  TTA-{n_aug}: {acc:.2%} (Δ{delta:+.2%})")

    return single_acc, tta_results


def main():
    # Load config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Test dataset ----------
    test_dataset = SpermDataset(
        config["data"]["test_dir"],
        transform=get_val_test_transforms()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4
    )

    # ---------- Load ResNet50 ----------
    resnet = get_model(
        model_type="resnet50",
        num_classes=config["training"]["num_classes"],
        pretrained=False
    ).to(device)

    # ---------- Load EfficientNet ----------
    efficient = get_model(
        model_type="efficientnet_b0",
        num_classes=config["training"]["num_classes"],
        pretrained=False
    ).to(device)

    # Load checkpoints (adjust names if needed)
    resnet_ckpt = torch.load(
        f"{config['output']['model_dir']}/best_resnet50.pth",
        map_location=device
    )
    efficient_ckpt = torch.load(
        f"{config['output']['model_dir']}/best_efficientnet_b0.pth2",
        map_location=device
    )

    resnet.load_state_dict(resnet_ckpt["model_state_dict"])
    efficient.load_state_dict(efficient_ckpt["model_state_dict"])

    resnet.eval()
    efficient.eval()

    # ---------- Run analysis ----------
    resnet_single, resnet_tta = analyze_tta(
        resnet, test_loader, device, "ResNet50"
    )

    efficient_single, efficient_tta = analyze_tta(
        efficient, test_loader, device, "EfficientNet-B0"
    )

    # ---------- Plot ----------
    os.makedirs("experiments/results", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    n_augs = [1, 3, 5, 7, 10]
    resnet_accs = [resnet_tta[n] for n in n_augs]
    efficient_accs = [efficient_tta[n] for n in n_augs]

    ax.plot(n_augs, resnet_accs, 'o-', label='ResNet50', linewidth=2)
    ax.plot(n_augs, efficient_accs, 's-', label='EfficientNet-B0', linewidth=2)

    ax.axhline(resnet_single, linestyle='--', alpha=0.5, label='ResNet50 baseline')
    ax.axhline(efficient_single, linestyle='--', alpha=0.5, label='EfficientNet baseline')

    ax.set_xlabel('Number of TTA Augmentations')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Impact of Test-Time Augmentation')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.savefig('experiments/results/tta_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved: experiments/results/tta_comparison.png")


if __name__ == "__main__":
    main()
