import torch
import yaml
import json
import csv
from torch.utils.data import DataLoader

from src.dataset import SpermDataset
from src.transforms import get_val_test_transforms
from src.model import get_model
from src.evaluate import evaluate_model
from validation.calibration import CalibrationAnalyzer


def load_model(model_type, ckpt_path, num_classes, device):
    model = get_model(
        model_type=model_type,
        num_classes=num_classes,
        pretrained=False
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def run_calibration(name, model, val_loader, test_loader, analyzer, device):
    print("\n" + "#"*80)
    print(f"CALIBRATION: {name}")
    print("#"*80)

    y_true_val, _, y_probs_val = evaluate_model(model, val_loader, device)
    y_true_test, _, y_probs_test = evaluate_model(model, test_loader, device)

    # BEFORE
    print("\nBEFORE CALIBRATION")
    ece_before = analyzer.expected_calibration_error(y_true_test, y_probs_test)

    # Temperature
    T, temp_probs = analyzer.temperature_scaling(
        y_probs_val,
        y_true_val,
        y_probs_test
    )

    print("\nAFTER TEMPERATURE")
    ece_temp = analyzer.expected_calibration_error(y_true_test, temp_probs)

    analyzer.plot_calibration_curve(
        y_true_test,
        temp_probs,
        save_path=f"experiments/results/{name}_temp.png",
        title=f"{name} Temperature"
    )

    # Dirichlet
    dir_probs = analyzer.dirichlet_calibration(
        y_probs_val,
        y_true_val,
        y_probs_test
    )

    print("\nAFTER DIRICHLET")
    ece_dir = analyzer.expected_calibration_error(y_true_test, dir_probs)

    analyzer.plot_calibration_curve(
        y_true_test,
        dir_probs,
        save_path=f"experiments/results/{name}_dirichlet.png",
        title=f"{name} Dirichlet"
    )

    return {
        "model": name,
        "temperature": float(T),
        "ece_before": float(ece_before),
        "ece_temperature": float(ece_temp),
        "ece_dirichlet": float(ece_dir)
    }


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = SpermDataset(
        config["data"]["val_dir"],
        transform=get_val_test_transforms()
    )

    test_dataset = SpermDataset(
        config["data"]["test_dir"],
        transform=get_val_test_transforms()
    )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    analyzer = CalibrationAnalyzer()

    # Load models
    resnet = load_model(
        "resnet50",
        f"{config['output']['model_dir']}/best_resnet50.pth",
        config["training"]["num_classes"],
        device
    )

    efficient = load_model(
        "efficientnet_b0",
        f"{config['output']['model_dir']}/best_efficientnet_b0.pth2",
        config["training"]["num_classes"],
        device
    )

    results = []
    results.append(run_calibration("ResNet50", resnet, val_loader, test_loader, analyzer, device))
    results.append(run_calibration("EfficientNet-B0", efficient, val_loader, test_loader, analyzer, device))

    # =============================
    # SAVE RESULTS
    # =============================
    json_path = "experiments/results/calibration_results.json"
    csv_path = "experiments/results/calibration_results.csv"

    # JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary
    print("\n" + "="*80)
    print("CALIBRATION SUMMARY")
    print("="*80)

    for r in results:
        print(
            f"{r['model']}: "
            f"T={r['temperature']:.3f} | "
            f"ECE {r['ece_before']:.3f} → "
            f"Temp {r['ece_temperature']:.3f} | "
            f"Dir {r['ece_dirichlet']:.3f}"
        )

    print("\nResults saved:")
    print(f"  {json_path}")
    print(f"  {csv_path}")