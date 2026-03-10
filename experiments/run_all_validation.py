""""
import os
import yaml
import json
import pickle
from torch.utils.data import ConcatDataset

from validation.cross_validation import CrossValidator
from src.model import get_model
from src.dataset import SpermDataset
from src.transforms import get_val_test_transforms

# ── directories that must exist before training ──────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("cv_results", exist_ok=True)


def build_full_dataset(config):
    train_dataset = SpermDataset(
        config["data"]["train_dir"],
        transform=get_val_test_transforms()
    )
    val_dataset = SpermDataset(
        config["data"]["val_dir"],
        transform=get_val_test_transforms()
    )
    return ConcatDataset([train_dataset, val_dataset])


def run_cv_with_cache(model_name: str, config: dict, full_dataset) -> dict:
    
    Run cross-validation for a model, using a cached result if it exists.
    This prevents re-running a model that already finished successfully.
    
    cache_path = f"cv_results/{model_name}_cv_results.pkl"

    if os.path.exists(cache_path):
        print(f"\n[CACHE HIT] Loading existing results for {model_name} from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"\nRunning {model_name} 5-Fold CV...")
    validator = CrossValidator(
        model_fn=lambda: get_model(
            model_name,
            num_classes=config["training"]["num_classes"]
        ),
        dataset=full_dataset,
        n_splits=5
    )

    cv_results = validator.run_cross_validation(config)

    # ── persist results so a later crash doesn't lose this work ──────────────
    with open(cache_path, "wb") as f:
        pickle.dump(cv_results, f)
    print(f"[SAVED] Results cached to {cache_path}")

    return cv_results


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    full_dataset = build_full_dataset(config)

    resnet_cv     = run_cv_with_cache("resnet50",        config, full_dataset)
    efficient_cv  = run_cv_with_cache("efficientnet_b0", config, full_dataset)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION COMPARISON")
    print("=" * 60)
    print(
        f"ResNet50:        "
        f"{resnet_cv['summary']['mean_val_acc']:.2%} ± "
        f"{resnet_cv['summary']['std_val_acc']:.2%}"
    )
    print(
        f"EfficientNet-B0: "
        f"{efficient_cv['summary']['mean_val_acc']:.2%} ± "
        f"{efficient_cv['summary']['std_val_acc']:.2%}"
    )


if __name__ == "__main__":
    main() 
    */ """

import os
import yaml
from torch.utils.data import ConcatDataset

from validation.cross_validation import CrossValidator
from src.model import get_model
from src.dataset import SpermDataset
from src.transforms import get_val_test_transforms

# ── ensure required directories exist ─────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("cv_results", exist_ok=True)


def build_full_dataset(config):
    train_dataset = SpermDataset(
        config["data"]["train_dir"],
        transform=get_val_test_transforms()
    )
    val_dataset = SpermDataset(
        config["data"]["val_dir"],
        transform=get_val_test_transforms()
    )
    return ConcatDataset([train_dataset, val_dataset])


def run_efficientnet_cv(config, full_dataset):
    """
    Run ONLY EfficientNet-B0 5-fold cross-validation
    """
    print("\nRunning EfficientNet-B0 5-Fold CV...")

    validator = CrossValidator(
        model_fn=lambda: get_model(
            "efficientnet_b0",
            num_classes=config["training"]["num_classes"]
        ),
        dataset=full_dataset,
        n_splits=5
    )

    cv_results = validator.run_cross_validation(config)

    print("\nEfficientNet CV completed ✅")
    print(
        f"Mean Val Acc: {cv_results['summary']['mean_val_acc']:.2%} ± "
        f"{cv_results['summary']['std_val_acc']:.2%}"
    )

    return cv_results


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    full_dataset = build_full_dataset(config)

    # ✅ RUN ONLY EFFICIENTNET (ResNet already done)
    efficient_cv = run_efficientnet_cv(config, full_dataset)

    print("\n" + "=" * 60)
    print("EFFICIENTNET CROSS-VALIDATION DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()