"""
5-Fold Stratified Cross-Validation for robust performance estimation
"""

import logging
from typing import Dict, Callable, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossValidator:
    """
    Performs stratified k-fold cross-validation for robust performance estimation
    """

    def __init__(
        self,
        model_fn: Callable[[], torch.nn.Module],
        dataset: Dataset,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.model_fn = model_fn
        self.dataset = dataset
        self.n_splits = n_splits
        self.random_state = random_state

        logger.info("Extracting labels from dataset...")

        # Safe label extraction
        self.labels = self._extract_labels(dataset)

        self.skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        logger.info(f"CrossValidator initialized with {n_splits} folds")
        logger.info(f"Dataset size: {len(dataset)}") # type: ignore
        logger.info(f"Class distribution: {np.bincount(self.labels)}")

    def _extract_labels(self, dataset: Dataset) -> np.ndarray:
        labels: List[int] = []
        for i in range(len(dataset)): # type: ignore
            _, label = dataset[i]
            labels.append(int(label))
        return np.array(labels)

    def run_cross_validation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, List[float]] = {
            "fold": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1_normal": [],
            "val_f1_abnormal": [],
            "val_f1_nonsperm": [],
            "val_roc_auc": [],
            "val_loss": [],
        }

        print("\n" + "=" * 70)
        print(f"{'5-FOLD STRATIFIED CROSS-VALIDATION':^70}")
        print("=" * 70)

        for fold, (train_idx, val_idx) in enumerate(
            self.skf.split(np.zeros(len(self.labels)), self.labels)
        ):
            print(f"\n{'=' * 70}")
            print(f"FOLD {fold + 1}/{self.n_splits}")
            print(f"{'=' * 70}")
            print(f"Training samples: {len(train_idx)}")
            print(f"Validation samples: {len(val_idx)}")

            train_dist = np.bincount(self.labels[train_idx])
            val_dist = np.bincount(self.labels[val_idx])
            print(f"Train distribution: {train_dist}")
            print(f"Val distribution: {val_dist}")

            train_subset = Subset(self.dataset, train_idx)
            val_subset = Subset(self.dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=config.get("batch_size", 32),
                shuffle=True,
                num_workers=config.get("num_workers", 4),
                pin_memory=True,
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=config.get("batch_size", 32),
                shuffle=False,
                num_workers=config.get("num_workers", 4),
                pin_memory=True,
            )

            model = self.model_fn()
            device = torch.device(
                config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            )
            model.to(device)

            print(f"\nTraining fold {fold + 1}...")
            train_metrics = self._train_fold(
                model, train_loader, val_loader, config, fold + 1
            )

            print(f"Evaluating fold {fold + 1}...")
            val_metrics = self._evaluate_fold(model, val_loader, device)

            results["fold"].append(fold + 1)
            results["train_acc"].append(train_metrics["best_train_acc"])
            results["val_acc"].append(val_metrics["accuracy"])
            results["val_f1_normal"].append(val_metrics["f1_normal"])
            results["val_f1_abnormal"].append(val_metrics["f1_abnormal"])
            results["val_f1_nonsperm"].append(val_metrics["f1_nonsperm"])
            results["val_roc_auc"].append(val_metrics["roc_auc"])
            results["val_loss"].append(val_metrics["val_loss"])

            print(f"\n{'Fold ' + str(fold + 1) + ' Results':^70}")
            print("-" * 70)
            print(f"  Validation Accuracy:  {val_metrics['accuracy']:.4f}")
            print(f"  Normal F1:            {val_metrics['f1_normal']:.4f}")
            print(f"  Abnormal F1:          {val_metrics['f1_abnormal']:.4f}")
            print(f"  Non-sperm F1:         {val_metrics['f1_nonsperm']:.4f}")
            print(f"  ROC-AUC:              {val_metrics['roc_auc']:.4f}")

        results_df = pd.DataFrame(results)

        summary_stats = {
            "mean_val_acc": results_df["val_acc"].mean(),
            "std_val_acc": results_df["val_acc"].std(),
            "mean_f1_normal": results_df["val_f1_normal"].mean(),
            "std_f1_normal": results_df["val_f1_normal"].std(),
            "mean_f1_abnormal": results_df["val_f1_abnormal"].mean(),
            "std_f1_abnormal": results_df["val_f1_abnormal"].std(),
            "mean_f1_nonsperm": results_df["val_f1_nonsperm"].mean(),
            "std_f1_nonsperm": results_df["val_f1_nonsperm"].std(),
            "mean_roc_auc": results_df["val_roc_auc"].mean(),
            "std_roc_auc": results_df["val_roc_auc"].std(),
        }

        print("\n" + "=" * 70)
        print(f"{'CROSS-VALIDATION SUMMARY':^70}")
        print("=" * 70)

        print(
            f"\nValidation Accuracy:  {summary_stats['mean_val_acc']:.4f} ± {summary_stats['std_val_acc']:.4f}"
        )
        print(
            f"Normal F1:            {summary_stats['mean_f1_normal']:.4f} ± {summary_stats['std_f1_normal']:.4f}"
        )
        print(
            f"Abnormal F1:          {summary_stats['mean_f1_abnormal']:.4f} ± {summary_stats['std_f1_abnormal']:.4f}"
        )
        print(
            f"Non-sperm F1:         {summary_stats['mean_f1_nonsperm']:.4f} ± {summary_stats['std_f1_nonsperm']:.4f}"
        )
        print(
            f"ROC-AUC:              {summary_stats['mean_roc_auc']:.4f} ± {summary_stats['std_roc_auc']:.4f}"
        )

        cv_acc = (summary_stats["std_val_acc"] / summary_stats["mean_val_acc"]) * 100
        cv_abnormal_f1 = (
            summary_stats["std_f1_abnormal"] / summary_stats["mean_f1_abnormal"]
        ) * 100

        summary_stats["cv_acc"] = cv_acc
        summary_stats["cv_abnormal_f1"] = cv_abnormal_f1

        print("\nCoefficient of Variation:")
        print(f"  Accuracy CV:    {cv_acc:.2f}%")
        print(f"  Abnormal F1 CV: {cv_abnormal_f1:.2f}%")

        results["summary"] = summary_stats # type: ignore

        results_df.to_csv("cross_validation_results.csv", index=False)
        logger.info("✓ Results saved to: cross_validation_results.csv")

        return results

    def _train_fold(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        fold_num: int,
    ) -> Dict[str, float]:
        """
        Full training loop for one fold
        """
        from src.train import train_epoch  # your epoch trainer

        device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        num_epochs = config.get("num_epochs", 10)

        best_val_acc = 0.0
        best_train_acc = 0.0

        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            val_metrics = self._evaluate_fold(model, val_loader, device)
            val_acc = val_metrics["accuracy"]

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc

                torch.save(
                    model.state_dict(),
                    f"models/fold_{fold_num}_best.pth",
                )

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        return {
            "best_train_acc": float(best_train_acc),
            "best_val_acc": float(best_val_acc),
        }

    def _evaluate_fold(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        model.eval()

        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[List[float]] = []
        total_loss = 0.0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)

        accuracy = float(np.mean(all_preds_np == all_labels_np))

        f1_scores = f1_score(all_labels_np, all_preds_np, average=None)

        try:
            roc_auc = float(
                roc_auc_score(all_labels_np, all_probs_np, multi_class="ovr", average="macro")
            )
        except Exception:
            roc_auc = 0.0
            logger.warning("Could not calculate ROC-AUC")

        return {
            "accuracy": accuracy,
            "f1_normal": float(f1_scores[0]), # type: ignore
            "f1_abnormal": float(f1_scores[1]), # type: ignore
            "f1_nonsperm": float(f1_scores[2]), # type: ignore
            "roc_auc": roc_auc,
            "val_loss": float(total_loss / len(dataloader)),
        }