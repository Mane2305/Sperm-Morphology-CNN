"""
Model calibration analysis including:
- Expected Calibration Error (ECE)
- Temperature scaling
- Dirichlet calibration
- Brier score
- Reliability diagrams
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationAnalyzer:
    """
    Analyzes and improves model probability calibration.

    Includes:
    - Expected Calibration Error (ECE)
    - Temperature scaling
    - Dirichlet calibration (multiclass)
    - Brier score
    - Reliability diagrams
    """

    # ============================================================
    # Expected Calibration Error
    # ============================================================
    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        n_bins: int = 10
    ) -> float:

        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0

        print("\n" + "="*70)
        print("EXPECTED CALIBRATION ERROR (ECE)")
        print("="*70)
        print(f"\n{'Bin':^15} {'Pred Conf':^12} {'Actual Acc':^12} {'Gap':^10} {'Count':^8}")
        print("─" * 70)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                acc = np.mean(predictions[in_bin] == y_true[in_bin])
                conf = np.mean(confidences[in_bin])
                gap = abs(conf - acc)

                ece += gap * prop_in_bin

                print(
                    f"{bin_lower*100:5.0f}%-{bin_upper*100:3.0f}%"
                    f"{conf:^12.2%}"
                    f"{acc:^12.2%}"
                    f"{gap:^10.2%}"
                    f"{np.sum(in_bin):^8d}"
                )

        print("─" * 70)
        print(f"\nECE: {ece:.4f} ({ece*100:.2f}%)")

        if ece < 0.05:
            print("✓ Excellent calibration")
        elif ece < 0.10:
            print("⚠ Moderate calibration")
        else:
            print("✗ Poor calibration")

        return float(ece)

    # ============================================================
    # Temperature Scaling
    # ============================================================
    @staticmethod
    def temperature_scaling(
        val_outputs: np.ndarray,
        labels_val: np.ndarray,
        test_outputs: np.ndarray = None # type: ignore
    ) -> Tuple[float, np.ndarray]:

        print("\n" + "="*70)
        print("TEMPERATURE SCALING")
        print("="*70)

        val_tensor = torch.FloatTensor(val_outputs)
        labels_tensor = torch.LongTensor(labels_val)

        # convert probs → logits if needed
        if torch.all((val_tensor >= 0) & (val_tensor <= 1)):
            val_tensor = torch.log(val_tensor + 1e-12)

        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(val_tensor / temperature, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)

        T = temperature.item()
        print(f"Optimal Temperature: {T:.4f}")

        if test_outputs is not None:
            test_tensor = torch.FloatTensor(test_outputs)
            if torch.all((test_tensor >= 0) & (test_tensor <= 1)):
                test_tensor = torch.log(test_tensor + 1e-12)
            calibrated = torch.softmax(test_tensor / T, dim=1).numpy()
        else:
            calibrated = torch.softmax(val_tensor / T, dim=1).numpy()

        return T, calibrated

    # ============================================================
    # Dirichlet Calibration
    # ============================================================
    @staticmethod
    def dirichlet_calibration(
        val_probs: np.ndarray,
        labels_val: np.ndarray,
        test_probs: np.ndarray = None # type: ignore
    ) -> np.ndarray:
        """
        Dirichlet calibration using multinomial logistic regression
        in log-probability space.

        Reference:
        Kull et al., NeurIPS 2019
        """

        print("\n" + "="*70)
        print("DIRICHLET CALIBRATION")
        print("="*70)

        eps = 1e-12

        def transform(p):
            p = np.clip(p, eps, 1.0)
            return np.log(p)

        X_val = transform(val_probs)

        clf = LogisticRegression(
    solver="lbfgs",
    max_iter=1000
)

        clf.fit(X_val, labels_val)

        if test_probs is not None:
            X_test = transform(test_probs)
            calibrated = clf.predict_proba(X_test)
        else:
            calibrated = clf.predict_proba(X_val)

        print("Dirichlet calibration fitted.")
        return calibrated

    # ============================================================
    # Brier Score
    # ============================================================
    @staticmethod
    def brier_score(
        y_true: np.ndarray,
        y_probs: np.ndarray
    ) -> float:

        y_true_onehot = np.zeros_like(y_probs)
        y_true_onehot[np.arange(len(y_true)), y_true] = 1

        brier = np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1))

        print(f"\nBrier Score: {brier:.4f}")
        return float(brier)

    # ============================================================
    # Reliability Diagram
    # ============================================================
    @staticmethod
    def plot_calibration_curve(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        n_bins: int = 10,
        save_path: str = "calibration_curve.png",
        title: str = "Calibration Curve"
    ):

        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_conf = []
        bin_acc = []

        for bl, bu in zip(bin_lowers, bin_uppers):
            mask = (confidences > bl) & (confidences <= bu)
            if np.sum(mask) > 0:
                bin_conf.append(np.mean(confidences[mask]))
                bin_acc.append(np.mean(predictions[mask] == y_true[mask]))

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.plot(bin_conf, bin_acc, "o-", label="Model")

        ax.set_xlabel("Predicted Confidence")
        ax.set_ylabel("Actual Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

        ece = CalibrationAnalyzer.expected_calibration_error(y_true, y_probs)
        ax.text(0.05, 0.95, f"ECE={ece:.3f}", transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        logger.info(f"✓ Calibration curve saved: {save_path}")

        return bin_conf, bin_acc


if __name__ == "__main__":
    print("CalibrationAnalyzer ready with Dirichlet + Temperature scaling")