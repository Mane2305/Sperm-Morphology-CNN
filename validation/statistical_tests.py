# validation/statistical_tests.py
"""
Statistical significance testing for model comparison.

Includes:
- Paired t-test (cross-validation comparison)
- Wilcoxon signed-rank test (non-parametric)
- McNemar's test (paired predictions)
- Bootstrap confidence intervals (metrics uncertainty)

Designed for ML / medical imaging experiments.
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalTester:
    """
    Statistical comparison utilities for ML model evaluation.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    # -------------------------------------------------
    # Paired t-test (CV folds)
    # -------------------------------------------------
    def paired_t_test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        model_a: str = "Model A",
        model_b: str = "Model B",
        metric: str = "Score",
    ) -> Dict:
        """
        Paired t-test comparing two models on identical folds.

        scores_a: fold scores model A
        scores_b: fold scores model B
        """

        scores_a = np.asarray(scores_a)
        scores_b = np.asarray(scores_b)

        if len(scores_a) != len(scores_b):
            raise ValueError("Scores must have equal length for paired test")

        diff = scores_b - scores_a  # B − A
        mean_diff = np.mean(diff)

        t_stat, p_value = stats.ttest_rel(scores_b, scores_a)

        ci_low, ci_high = stats.t.interval(
            confidence=0.95,
            df=len(diff) - 1,
            loc=mean_diff,
            scale=stats.sem(diff),
        )

        std_diff = np.std(diff, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else np.inf

        significant = p_value < self.alpha

        # Effect size label
        ad = abs(cohens_d)
        if ad < 0.2:
            effect = "negligible"
        elif ad < 0.5:
            effect = "small"
        elif ad < 0.8:
            effect = "medium"
        elif ad < 2.0:
            effect = "large"
        else:
            effect = "very large"

        print("\n" + "=" * 70)
        print(f"PAIRED T-TEST: {model_a} vs {model_b}")
        print("=" * 70)

        print(f"\n{model_a} scores: {scores_a}")
        print(f"{model_b} scores: {scores_b}")

        print("\nStatistics:")
        print(f"  Mean diff ({model_b} − {model_a}): {mean_diff:.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.4e}")
        print(f"  95% CI:      [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"  Cohen's d:   {cohens_d:.4f} ({effect})")

        if significant:
            better = model_b if mean_diff > 0 else model_a
            print(f"\n✓ Significant (p < {self.alpha}) → {better} better")
        else:
            print(f"\n✗ Not significant (p ≥ {self.alpha})")

        return {
            "model_A": model_a,
            "model_B": model_b,
            "metric": metric,
            "mean_difference": float(mean_diff),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "ci_95_lower": float(ci_low),
            "ci_95_upper": float(ci_high),
            "cohens_d": float(cohens_d),
            "effect_size": effect,
            "significant": bool(significant),
        }

    # -------------------------------------------------
    # Wilcoxon signed-rank
    # -------------------------------------------------
    def wilcoxon_test(
        self,
        scores_a: np.ndarray,
        scores_b: np.ndarray,
        model_a: str = "Model A",
        model_b: str = "Model B",
    ) -> Dict:
        """
        Non-parametric paired comparison.
        """

        scores_a = np.asarray(scores_a)
        scores_b = np.asarray(scores_b)

        stat, p_value = stats.wilcoxon(scores_b, scores_a)

        mean_diff = np.mean(scores_b - scores_a)
        significant = p_value < self.alpha # type: ignore

        print("\n" + "=" * 70)
        print(f"WILCOXON TEST: {model_a} vs {model_b}")
        print("=" * 70)

        print(f"\nStatistic: {stat:.4f}")
        print(f"p-value:   {p_value:.4e}")

        if significant:
            better = model_b if mean_diff > 0 else model_a
            print(f"\n✓ Significant → {better} better")
        else:
            print("\n✗ Not significant")

        return {
            "wilcoxon_statistic": float(stat), # type: ignore
            "wilcoxon_p_value": float(p_value), # type: ignore
            "significant": bool(significant),
        }

    # -------------------------------------------------
    # McNemar test
    # -------------------------------------------------
    def mcnemar_test(
        self,
        y_true: np.ndarray,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
        model_a: str = "Model A",
        model_b: str = "Model B",
    ) -> Dict:

        print("\n" + "=" * 70)
        print(f"MCNEMAR TEST: {model_a} vs {model_b}")
        print("=" * 70)

        a_correct = preds_a == y_true
        b_correct = preds_b == y_true

        both_correct = np.sum(a_correct & b_correct)
        both_wrong = np.sum(~a_correct & ~b_correct)
        a_only = np.sum(a_correct & ~b_correct)
        b_only = np.sum(~a_correct & b_correct)

        if (a_only + b_only) == 0:
            logger.warning("No disagreements between models")
            return {"p_value": 1.0}

        chi2 = (a_only - b_only) ** 2 / (a_only + b_only)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        print("\nContingency:")
        print(f"{model_a} only correct: {a_only}")
        print(f"{model_b} only correct: {b_only}")

        print(f"\nχ² = {chi2:.4f}")
        print(f"p = {p_value:.4e}")

        if p_value < self.alpha:
            print("✓ Significant prediction difference")
        else:
            print("✗ No significant difference")

        return {
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "model_a_only_correct": int(a_only),
            "model_b_only_correct": int(b_only),
        }

    # -------------------------------------------------
    # Bootstrap CI
    # -------------------------------------------------
    def bootstrap_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_iterations: int = 10000,
        confidence: float = 0.95,
        random_state: int = 42,
    ) -> Dict:

        print("\n" + "=" * 70)
        print(f"BOOTSTRAP CI ({n_iterations} iterations)")
        print("=" * 70)

        rng = np.random.default_rng(random_state)
        n = len(y_true)

        acc_vals = []

        for _ in range(n_iterations):
            idx = rng.choice(n, n, replace=True)
            acc_vals.append(np.mean(y_true[idx] == y_pred[idx]))

        acc_vals = np.array(acc_vals)

        alpha = (1 - confidence) / 2
        ci_low = np.percentile(acc_vals, alpha * 100)
        ci_high = np.percentile(acc_vals, (1 - alpha) * 100)

        mean = np.mean(acc_vals)
        std = np.std(acc_vals)

        print(f"\nAccuracy mean: {mean:.4f}")
        print(f"Std: {std:.4f}")
        print(f"{confidence*100:.0f}% CI: [{ci_low:.4f}, {ci_high:.4f}]")

        return {
            "accuracy_mean": float(mean),
            "accuracy_std": float(std),
            "ci_lower": float(ci_low),
            "ci_upper": float(ci_high),
        }