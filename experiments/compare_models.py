# experiments/compare_models.py

import numpy as np
import json
import csv
from validation.statistical_tests import StatisticalTester

# 5-fold CV accuracies (%)
resnet_accs = np.array([81.37, 78.43, 83.33, 75.00, 79.41])
efficient_accs = np.array([85.49, 84.12, 82.91, 84.87, 88.02])

tester = StatisticalTester()

# -----------------------------
# Paired t-test
# baseline = ResNet
# improved = EfficientNet
# -----------------------------
t_results = tester.paired_t_test(
    baseline_scores=resnet_accs, # type: ignore
    improved_scores=efficient_accs # type: ignore
)

# -----------------------------
# Wilcoxon test (your class version)
# -----------------------------
w_results = tester.wilcoxon_test(
    baseline_scores=resnet_accs, # type: ignore
    improved_scores=efficient_accs # type: ignore
)

# -----------------------------
# Collect results
# -----------------------------
results = {
    "model_A": "ResNet50",
    "model_B": "EfficientNet-B0",
    "metric": "CV Accuracy (%)",
    "mean_diff": float(np.mean(efficient_accs - resnet_accs)),
    "t_test_p": float(t_results["p_value"]),
    "cohens_d": float(t_results["cohens_d"]),
    "wilcoxon_stat": float(w_results["statistic"]),
    "wilcoxon_p": float(w_results["p_value"]),
    "significant_t": bool(t_results["significant"]),
    "significant_w": bool(w_results["significant"])
}

# -----------------------------
# Save JSON
# -----------------------------
with open("results_resnet_vs_efficient.json", "w") as f:
    json.dump(results, f, indent=4)

# -----------------------------
# Save CSV
# -----------------------------
with open("results_resnet_vs_efficient.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results.keys())
    writer.writeheader()
    writer.writerow(results)

print("\nResults saved:")
print("  results_resnet_vs_efficient.json")
print("  results_resnet_vs_efficient.csv")