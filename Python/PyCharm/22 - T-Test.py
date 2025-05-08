import pandas as pd
from scipy.stats import ttest_rel
import itertools

# Load the CSVs with 10-fold results
df_acc = pd.read_csv("10 Fold results\model_accuracy_10fold.csv")
df_f1 = pd.read_csv("10 Fold results\model_f1_10fold.csv")

# Ensure only the 10 folds are included (no average row)
df_acc = df_acc.iloc[:10]
df_f1 = df_f1.iloc[:10]

# Get model names
models = df_acc.columns.tolist()

# Generate all unique model pairs
model_pairs = list(itertools.combinations(models, 2))

# Store results
t_test_acc_results = []
t_test_f1_results = []

# Accuracy-based T-tests
for m1, m2 in model_pairs:
    stat, p = ttest_rel(df_acc[m1], df_acc[m2])
    t_test_acc_results.append({
        'Model 1': m1,
        'Model 2': m2,
        'T-Statistic': stat,
        'p-value': p
    })

# F1-score-based T-tests
for m1, m2 in model_pairs:
    stat, p = ttest_rel(df_f1[m1], df_f1[m2])
    t_test_f1_results.append({
        'Model 1': m1,
        'Model 2': m2,
        'T-Statistic': stat,
        'p-value': p
    })

# Save to CSV
pd.DataFrame(t_test_acc_results).to_csv("ttest_accuracy_results.csv", index=False)
pd.DataFrame(t_test_f1_results).to_csv("ttest_f1_results.csv", index=False)

print("Paired T-tests completed and results saved to:")
print(" - ttest_accuracy_results.csv")
print(" - ttest_f1_results.csv")
