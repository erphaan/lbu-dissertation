import pandas as pd
from scipy.stats import wilcoxon
import itertools

# Load 10-fold accuracy and F1-score results for all 5 models
df_acc = pd.read_csv("10 Fold results\model_accuracy_10fold.csv")
df_f1 = pd.read_csv("10 Fold results\model_f1_10fold.csv")

# Ensure you're only using the 10 actual folds (no average row)
df_acc = df_acc.iloc[:10]
df_f1 = df_f1.iloc[:10]

# List of model names (columns)
models = df_acc.columns.tolist()

# Generate all unique pairs of models
model_pairs = list(itertools.combinations(models, 2))

# Initialize lists to store Wilcoxon results
wilcoxon_acc_results = []
wilcoxon_f1_results = []

# Perform Wilcoxon Signed-Rank Test on Accuracy
for model1, model2 in model_pairs:
    stat, p = wilcoxon(df_acc[model1], df_acc[model2])
    wilcoxon_acc_results.append({
        'Model 1': model1,
        'Model 2': model2,
        'Statistic': stat,
        'p-value': p
    })

# Perform Wilcoxon Signed-Rank Test on F1-score
for model1, model2 in model_pairs:
    stat, p = wilcoxon(df_f1[model1], df_f1[model2])
    wilcoxon_f1_results.append({
        'Model 1': model1,
        'Model 2': model2,
        'Statistic': stat,
        'p-value': p
    })

# Convert to DataFrames
df_wilcoxon_acc = pd.DataFrame(wilcoxon_acc_results)
df_wilcoxon_f1 = pd.DataFrame(wilcoxon_f1_results)

# Save to CSV
df_wilcoxon_acc.to_csv("wilcoxon_accuracy_results.csv", index=False)
df_wilcoxon_f1.to_csv("wilcoxon_f1_results.csv", index=False)

# Print summary
print("Wilcoxon tests completed and saved to CSV.")