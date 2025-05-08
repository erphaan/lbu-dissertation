import pandas as pd
from scipy.stats import friedmanchisquare

# Load 10-fold accuracy and F1-score CSVs (replace paths if needed)
df_acc = pd.read_csv("10 Fold results/model_accuracy_10fold.csv")
df_f1 = pd.read_csv("10 Fold results/model_f1_10fold.csv")

# Ensure only the 10 numeric folds are included (no 'Average' row)
df_acc = df_acc.iloc[:10]
df_f1 = df_f1.iloc[:10]

# Run Friedman test on Accuracy scores across all 5 models
friedman_acc = friedmanchisquare(
    df_acc['Logistic'],
    df_acc['SVM'],
    df_acc['LSTM'],
    df_acc['BERT'],
    df_acc['RoBERTa']
)

# Run Friedman test on F1 scores across all 5 models
friedman_f1 = friedmanchisquare(
    df_f1['Logistic'],
    df_f1['SVM'],
    df_f1['LSTM'],
    df_f1['BERT'],
    df_f1['RoBERTa']
)

# Output results
print("Friedman Test on Accuracy:")
print(f"  χ² = {friedman_acc.statistic:.4f}, p = {friedman_acc.pvalue:.6f}")
if friedman_acc.pvalue < 0.05:
    print("Statistically significant difference in Accuracy.")

print("\nFriedman Test on F1 Score:")
print(f"  χ² = {friedman_f1.statistic:.4f}, p = {friedman_f1.pvalue:.6f}")
if friedman_f1.pvalue < 0.05:
    print("Statistically significant difference in F1 Scores.")
