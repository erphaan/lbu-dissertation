import pandas as pd
from scipy.stats import f_oneway

# Load model accuracy and F1-score results
df_acc = pd.read_csv("10 Fold results\model_accuracy_10fold.csv")  # 10 rows for each model
df_f1 = pd.read_csv("10 Fold results\model_f1_10fold.csv")

# Ensure only 10-fold data is included
df_acc = df_acc.iloc[:10]
df_f1 = df_f1.iloc[:10]

# Run one-way ANOVA on Accuracy
anova_acc = f_oneway(
    df_acc['Logistic'],
    df_acc['SVM'],
    df_acc['LSTM'],
    df_acc['BERT'],
    df_acc['RoBERTa']
)

# Run one-way ANOVA on F1-score
anova_f1 = f_oneway(
    df_f1['Logistic'],
    df_f1['SVM'],
    df_f1['LSTM'],
    df_f1['BERT'],
    df_f1['RoBERTa']
)

# Print results
print("ANOVA on Accuracy:")
print(f"  F-statistic: {anova_acc.statistic:.4f}")
print(f"  p-value: {anova_acc.pvalue:.6f}")
if anova_acc.pvalue < 0.05:
    print("Significant difference in model accuracy.")

print("\nANOVA on F1-score:")
print(f"  F-statistic: {anova_f1.statistic:.4f}")
print(f"  p-value: {anova_f1.pvalue:.6f}")
if anova_f1.pvalue < 0.05:
    print("Significant difference in model F1-scores.")
