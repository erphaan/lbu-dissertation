import pandas as pd
import matplotlib.pyplot as plt

# Load your CSVs
df_acc = pd.read_csv("10 Fold results/model_accuracy_10fold.csv")
df_f1 = pd.read_csv("10 Fold results/model_f1_10fold.csv")

# Compute averages
mean_accuracy = df_acc.mean()
mean_f1 = df_f1.mean()

# Create DataFrame
comparison_df = pd.DataFrame({
    "Model": mean_accuracy.index,
    "Accuracy": mean_accuracy.values,
    "F1 Macro": mean_f1.values
}).sort_values(by="Accuracy", ascending=False)

# Plot
labels = comparison_df["Model"]
x = range(len(labels))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
plt.bar(x, comparison_df["Accuracy"], width=bar_width, label='Accuracy')
plt.bar([i + bar_width for i in x], comparison_df["F1 Macro"], width=bar_width, label='F1 Macro')

plt.xlabel("Model")
plt.ylabel("Score")
plt.title("10-Fold Comparison: Accuracy vs F1 Macro")
plt.xticks([i + bar_width / 2 for i in x], labels, rotation=45)
plt.ylim(0.8, 0.9)
plt.legend()
plt.tight_layout()
plt.savefig("10fold_model_comparison.png")
plt.show()
