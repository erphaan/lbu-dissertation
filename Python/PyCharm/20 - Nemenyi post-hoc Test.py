import pandas as pd
import scikit_posthocs as sp

# Load the accuracy scores (10 folds for 5 models)
df = pd.read_csv("10 Fold results\model_accuracy_10fold.csv")  # or model_f1_10fold.csv

# Apply Nemenyi post-hoc test
nemenyi_results = sp.posthoc_nemenyi_friedman(df)

# Show the comparison matrix
print("\nNemenyi Test (p-values):")
print(nemenyi_results.round(4))