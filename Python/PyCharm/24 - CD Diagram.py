import pandas as pd
import Orange.evaluation.graph_ranks as orange_graph
import matplotlib.pyplot as plt

# Load your results
df = pd.read_csv("10 Fold results/model_accuracy_10fold.csv")  # Or use F1-scores
df = df.iloc[:10]  # Ensure it's 10 folds

# Calculate average rank for each model (lower rank is better)
ranks = df.rank(axis=1, ascending=False)
avg_ranks = ranks.mean().to_dict()

# Prepare data for Orange3's CD plot
names = list(avg_ranks.keys())
rank_values = [avg_ranks[model] for model in names]

# Plot Critical Difference diagram
plt.figure(figsize=(10, 4))
orange_graph.graph_ranks(rank_values, names, cd=1.0, width=6)
plt.title("Critical Difference Diagram (Nemenyi Test)")
plt.tight_layout()
plt.savefig("cd_diagram_accuray.png")
plt.show()



# Load your results
df = pd.read_csv("10 Fold results/model_f1_10fold.csv")  # Or use F1-scores
df = df.iloc[:10]  # Ensure it's 10 folds

# Calculate average rank for each model (lower rank is better)
ranks = df.rank(axis=1, ascending=False)
avg_ranks = ranks.mean().to_dict()

# Prepare data for Orange3's CD plot
names = list(avg_ranks.keys())
rank_values = [avg_ranks[model] for model in names]

# Plot Critical Difference diagram
plt.figure(figsize=(10, 4))
orange_graph.graph_ranks(rank_values, names, cd=1.0, width=6)
plt.title("F1 Critical Difference Diagram (Nemenyi Test)")
plt.tight_layout()
plt.savefig("cd_diagram_f1.png")
plt.show()