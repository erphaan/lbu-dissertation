from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("sentiment_results.csv", encoding="latin1")
df['processed_text'] = df['processed_text'].fillna('')
df['sentiment_label'] = df['sentiment_label'].fillna("Neutral")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentiment_label'])

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_text'])

# Models
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='rbf', C=10, gamma='scale')
}

# Results storage
results_acc = {model: [] for model in models}
results_f1 = {model: [] for model in models}

# 10-Fold CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X, y):
    X_train_raw, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train_raw, y_train)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results_acc[name].append(acc)
        results_f1[name].append(f1)

# Convert to DataFrame
df_acc = pd.DataFrame(results_acc)
df_f1 = pd.DataFrame(results_f1)

# Save to CSV
df_acc.to_csv("model_accuracy_10fold.csv", index=False)
df_f1.to_csv("model_f1_10fold.csv", index=False)

print("✅ Saved accuracy and F1-score results to CSV.")
