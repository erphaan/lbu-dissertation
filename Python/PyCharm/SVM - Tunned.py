# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('F:\Personal Files\Student LBU\OneDrive - Leeds Beckett University\MSc Dissertation\Python\sentiment_results.csv', encoding="latin1")

# Ensure text column exists and handle missing values
# If 'processed_text' column doesn't exist, create it and fill NaN with empty strings
if 'processed_text' not in df.columns:
    df['processed_text'] = df['text'].fillna("")  # Use raw text if preprocessing was lost
# If 'processed_text' column exists, fill NaN with empty strings
else:
    df['processed_text'] = df['processed_text'].fillna("")

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features for efficiency
X_tfidf = vectorizer.fit_transform(df['processed_text'])

# Encode sentiment labels
label_encoder = LabelEncoder()
df['sentiment_label'] = df['sentiment_label'].fillna("Neutral")  # Handle missing labels
y_encoded = label_encoder.fit_transform(df['sentiment_label'])

# Apply SMOTE to balance sentiment classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y_encoded)

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define hyperparameter grid for tuning SVM
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Different kernels
    'gamma': ['scale', 'auto']  # Kernel coefficient (only for 'rbf' and 'poly')
}

# Initialize GridSearchCV to find the best parameters
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters and best estimator
best_params = grid_search.best_params_
best_svm_model = grid_search.best_estimator_

# Make predictions with the optimized model
y_pred_best_svm = best_svm_model.predict(X_test)

# Evaluate the optimized model
best_svm_accuracy = accuracy_score(y_test, y_pred_best_svm)
best_svm_report = classification_report(y_test, y_pred_best_svm, target_names=label_encoder.classes_)

# Display best parameters and updated performance
print(f"🔹 Best SVM Parameters: {best_params}")
print(f"🔹 Optimized SVM Accuracy: {best_svm_accuracy:.4f}")
print("\n🔹 Updated Classification Report:\n", best_svm_report)
