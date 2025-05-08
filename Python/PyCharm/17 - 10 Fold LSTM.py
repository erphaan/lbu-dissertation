import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load and prepare dataset
df = pd.read_csv("sentiment_results.csv", encoding="latin1")
df = df[['processed_text', 'sentiment_label']].dropna()
label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['label'] = df['sentiment_label'].map(label_map)

texts = df['processed_text'].tolist()
labels = df['label'].values

# Tokenize and pad sequences
max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)
y = np.array(labels)

# Cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n Fold {fold}/10")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = Sequential()
    model.add(Embedding(max_words, 128, input_length=max_len))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    acc = accuracy_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes, average='macro')
    results.append({'Fold': fold, 'Accuracy': acc, 'F1_Score': f1})

# Add average results
results.append({
    'Fold': 'Average',
    'Accuracy': np.mean([r['Accuracy'] for r in results if isinstance(r['Fold'], int)]),
    'F1_Score': np.mean([r['F1_Score'] for r in results if isinstance(r['Fold'], int)])
})

pd.DataFrame(results).to_csv("lstm_10fold_results.csv", index=False)
print("\n Results saved to 'lstm_10fold_results.csv'")
