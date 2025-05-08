# Import libraries
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load your preprocessed dataset
df = pd.read_csv("balanced_sentiment_data.csv", encoding="latin1")
df = df[['processed_text', 'sentiment_label']].dropna()

# Encode sentiment labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['sentiment_label'])

# Split into training and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['processed_text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# Load RoBERTa tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the text
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Convert to HuggingFace Datasets
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# Load pre-trained RoBERTa model for classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluate
predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(axis=1)

model.save_pretrained("roberta_sentiment_model")
tokenizer.save_pretrained("roberta_sentiment_model")

# Print results
acc = accuracy_score(test_labels, y_pred)
print(f"\n🔹 RoBERTa Accuracy: {acc:.4f}")
print("\n🔹 Classification Report:\n")
print(classification_report(test_labels, y_pred, target_names=label_encoder.classes_))