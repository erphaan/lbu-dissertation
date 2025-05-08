# Install required libraries (if not installed)
#!pip install transformers datasets torch scikit - learn

# Import necessary libraries
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("F:\Personal Files\Student LBU\OneDrive - Leeds Beckett University\MSc Dissertation\Python\sentiment_results.csv", encoding="latin1")

# Ensure text column exists and handle missing values
# If 'processed_text' column doesn't exist, create it and fill NaN with empty strings
if 'processed_text' not in df.columns:
    df['processed_text'] = df['text'].fillna("")  # Use raw text if preprocessing was lost
# If 'processed_text' column exists, fill NaN with empty strings
else:
    df['processed_text'] = df['processed_text'].fillna("")

# Encode sentiment labels
label_encoder = LabelEncoder()
df['sentiment_label'] = df['sentiment_label'].fillna("Neutral")  # Handle missing labels
df['label'] = label_encoder.fit_transform(df['sentiment_label'])

# Split dataset into training & testing
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['processed_text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# Load pre-trained BERT tokenizer
model_name = "distilbert-base-uncased"  # or "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize text for BERT
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict(
    {"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"],
     "labels": train_labels})
test_dataset = Dataset.from_dict(
    {"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"],
     "labels": test_labels})

# Load pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

# Train the model using Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("🔹 BERT Sentiment Classification Results:")
print(results)

# Save trained model
model.save_pretrained("./bert_sentiment_model")
tokenizer.save_pretrained("./bert_sentiment_model")

from sklearn.metrics import accuracy_score, classification_report

# Get BERT model predictions
predictions = trainer.predict(test_dataset)
y_pred_bert = predictions.predictions.argmax(axis=1)

# Compute Accuracy and Classification Report
bert_accuracy = accuracy_score(test_labels, y_pred_bert)
bert_report = classification_report(test_labels, y_pred_bert, target_names=label_encoder.classes_)

# Print the results
print(f"🔹 BERT Sentiment Classification Accuracy: {bert_accuracy:.4f}")
print("\n🔹 Classification Report:\n", bert_report)
