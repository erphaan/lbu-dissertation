# Install required libraries (if not installed)
#!pip install transformers datasets torch scikit-learn

# Import necessary libraries
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("sentiment_results.csv", encoding="latin1")

# Ensure the 'processed_text' column exists
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

# Load pre-trained tokenizer (Using RoBERTa for better performance)
model_name = "roberta-base"  # You can also try "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize text for BERT
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "labels": train_labels})
test_dataset = Dataset.from_dict({"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"], "labels": test_labels})

# Load fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Define fine-tuning arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,  # Increased batch size
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # Increased training epochs
    weight_decay=0.01,  # Regularization
    learning_rate=3e-5,  # Fine-tuned learning rate
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

# Train the model using Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Evaluate the fine-tuned model
results = trainer.evaluate()
print("🔹 Fine-Tuned BERT Sentiment Classification Results:")
print(results)

# Save the final model
model.save_pretrained("./fine_tuned_bert_model")
tokenizer.save_pretrained("./fine_tuned_bert_model")
