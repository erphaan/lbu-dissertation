# Optimized BERT/RoBERTa sentiment classifier with Focal Loss and enhancements

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
import numpy as np

# Load dataset
df = pd.read_csv("sentiment_results.csv", encoding="latin1")
df['processed_text'] = df['processed_text'].fillna('')
df['sentiment_label'] = df['sentiment_label'].fillna("Neutral")

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['sentiment_label'])

# Stratified split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['processed_text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label']
)

# Tokenization
model_name = "bert-base-uncased"  # or use 'vinai/bertweet-base' or 'roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

# Prepare datasets with tensor conversion
train_dataset = Dataset.from_dict({
    "input_ids": torch.tensor(train_encodings["input_ids"]),
    "attention_mask": torch.tensor(train_encodings["attention_mask"]),
    "labels": torch.tensor(train_labels)
})
test_dataset = Dataset.from_dict({
    "input_ids": torch.tensor(test_encodings["input_ids"]),
    "attention_mask": torch.tensor(test_encodings["attention_mask"]),
    "labels": torch.tensor(test_labels)
})

# Compute class weights
classes = np.unique(train_labels)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_labels)
weights = torch.tensor(weights, dtype=torch.float)

# Define Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# Load model and use Focal Loss
class FocalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        self.focal_loss = FocalLoss(gamma=2.0, weight=weights)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        logits = outputs.logits
        loss = self.focal_loss(logits, labels) if labels is not None else None
        return {'loss': loss, 'logits': logits}

model = FocalModel()

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    fp16=True
)

# Evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_weighted': f1_score(labels, preds, average='weighted')
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train and evaluate
trainer.train()
results = trainer.evaluate()
print("\n🔹 Optimized BERT with Focal Loss Evaluation Results:")
print(results)

# Save model
model.model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
