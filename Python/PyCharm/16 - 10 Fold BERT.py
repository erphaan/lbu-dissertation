import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch

# Load and prepare dataset
df = pd.read_csv("sentiment_results.csv", encoding="latin1")
df = df[['processed_text', 'sentiment_label']].dropna()

# Encode sentiment labels
label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
df['label'] = df['sentiment_label'].map(label_map)

# Convert to Hugging Face Dataset
full_dataset = Dataset.from_pandas(df[['processed_text', 'label']])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(batch):
    return tokenizer(batch["processed_text"], padding="max_length", truncation=True, max_length=128)

# Set up 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
X = df['processed_text'].tolist()
y = df['label'].tolist()

results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n Fold {fold}/10")

    train_dataset = full_dataset.select(train_idx).map(tokenize_function, batched=True)
    test_dataset = full_dataset.select(test_idx).map(tokenize_function, batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    training_args = TrainingArguments(
        output_dir=f"./results/fold_{fold}",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir=f"./logs/fold_{fold}",
        save_strategy="no",
        logging_steps=10,
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_dataset["label"]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    results.append({'Fold': fold, 'Accuracy': acc, 'F1_Score': f1})

# Add average at the end
results.append({
    'Fold': 'Average',
    'Accuracy': np.mean([r['Accuracy'] for r in results if isinstance(r['Fold'], int)]),
    'F1_Score': np.mean([r['F1_Score'] for r in results if isinstance(r['Fold'], int)])
})

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("bert_10fold_results.csv", index=False)

print("\n Results saved to 'bert_10fold_results.csv'")
