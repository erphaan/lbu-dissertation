import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

# Load and prepare dataset
df = pd.read_csv("sentiment_results.csv", encoding="latin1")
df = df[['processed_text', 'sentiment_label']].dropna()
df['label'] = df['sentiment_label'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2})

dataset = Dataset.from_pandas(df[['processed_text', 'label']])
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize(batch):
    return tokenizer(batch["processed_text"], padding="max_length", truncation=True, max_length=128)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
X = df['processed_text'].tolist()
y = df['label'].tolist()

results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n Fold {fold}/10")

    train_data = dataset.select(train_idx).map(tokenize, batched=True)
    test_data = dataset.select(test_idx).map(tokenize, batched=True)

    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

    training_args = TrainingArguments(
        output_dir=f"./results_roberta/fold_{fold}",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir=f"./logs_roberta/fold_{fold}",
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()
    preds = trainer.predict(test_data)
    pred_labels = np.argmax(preds.predictions, axis=1)

    acc = accuracy_score(test_data['label'], pred_labels)
    f1 = f1_score(test_data['label'], pred_labels, average='macro')

    results.append({'Fold': fold, 'Accuracy': acc, 'F1_Score': f1})

results.append({
    'Fold': 'Average',
    'Accuracy': np.mean([r['Accuracy'] for r in results if isinstance(r['Fold'], int)]),
    'F1_Score': np.mean([r['F1_Score'] for r in results if isinstance(r['Fold'], int)])
})

pd.DataFrame(results).to_csv("roberta_10fold_results.csv", index=False)
print("\n Results saved to 'roberta_10fold_results.csv'")
