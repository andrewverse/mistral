from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate

csv_files = {
    "train": "data/train_data.csv",
    "test": "data/test_data.csv",
    "validation": "data/validation_data.csv",
}

# Load the dataset
dataset = load_dataset("csv", data_files=csv_files)

# Initialize Tokenizer and Model for Mistral-7B
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1") 
model = AutoModelForSequenceClassification.from_pretrained("mistralai/Mistral-7B-v0.1", num_labels=1)  # Adjust num_labels if needed

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["tweetText"], padding="max_length", truncation=True)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Creating datasets
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Training parameters
training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # Might need to be adjusted
    per_device_eval_batch_size=16,   # Might need to be adjusted
    num_train_epochs=3,
    weight_decay=0.01
)

# Metric for evaluation
metric = evaluate.load("accuracy")

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Trainer
trainer = Trainer(
    model=model,
    # args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
