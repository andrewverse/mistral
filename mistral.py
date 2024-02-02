import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate


# Define file paths for your dataset
csv_files = {
    "train": "data/train_data.csv",
    "test": "data/test_data.csv",
    "validation": "data/validation_data.csv",
}

# Load the dataset from CSV files
dataset = load_dataset("csv", data_files=csv_files)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Load the model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16)

# Tokenization function for processing the text
def tokenize_function(examples):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(examples["tweetText"], padding="max_length", truncation=True, max_length=64)

# Apply tokenization to each dataset split
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Select a manageable subset for training and evaluation
def select_subset(dataset_split, num_samples=100):
    return dataset_split.shuffle(seed=42).select(range(min(num_samples, len(dataset_split))))

small_train_dataset = select_subset(tokenized_datasets["train"], 100)
small_eval_dataset = select_subset(tokenized_datasets["validation"], 100)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    fp16=True,
    gradient_accumulation_steps=4,
)

# Define a dummy metric for demonstration purposes
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"dummy_metric": float(np.mean(predictions == labels))}

# Initialize the Trainer for training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
