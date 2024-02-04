# Import necessary libraries
from datasets import load_dataset
import wandb, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Load the dataset
dataset = load_dataset("csv", data_files={
    "train": "data/train_data.csv",
    "validation": "data/validation_data.csv",
})

# Login to Weights & Biases for experiment tracking
wandb.login()

# Set the Weights & Biases project name
wandb_project = "x-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

# Base model and tokenizer initialization
base_model_id = "mistralai/Mistral-7B-v0.1"

# Configuration for BitsAndBytes to enable efficient quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the pre-trained model with quantization config
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto")

# Initialize the tokenizer with specific settings for padding and special tokens
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["tweetText"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Prepare the model for k-bit training and apply LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "lm_head"],
    bias="none", lora_dropout=0.1, task_type="CAUSAL_LM"))

# Trainer setup
training_args = TrainingArguments(
    output_dir="./results",
    logging_dir="./logs",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    learning_rate=2e-05,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Start training
trainer.train()