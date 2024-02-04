# Import necessary libraries
from datasets import load_dataset
import wandb, os
import torch
import matplotlib.pyplot as plt
import transformers
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

# Define paths to your dataset files
csv_files = {
    "train": "data/train_data.csv",
    "test": "data/test_data.csv",
    "validation": "data/validation_data.csv",
}

# Load the dataset from CSV files using the Hugging Face datasets library
dataset = load_dataset("csv", data_files=csv_files)

# Login to Weights & Biases for experiment tracking
wandb.login()

# Set the Weights & Biases project name
wandb_project = "x-finetune"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

# Define a formatting function for the dataset examples
def formatting_func(example):
    text = f"### The follow is a Tweet on the X platform: {example['tweetText']}"
    return text

# Specify the base model to be fine-tuned
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

# Function to tokenize dataset examples using the formatting function
def generate_and_tokenize_prompt(prompt):
    return tokenizer(formatting_func(prompt))

# Tokenize the training and validation datasets
tokenized_train_dataset = dataset["train"].map(generate_and_tokenize_prompt)
tokenized_val_dataset = dataset["validation"].map(generate_and_tokenize_prompt)

# Function to plot the distribution of sequence lengths in the dataset
def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
    lengths = [len(x['input_ids']) for x in tokenized_train_dataset] + [len(x['input_ids']) for x in tokenized_val_dataset]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    plt.xlabel('Length of input_ids')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lengths of input_ids')
    plt.show()
    return lengths

# Plot the distribution of sequence lengths and get lengths
lengths = plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)

# Calculate the 95th percentile of sequence lengths
max_length = int(np.percentile(lengths, 95))

print("Max Length:", max_length)

# Tokenize the prompts with truncation and padding to a maximum length
def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

# Re-tokenize the datasets with the new function
tokenized_train_dataset = dataset["train"].map(generate_and_tokenize_prompt2)
tokenized_eval_dataset = dataset["validation"].map(generate_and_tokenize_prompt2)

# Select a manageable subset for training and evaluation
def select_subset(dataset_split, num_samples=100):
    return dataset_split.shuffle(seed=42).select(range(min(num_samples, len(dataset_split))))

small_train_dataset = select_subset(tokenized_train_dataset, 100)
small_eval_dataset = select_subset(tokenized_eval_dataset, 100)


# Plot the distribution of sequence lengths again to see the effect of truncation
plot_data_lengths(small_train_dataset, small_eval_dataset)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Prepare the model for k-bit training, which further reduces memory usage
model = prepare_model_for_kbit_training(model)

# Configure LoRA (Low-Rank Adaptation) for efficient parameterization
config = LoraConfig(
    r=16, # was 32, adjusted for small dataset test run #change_me
    lora_alpha=32, # was 64, adjusted for small dataset test run #change_me
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.1, # was 0.05, adjusted for small dataset test run #change_me
    task_type="CAUSAL_LM",
)

# Apply the LoRA configuration to the model
model = get_peft_model(model, config)

# Function to print the number of trainable parameters
def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_param = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

# Print model parameters
print_trainable_parameters(model)

# Check for multiple GPUs and enable model parallelism if available
if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True
    
project = "x-finetune"
base_model_name = "mistral"
run_name = project + "-" + base_model_name
output_dir = "./" + run_name

# Initialize the Hugging Face Trainer with training arguments
trainer = transformers.Trainer(
    model=model,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=2.5e-5,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=25,
        evaluation_strategy="steps",
        eval_steps=25,
        do_eval=True,
        report_to="wandb",
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Start training
trainer.train()

# Re-load the base model with the same quantization settings for evaluation
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Initialize the tokenizer for evaluation
eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

# Prepare the prompt for evaluation
eval_prompt = "The following is a tweet from the X platform # "
model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

# Load the fine-tuned model checkpoint for generation
ft_model = PeftModel.from_pretrained(base_model, "mistral-tweets-finetune/checkpoint-300")

# Generate text based on the provided prompt
ft_model.eval()
with torch.no_grad():
    print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True))
