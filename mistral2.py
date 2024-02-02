import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

csv_files = {
    "train": "data/train_data.csv",
    "test": "data/test_data.csv",
    "validation": "data/validation_data.csv",
}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", load_in_4bit=True, torch_dtype=torch.float16, device_map="auto")

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Set pad token
tokenizer.pad_token = "!" # Choose an appropriate pad token for your dataset

# Dataset-specific configurations
CUTOFF_LEN = 256  # Adjust based on the length of your tweets



# Load your Twitter dataset
dataset = load_dataset('csv', data_files=csv_files)

# Assume your dataset has a 'tweetText' column
def process_tweet(tweet):
    # Here you define what you want to do with each tweet
    # For example, let's just prepare it for language modeling
    return "<s>" + tweet['tweetText'] + "</s>"

def tokenize_and_prepare(tweet):
    processed_tweet = process_tweet(tweet)
    return tokenizer(
        processed_tweet + tokenizer.eos_token,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding="max_length"
    )

# Process and tokenize the dataset

# Assuming dataset is a DatasetDict with splits 'train', 'test', 'validation'
for split in dataset.keys():  # This iterates over each split
    processed_data = dataset[split].map(lambda x: tokenize_and_prepare(x), remove_columns=['tweetText'])
    # Here, we're removing the 'tweetText' column after processing, assuming you've extracted and tokenized the data as needed.

# Training configuration
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=6,
    learning_rate=1e-4,
    logging_steps=2,
    optim="adamw_torch",
    save_strategy="epoch",
    output_dir="twitter-data-processing-output"
)

split_data = processed_data.train_test_split(test_size=0.1)  # This splits the dataset, for example, 90% train, 10% test

trainer = Trainer(
    model=model,
    train_dataset=split_data["train"], 
    eval_dataset=split_data["test"], 
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)


model.config.use_cache = False

# Start training
trainer.train()
