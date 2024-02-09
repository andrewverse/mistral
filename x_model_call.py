import torch

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

from transformers import pipeline
from transformers import AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"
hf_finetuned_model = "giordanorogers/mistral-X-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"

config = PeftConfig.from_pretrained(hf_finetuned_model)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model = PeftModel.from_pretrained(model, hf_finetuned_model)

tokenizer = AutoTokenizer.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, config=config, device=0 if torch.cuda.is_available() else -1)
# Few-shot examples
few_shot_examples = """
"Food for thought:

The stuff people wrote on the internet from the 90s to the 2020s was the training data for all future AI content.

Most content created after the 2020s will likely just be a repetition of those original 30 years of data.

Crazy."

"New research can now track full body movements of multiple people using only WiFi.

A deep neural network has been developed that maps the phase and amplitude of WiFi signals to specific body regions.

We no longer need cameras to track body movements - just a WiFi signal."

"The mammoth of an AI project that everyone is forgetting about in light of chatGPT:

Google's PaLM (Pathways Language Model) claims to have 560 billion parameters, nearly 3x that of OpenAI's GPT-3 "
"""

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

clear_cuda_cache()  # Clear any cached memory to free up RAM and GPU memory


def generate_response(user_input):
    prompt = user_input + few_shot_examples
    responses = generator(prompt, max_new_tokens=60, num_return_sequences=1, do_sample=True, temperature=0.8, top_p=0.9, num_beams=5, early_stopping=True, length_penalty=0.9)
    for response in responses:
        print(response['generated_text'])
    clear_cuda_cache()

user_input = input("Enter your prompt: ")
generate_response(user_input)
