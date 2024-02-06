from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM

from transformers import pipeline
from transformers import AutoTokenizer

config = PeftConfig.from_pretrained("giordanorogers/mistral-X-v2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = PeftModel.from_pretrained(model, "giordanorogers/mistral-X-v2")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, config=config, )

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

# Your actual prompt, appended after the examples
prompt = few_shot_examples + "\nWrite a concise tweet about huggingface, its features, emphasizing AI and models."

# Generating response with adjusted prompt and specifying max_length
responses = generator(prompt,
                      max_new_tokens=60,
                      num_return_sequences=1,  # Generate a single output
                      do_sample=True,  # Allow for temperature setting
                      temperature=0.8, # Predictable output
                      top_p=0.9,
                      num_beams=5,  # Enable beam search by setting num_beams > 1
                      early_stopping=True,  # Now relevant because we're using beam search
                      length_penalty=0.9) # Encourage concise output

# Print the generated text
for response in responses:
    print(response['generated_text'])
