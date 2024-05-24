import os
import time
import random
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
from huggingface_hub import login
from datasets import load_dataset
import time

def load_model_huggingface(model_name):
  """
  Load a Hugging Face model with specific quantization settings.

  Parameters:
      model_name (str): The name of the model as listed on Hugging Face. Example:

  Returns:
      tokenizer (AutoTokenizer): Tokenizer for the specified model.
      model (AutoModelForCausalLM): Quantized model loaded from Hugging Face.
  """

  # Load tokenizer and base model
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  #model = AutoModel.from_pretrained(model_name, token="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")

  # Define quantization configuration
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Load model weights in 4-bit
    bnb_4bit_quant_type="nf4",  # Use nf4 quantization type
    bnb_4bit_use_double_quant=True,  # Enable nested quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computation
    token="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",
  )

  # Load quantized model
  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    token="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",
    low_cpu_mem_usage=True
  )

  # Disable caching in model configuration
  model.config.use_cache = False

  # Update tokenizer settings
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token

  return tokenizer, model

def load_dataset_huggingface(dataset_name):
  """
  Load a Hugging Face data

  Parameters:
      dataset_name (str): The name of the model as listed on Hugging Face. Example:

  Returns:
      dataset (dataset): Dataset loaded from Hugging Face.
  """
  dataset = load_dataset(dataset_name)
  return dataset

#model = 'mistralai/Mistral-7B-Instruct-v0.2'
model = 'meta-llama/Meta-Llama-3-8B'
#model = "tiiuae/falcon-11B"
login(token = "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")

# Use this functon if you allready downloaded model
print("Getting model")
tokenizer,model = load_model_huggingface(model)
dataset_squad = load_dataset("rajpurkar/squad")
from transformers import pipeline



generator = pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.001,
    max_new_tokens=256,
    repetition_penalty=1.1
)

with open("prompts_squad_llama3_COT.txt", "w") as file:
    for i in range(100):
        text = dataset_squad["train"]["question"][i]
        COT = """
CoT Prompts for SQuAD Examples

Question: What is the capital of France?

CoT Prompt: Read the passage carefully. Identify the sentence that directly states the capital of France. Write down your answer based on this sentence.


Passage:  Whales are mammals, not fish. They breathe air through blowholes located on top of their heads. Whales are warm-blooded animals and give birth to live young.

Question:  How do whales breathe?

CoT Prompt: Read the passage and focus on the characteristics of whales. Identify the sentence that explains how whales take in oxygen. Briefly explain why this sentence tells you how they breathe. Write your answer based on this information.


Passage: The Great Wall of China is a series of fortifications made of stone, brick, wood, and earth. It was built over centuries to protect the Chinese Empire from invaders. The wall snakes across mountains, deserts, and grasslands, with watchtowers and barracks positioned along its length.

Question:  What was the Great Wall of China built for?

CoT Prompt: Read the passage carefully. Find the sentences that describe the purpose and function of the Great Wall. Briefly explain how these sentences together tell you why the wall was built. Write your answer by summarizing this information.


Passage:  The Amazon rainforest is home to ten percent of the world's known species. It is a vast ecosystem with a variety of plants and animals, many of which are  still undiscovered.  The rainforest plays a crucial role in regulating the Earth's climate.

Question:  What percentage of the world's known species live in the Amazon rainforest?

CoT Prompt:  Read the passage and find the sentence mentioning the percentage of known species found in the Amazon rainforest.  Write down the specific number  mentioned in this sentence as your answer.


Passage:  William Shakespeare was a famous playwright from England. He wrote many classic plays, including Hamlet, Romeo and Juliet, and Macbeth. His works are still performed around the world today.

Question:  When was William Shakespeare born?

CoT Prompt: Read the passage carefully. While the passage doesn't mention Shakespeare's birthdate directly, see if there are any clues  that could help you determine the time period he lived in.  Write down your answer and explain why the passage doesn't provide a specific birthdate.
                  
          """
        prompt = f"""Passage:{dataset_squad["train"]["context"][i]}
            Question: {text}
        Answer:
        """
        prompt = COT+prompt
        start = time.time()
        res = generator(prompt)
        file.write("Question number "+str(i))
        file.write("Prompt\n")
        file.write(prompt+"\n")
        file.write("Generated output\n")
        file.write(res[0]["generated_text"][len(prompt):]+"\n")
        end = time.time()
        file.write("Time need to generate prompt "+str(end - start)+"\n")
        