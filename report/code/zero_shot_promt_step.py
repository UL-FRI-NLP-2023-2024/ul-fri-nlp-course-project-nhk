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

model = 'mistralai/Mistral-7B-Instruct-v0.2'
#model = 'mistralai/Mixtral-8x7B-v0.1'
login(token = "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")

# Use this functon if you allready downloaded model
print("Getting model")
tokenizer,model = load_model_huggingface(model)
dataset = load_dataset("winograd_wsc","wsc285")
from transformers import pipeline



generator = pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.001,
    max_new_tokens=256,
    repetition_penalty=1.1
)

with open("prompts_winogrand_minstral_zeroshot_step.txt", "w") as file:
    for i in range(len(dataset["test"]["text"])):
        
        text = dataset["test"]["text"][i]
        options = dataset["test"]["options"][i]
        prompt = f"""Classify the text into A or B based on pronoun. Solve the problem step by step please
    Text: {text}
    Options:
    A) {options[0]}
    B) {options[1]}
    Answer:
    """
        start = time.time()
        res = generator(prompt)
        end = time.time()
        file.write("Question number "+str(i))
        file.write("Prompt\n")
        file.write(prompt+"\n")
        file.write("Generated output\n")
        file.write(res[0]["generated_text"][len(prompt):]+"\n")
        file.write("Time need to generate prompt "+str(end - start)+" \n")
        