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

#model = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
model = 'mistralai/Mistral-7B-Instruct-v0.2'
login(token = "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")

# Use this functon if you allready downloaded model
print("Getting model")
tokenizer,model = load_model_huggingface(model)
dataset_math = load_dataset("lighteval/MATH")
dataset = load_dataset("winograd_wsc","wsc285")
dataset_matharith = load_dataset("ChilleD/MultiArith")
from transformers import pipeline



generator = pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    max_new_tokens=300,
    do_sample=True,
    top_p=0.8,
    no_repeat_ngram_size=3,
    early_stopping=True
)

base_promts = []
allready_used = set([])
with open("prompts_multArith_minstral_COT.txt", "w") as file:
    for i in range(200):
        print("Question number",i)
        COT = """ Q: In a botanical garden, there are 80 plants being tended by gardeners. If 40 of them required special care due to pests, and the remaining plants were divided into 8 sections for different treatments, how many plants would be in each section?
A:5
Steps:

    Identify Total Entities: In this scenario, the entities are the plants in the botanical garden.

    Determine Action: The action described is the gardeners tending to the plants.

    Recognize Outcome or Observation: Some of the plants required special care due to pests.

    Calculate Remaining Plants: Subtract the plants requiring special care from the total to find out how many plants were left to be divided: 80−40=4080−40=40 plants.

    Determine Number of Sections: The remaining 40 plants were divided into 8 sections for different treatments.

    Calculate Plants per Section: To find out how many plants are in each section, divide the total remaining plants by the number of sections: 408=5840​=5 plants.

So, there would be 5 plants in each section.

Q: A farmer has 120 eggs from their chickens. If 60 of them were set aside for hatching, and the remaining eggs were packed into cartons with 6 eggs each, how many cartons were filled?
A:10
Steps:

Identify Total Entities: In this scenario, the entities are the eggs from the farmer's chickens.

Determine Action: The action described is the farmer setting aside some eggs for hatching and packing the remaining eggs into cartons.

Recognize Outcome or Observation: Some of the eggs were set aside for hatching.

Calculate Remaining Eggs: Subtract the eggs set aside for hatching from the total to find out how many eggs were left to be packed: 120−60=60120−60=60 eggs.

Determine Number of Cartons: The remaining 60 eggs were packed into cartons with 6 eggs each.

Calculate Cartons Filled: To find out how many cartons were filled, divide the total remaining eggs by the number of eggs per carton: 606=10660​=10 cartons.

So, 10 cartons were filled with eggs.

Q: In a bookstore, there are 96 books on a shelf. If 48 of them are fiction novels, and the rest are divided equally between history and science books, how many books are there in each category?
A: 24
Steps:

Identify Total Entities: In this scenario, the entities are the books on the shelf.

Determine Categorization: The books are categorized into fiction, history, and science.

Recognize Distribution: Some books are identified as fiction, leaving the rest to be distributed between history and science.

Calculate Remaining Books: Subtract the fiction novels from the total to find out how many books are left to be divided: 96−48=4896−48=48 books.

Determine Equal Distribution: The remaining 48 books are divided equally between history and science categories.

Calculate Books per Category: To find out how many books are in each category, divide the total remaining books by the number of categories: 482=24248​=24 books
        
"""
        prompt = dataset_matharith["train"]["question"][i]
        prompt = COT+"\n"+prompt
        start = time.time()
        res = generator(prompt)
        end = time.time()
        file.write("Question number "+str(i))
        file.write("Prompt\n")
        file.write(prompt+"\n")
        file.write("Generated output\n")
        file.write(res[0]["generated_text"][len(prompt):]+"\n")
        file.write("Time need to generate prompt "+str(end - start)+"\n")
        