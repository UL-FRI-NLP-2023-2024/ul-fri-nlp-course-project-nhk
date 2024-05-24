from bert_score import score
from datasets import load_dataset
from huggingface_hub import login
from colorama import Fore, Style
import os
import numpy as np
from rouge_score import rouge_scorer
import nltk
from nltk.metrics import scores
from bert_score import BERTScorer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu


def load_predictions(path):
    LLM_output = []
    text_output = []
    sum_time = 0.0
    output = ""
    question_num=0
    with open(path, "r",encoding="utf8") as file:
        new_output = 0
        for line in file:
            if "Generated output"==line.strip() or new_output==1:
                if(new_output!=1):
                    new_output = 1
                    question_num += 1
                text_output.append(line)
                if("Question number "+str(question_num)+"Prompt" == line.strip()):
                    for i in range(1,len(text_output)-2):
                        output+=text_output[i]
                    new_output = 0
                    sum_time+=float(text_output[len(text_output) - 2].strip().split(" ")[-1])
                    text_output = []
                    LLM_output.append(output)
                    output = ""


    return LLM_output, sum_time/len(LLM_output)

def generate_gt_winogrand(dataset):
    gt = []
    for i in range(len(dataset["test"]["text"])-1):
        sample = dataset["test"]["text"][i]
        options = dataset["test"]["options"][i]
        label = int(dataset["test"]["label"][i])
        prompt = f"""Classify the text into A or B based on pronoun.
        Text: {sample}
        Options:
        A) {options[0]}
        B) {options[1]}
        Answer:
        {options[label]}
        """
        prompt = f"""
        # {options[label]}
        # """
        gt.append(prompt)

    return gt


def generate_gt_multiArth(dataset):

    gt = []
    for i in range(199):
        prompt = "Q:"+dataset["train"]["question"][i] +"\nA:" + dataset["train"]["final_ans"][i]+"\n"
        gt.append(prompt)

    return gt

def eval_bertscore(pred,gt):
    P, R, F1 = score(pred, gt, lang='en')
    print("Bert score eval:")
    print("Precision: "+str(P.mean()))
    print("Recall: "+str((R.mean())))
    print("F1: "+str((F1.mean())))


def bleu(gen, ref):
    ref_bleu = []
    gen_bleu = []
    for l in gen:
        gen_bleu.append(l.split())
    for i,l in enumerate(ref):
        ref_bleu.append([l.split()])
    cc = SmoothingFunction()
    score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
    print("BLEU score: "+str(score_bleu))

def eval_rouge(pred,gt):
    scorer = rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)
    scores_R = []
    scores_P = []
    scores_F = []
    for pred,gt in zip(pred,gt):
        scores = scorer.score(gt,pred)
        scores_R.append(scores["rouge2"].recall)
        scores_P.append(scores["rouge2"].precision)
        scores_F.append(scores["rouge2"].fmeasure)

    print("Rouge")
    mean_R = sum(scores_R)/ (len(scores_R))
    mean_P = sum(scores_P) / len(scores_P)
    mean_F = sum(scores_F) / len(scores_F)
    print("Recall: "+str(mean_R))
    print("Precision: " + str(mean_P))
    print("F1: " + str(mean_F))

base_dir = "./Outputed_prompts/"
list = os.listdir(base_dir)
txt_files = filter(lambda x: x[-4:] == '.txt', list)
login(token="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")
dataset_win = load_dataset("winograd_wsc", "wsc285")
dataset_arth = load_dataset("ChilleD/MultiArith")
print("\n")
print("Starting eval\n")
dataset_to_eval = ["multArith"]
for file_path in list:
    pred, time = load_predictions(base_dir+file_path)
    dataset_name = file_path.split("_")[1]
    model_name = file_path.split("_")[2]
    start_name = file_path.split("_")[3]
    print(Fore.RED+f"{dataset_name} dataset ")
    print(Fore.GREEN+f"model {model_name}")
    print(Fore.YELLOW+f"Strategy {start_name}")
    print(Fore.WHITE + f"")
    print("_" * 50)

    if dataset_name == "multArith":
        gt = generate_gt_multiArth(dataset_arth)
    elif dataset_name == "winogrand":
        gt = generate_gt_winogrand(dataset_win)

    eval_rouge(pred,gt)

    bleu(pred,gt)

    eval_bertscore(pred,gt)

    print("Average time to generate prompt: "+str(time))

    print("_" * 50)