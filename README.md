# Natural language processing course 2023/24: `LLM Prompt Strategies for Commonsense-Reasoning Task`

### Authors:
- Matic Pristavnik Vrešnjak
- Mitja Kocjančič
- Songeun Hong

### Advisors:
- Slavko Žitnik

### Project File Structure
```
root/
│
└── report/
    ├── code/                           # Directory containing project code
    │   ├── promt_COT.py.py
    │   ├── few_shot.py
    │   └── ...                         # You can see the more files in the folder
    │
    └── Reports_compiled_submission/    # Directory containing PDF reports
        ├── submisson1_report.pdf
        ├── submisson2_report.pdf
        └── submisson3_report.pdf
```

### Overview
This project explores various prompting strategies to enhance the performance of Large Language Models (LLMs) on commonsense reasoning tasks. With the increasing use of LLMs in personal and commercial domains, developing effective prompts is crucial for generating relevant and informative responses. This study provides a comprehensive comparison of different prompting strategies to determine their effectiveness and applicability.

### Introduction
The surge in popularity of LLMs like ChatGPT, PaLM, and Gemini has led to their widespread use. These models, often based on the transformer architecture, are capable of handling tasks that require commonsense reasoning—using everyday knowledge to solve problems. Several prompting strategies have been developed to improve LLM performance on these tasks.

### Prompting Strategies
1. **Zero-shot**
   - Relies on the model being sufficiently trained with minimal prompt modifications.
   - *Example:* "Classify the text into neutral, negative, or positive. Text: I think the vacation is okay. Sentiment:"

2. **Few-shot**
   - Enhances the prompt by including examples of previously solved problems.
   - *Example:* "This is awesome! //Negative This is bad! //Positive Wow that movie was rad! //Positive What a horrible show! //"

3. **Chain-of-thought**
   - Combines with few-shot, adding step-by-step reasoning to the examples.
   - *Example:* "The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1. A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False."

### Methodology
- Various models were evaluated, including Command (Cohere AI), Mistral (Mistral AI), LLAMA3 (Meta), and T5 (Google Research).
- Performance metrics included ROUGE, BLEU, and BERT Score.
- Datasets used: Winograd Schema Challenge, MultiArith, SQuAD, among others.

### Conclusion
This project offers valuable insights into the effectiveness of different prompting strategies for improving LLM performance on commonsense reasoning tasks. The comprehensive comparison highlights the potential benefits and limitations of each strategy, guiding future research and practical implementations.
