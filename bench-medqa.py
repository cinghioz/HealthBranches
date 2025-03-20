import os
import pandas as pd
import ast
import argparse
from typing import Dict, List
from alive_progress import alive_bar

from classes.vector_store import VectorStore
from classes.llm_inference import LLMinference
from classes.utils import check_results
from prompt import *
import json

parser = argparse.ArgumentParser(description="LLM inference with optional quiz mode.")
parser.add_argument("-quiz", action="store_true", help="Run in quiz mode.")
args = parser.parse_args()

QUIZ = args.quiz
PATH = "/home/cc/PHD/HealthBranches/"
EXT = "QUIZ" if QUIZ else "OPEN"

print("##### QUIZ EXP #####\n" if QUIZ else "##### OPEN EXP #####\n")

# Create an empty vector store in the indicated path. If the path already exists, load the vector store
vector_store = VectorStore(f'{PATH}indexes/kgbase-new/')

# Add documents in vector store (comment this line after the first add)
# vector_store.add_documents('/home/cc/PHD/ragkg/data/kgbase')


folder_path = f"{PATH}MedQA/test_with_paths/top_100_test_all_similarity.jsonl"
# folder_path = f"{PATH}MedQA/test_with_index/"

models = ["mistral:7b", "llama3.1:8b", "llama2:7b", "gemma:7b", "gemma2:9b", "qwen2.5:7b", "phi4-mini:3.8b", "gemma3:4b"]
models = check_results(PATH+"results-medqa/", f"results_{EXT}_medqa_*.csv", models)

templates = [PROMPT_QUIZ_MEDQA, PROMPT_QUIZ_RAG_MEDQA] if QUIZ else [PROMPT_OPEN, PROMPT_OPEN_RAG]

cnt_rag = 0
cnt = 0

rows = []
questions = []

# for root, _, files in os.walk(folder_path):
#     for file in files:
#         if file.startswith("top_15_") and file.endswith(".jsonl"):
#             file_path = os.path.join(root, file)
#             with open(file_path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     questions.append(json.loads(line))

with open(folder_path, "r", encoding="utf-8") as f:
    for line in f:
        questions.append(json.loads(line))

for model_name in models:
    llm = LLMinference(llm_name=model_name)

    cnt = 0
    rows = []
    print(f"Running model {model_name}...")
    with alive_bar(len(questions)) as bar:
        for row in questions:
            res = []
            opts = []

            if QUIZ:
                try:
                    opts = list(questions[0]['options'].values())
                    
                    if not isinstance(opts, list) or len(opts) != 5:
                        print(f"Skipping row due to invalid options")
                        continue  # Skip this iteration if the condition is not met

                except (ValueError, SyntaxError):
                    print(f"Skipping row due to value/syntax error")
                    continue  # Skip if there's an issue with evaluation
            
            for template in templates:
                try:
                    res.append(llm.qea_evaluation(row['question'], template, "", "", opts, "", vector_store)) # Baseline
                except Exception:
                    print(row)

            if QUIZ:
                res.append(row["answer_idx"])
            else:
                res.append(row["answer"])

            res.append(row['question'])

            rows.append(res)
            bar()

        df = pd.DataFrame(rows, columns=["zero_shot", "zero_shot_rag", "real", "question"])
        df.to_csv(f"{PATH}/results-medqa/results_{EXT}_medqa_{model_name}.csv", index=False)

    print(f"Model {model_name} done!\n")