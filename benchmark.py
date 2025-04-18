import os
import pandas as pd
import ast
import argparse
from typing import Dict, List
from alive_progress import alive_bar

from classes.vector_store import VectorStore
from classes.llm_inference import LLMinference
from utils import check_results
from prompt import *

parser = argparse.ArgumentParser(description="LLM inference with different modalities.")
parser.add_argument("-base", action="store_true", help="Run in baseline mode.")
parser.add_argument("-quiz", action="store_true", help="Run in quiz mode.")
args = parser.parse_args()

# Set BASELINE based on the argument
BASELINE = args.base
QUIZ = args.quiz
PATH = "/home/cc/PHD/HealthBranches/" # Prendi da arg o altro
EXT = "QUIZ" if QUIZ else "OPEN"

print("##### BASELINE MODE #####\n" if BASELINE else "##### BENCHMARK MODE #####\n")
print("##### QUIZ EXP #####\n" if QUIZ else "##### OPEN EXP #####\n")

# Create an empty vector store in the indicated path. If the path already exists, load the vector store
vector_store = VectorStore(f'{PATH}indexes/kgbase/')

# Add documents in vector store (comment this line after the first add)
# vector_store.add_documents(f'{PATH}data/kgbase')

folder_path = f"{PATH}questions_pro/final_dataset.csv"
# folder_path = f"{PATH}questions_pro/dataset_updated.csv"

questions = pd.read_csv(folder_path)

models = ["mistral:7b", "gemma:7b", "gemma2:9b", "gemma3:4b", "llama3.1:8b",
           "qwen2.5:7b", "phi4:14b", "mistral-nemo:12b", "llama2:7b", "deepseek-r1:8b"]

models = check_results(PATH+"results/", f"results_{EXT}_baseline_*.csv" if BASELINE else f"results_{EXT}_bench_*.csv", models)

templates = [PROMPT_QUIZ, PROMPT_QUIZ_RAG] if QUIZ else [PROMPT_OPEN, PROMPT_OPEN_RAG]

if BASELINE:
    templates = [PROMPT_QUIZ_BASELINE_PATH, PROMPT_QUIZ_BASELINE_TEXT, PROMPT_QUIZ_BASELINE] if QUIZ \
                    else [PROMPT_OPEN_BASELINE_PATH,  PROMPT_OPEN_BASELINE_TEXT, PROMPT_OPEN_BASELINE]

cnt_rag = 0
cnt = 0

rows = []
questions = pd.read_csv(folder_path)

for model_name in models:
    if "deepseek" in model_name.lower():
        llm = LLMinference(llm_name=model_name, num_predict=1024)
    else:
        llm = LLMinference(llm_name=model_name)

    cnt = 0
    rows = []
    print(f"Running model {model_name}...")
    with alive_bar(len(questions)) as bar:
        for index, row in questions.iterrows():
            res = []
            opts = []

            try:
                opts = ast.literal_eval(row['options'].replace("['", '["').replace("']", '"]').replace("', '", '", "'))
                if not isinstance(opts, list) or len(opts) != 5:
                    print(f"Skipping row {index} due to invalid options")
                    continue  # Skip this iteration if the condition is not met

            except (ValueError, SyntaxError):
                print(f"Skipping row {index} due to value/syntax error")
                continue  # Skip if there's an issue with evaluation

            txt_name = row['condition'].upper()+".txt"
            txt_folder_name = f"{PATH}data/kgbase/"

            try:
                with open(os.path.join(txt_folder_name, txt_name), 'r') as file:
                    text = file.readlines()
            except Exception:
                print(os.path.join(txt_folder_name, txt_name))
                print(f"{txt_name} text is EMPTY!")
                continue    
            
            for template in templates:
                if BASELINE:
                    try:
                        res.append(llm.qea_evaluation(row['question'], template, row['path'], text, opts, row['condition'].lower(), vector_store)) # Baseline
                    except Exception:
                        print(row)
                else:
                    try:
                        res.append(llm.qea_evaluation(row['question'], template, "", "", opts, row['condition'].lower(), vector_store))
                    except Exception:
                        print(row)

            if QUIZ:
                res.append(row["correct_option"])
            else:
                res.append(opts[ord(row["correct_option"].strip().upper()) - ord('A')])

            res.append(row['question'])
            res.append(row['path'])
            res.insert(0, row['condition'].lower())

            rows.append(res)
            bar()

        if BASELINE:
            df = pd.DataFrame(rows, columns=["name", "zero_shot_path", "zero_shot_text", "zero_shot_all", 
                                             "real", "question", "path"]) # Baseline
            
            df.to_csv(f"{PATH}/results/results_{EXT}_baseline_{model_name.replace(":", "_")}.csv", index=False) # Baseline# Baseline only path
        else:
            df = pd.DataFrame(rows, columns=["name", "zero_shot", "zero_shot_rag", "real", "question", "path"])
            df.to_csv(f"{PATH}/results/results_{EXT}_bench_{model_name.replace(":", "_")}.csv", index=False)

    print(f"Model {model_name} done!\n")