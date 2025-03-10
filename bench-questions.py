import os
import glob
import pandas as pd
import ast
import argparse
import torch
from typing import Dict, List
from alive_progress import alive_bar

from classes.vector_store import VectorStore
from classes.llm_inference import LLMinference
from prompt import *

parser = argparse.ArgumentParser(description="LLM inference with optional baseline mode.")
parser.add_argument("-base", action="store_true", help="Run in baseline mode.")
args = parser.parse_args()

# Set BASELINE based on the argument
BASELINE = args.base
PATH = "/home/cc/PHD/HealthBranches/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("##### BASELINE MODE #####\n" if BASELINE else "##### BENCHMARK MODE #####\n")

def check_results(search_directory: str, string_to_check: str, search_strings: List[str]) -> List[str]:
    # Get all matching files
    matching_files = glob.glob(os.path.join(search_directory, string_to_check))

    # Find matching files and remove matched strings
    found_files = [file for file in matching_files if any(name in os.path.basename(file) for name in search_strings)]
    remaining_strings = [name for name in search_strings if not any(name in os.path.basename(file) for file in matching_files)]

    print("Remaining models to run: ", remaining_strings)

    return remaining_strings

# Create an empty vector store in the indicated path. If the path already exists, load the vector store
vector_store = VectorStore(f'{PATH}indexes/kgbase-new/')

# Add documents in vector store (comment this line after the first add)
# vector_store.add_documents('/home/cc/PHD/ragkg/data/kgbase')

folder_path = f"{PATH}questions_pro/ultimate_questions_v3_full_balanced.csv"
questions = pd.read_csv(folder_path)

models = ["mistral", "llama3.1:8b", "llama2:7b", "gemma:7b", "gemma2:9b", "qwen2.5:7b", "phi4:14b"]
models = check_results(PATH+"results/", "results_open_baseline_*.csv" if BASELINE else "results_open_*.csv", models)

templates = [PROMPT_OPEN, PROMPT_OPEN_RAG]

if BASELINE:
    templates = [PROMPT_OPEN_BASELINE]

cnt_rag = 0
cnt = 0

rows = []
questions = pd.read_csv(folder_path)

for model_name in models:
    llm = LLMinference(llm_name=model_name)

    cnt = 0
    rows = []
    print(f"Running model {model_name}...")
    with alive_bar(len(questions)) as bar:
        for index, row in questions.iterrows():
            res = []
            try:
                opts = ast.literal_eval(row['options'].replace("['", '["').replace("']", '"]').replace("', '", '", "'))
                
                if not isinstance(opts, list) or len(opts) != 5:
                    print(f"Skipping row {index} due to invalid options")
                    continue  # Skip this iteration if the condition is not met

            except (ValueError, SyntaxError):
                print(f"Skipping row {index} due to value/syntax error")
                continue  # Skip if there's an issue with evaluation

            txt_name = row['condition'].upper()+".txt"
            txt_folder_name = f"{PATH}data/kgbase-new/"

            try:
                with open(os.path.join(txt_folder_name, txt_name), 'r') as file:
                    text = file.readlines()
            except Exception:
                print(os.path.join(txt_folder_name, txt_name))
                print(f"{txt_name} text is EMPTY!")
                continue    
            
            for template in templates:
                if BASELINE:
                    res.append(llm.qea_evaluation(row['question'], template, row['path'], text, [], row['condition'].lower(), vector_store)) # Baseline
                else:
                    res.append(llm.qea_evaluation(row['question'], template, "", "", [], row['condition'].lower(), vector_store))

            res.append(row["answer"])
            res.append(row['question'])
            res.append(row['path'])
            res.insert(0, row['condition'].lower())

            rows.append(res)
            bar()

    if BASELINE:
        df = pd.DataFrame(rows, columns=["name", "zero_shot", "real", "question", "path"]) # Baseline
        df.to_csv(f"{PATH}/results/results_open_baseline_{model_name}.csv", index=False) # Baseline
    else:
        df = pd.DataFrame(rows, columns=["name", "zero_shot", "zero_shot_rag", "real", "question", "path"])
        df.to_csv(f"{PATH}/results/results_open_{model_name}.csv", index=False)

    print(f"Model {model_name} done!\n")