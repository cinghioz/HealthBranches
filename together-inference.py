import os
import pandas as pd
import numpy as np
import argparse
import ast
from together import Together
from alive_progress import alive_bar

from classes.vector_store import VectorStore
from classes.utils import check_results
from prompt import *

client = Together(api_key="de1d1c231987694e2b9111e06e048732d206ecaee729b8aee41e2121006f2cfc")

parser = argparse.ArgumentParser(description="LLM inference with optional baseline mode.")
parser.add_argument("-base", action="store_true", help="Run in baseline mode.")
parser.add_argument("-quiz", action="store_true", help="Run in baseline mode.")
args = parser.parse_args()

# Set BASELINE based on the argument
BASELINE = args.base
QUIZ = args.quiz
PATH = "/home/cc/PHD/HealthBranches/"
EXT = "QUIZ" if QUIZ else "OPEN"

print("##### BASELINE MODE #####\n" if BASELINE else "##### BENCHMARK MODE #####\n")
print("##### QUIZ EXP #####\n" if QUIZ else "##### OPEN EXP #####\n")

def together_inference(model, query, template, path, text, choices, cond, vector_store):
    context_text = vector_store.search(query=query, k=3)

    if choices: # quiz
        if path != "" and text != "":
            prompt = template.format(context=context_text, question=query, path=path, text=text, condition=cond, o1=choices[0], o2=choices[1], o3=choices[2], o4=choices[3], o5=choices[4])
        else:
            prompt = template.format(context=context_text, question=query, condition=cond, o1=choices[0], o2=choices[1], o3=choices[2], o4=choices[3], o5=choices[4])
    else: # open question
        if path != "" and text != "":
            prompt = template.format(context=context_text, question=query, path=path, text=text, condition=cond)
        else:
            prompt = template.format(context=context_text, question=query, condition=cond)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content

# Create an empty vector store in the indicated path. If the path already exists, load the vector store
vector_store = VectorStore(f'{PATH}indexes/kgbase-new/')

folder_path = f"{PATH}questions_pro/ultimate_questions_v3_full_balanced.csv"
questions = pd.read_csv(folder_path)

templates = [PROMPT_QUIZ, PROMPT_QUIZ_RAG] if QUIZ else [PROMPT_OPEN, PROMPT_OPEN_RAG]

if BASELINE:
    templates = [PROMPT_QUIZ_BASELINE] if QUIZ else [PROMPT_OPEN_BASELINE]

cnt_rag = 0
cnt = 0

rows = []
questions = pd.read_csv(folder_path)

models = ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"]
models = check_results(PATH+"results/", f"results_{EXT}_baseline_*.csv" if BASELINE else f"results_{EXT}_*.csv", models)

for model in models:
    with alive_bar(len(questions)) as bar:
        for index, row in questions.iterrows():
            res = []
            opts = []

            if QUIZ:
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
                    res.append(together_inference(model, row['question'], template, row['path'], text, opts, row['condition'].lower(), vector_store)) # Baseline
                else:
                    res.append(together_inference(model, row['question'], template, "", "", opts, row['condition'].lower(), vector_store))

            if QUIZ:
                res.append(row["correct_option"])
            else:
                res.append(row["answer"])

            res.append(row['question'])
            res.append(row['path'])
            res.insert(0, row['condition'].lower())

            rows.append(res)
            bar()

        if BASELINE:
            df = pd.DataFrame(rows, columns=["name", "zero_shot", "real", "question", "path"]) # Baseline
            df.to_csv(f"{PATH}/results/results_{EXT}_baseline_{model.split('/')[1]}.csv", index=False) # Baseline
        else:
            df = pd.DataFrame(rows, columns=["name", "zero_shot", "zero_shot_rag", "real", "question", "path"])
            df.to_csv(f"{PATH}/results/results_{EXT}_bench_{model.split('/')[1]}.csv", index=False)

        print(f"Model {model} done!\n")


