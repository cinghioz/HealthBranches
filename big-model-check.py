import os
import pandas as pd
import numpy as np
import argparse
import ast
from together import Together
from alive_progress import alive_bar

from prompt import *
from classes.utils import extract_option

client = Together(api_key="de1d1c231987694e2b9111e06e048732d206ecaee729b8aee41e2121006f2cfc")
    
parser = argparse.ArgumentParser(description="LLM inference with optional baseline mode.")
parser.add_argument("-base", action="store_true", help="Run in baseline mode.")
args = parser.parse_args()

# Set BASELINE based on the argument
BASELINE = args.base
PATH = "/home/cc/PHD/HealthBranches/"

def together_inference(client, query, template, path, text, choices, condition):
    if BASELINE:
        prompt = template.format(question=query, condition=condition, path=path, text=text, o1=choices[0], o2=choices[1], o3=choices[2], o4=choices[3], o5=choices[4])
    else:
        prompt = template.format(question=query, condition=condition, o1=choices[0], o2=choices[1], o3=choices[2], o4=choices[3], o5=choices[4])
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
    )

    # print(response.choices[0].message.content)
    return response.choices[0].message.content

df = pd.read_csv("/home/cc/PHD/HealthBranches/results/results_quiz_Llama-3.3-70B-Instruct-Turbo-Free.csv")
df["zero_shot_new"] = df["zero_shot"].apply(extract_option)

big_filtered = df[df["zero_shot_new"].str.upper() != df["real"].str.upper()]

# %%
folder_path = f"{PATH}questions_pro/ultimate_questions_v3_full_balanced.csv"
questions = pd.read_csv(folder_path)

templates = [PROMPT_QUIZ]

if BASELINE:
    templates = [PROMPT_QUIZ_BASELINE]

cnt_rag = 0
cnt = 0

rows = []
questions = pd.read_csv(folder_path)
questions = questions[questions["question"].isin(big_filtered["question"])] # Per test su 405B

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
                res.append(together_inference(client, row['question'], template, row['path'], text, opts, row['condition'].lower())) # Baseline
            else:
                res.append(together_inference(client, row['question'], template, "", "", opts, row['condition'].lower()))

        res.append(row["correct_option"])
        res.append(row['question'])
        res.append(row['path'])
        res.insert(0, row['condition'].lower())

        rows.append(res)
        bar()

if BASELINE:
    df = pd.DataFrame(rows, columns=["name", "zero_shot", "real", "question", "path"]) # Baseline
    df.to_csv(f"{PATH}results/results_quiz_baseline_405B.csv", index=False) # Baseline
else:
    df = pd.DataFrame(rows, columns=["name", "zero_shot", "real", "question", "path"])
    df.to_csv(f"{PATH}results/results_quiz_405B.csv", index=False)

print(f"Model BIG done!\n")


