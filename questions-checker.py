import os
import pandas as pd
import numpy as np
import argparse
import ast
from together import Together
from alive_progress import alive_bar

from prompt import *
from utils import extract_option

client = Together(api_key="de1d1c231987694e2b9111e06e048732d206ecaee729b8aee41e2121006f2cfc")
    
parser = argparse.ArgumentParser(description="LLM inference with optional baseline mode.")
parser.add_argument("-base", action="store_true", help="Run in baseline mode.")
args = parser.parse_args()

# Set BASELINE based on the argument
BASELINE = args.base
PATH = "/home/cc/PHD/HealthBranches/"

def together_inference(template, query, answer, path, text, choices, condition):
    if text != "":
        prompt = template.format(question=query, condition=condition, context=text, options=choices, correct_option=answer)
    else:
        prompt = template.format(question=query, condition=condition, options=choices, correct_option=answer)
    response = client.chat.completions.create(
        # model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "user", "content": prompt}],
    )

    # print(response.choices[0].message.content)
    return response.choices[0].message.content

questions = pd.read_csv("/home/cc/PHD/HealthBranches/questions_to_check.csv")
rows = []

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
        

        res.append(together_inference(CHECK_QUESTION_RATE, row['question'], row['answer'], row['path'], text, opts, row['condition'].lower())) # Baseline

        res.append(row["answer"])
        res.append(row['question'])
        res.append(row['path'])
        res.insert(0, row['condition'].lower())

        rows.append(res)
        bar()


df = pd.DataFrame(rows, columns=["name", "check", "answer", "question", "path"])
df.to_csv(f"{PATH}results/results_checker_70B.csv", index=False)

print(f"Model 70B done!\n")