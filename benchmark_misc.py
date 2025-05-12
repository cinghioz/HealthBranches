import os
import pandas as pd
import ast
import argparse
import time                      
from typing import Dict, List
from alive_progress import alive_bar

from classes.vector_store import VectorStore
from classes.llm_inference import LLMinference
from utils import check_results
from prompt import *
from models import *

parser = argparse.ArgumentParser(description="LLM inference with different modalities.")
parser.add_argument("-base", action="store_true", help="Run in baseline mode.")
parser.add_argument("-quiz", action="store_true", help="Run in quiz mode.")
args = parser.parse_args()

BASELINE = args.base
QUIZ     = args.quiz
PATH     = "/home/cc/PHD/HealthBranches/"
EXT      = "QUIZ" if QUIZ else "OPEN"

print("##### BASELINE MODE #####\n" if BASELINE else "##### BENCHMARK MODE #####\n")
print("##### QUIZ EXP #####\n"     if QUIZ     else "##### OPEN EXP #####\n")

vector_store = VectorStore(f'{PATH}indexes/kgbase/')
questions   = pd.read_csv(f"{PATH}questions_pro/final_dataset.csv")
models      = MODELS
models      = check_results(
    PATH+"results_misc/",
    f"results_{EXT}_baseline_*.csv" if BASELINE else f"results_{EXT}_bench_*.csv",
    models
)

templates = [PROMPT_QUIZ, PROMPT_QUIZ_RAG] if QUIZ else [PROMPT_OPEN, PROMPT_OPEN_RAG]
if BASELINE:
    templates = (
        [PROMPT_QUIZ_BASELINE_PATH, PROMPT_QUIZ_BASELINE_TEXT, PROMPT_QUIZ_BASELINE]
        if QUIZ
        else
        [PROMPT_OPEN_BASELINE_PATH,  PROMPT_OPEN_BASELINE_TEXT, PROMPT_OPEN_BASELINE]
    )

# Prepare a summary list to hold (model_name, mean_time)
summary = []

# Shuffle & limit questions
questions = questions.sample(frac=1, random_state=42).reset_index(drop=True)[:100]

for model_name in models:
    print(f"Running model {model_name}...")

    # Initialize LLM and timing list
    if "deepseek" in model_name.lower():
        llm = LLMinference(llm_name=model_name, temperature=0.0, num_predict=512)
    else:
        llm = LLMinference(llm_name=model_name, temperature=0.0)

    timings = []      # ← collect each inference duration
    rows    = []

    with alive_bar(len(questions)) as bar:
        for _, row in questions.iterrows():
            # parse options
            try:
                opts = ast.literal_eval(
                    row['options']
                    .replace("['", '["')
                    .replace("']", '"]')
                    .replace("', '", '", "')
                )
                if not isinstance(opts, list) or len(opts) != 5:
                    continue
            except Exception:
                continue

            # load KG text
            txt_name        = row['condition'].upper() + ".txt"
            txt_folder_name = f"{PATH}data/kgbase/"
            try:
                with open(os.path.join(txt_folder_name, txt_name), 'r') as f:
                    text = f.readlines()
            except Exception:
                continue

            # run each prompt template
            res = []
            for template in templates:
                start = time.time()                # ← start timer
                if BASELINE:
                    out = llm.qea_evaluation(
                        row['question'], template,
                        row['old_path'], text, opts,
                        row['condition'].lower(), vector_store
                    )
                else:
                    out = llm.qea_evaluation(
                        row['question'], template,
                        "", "", opts,
                        row['condition'].lower(), vector_store
                    )
                duration = time.time() - start     # ← stop timer
                timings.append(duration)           # ← record
                res.append(out)

            # append gold, question, etc.
            if QUIZ:
                res.append(row["correct_option"])
            else:
                res.append(opts[ord(row["correct_option"].strip().upper()) - ord('A')])
            res.append(row['question'])
            res.append(row['old_path'])
            res.insert(0, row['condition'].lower())

            rows.append(res)
            bar()

    # Build DataFrame and save per-model results
    if BASELINE:
        cols = ["name", "zero_shot_path", "zero_shot_text", "zero_shot_all",
                "real", "question", "path"]
        out_name = f"results_{EXT}_baseline_{model_name.replace(':', '_')}.csv"
    else:
        cols = ["name", "zero_shot", "zero_shot_rag",
                "real", "question", "path"]
        out_name = f"results_{EXT}_bench_{model_name.replace(':', '_')}.csv"

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(os.path.join(PATH, "results_misc", out_name), index=False)

    # Compute mean and record in summary
    mean_time = sum(timings) / len(timings) if timings else 0.0
    summary.append({"model": model_name, "mean_inference_time_s": mean_time})

    print(f"Model {model_name} done! Mean inference time: {mean_time:.3f}s\n")

# After all models, write a summary CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(PATH, "results_misc", f"mean_inference_times_{EXT}.csv"), index=False)

print("All models complete. Summary CSV saved.")  
