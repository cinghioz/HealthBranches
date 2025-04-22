from evaluation.quiz_evaluator import QuizEvaluator
import argparse
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Evaluation quiz results")
parser.add_argument('-res_dir', type=str, default="results", help='Folder with results')
args = parser.parse_args()

PATH = os.getcwd()+'/'
BENCH = f"{PATH}{args.res_dir}/results_QUIZ_bench_"
BASE = f"{PATH}{args.res_dir}/results_QUIZ_baseline_"

qe = QuizEvaluator(PATH, args.res_dir, "/home/cc/PHD/HealthBranches/category.json")

models = ["mistral_7b", "gemma_7b", "gemma2_9b", "gemma3_4b", "llama3.1_8b", "llama3.3_70b",
           "qwen2.5_7b", "phi4_14b", "nemo_12b", "llama2_7b", "deepseek-r1_8b"]



res = []

for condition in tqdm(qe.get_conditions()):
    res.extend(qe.evaluate_models(models, condition))

qe.plot_by_conditions(res)

# res = qe.evaluate_models(models)

# qe.plot_by_models(res)