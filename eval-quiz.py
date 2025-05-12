from classes.quiz_evaluator import QuizEvaluator
import argparse
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from models import *

parser = argparse.ArgumentParser(description="Evaluation quiz results")
parser.add_argument('-res_dir', type=str, default="results", help='Folder with results')
args = parser.parse_args()

PATH = os.getcwd()+'/'
BENCH = f"{PATH}{args.res_dir}/results_QUIZ_bench_"
BASE = f"{PATH}{args.res_dir}/results_QUIZ_baseline_"

qe = QuizEvaluator(PATH, "/home/cc/PHD/HealthBranches/category.json", args.res_dir)

res = []

for condition in tqdm(qe.get_conditions()):
    res.extend(qe.evaluate_models(MODELS_EVAL, condition))

qe.plot_by_conditions(res)

# res = qe.evaluate_models(MODELS_EVAL)

# qe.plot_by_models(res)