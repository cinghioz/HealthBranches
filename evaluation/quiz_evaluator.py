import re
import pandas as pd
from collections import Counter
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import defaultdict
import argparse
import json

from utils import check_options, extract_option

parser = argparse.ArgumentParser(description="Evaluation quiz results")
parser.add_argument('-res_dir', type=str, default="results", help='Folder with results')
args = parser.parse_args()

PATH = os.getcwd()+'/'
BENCH = f"{PATH}{args.res_dir}/results_QUIZ_bench_"
BASE = f"{PATH}{args.res_dir}/results_QUIZ_baseline_"

with open('/Users/cinghio/Documents/PHD/HealthBranches/category.json', 'r') as f:
    category_map =  json.load(f)

def get_conditions():
    return list(set(category_map.values()))

def evaluate_answers(file_path: str, model: str):
    df = pd.read_csv(file_path)
    accs = []

    # Loop through each relevant column (those starting with "zero_shot" or "one_shot").
    for col in [c for c in df.columns if c.startswith(("zero_shot", "one_shot"))]:
        df[f'{col}_choice'] = df[col].apply(extract_option)
        df[f'{col}_is_correct'] = df[f'{col}_choice'] == df['real']
        accuracy = df[f'{col}_is_correct'].mean()
        # print(f'Accuracy for {col}: {accuracy:.2%}')
        accs.append(accuracy)
    
    accs.insert(0, model)

    return accs

def evaluate_answers_by_cond(file_path, model, condition_value=None):
    df = pd.read_csv(file_path)
    
    # Filter the DataFrame for the specific condition if provided.
    if condition_value is not None:
        df = df[df['name'] == condition_value]
    
    accs = []

    # Loop through each relevant column (those starting with "zero_shot" or "one_shot").
    for col in [c for c in df.columns if c.startswith(("zero_shot", "one_shot"))]:
        df[f'{col}_choice'] = df[col].apply(extract_option)
        df[f'{col}_is_correct'] = df[f'{col}_choice'] == df['real']
        accuracy = df[f'{col}_is_correct'].mean()
        # print(f'Accuracy for {col}: {accuracy:.2%}')
        accs.append(accuracy)
    
    accs.insert(0, model)
    accs.insert(0, condition_value)

    return accs

def find_incorrect_indices(file_path: str, col: str = "zero_shot"):
    df = pd.read_csv(file_path)

    incorrect_indices = df.index[df[col].apply(extract_option) != df["real"]].tolist()
    return incorrect_indices

def find_common_wrongs(lists_of_indices):
    if not lists_of_indices:
        return []
    
    # Start with the set of indices from the first list
    common = set(lists_of_indices[0])
    
    # Intersect with the indices from each subsequent list
    for indices in lists_of_indices[1:]:
        common &= set(indices)
    
    return sorted(common)

def merge_lists(data : list[list], condition: bool = False):
    merged_data = {}

    for row in data:
        key = row[0]
        second_elem = row[1] if isinstance(row[1], str) else None
        values = row[2:] if second_elem else row[1:]

        if key not in merged_data:
            merged_data[key] = {'string': second_elem, 'values': values.copy()}
        else:
            merged_data[key]['values'].extend(values)

    # Construct the final result list
    result = []
    for key, content in merged_data.items():
        if condition and content['string'] is not None:
            result.append([key, content['string']] + content['values'])
        else:
            result.append([key] + content['values'])

    return result

def evaluate_models(models: list[str], condition: str = None):
    bench = [evaluate_answers(f"{BENCH}{model}.csv", model) for model in models]
    baseline = [evaluate_answers(f"{BASE}{model}.csv", model) for model in models]
    
    if condition:
        bench = [evaluate_answers_by_cond(f"{BENCH}{model}.csv", model) for model in models]
        baseline = [evaluate_answers_by_cond(f"{BASE}{model}.csv", model) for model in models]

    return merge_lists(bench + baseline, condition)

    


