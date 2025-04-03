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

qe = QuizEvaluator(PATH, args.res_dir, "/Users/cinghio/Documents/PHD/HealthBranches/category.json")

# models = ["mistral_7b", "gemma_7b", "gemma2_9b", "gemma3_4b", "llama3.1_8b", "qwen2.5_7b", 
#           "phi4_14b", "llama2_7b", "Llama-3.3-70B-Instruct-Turbo-Free"]

models = ["mistral_7b", "gemma_7b", "gemma2_9b", "gemma3_4b", "llama3.1_8b", "qwen2.5_7b", 
          "phi4_14b", "llama2_7b"]

res = []

for condition in tqdm(qe.get_conditions()):
    res.extend(qe.evaluate_models(models, condition))

category_values1 = defaultdict(list)
category_values2 = defaultdict(list)
category_values3 = defaultdict(list)

# 2 -> no rag, 3 -> rag, 4 -> topline
for item in res:
    category = qe.map_condition(item[0])
    value1 = float(item[2])
    value2 = float(item[3]) 
    value3 = float(item[4])

    category_values1[category].append(value1)
    category_values2[category].append(value2)
    category_values3[category].append(value3)

# Compute mean for each category for all three value columns
mean_values1 = {category: np.mean(values) for category, values in category_values1.items()}
mean_values2 = {category: np.mean(values) for category, values in category_values2.items()}
mean_values3 = {category: np.mean(values) for category, values in category_values3.items()}

# Extract names and mean values for plotting
names = list(mean_values1.keys())
values1 = [mean_values1[category] for category in names]
values2 = [mean_values2[category] for category in names]
values3 = [mean_values3[category] for category in names]

# Create y positions for categories and bar offsets
y_pos = np.arange(len(names))
bar_height = 0.3

plt.figure(figsize=(14, 10))
# Plot value1 bars shifted up
bars1 = plt.barh(y_pos - bar_height, values1, height=bar_height, color='skyblue', label='Zero Shot')
# Plot value2 bars in the middle
bars2 = plt.barh(y_pos, values2, height=bar_height, color='orange', label='Zero Shot + RAG')
# Plot value3 bars shifted down
bars3 = plt.barh(y_pos + bar_height, values3, height=bar_height, color='green', label='Topline')

# Set x-axis limits for proper scaling
all_values = values1 + values2 + values3
plt.xlim(0, max(all_values) + 0.1)

plt.xlabel('Mean Accuracy')
plt.ylabel('Category')
plt.title('Mean Values per Category Across All Models')
plt.yticks(y_pos, names)
plt.legend()

# Invert y-axis to keep the first item on top
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Show plot
plt.show()