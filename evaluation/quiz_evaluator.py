import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from utils import extract_option

sns.set(style="darkgrid")

class QuizEvaluator:
    def __init__(self, path: str, res_dir: str = "results", category_path : str = "/Users/cinghio/Documents/PHD/HealthBranches/category.json"):
        self.path = path
        self.bench_path = f"{self.path}{res_dir}/results_QUIZ_bench_"
        self.base_path = f"{self.path}{res_dir}/results_QUIZ_baseline_"
        
        with open(category_path, 'r') as f:
            self.category_map = json.load(f)

    def get_conditions(self):
        return list(self.category_map.keys())
    
    @staticmethod
    def _evaluate_answers(file_path: str, model: str):
        df = pd.read_csv(file_path)
        accs = []
        
        for col in [c for c in df.columns if c.startswith(("zero_shot", "one_shot"))]:
            df[f'{col}_choice'] = df[col].apply(extract_option)
            df[f'{col}_is_correct'] = df[f'{col}_choice'] == df['real']
            accuracy = df[f'{col}_is_correct'].mean()
            accs.append(accuracy)
        
        accs.insert(0, model)
        return accs
    
    @staticmethod
    def _evaluate_answers_by_cond(file_path: str, model: str, condition_value=None):
        df = pd.read_csv(file_path)
        
        if condition_value is not None:
            df = df[df['name'] == condition_value]
        
        accs = []
        
        for col in [c for c in df.columns if c.startswith(("zero_shot", "one_shot"))]:
            if condition_value is not None:
                df[f'{col}_choice'] = df[col].apply(extract_option)
                df[f'{col}_is_correct'] = df[f'{col}_choice'] == df['real']
                accuracy = df[f'{col}_is_correct'].mean()
                accs.append(accuracy)
            else:
                accs.append(np.float(0))
        
        accs.insert(0, condition_value)
        accs.insert(0, model)
        return accs
    
    def find_incorrect_indices(self, file_path: str, col: str = "zero_shot"):
        df = pd.read_csv(file_path)
        incorrect_indices = df.index[df[col].apply(extract_option) != df["real"]].tolist()
        return incorrect_indices
    
    def find_common_wrongs(lists_of_indices):
        if not lists_of_indices:
            return []
        
        common = set(lists_of_indices[0])
        for indices in lists_of_indices[1:]:
            common &= set(indices)
        
        return sorted(common)
    
    @staticmethod
    def _merge_lists(data1: list[list], data2: list[list]):
           # Build a mapping for list2 based on keys.
        mapping = {}
        for sub in data2:
            # Check if a second key is available; otherwise use None.
            if len(sub) >= 3:
                key = (sub[0], sub[1])
                num = sub[2]
            else:
                key = (sub[0], None)
                num = sub[1]
            mapping[key] = num

        # For each sublist in list1, build the corresponding key and append the numeric value if found.
        for sub in data1:
            if len(sub) >= 4:  # Assuming list1 contains two keys followed by numeric values.
                key = (sub[0], sub[1])
            else:
                key = (sub[0], None)
            if key in mapping:
                sub.append(mapping[key])

        return data1

    def evaluate_models(self, models: list[str], condition: str = None):
        bench = [self._evaluate_answers(f"{self.bench_path}{model}.csv", model) for model in models]
        baseline = [self._evaluate_answers(f"{self.base_path}{model}.csv", model) for model in models]
        
        if condition:
            bench = [self._evaluate_answers_by_cond(f"{self.bench_path}{model}.csv", model, condition) for model in models]
            baseline = [self._evaluate_answers_by_cond(f"{self.base_path}{model}.csv", model, condition) for model in models]
        
        return self._merge_lists(bench, baseline)
        # return bench
    
    def remap_condition(self, merged_list: list[list]):
        for sublist in merged_list:
            file_name = sublist[0]
            if file_name in self.category_map.keys():
                sublist[0] = self.category_map[file_name]
            else:
                # Handle the case where the file name is not found in the dictionary
                sublist[0] = 'Unknown Folder'

    def _map_condition(self, sub_con: str):
        return self.category_map[sub_con]
    
    def plot_by_conditions(self, data: list[list]):
        category_values1 = defaultdict(list)
        category_values2 = defaultdict(list)
        category_values3 = defaultdict(list)

        # 2 -> no rag, 3 -> rag, 4 -> topline
        for item in data:
            category = self._map_condition(item[1])
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

        plt.figure(figsize=(16, 12))
        # Plot value1 bars shifted up
        bars1 = plt.barh(y_pos - bar_height, values1, height=bar_height, color='skyblue', label='Zero Shot')
        # Plot value2 bars in the middle
        bars2 = plt.barh(y_pos, values2, height=bar_height, color='orange', label='Zero Shot + RAG')
        # Plot value3 bars shifted down
        bars3 = plt.barh(y_pos + bar_height, values3, height=bar_height, color='green', label='Topline')

        # Set x-axis limits for proper scaling
        all_values = values1 + values2 + values3
        plt.xlim(0, max(all_values) + 0.1)

        plt.xlabel('Accuracy')
        plt.ylabel('Category')
        plt.title('Mean Accuracy per Category Across All Models')
        plt.yticks(y_pos, names)
        plt.legend()

        # Invert y-axis to keep the first item on top
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # plt.show()
        plt.savefig("plot_conditions.png", dpi=400, bbox_inches='tight')

    def plot_by_models(self, data: list[list]):
        data.sort(key=lambda x: x[1], reverse=True)

        # Estrai le etichette e i valori
        labels = [x[0] for x in data]
        values1 = [x[1] for x in data]
        values2 = [x[2] for x in data]
        values3 = [x[3] for x in data]

        # Imposta la posizione delle barre con più spazio tra i gruppi
        x = np.arange(len(labels)) * 1.3  # Moltiplica per aumentare la distanza tra i gruppi
        width = 0.35  # Larghezza delle barre

        # Aumenta le dimensioni del grafico
        fig, ax = plt.subplots(figsize=(16, 12))
        bars1 = ax.bar(x - width, values1, width, label="Zero Shot")
        bars2 = ax.bar(x, values2, width, label="Zero Shot + RAG")
        bars3 = ax.bar(x + width, values3, width, label="Topline")    

        # Etichette e titolo
        ax.set_xlabel("Models")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Model")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        # plt.show()
        plt.savefig("plot_models.png", dpi=400, bbox_inches='tight')