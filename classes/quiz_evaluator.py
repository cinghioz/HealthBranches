import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Any, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from utils import extract_option


class QuizEvaluator:
    def __init__(self, path: str,  category_path : str, res_dir: str = "results"):
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
        # qc = pd.read_csv("/home/cc/PHD/HealthBranches/questions_checked.csv")
        # df = df[df['question'].isin(qc['question'])]

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
    def _merge_by_key_concat(*lists: List[List[Any]]) -> List[List[Any]]:

        buffer: defaultdict[Union[str, Tuple[str, str]], List[Any]] = defaultdict(list)

        for lst in lists:
            for rec in lst:
                if len(rec) >= 2 and isinstance(rec[1], str):
                    key = (rec[0], rec[1])
                    vals = rec[2:]
                else:
                    key = rec[0]
                    vals = rec[1:]

                buffer[key].extend(vals)

        merged: List[List[Any]] = []
        for key, vals in buffer.items():
            if isinstance(key, tuple):
                merged.append([key[0], key[1], *vals])
            else:
                merged.append([key, *vals])

        return merged

    def evaluate_models(self, models: list[str], condition: str = None):
        bench = [self._evaluate_answers(f"{self.bench_path}{model.replace(":", "_")}.csv", model) for model in models]
        baseline = [self._evaluate_answers(f"{self.base_path}{model.replace(":", "_")}.csv", model) for model in models]
        
        if condition:
            bench = [self._evaluate_answers_by_cond(f"{self.bench_path}{model.replace(":", "_")}.csv", model, condition) for model in models]
            baseline = [self._evaluate_answers_by_cond(f"{self.base_path}{model.replace(":", "_")}.csv", model, condition) for model in models]
        
        return self._merge_by_key_concat(bench, baseline)
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
        category_values4 = defaultdict(list)
        category_values5 = defaultdict(list)

        # 2 -> no rag, 3 -> rag, 4 -> topline
        for item in data:
            category = self._map_condition(item[1])
            value1 = float(item[2])
            value2 = float(item[3]) 
            value3 = float(item[4])
            value4 = float(item[5]) 
            value5 = float(item[6])

            category_values1[category].append(value1)
            category_values2[category].append(value2)
            category_values3[category].append(value3)
            category_values4[category].append(value4)
            category_values5[category].append(value5)

        # Compute mean for each category for all three value columns
        mean_values1 = {category: np.mean(values) for category, values in category_values1.items()}
        mean_values2 = {category: np.mean(values) for category, values in category_values2.items()}
        mean_values3 = {category: np.mean(values) for category, values in category_values3.items()}
        mean_values4 = {category: np.mean(values) for category, values in category_values4.items()}
        mean_values5 = {category: np.mean(values) for category, values in category_values5.items()}

        # Extract names and mean values for plotting
        names = list(mean_values1.keys())
        values1 = [mean_values1[category] for category in names]
        values2 = [mean_values2[category] for category in names]
        values3 = [mean_values3[category] for category in names]
        values4 = [mean_values4[category] for category in names]
        values5 = [mean_values5[category] for category in names]

        df = pd.DataFrame({
            "categories": names,
            "zero_shot": values1,
            "zero_shot_RAG": values2,
            "topline_path": values3,
            "topline_text": values4,
            "topline_all": values5,
        })

        # Save to CSV
        csv_path = "categories_accuracy.csv"
        df.to_csv(csv_path, index=False)

        # Create y positions for categories and bar offsets
        y_pos = np.arange(len(names))
        bar_height = 0.3

        plt.figure(figsize=(12, 6))
        # Plot value1 bars shifted up
        bars1 = plt.barh(y_pos - bar_height, values1, height=bar_height, color='skyblue', label='Zero Shot')
        # Plot value2 bars in the middle
        bars2 = plt.barh(y_pos, values2, height=bar_height, color='orange', label='Zero Shot RAG')
        # Plot value3 bars shifted down
        bars3 = plt.barh(y_pos + bar_height, values3, height=bar_height, color='green', label='Topline (path only)')

        # Set x-axis limits for proper scaling
        all_values = values1 + values2 + values3
        plt.xlim(0, max(all_values) + 0.05)

        plt.xlabel('Accuracy')
        plt.ylabel('Category')
        plt.yticks(y_pos, names)
        plt.legend(loc='lower right', title="Experiment Type")

        # Invert y-axis to keep the first item on top
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # plt.show()
        plt.savefig("plot_conditions.pdf", dpi=450, bbox_inches='tight')

    def plot_by_models(self, data: list[list]):

        data.sort(key=lambda x: x[1], reverse=False)

        # Extract labels and values
        labels = [x[0] for x in data]
        values1 = [x[1]*100 for x in data]
        values2 = [x[2]*100 for x in data]
        values3 = [x[3]*100 for x in data]
        values4 = [x[4]*100 for x in data]
        values5 = [x[5]*100 for x in data]

        df = pd.DataFrame({
            "model": labels,
            "zero_shot": values1,
            "zero_shot_RAG": values2,
            "topline_path": values3,
            "topline_text": values4,
            "topline_all": values5,
        })

        # Save to CSV
        csv_path = "models_accuracy.csv"
        df.to_csv(csv_path, index=False)

        # Bar settings
        width = 0.15               # bar width
        bar_spacing = 0.005        # gap between bars within a group
        num_series = 5

        # Compute offsets with minimal spacing
        offsets = (np.arange(num_series) - (num_series - 1) / 2) * (width + bar_spacing)

        # Increased spacing between groups
        group_spacing = (width + bar_spacing) * (num_series + 1.0)
        x = np.arange(len(labels)) * group_spacing

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each series
        ax.bar(x + offsets[0], values1, width, label="Zero Shot")
        ax.bar(x + offsets[1], values2, width, label="Zero Shot + RAG")
        ax.bar(x + offsets[2], values3, width, label="Topline (path only)")
        ax.bar(x + offsets[3], values4, width, label="Topline (text only)")
        ax.bar(x + offsets[4], values5, width, label="Topline (text+path)")

        # Annotate bars
        for bars in ax.containers:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=6)

        # Labels and legend
        ax.set_xlabel("Models")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation='vertical', ha='center')
        ax.legend(loc='lower right', title="Experiment Type")

        plt.tight_layout()
        plt.savefig("plot_models.pdf", dpi=450, bbox_inches='tight')



