import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Any, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from utils import extract_option

def set_neurips_style():
    """
    Apply NeurIPS-friendly styling to matplotlib plots with seaborn darkgrid aesthetics
    and Computer Modern fonts
    """
    # Add Computer Modern font family
    # This assumes the user has the Computer Modern fonts installed in the system
    # If not, you might need to install them or use matplotlib's builtin 'cm' family
    try:
        # Try to use the proper Computer Modern fonts if available
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman'] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern math font
    except:
        # Fallback to matplotlib's built-in Computer Modern
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'cm'
    
    # Figure size for NeurIPS papers (designed to fit in a single column)
    plt.rcParams['figure.figsize'] = (3.5, 2.625)  # 3.5 x 2.625 inches is good for a single column
    
    # Font sizes
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    
    # Line widths
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['grid.linewidth'] = 0.5
    
    # Marker size and style
    plt.rcParams['lines.markersize'] = 4
    plt.rcParams['scatter.marker'] = 'o'
    
    # Seaborn-inspired colors - maintains colorblind friendliness
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', [
        '#4C72B0',  # blue
        '#DD8452',  # orange
        '#55A868',  # green
        '#C44E52',  # red
        '#8172B3',  # purple
        '#937860',  # brown
        '#DA8BC3',  # pink
        '#8C8C8C',  # gray
        '#CCB974',  # khaki
        '#64B5CD',  # light blue
    ])
    
    # Darkgrid background and grid (seaborn style)
    plt.rcParams['axes.facecolor'] = '#EAEAF2'  # light gray background
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['grid.color'] = 'white'
    plt.rcParams['grid.alpha'] = 1.0
    plt.rcParams['grid.linewidth'] = 1.0
    
    # Spines and ticks - seaborn darkgrid typically has reduced spines
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    # plt.rcParams['axes.spines.color'] = ['#CCCCCC']  # Light gray spines
    
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['xtick.color'] = '#555555'
    plt.rcParams['ytick.color'] = '#555555'
    
    # Legend
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = '#CCCCCC'
    
    # LaTeX-like rendering for text with Computer Modern font
    plt.rcParams['text.usetex'] = True  # Set to True if you have LaTeX installed
    
    # Saving options
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.05

    # set figure dpi
    plt.rcParams['figure.dpi'] = 300

def use_latex_with_computer_modern():
    """
    Alternative function to use actual LaTeX for text rendering.
    This requires a working LaTeX installation with the Computer Modern fonts.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    
    # Call the rest of the styles
    set_neurips_style()

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
        use_latex_with_computer_modern()

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

        plt.figure(figsize=(10, 5))
        # Plot value1 bars shifted up
        bars1 = plt.barh(y_pos - bar_height, values1, height=bar_height, color='skyblue', label='Zero Shot')
        # Plot value2 bars in the middle
        bars2 = plt.barh(y_pos, values2, height=bar_height, color='orange', label='Zero Shot + RAG')
        # Plot value3 bars shifted down
        bars3 = plt.barh(y_pos + bar_height, values3, height=bar_height, color='green', label='Topline (path only)')

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
        plt.savefig("plot_conditions.png", dpi=450, bbox_inches='tight')

    def plot_by_models(self, data: list[list]):
        use_latex_with_computer_modern()

        data.sort(key=lambda x: x[1], reverse=False)

        # Extract labels and values
        labels = [x[0] for x in data]
        values1 = [x[1]*100 for x in data]
        values2 = [x[2]*100 for x in data]
        values3 = [x[3]*100 for x in data]
        values4 = [x[4]*100 for x in data]
        values5 = [x[5]*100 for x in data]  # New data series

        # Bar settings
        width = 0.15  # narrower bars to fit five series
        num_series = 5
        # Offsets for five bars centered around each x position
        offsets = (np.arange(num_series) - (num_series - 1) / 2) * width

        # Spacing between groups
        group_spacing = width * (num_series + 0.75)
        x = np.arange(len(labels)) * group_spacing

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each series with its offset and label, capture bar containers
        bars1 = ax.bar(x + offsets[0], values1, width, label="Zero Shot")
        bars2 = ax.bar(x + offsets[1], values2, width, label="Zero Shot + RAG")
        bars3 = ax.bar(x + offsets[2], values3, width, label="Topline (path only)")
        bars4 = ax.bar(x + offsets[3], values4, width, label="Topline (text only)")
        bars5 = ax.bar(x + offsets[4], values5, width, label="Topline (text+path)")

        # Annotate bar values
        for bar_container in [bars1, bars2, bars3, bars4, bars5]:
            for bar in bar_container:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=6)

        # Labels and title
        ax.set_xlabel("Models")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation='vertical', ha='center')
        ax.legend(loc='lower right')

        # Save figure
        plt.tight_layout()
        plt.savefig("plot_models.png", dpi=450, bbox_inches='tight')
