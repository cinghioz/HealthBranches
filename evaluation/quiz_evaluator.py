import os
import json
import pandas as pd
from utils import extract_option

class QuizEvaluator:
    def __init__(self, path: str, res_dir: str = "results", category_path : str = "/Users/cinghio/Documents/PHD/HealthBranches/category.json"):
        self.path = path
        self.bench_path = f"{self.path}{res_dir}/results_QUIZ_bench_"
        self.base_path = f"{self.path}{res_dir}/results_QUIZ_baseline_"
        
        with open(category_path, 'r') as f:
            self.category_map = json.load(f)
    
    def get_conditions(self):
        return list(set(self.category_map.keys()))
    
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
            df[f'{col}_choice'] = df[col].apply(extract_option)
            df[f'{col}_is_correct'] = df[f'{col}_choice'] == df['real']
            accuracy = df[f'{col}_is_correct'].mean()
            accs.append(accuracy)
        
        accs.insert(0, model)
        accs.insert(0, condition_value)
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
    def _merge_lists(data: list[list], condition: bool = False):
        merged_data = {}
        
        for row in data:
            key = row[0]
            second_elem = row[1] if isinstance(row[1], str) else None
            values = row[2:] if second_elem else row[1:]
            
            if key not in merged_data:
                merged_data[key] = {'string': second_elem, 'values': values.copy()}
            else:
                merged_data[key]['values'].extend(values)
        
        result = []
        for key, content in merged_data.items():
            if condition and content['string'] is not None:
                result.append([key, content['string']] + content['values'])
            else:
                result.append([key] + content['values'])
        
        return result

    def evaluate_models(self, models: list[str], condition: str = None):
        bench = [self._evaluate_answers(f"{self.bench_path}{model}.csv", model) for model in models]
        baseline = [self._evaluate_answers(f"{self.base_path}{model}.csv", model) for model in models]
        
        if condition:
            bench = [self._evaluate_answers_by_cond(f"{self.bench_path}{model}.csv", model, condition) for model in models]
            baseline = [self._evaluate_answers_by_cond(f"{self.base_path}{model}.csv", model, condition) for model in models]
        
        return self._merge_lists(bench + baseline, condition)
        # return bench
    
    def remap_condition(self, merged_list: list[list]):
        for sublist in merged_list:
            file_name = sublist[0]
            if file_name in self.category_map.keys():
                sublist[0] = self.category_map[file_name]
            else:
                # Handle the case where the file name is not found in the dictionary
                sublist[0] = 'Unknown Folder'

    def map_condition(self, sub_con: str):
        return self.category_map[sub_con]