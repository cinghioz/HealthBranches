from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import argparse

MODEL_NAME = 'BAAI/bge-m3'
FILES_DIR = '/home/cc/PHD/HealthBranches/results'
BATCH_SIZE = 32
COLS = ['zero_shot', 'zero_shot_rag', 'zero_shot_text', 'zero_shot_path', 'zero_shot_all']
SAVE_FILE = 'open_eval/similarity_res_real_bge.csv'

def argparse_args():
    parser = argparse.ArgumentParser(description='Scorer for BGE M3')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='Model name')
    parser.add_argument('--files_dir', type=str, default=FILES_DIR, help='Files directory')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--save_file', type=str, default=SAVE_FILE, help='Save file path')
    return parser.parse_args()

def find_model(x):
    if 'baseline' in x:
        return x.split('results_OPEN_baseline_')[1].split('.csv')[0]
    elif 'bench' in x:
        return x.split('results_OPEN_bench_')[1].split('.csv')[0]
    
def main():
    
    all_files = [f for f in os.listdir(FILES_DIR) if 'results_OPEN_bench' in f]

    print(f'Loading model: {MODEL_NAME}')
    model = model = BGEM3FlagModel(MODEL_NAME,  use_fp16=True)

    pbar = tqdm(all_files, total=len(all_files))
    for f in pbar:
        pbar.set_postfix_str(f'{find_model(f)}')

        bench = pd.read_csv(os.path.join(FILES_DIR, f))
        base = pd.read_csv(os.path.join(FILES_DIR, f).replace('bench', 'baseline'))
        # if 'nemo' in f:
        #     base = pd.read_csv(os.path.join(FILES_DIR+'/baseline', 'results_OPEN_baseline_mistral-nemo_12b.csv').replace('bench', 'baseline'))
        # else:
        #     base = pd.read_csv(os.path.join(FILES_DIR+'/baseline', f).replace('bench', 'baseline'))
        
        res = []
        for c in COLS:

            sentences_baseline = base.real.astype(str).tolist()

            if c in ['zero_shot_text', 'zero_shot_path', 'zero_shot_all']:
                sentences_bench = base[c].astype(str).tolist()
            else:
                sentences_bench = bench[c].astype(str).tolist()
            
            sentence_pairs = list(zip(sentences_baseline, sentences_bench))

            sim_dict = model.compute_score(sentence_pairs,
                          batch_size=32,
                          max_passage_length=2048,
                          weights_for_different_modes=[0.33, 0.33, 0.33])
            
            sim = pd.DataFrame(sim_dict)
            sim['model'] = find_model(f)
            sim['exp'] = c
            sim['idx'] = list(range(len(bench)))
            sim = sim[['idx', 'model', 'exp']+list(sim_dict.keys())]

            res.append(sim)

        res = pd.concat(res)
    
        if os.path.exists(SAVE_FILE):
            res.to_csv(SAVE_FILE, mode='a', header=False, index=False)
        else:
            res.to_csv(SAVE_FILE, index=False)


if __name__ == '__main__':
    args = argparse_args()
    MODEL_NAME = args.model_name
    FILES_DIR = args.files_dir
    BATCH_SIZE = args.batch_size
    SAVE_FILE = args.save_file
    main()