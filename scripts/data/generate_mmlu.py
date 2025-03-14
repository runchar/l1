import pandas as pd
import numpy as np
from datasets import load_dataset

ds_mmlu = load_dataset("cais/mmlu", "all")

for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    all_data = []
    for i in range(len(ds_mmlu['test'])):
        row = ds_mmlu['test'][i]
        options = row['choices']
        options_str  = ""
        for j in range(len(options)):
            options_str += f"{chr(65 + j)}. {options[j]}\n"
        # Add Options to the question
        question = row['question'] + "\n\nOptions:\n" + options_str
        correct_choice = chr(65 + row['answer'])
        if num_tokens < -1:
            question = f"{question}"+"\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + f" Think for maximum {num_tokens} tokens."
        else:
            question = f"{question}"+"\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + f" Think for {num_tokens} tokens."

        all_data.append({
                    "data_source": "mmlu",
                    "prompt": [{
                        "role": "user",
                        "content": question
                    }],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": correct_choice,
                        "num_tokens": num_tokens
                    },
                    "extra_info": {
                        'split': 'test',
                        'index': i,
                        'subject': row['subject']
                    }
                })
    # Suffle all_data randomly
    np.random.seed(42)
    indices = np.arange(len(all_data))
    np.random.shuffle(indices)
    all_data = [all_data[i] for i in indices[:1000]]
    if num_tokens != -1:
        if num_tokens < -1:
            pd.DataFrame(all_data).to_parquet(f'~/deepscaler/data9_{num_tokens}/mmlu_1000.parquet')
        else:
            pd.DataFrame(all_data).to_parquet(f'~/deepscaler/data_{num_tokens}/mmlu_1000.parquet')
    else:
        pd.DataFrame(all_data).to_parquet(f'~/deepscaler/data/mmlu_1000.parquet')