import pandas as pd

from datasets import load_dataset, concatenate_datasets
import random

ds1 = load_dataset("opencompass/AIME2025", "AIME2025-I")['test']
ds2 = load_dataset("opencompass/AIME2025", "AIME2025-II")['test']
# Concatenate the two datasets
ds = concatenate_datasets([ds1, ds2])


for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    all_data = []
    for i in range(len(ds)):
        question = ds[i]['question'].strip()
        if num_tokens >=-1:
            question = f"{question}"+"\n\nLet's think step by step and output the final answer within \\boxed{}." + (f" Think for {num_tokens} tokens." if num_tokens != -1 else "")
        else:
            question = f"{question}"+"\n\nLet's think step by step and output the final answer within \\boxed{}." + (f" Think for maximum {abs(num_tokens)} tokens.")


        all_data.append({
                    "data_source": "aime2025",
                    "prompt": [{
                        "role": "user",
                        "content": question
                    }],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": ds[i]['answer'],
                        "num_tokens": num_tokens
                    },
                    "extra_info": {
                        'split': 'test',
                        'index': i
                    }
                })
    if num_tokens == -1:
        pd.DataFrame(all_data).to_parquet(f'~/deepscaler/data/aime2025.parquet')
    else:
        if num_tokens < 0:
            pd.DataFrame(all_data).to_parquet(f'~/deepscaler/data9_{num_tokens}/aime2025.parquet')
        else:
            pd.DataFrame(all_data).to_parquet(f'~/deepscaler/data_{num_tokens}/aime2025.parquet')
    