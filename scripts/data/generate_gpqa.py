import pandas as pd

from datasets import load_dataset
import random

ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")



for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    all_data = []
    for i in range(len(ds['train'])):
        correct_answer = ds['train'][i]['Correct Answer'].strip()
        incorrect_answers = []
        incorrect_answers.append(ds['train'][i]['Incorrect Answer 1'].strip())
        incorrect_answers.append(ds['train'][i]['Incorrect Answer 2'].strip())
        incorrect_answers.append(ds['train'][i]['Incorrect Answer 3'].strip())
        # Get shuffled choices 
        shuffled_choices = incorrect_answers + [correct_answer]
        random.shuffle(shuffled_choices)

        # Add Options to the question
        question = ds['train'][i]['Question']
        question += f"\n\nOptions:\n"
        for i, choice in enumerate(shuffled_choices):
            question += f"{chr(65 + i)}. {choice}\n"
        correct_choice = chr(65 + shuffled_choices.index(correct_answer))
        if num_tokens < -1:
            question = f"{question}"+"\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + f" Think for maximum {abs(num_tokens)} tokens."
        else:
            question = f"{question}"+"\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + f" Think for {num_tokens} tokens."


        all_data.append({
                    "data_source": "gpqa",
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
                        'index': i
                    }
                })
    if num_tokens < -1:
        pd.DataFrame(all_data).to_parquet(f'~/deepscaler/data9_{num_tokens}/gpqa.parquet')
    else:
        pd.DataFrame(all_data).to_parquet(f'~/deepscaler/data_{num_tokens}/gpqa.parquet')
    