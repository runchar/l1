# python scripts/data/deepscaler_dataset.py --num_tokens 512
# python scripts/data/deepscaler_dataset.py --num_tokens 1024
# python scripts/data/deepscaler_dataset.py --num_tokens 2048
# python scripts/data/deepscaler_dataset.py --num_tokens 3600

python scripts/data/deepscaler_dataset.py --num_tokens -512 --local_dir ~/deepscaler/data_max
# python scripts/data/deepscaler_dataset.py --num_tokens -1024 --local_dir ~/deepscaler/data_max
# python scripts/data/deepscaler_dataset.py --num_tokens -2048 --local_dir ~/deepscaler/data_max
# python scripts/data/deepscaler_dataset.py --num_tokens -3600 --local_dir ~/deepscaler/data_max


# python scripts/data/generate_aime.py
# python scripts/data/generate_gpqa.py
# python scripts/data/generate_lsat.py
# python scripts/data/generate_mmlu.py