#!/bin/bash
set -x
HYDRA_FULL_ERROR=1
# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done


# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/mnt/f/WorkSpace/Thinking/Qwen/Qwen3-0.6B"
fi


# if [ ! -d "./Qwen3-0.6B" ]; then
#     echo "Downloading Qwen3-0.6B from HuggingFace..."
#     git lfs install
#     git clone https://huggingface.co/Qwen/Qwen3-0.6B
# fi

# Train over a single node, 8 A100-80GB GPUs.
python3 main_ppo.py \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/deepscaler/data/train.parquet \
    data.val_files=$HOME/deepscaler/data/aime.parquet \
    data.train_batch_size=1 \
    data.val_batch_size=2 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_k=0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='deepscaler' \
    trainer.experiment_name='l1_exact' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=3 "${@:1}" \
    reward_config.sigmoid_reward=False \
    reward_config.linear_reward=True \
    reward_config.multiplier_reward=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    reward_config.alpha=0.0003 \
    critic.ppo_max_token_len_per_gpu=4096 