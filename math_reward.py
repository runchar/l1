"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union
import re



from rewards_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from utils import extract_answer, grade_answer_sympy, grade_answer_mathd
import random
import numpy as np

import math



THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput, ignore_think_token = False) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # Extract solution.
        if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        elif THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            if not ignore_think_token:
                return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
            else:
                model_solution = model_response
        
        model_answer = extract_answer(model_solution)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)

     
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def get_delta_score(num_tokens: int, used_tokens: int):
    # Stddev = num_tokens/5
    # Calculate z-score based on how far used_tokens deviates from target (num_tokens)
    z_score = (used_tokens - num_tokens) / (500)
    # Simple Gaussian function that peaks at 1.0 when used_tokens matches target
    delta_score = math.exp(-z_score**2 / 2)
    return max(0.1, delta_score)

def get_delta_score_linear(num_tokens: int, used_tokens: int, alpha = 1/3000):
    # z_score = abs(used_tokens - num_tokens) / (num_tokens/2)
    z_score = abs(used_tokens - num_tokens) * alpha
    
    delta_score = 1 - z_score
    # return max(0, min(1, delta_score))
    return delta_score - 1

def get_delta_score_linear_both(num_tokens: int, used_tokens: int, alpha = 0.002):
    # If used_tokens is negative, we have to setup maximum budget constraint
    if num_tokens < 0:
        beta = alpha

        delta = used_tokens - abs(num_tokens)
        sc = 0
        if delta < 0:
            sc = beta * delta * -1
        else:
            sc = alpha * delta * -1

        # Clip sc to [-1, 1]
        sc = max(-1, min(1, sc))
        return (sc + 1)/2
    else:
        return get_delta_score_linear(num_tokens, used_tokens, alpha)

def get_delta_score_sigmoid(num_tokens: int, used_tokens: int, alpha = 0.01):
    delta = abs(num_tokens) - used_tokens
    if delta < 0:
        delta = delta*alpha
        sigma_score = 1 / (1 + math.exp(-delta))
    else:
        delta = delta*alpha
        sigma_score = 1 / (1 + math.exp(-delta))
        sigma_score += 0.1 # Small bonus
    return max(0, min(1, sigma_score))

def get_delta_score_sigmoid_exact(num_tokens: int, used_tokens: int, alpha = 0.01):
    delta = abs(num_tokens - used_tokens)
    delta = delta*alpha
    sigma_score = 1 / (1 + math.exp(-delta))
    return max(0, min(1, sigma_score))

def get_binary_score(num_tokens: int, used_tokens: int):
    if used_tokens > num_tokens:
        return 0.0
    else:
        return 1.0

def gpqa_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False, num_tokens = -1, valid_response_length = -1):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    def get_model_choice(res):
        for i in range(len(res)-1, -1, -1):
            if res[i] == 'A' or res[i] == 'B' or res[i] == 'C' or res[i] == 'D':
                # Check if res[i-1] is not a character
                if not (res[i-1] >= 'a' and res[i-1] <= 'z') and not (res[i-1] >= 'A' and res[i-1] <= 'Z'):
                    return res[i]
                    break
        return ''
    model_choice = get_model_choice(solution_str)
    if model_choice == ground_truth:
        return 1.0
    else:
        return 0.0

def math_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], num_tokens = -1, valid_response_length = -1, ignore_think_token = False, reward_config : RewardConfig = RewardConfig(), return_delta_score = False):
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}), ignore_think_token=ignore_think_token)
    # Compute number of words in solution_str
    if not reward_config.linear_reward and not reward_config.multiplier_reward and not reward_config.sigmoid_reward: 
        return reward_response.is_correct

    if num_tokens != -1:
        if num_tokens < 0:
            # LCPO-Max
            if reward_config.sigmoid_reward:
                delta_score = get_delta_score_sigmoid(num_tokens, float(valid_response_length), reward_config.alpha)
            else:
                delta_score = get_delta_score_linear_both(num_tokens, float(valid_response_length), reward_config.alpha)
        else:
            # LCPO-Exact
            if reward_config.sigmoid_reward:
                delta_score = get_delta_score_sigmoid_exact(num_tokens, float(valid_response_length), reward_config.alpha)
            else:
                delta_score = get_delta_score_linear(num_tokens, float(valid_response_length), reward_config.alpha)
        print(f"delta_score: {delta_score}, reward_response.is_correct: {reward_response.is_correct}, num_tokens: {num_tokens}, valid_response_length: {valid_response_length}")
        correctness_score = 0 if not reward_response.is_correct else 1
        if reward_config.multiplier_reward:
            if return_delta_score:
                return max(0, delta_score) * correctness_score, delta_score
            else:
                return max(0, delta_score) * correctness_score
        else:
            if return_delta_score:
                return delta_score + correctness_score, delta_score
            else:
                return delta_score + correctness_score
    else:
        return reward_response.is_correct

def majority_at_k(generations: List[str], ground_truths: Union[str, List[str]], k: int = -1, problem: str = "", enable_llm: bool = False, ignore_think_token: bool = False, shuffle: bool = False) -> str:
    """
    Perform majority@k voting on a list of generated answers.
    
    Args:
        generations: List of generated answers from the model
        ground_truths: The ground truth answer(s) - used only for answer extraction patterns
        k: Number of top answers to consider. If -1, use all answers
        problem: The original problem text (used for ORM if enabled)
        enable_llm: Whether to use LLM as ORM for grading
        ignore_think_token: Whether to ignore the thinking token when processing answers
        
    Returns:
        The most common answer based on equivalence classes
    """
    if not isinstance(ground_truths, list) and not isinstance(ground_truths, np.ndarray):
        ground_truths = [ground_truths]
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)
    # Limit to top k if specified
    if k > 0 and k < len(generations):
        if shuffle:
            # create a copy of generations
            generations_copy = generations.copy()
            # shuffle the copy
            random.shuffle(generations_copy)
            generations = generations_copy[:k]
        else:
            generations = generations[:k]
    
    # Process each generation to extract answers
    processed_answers = []
    for gen in generations:
        # Remove thinking tokens if needed
        if ignore_think_token:
            gen = re.sub(r'<think>.*?</think>', '', gen, flags=re.DOTALL)
        
        # Extract answer if it's in a \boxed format
        if "\\boxed" in gen:
            extracted = extract_answer(gen)
            if extracted is not None:
                processed_answers.append(extracted)
        else:
            processed_answers.append(gen)
    
    # Group equivalent answers into clusters
    answer_clusters = []
    cluster_counts = []
    
    for answer in processed_answers:
        found_cluster = False
        
        # Check if the answer belongs to any existing cluster
        for i, cluster_representative in enumerate(answer_clusters):
            # Use the grading functions to check equivalence
            if grade_answer_mathd(answer, cluster_representative) or grade_answer_sympy(answer, cluster_representative):
                cluster_counts[i] += 1
                found_cluster = True
                break
        
        # If not found in any cluster, create a new one
        if not found_cluster:
            answer_clusters.append(answer)
            cluster_counts.append(1)
    # print(answer_clusters, cluster_counts)
    # Find the cluster with the highest count
    if not answer_clusters:
        return 0.0
    
    max_count_index = cluster_counts.index(max(cluster_counts))
    final_answer = answer_clusters[max_count_index]
    for truth in processed_ground_truths:
        if grade_answer_mathd(final_answer, truth) or grade_answer_sympy(final_answer, truth):
            return 1.0
    return 0.0

if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig)
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    output = reward(input)
    print(output)
