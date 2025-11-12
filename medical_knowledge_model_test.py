# ------------------------------------------------------------
# Title: Medical Knowledge Multiple-Choice Evaluation (vLLM)
# Author: Weidong Li
# Date: 2025-11-12
# Description:
#   This script evaluates a medical knowledge multiple-choice dataset
#   using the DeepSeek-R1-7B model via vLLM. It performs deterministic
#   or stochastic inference, extracts answer letters (A–D), and computes
#   accuracy across multiple epochs.
# ------------------------------------------------------------

import os
import multiprocessing
import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Ensure multiprocessing setup before importing torch/vLLM
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
multiprocessing.set_start_method("spawn", force=True)


def load_data(question_path, answer_path):
    """Load question and answer JSON files."""
    with open(question_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(answer_path, "r", encoding="utf-8") as f:
        answers = json.load(f)
    return questions, answers


def main():
    # -----------------------------
    # 1. Paths and model config
    # -----------------------------
    model_path = "checkpoints_7b"
    question_path = "medical_knowledge_question.json"
    answer_path = "medical_knowledge_answer.json"

    questions, answers = load_data(question_path, answer_path)

    # -----------------------------
    # 2. Initialize tokenizer and model (vLLM)
    # -----------------------------
    llm = LLM(
        model=model_path,
        dtype="bfloat16",             
        tensor_parallel_size=1,
        disable_log_stats=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # -----------------------------
    # 3. Sampling parameters
    # -----------------------------
    sampling_params = SamplingParams(
        temperature=0.7,               # slightly stochastic
        top_p=0.9,
        max_tokens=32,
    )

    # -----------------------------
    # 4. Build prompts
    # -----------------------------
    base_prompt = (
        "请严格按照以下格式回答问题：\n"
        "1. 只输出选项字母（A/B/C/D），不要包含任何解释、推理或其他内容。\n"
        "2. 不要添加任何额外字符或符号。\n"
        "以下是睡眠医学试题及其选项ABCD的内容："
    )

    prompts = []
    correct_answers = []
    for key, question in questions.items():
        full_prompt = f"{base_prompt}\n{question}\n请直接给出答案："
        prompts.append(full_prompt)
        correct_answers.append(answers[key])

    # -----------------------------
    # 5. Multi-epoch evaluation
    # -----------------------------
    epochs = 10
    results = []

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        outputs = llm.generate(prompts, sampling_params)
        correct_num = 0

        for i, output in enumerate(outputs):
            raw = output.outputs[0].text.strip()
            matches = re.findall(r"[A-D]", raw.upper())
            pred = matches[-1] if matches else ""
            print(f"Q{i + 1} -> Raw: {raw!r} -> Parsed: {pred}, Correct: {correct_answers[i]}")
            print("-" * 60)

            if pred == correct_answers[i].upper():
                correct_num += 1

        acc = correct_num / len(prompts)
        results.append(acc)
        print(f"Epoch {epoch + 1} Accuracy: {acc:.2%}")

    # -----------------------------
    # 6. Summary
    # -----------------------------
    print("\nAll results:", results)
    print("Average accuracy:", sum(results) / len(results))


if __name__ == "__main__":
    main()
