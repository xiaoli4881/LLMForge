# ------------------------------------------------------------
# Title: Medical Ethics Single-Choice Evaluation with vLLM
# Author: Weidong Li
# Date: 2025-11-12
# Description:
#   This script evaluates a JSON-based medical ethics question set
#   using the Qwen2.5-0.5B model via vLLM. It performs deterministic
#   multiple-choice inference and computes accuracy across epochs.
# ------------------------------------------------------------

import os
import multiprocessing

# Set multiprocessing method before any CUDA/vLLM initialization
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
multiprocessing.set_start_method("spawn", force=True)

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import re


def main():
    data_path = "医学伦理测试单选题-上交大.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_path = "Qwen2.5-0.5B"

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        disable_log_stats=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
    )

    prompt_prefix = (
        "以下是一道医学伦理单选题。"
        "请只回答选项对应的英文字母（A、B、C、D或E），"
        "禁止写任何理由或分析，不要解释，只输出最后的选择结果即一个大写字母。"
        "只输出最后的选择结果"
        "题目如下："
    )

    all_questions = []
    correct_answers = []

    for key, value in data.items():
        input_text = prompt_prefix + key
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_text},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_questions.append(chat_text)
        correct_answers.append(value)

    epochs = 10
    results = []

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        outputs = llm.generate(all_questions, sampling_params)
        correct_num = 0

        for i, output in enumerate(outputs):
            raw = output.outputs[0].text.strip()
            matches = re.findall(r"[A-E]", raw.upper())
            pred = matches[-1] if matches else ""
            print(f"Q{i + 1} -> Raw: {raw!r} -> Parsed: {pred}, Correct: {correct_answers[i]}")
            print("-" * 60)

            if pred == correct_answers[i].upper():
                correct_num += 1

        acc = correct_num / len(all_questions)
        results.append(acc)
        print(f"Epoch {epoch + 1} Accuracy: {acc:.2%}")

    print("\nAll results:", results)
    print("Average accuracy:", sum(results) / len(results))


if __name__ == "__main__":
    main()
