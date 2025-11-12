# pubmedqa_vllm_eval.py
# ------------------------------------------------------------
# Title: PubMedQA Evaluation with Baichuan-M1-14B using vLLM
# Author: Weidong Li
# Date: 2025-11-12
# Description:
#   This script evaluates a PubMedQA-style dataset using the
#   Baichuan-M1-14B model via vLLM. It performs multi-epoch inference
#   across questions, extracting yes/no/maybe responses and computing
#   accuracy.
# ------------------------------------------------------------

import os
import multiprocessing
import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 必须在导入 vLLM 前设置多进程方式（否则会报错）
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
multiprocessing.set_start_method("spawn", force=True)


def load_pubmedqa_data(file_path):
    """Load PubMedQA data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        data["questions"],
        data["contexts"],
        data["long_answer"],
        data["answers"],
    )


def main():
    # -----------------------------
    # 1. 数据与模型路径设置
    # -----------------------------
    data_path = "PubMedQA_data.json"
    model_path = "checkpoints_7b"

    questions, contexts, long_answers, answers = load_pubmedqa_data(data_path)

    # -----------------------------
    # 2. 初始化 vLLM 模型
    # -----------------------------
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        disable_log_stats=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # -----------------------------
    # 3. 推理参数设置
    # -----------------------------
    sampling_params = SamplingParams(
        temperature=0.2,     # 小温度，输出更稳定
        top_p=1.0,
        max_tokens=16,       # 仅需输出 yes/no/maybe
    )

    # -----------------------------
    # 4. 构造输入提示
    # -----------------------------
    prompts = []
    correct_labels = []

    for q, c, la, ans in zip(questions, contexts, long_answers, answers):
        prompt = (
            f"这是一些医学方面的试题，问题为: {q}，"
            f"与问题相关的上下文为：{c}，"
            f"与问题相关的文本回答为：{la}。"
            "你需要根据问题、上下文及文本回答，选择 yes、no、maybe 中最符合的一个。"
            "只输出选项（yes/no/maybe），不要任何解释。"
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)
        correct_labels.append(ans.strip().lower())

    # -----------------------------
    # 5. 多轮评估
    # -----------------------------
    epochs = 10
    results = []

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        outputs = llm.generate(prompts, sampling_params)
        correct_num = 0

        for i, output in enumerate(outputs):
            raw = output.outputs[0].text.strip().lower()
            match = re.search(r"\b(yes|no|maybe)\b", raw)
            pred = match.group(1) if match else ""
            print(f"Q{i + 1}: {raw!r} -> Parsed: {pred}, Correct: {correct_labels[i]}")
            print("-" * 60)

            if pred == correct_labels[i]:
                correct_num += 1

        acc = correct_num / len(prompts)
        results.append(acc)
        print(f"Epoch {epoch + 1} Accuracy: {acc:.2%}")

    # -----------------------------
    # 6. 总结结果
    # -----------------------------
    print("\nAll Results:", results)
    print("Average Accuracy:", sum(results) / len(results))


if __name__ == "__main__":
    main()
