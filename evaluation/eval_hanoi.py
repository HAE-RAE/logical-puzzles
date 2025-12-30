import os
import sys
import json
import re
import pandas as pd
from litellm import batch_completion
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

load_dotenv()


# ======================================================
# (1) ------------  RULE-BASED HANOI ENGINE  ------------
# ======================================================

Move = Tuple[int, int, int]


def build_hanoi_moves(n: int, src: int, aux: int, dst: int, acc: List[Move]):
    if n == 0:
        return
    build_hanoi_moves(n - 1, src, dst, aux, acc)
    acc.append((n, src, dst))
    build_hanoi_moves(n - 1, aux, src, dst, acc)


def get_hanoi_moves(n: int, src: int, aux: int, dst: int) -> List[Move]:
    acc: List[Move] = []
    build_hanoi_moves(n, src, aux, dst, acc)
    return acc


def create_hanoi_dataset(num_samples: int = 1000) -> pd.DataFrame:
    """
    완전 Rule-Based Hanoi 문제/정답 JSONL 데이터를 생성.
    (문제: question, 정답: answer)
    """
    records = []
    for i in range(num_samples):
        n = random.randint(3, 7)
        src, aux, dst = random.sample([0, 1, 2], 3)

        moves = get_hanoi_moves(n, src, aux, dst)
        total_moves = len(moves)

        # 하나의 랜덤 질문 유형 적용 (여기서는 "k번째 이동" 문제)
        k = random.randint(1, total_moves)
        disk_k, from_k, to_k = moves[k - 1]

        question = (
            f"There are {n} disks initially on Peg {src}. "
            f"They must be moved to Peg {dst} using Peg {aux} as auxiliary. "
            f"All moves follow the optimal Tower of Hanoi solution.\n\n"
            f"Question: On the {k}-th move, which disk moves, and from which peg to which peg?\n"
            f"Answer format: (disk, from, to)"
        )

        answer = f"({disk_k}, {from_k}, {to_k})"

        records.append({"question": question, "answer": answer})

    df = pd.DataFrame(records)
    df.to_json("HANOI_RULE_BASED.jsonl", orient="records", lines=True, force_ascii=False)
    return df


# ======================================================
# (2) --------- QUERY PREPARATION (LLM INPUT) ----------
# ======================================================

def load_hanoi_data(jsonl_path: str) -> pd.DataFrame:
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)


def get_hanoi_queries(model_name: str, add_chat_template: bool = False):
    df = load_hanoi_data("HANOI_RULE_BASED.jsonl")

    if add_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = None

    def prepare_queries():
        queries = []
        for _, row in df.iterrows():
            messages = [
                {"role": "system", "content": "You must answer ONLY in the format (disk, from, to)."},
                {"role": "user", "content": row["question"]}
            ]
            if add_chat_template:
                messages = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            queries.append(messages)
        return queries

    return prepare_queries(), df


# ======================================================
# (3) -------------- LLM RESPONSE HANDLING -------------
# ======================================================

def get_response(qrys, model_name: str, engine: str):
    if engine == "vllm":
        tensor_parallel_size = torch.cuda.device_count()
        sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)
        llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
        output = llm.generate(qrys, sampling_params)
        return [o.outputs[0].text for o in output]

    elif engine == "litellm":
        response = batch_completion(model=model_name, messages=qrys, temperature=0.0)

        contents = []
        for item in response:
            contents.append(item.choices[0].message.content)
        return contents
    else:
        raise ValueError("engine must be 'vllm' or 'litellm'")


# ======================================================
# (4) ----------- EXTRACT MODEL FINAL ANSWER -----------
# ======================================================

def extract_triplet(ans: str) -> Optional[Tuple[int, int, int]]:
    """
    (disk, from, to) 형태만 추출
    """
    nums = re.findall(r"\d+", ans)
    if len(nums) >= 3:
        return int(nums[0]), int(nums[1]), int(nums[2])
    return None


# ======================================================
# (5) ---------------- MODEL EVALUATION ----------------
# ======================================================

def evaluate_hanoi_model(model_name, save_to_excel=True):
    qrys, data = get_hanoi_queries(model_name, add_chat_template=False)

    print(f"총 {len(qrys)}개 문제 평가 시작…")

    responses = get_response(qrys, model_name, engine="litellm")

    results = []
    correct = 0

    for i, row in data.iterrows():
        gold = extract_triplet(row["answer"])
        model_ans = extract_triplet(responses[i])

        ok = gold == model_ans

        if ok:
            correct += 1

        results.append({
            "id": i + 1,
            "question": row["question"],
            "gold_answer": row["answer"],
            "model_raw": responses[i],
            "model_extracted": model_ans,
            "exact_match": ok
        })

    acc = correct / len(data) * 100
    df = pd.DataFrame(results)

    print(f"\n=== 최종 정확도: {acc:.2f}% ===")

    if save_to_excel:
        save_hanoi_excel(df, model_name, acc)

    return acc, df


# ======================================================
# (6) ---------------- SAVE RESULTS --------------------
# ======================================================

def save_hanoi_excel(df, model_name: str, acc: float):
    os.makedirs("results_hanoi", exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    fname = f"{model_name.replace('/', '_')}_HANOI_{ts}__{acc:.2f}.xlsx"

    path = os.path.join("results_hanoi", fname)
    df.to_excel(path, index=False)
    print(f"결과 저장: {path}")


# ======================================================
# (7) ---------------- MAIN ENTRY ----------------------
# ======================================================

if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Hanoi LLM 평가")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--generate", action="store_true", help="새로운 Hanoi 데이터 생성")

    args = parser.parse_args()

    if args.generate:
        print("새로운 Hanoi Rule-Based 데이터 생성 중…")
        create_hanoi_dataset(1000)
        print("생성 완료.")

    evaluate_hanoi_model(args.model, save_to_excel=True)
