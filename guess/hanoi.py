"""
Tower of Hanoi Rule-Based Problem Generator
- 모든 수학/논리 계산은 규칙 기반으로만 수행
- LLM(gpt mini)은 '문장 다듬기 / 포맷팅'만 담당하는 선택적 단계

필요:
  pip install openai
  환경변수: OPENAI_API_KEY=<your key>
"""

import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Any

# -----------------------------
# 1. 순수 규칙 기반 Hanoi 엔진
# -----------------------------

Move = Tuple[int, int, int]  # (disk, from_peg, to_peg)


def build_hanoi_moves(
    n: int, src: int, aux: int, dst: int, acc: List[Move]
) -> None:
    """
    재귀적으로 n개의 디스크를 src -> dst (aux 보조) 로 옮기는
    최적 해법의 이동 시퀀스를 acc 리스트에 (disk, from, to) 형태로 저장.
    """
    if n == 0:
        return
    # 1) n-1개: src -> aux (dst 보조)
    build_hanoi_moves(n - 1, src, dst, aux, acc)
    # 2) n번 디스크 1회: src -> dst
    acc.append((n, src, dst))
    # 3) n-1개: aux -> dst (src 보조)
    build_hanoi_moves(n - 1, aux, src, dst, acc)


def get_hanoi_moves(n: int, src: int, aux: int, dst: int) -> List[Move]:
    moves: List[Move] = []
    build_hanoi_moves(n, src, aux, dst, moves)
    return moves


def simulate_pegs(
    n: int, src: int, aux: int, dst: int, moves: List[Move], steps: int
) -> Dict[int, List[int]]:
    """
    처음 상태: 모든 디스크(n..1)가 src에 쌓여 있고(aux,dst는 비어 있음).
    steps 번의 move를 적용한 후, 각 Peg에 어떤 디스크들이 있는지 반환.
    결과: {peg: [bottom..top]} 형태
    """
    pegs: Dict[int, List[int]] = {
        src: list(range(n, 0, -1)),  # bottom..top = n..1
        aux: [],
        dst: [],
    }
    for idx in range(min(steps, len(moves))):
        disk, from_peg, to_peg = moves[idx]
        popped = pegs[from_peg].pop()
        assert (
            popped == disk
        ), f"Internal error: expected disk {disk}, got {popped}"
        pegs[to_peg].append(disk)
    return pegs


# -----------------------------
# 2. 다양한 문제 템플릿 정의
# -----------------------------

Context = Dict[str, Any]
TemplateFn = Callable[[Context], Tuple[str, str]]


def template_min_moves(ctx: Context) -> Tuple[str, str]:
    n = ctx["n"]
    src, aux, dst = ctx["src"], ctx["aux"], ctx["dst"]
    total = ctx["total_moves"]
    problem = (
        f"In a Tower of Hanoi puzzle with {n} disks, all disks start on Peg {src}.\n"
        f"The goal is to move all disks to Peg {dst} using Peg {aux} as an auxiliary peg,\n"
        f"following the usual rules (move one disk at a time, never placing a larger disk on a smaller one).\n"
        f"What is the minimum number of moves needed to complete the puzzle?"
    )
    answer = f"The minimum number of moves is {total}."
    return problem, answer


def template_kth_disk(ctx: Context) -> Tuple[str, str]:
    n = ctx["n"]
    src, aux, dst = ctx["src"], ctx["aux"], ctx["dst"]
    k = ctx["k"]
    disk_k = ctx["disk_k"]
    problem = (
        f"Consider the optimal solution of a Tower of Hanoi puzzle with {n} disks.\n"
        f"All disks start on Peg {src} and must be moved to Peg {dst} (Peg {aux} is auxiliary).\n"
        f"In this optimal sequence, which disk is moved on the {k}-th move?"
    )
    answer = f"On the {k}-th move, Disk {disk_k} is moved."
    return problem, answer


def template_kth_from_to(ctx: Context) -> Tuple[str, str]:
    n = ctx["n"]
    src, aux, dst = ctx["src"], ctx["aux"], ctx["dst"]
    k = ctx["k"]
    i_k, j_k = ctx["from_k"], ctx["to_k"]
    problem = (
        f"In the optimal {n}-disk Tower of Hanoi solution from Peg {src} to Peg {dst}\n"
        f"(with Peg {aux} as auxiliary), consider the {k}-th move in the sequence.\n"
        f"From which peg to which peg does the disk move on that {k}-th move?"
    )
    answer = f"On the {k}-th move, the disk moves from Peg {i_k} to Peg {j_k}."
    return problem, answer


def template_kth_full_triplet(ctx: Context) -> Tuple[str, str]:
    n = ctx["n"]
    src, aux, dst = ctx["src"], ctx["aux"], ctx["dst"]
    k = ctx["k"]
    disk_k, i_k, j_k = ctx["disk_k"], ctx["from_k"], ctx["to_k"]
    problem = (
        f"In an optimal Tower of Hanoi puzzle with {n} disks, all disks start on Peg {src}\n"
        f"and must be moved to Peg {dst} using Peg {aux} as auxiliary.\n"
        f"Describe the {k}-th move in the form (disk, from_peg, to_peg)."
    )
    answer = f"The {k}-th move is (disk, from_peg, to_peg) = ({disk_k}, {i_k}, {j_k})."
    return problem, answer


def template_largest_disk_move(ctx: Context) -> Tuple[str, str]:
    n = ctx["n"]
    moves: List[Move] = ctx["moves"]
    # 가장 큰 디스크는 단 한 번, src -> dst 로 이동
    largest = n
    move_index = next(
        idx for idx, (d, _, _) in enumerate(moves) if d == largest
    ) + 1  # 1-based
    problem = (
        f"In the optimal solution of a Tower of Hanoi puzzle with {n} disks,\n"
        f"on which move number does the largest disk (Disk {n}) move?"
    )
    answer = f"The largest disk (Disk {n}) moves on move {move_index}."
    return problem, answer


def template_disk_move_count(ctx: Context) -> Tuple[str, str]:
    n = ctx["n"]
    moves: List[Move] = ctx["moves"]
    # 1 이상 n 이하 아무 디스크 하나 선택
    disk = random.randint(1, n)
    count = sum(1 for d, _, _ in moves if d == disk)
    problem = (
        f"In the optimal solution for a Tower of Hanoi puzzle with {n} disks,\n"
        f"how many times does Disk {disk} move in total?"
    )
    answer = f"In the optimal solution, Disk {disk} moves {count} times."
    return problem, answer


def template_disks_on_peg_after_k(ctx: Context) -> Tuple[str, str]:
    n = ctx["n"]
    src, aux, dst = ctx["src"], ctx["aux"], ctx["dst"]
    k = ctx["k"]
    pegs_after_k: Dict[int, List[int]] = ctx["pegs_after_k"]

    # 임의의 Peg 선택
    peg = random.choice([src, aux, dst])
    disks_on_peg = sorted(pegs_after_k[peg])
    if disks_on_peg:
        disk_list_str = ", ".join(str(d) for d in disks_on_peg)
        ans_str = f"Disks {disk_list_str} are on Peg {peg}."
    else:
        ans_str = f"No disks are on Peg {peg}."

    problem = (
        f"In a Tower of Hanoi puzzle with {n} disks (Peg {src} → Peg {dst}, Peg {aux} auxiliary),\n"
        f"consider the optimal sequence of moves. After exactly {k} moves have been performed,\n"
        f"which disks are on Peg {peg}?"
    )
    answer = ans_str
    return problem, answer


def template_inverse_find_n(ctx: Context) -> Tuple[str, str]:
    n = ctx["n"]
    src, aux, dst = ctx["src"], ctx["aux"], ctx["dst"]
    k = ctx["k"]
    disk_k, i_k, j_k = ctx["disk_k"], ctx["from_k"], ctx["to_k"]
    problem = (
        f"In a certain optimal Tower of Hanoi puzzle, all disks start on Peg {src}\n"
        f"and the goal is to move them to Peg {dst} using Peg {aux} as auxiliary.\n"
        f"It is known that on move {k}, Disk {disk_k} moves from Peg {i_k} to Peg {j_k}.\n"
        f"Assuming the puzzle uses the standard rules and the solution is minimal,\n"
        f"how many disks are in this Tower of Hanoi puzzle?"
    )
    answer = f"The puzzle has {n} disks."
    return problem, answer


def template_where_is_disk_after_k(ctx: Context) -> Tuple[str, str]:
    n = ctx["n"]
    src, aux, dst = ctx["src"], ctx["aux"], ctx["dst"]
    k = ctx["k"]
    pegs_after_k: Dict[int, List[int]] = ctx["pegs_after_k"]

    # 임의의 디스크 하나 선택
    disk = random.randint(1, n)
    peg_of_disk = None
    for peg, stack in pegs_after_k.items():
        if disk in stack:
            peg_of_disk = peg
            break

    assert peg_of_disk is not None, "Internal error: disk not found on any peg."

    problem = (
        f"In an optimal Tower of Hanoi solution with {n} disks, all disks start on Peg {src}\n"
        f"and must be moved to Peg {dst}, using Peg {aux} as auxiliary.\n"
        f"After exactly {k} moves, on which peg is Disk {disk} located?"
    )
    answer = f"After {k} moves, Disk {disk} is on Peg {peg_of_disk}."
    return problem, answer


def template_disk_k_total_moves(ctx: Context) -> Tuple[str, str]:
    """
    k번째 move에 움직인 디스크가 전체에서 몇 번 움직이는지 묻는 변형.
    """
    moves: List[Move] = ctx["moves"]
    k = ctx["k"]
    disk_k = ctx["disk_k"]
    count = sum(1 for d, _, _ in moves if d == disk_k)

    problem = (
        f"In an optimal Tower of Hanoi solution, look at the {k}-th move of the sequence.\n"
        f"Let the disk moved at this step be called Disk X (here, X = Disk {disk_k}).\n"
        f"In the entire solution, how many times does this Disk X move?"
    )
    answer = f"Disk {disk_k} (the disk moved on step {k}) moves {count} times in total."
    return problem, answer


TEMPLATES: List[TemplateFn] = [
    template_min_moves,
    template_kth_disk,
    template_kth_from_to,
    template_kth_full_triplet,
    template_largest_disk_move,
    template_disk_move_count,
    template_disks_on_peg_after_k,
    template_inverse_find_n,
    template_where_is_disk_after_k,
    template_disk_k_total_moves,
]


# -----------------------------
# 3. 문제 한 개 생성 함수
# -----------------------------

def generate_raw_hanoi_problem() -> Tuple[str, str, Dict[str, Any]]:
    """
    완전히 규칙 기반으로 하노이 문제 1개와 정답 1개를 생성.

    반환:
      problem_text, answer_text, internal_metadata
    """
    # 디스크 수 및 peg 매핑 랜덤
    n = random.randint(3, 7)
    src, aux, dst = random.sample([0, 1, 2], 3)

    moves = get_hanoi_moves(n, src, aux, dst)
    total_moves = len(moves)

    # k번째 move 랜덤 선택
    k = random.randint(1, total_moves)
    disk_k, from_k, to_k = moves[k - 1]

    # k번째 move 이후 상태
    pegs_after_k = simulate_pegs(n, src, aux, dst, moves, k)

    ctx: Context = {
        "n": n,
        "src": src,
        "aux": aux,
        "dst": dst,
        "moves": moves,
        "total_moves": total_moves,
        "k": k,
        "disk_k": disk_k,
        "from_k": from_k,
        "to_k": to_k,
        "pegs_after_k": pegs_after_k,
    }

    template_fn = random.choice(TEMPLATES)
    problem_text, answer_text = template_fn(ctx)

    return problem_text, answer_text, ctx


# -----------------------------
# 4. gpt mini 연계 (포맷팅 전용)
# -----------------------------

# gpt mini 사용을 원하지 않으면, 이 import / 함수 / 플래그는 무시해도 된다.
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False


GPT_MINI_MODEL = "gpt-4.1-mini"  # 또는 계정에서 사용 가능한 작은 모델로 변경


def format_with_gpt_mini(problem: str, answer: str) -> str:
    """
    gpt mini에게 '아주 쉬운 task'만 시킨다:
      - 이미 규칙 기반으로 계산된 문제/정답을 살짝 자연스럽게 다듬고
      - 동일한 내용으로 JSON {"problem": "...", "answer": "..."} 만 출력하도록.

    반환:
      gpt mini가 만들어준 JSON 문자열 (또는 그대로 problem/answer가 들어간 텍스트)
    """
    if not _OPENAI_AVAILABLE:
        # OpenAI 라이브러리가 없는 경우, 그냥 원본을 JSON처럼 싸서 반환
        import json
        return json.dumps({"problem": problem, "answer": answer}, ensure_ascii=False)

    client = OpenAI()

    prompt = f"""
You are a very simple formatting assistant.

I will give you a Tower of Hanoi problem and its **correct** answer.
Both were computed by a rule-based algorithm, so DO NOT change any numbers,
pegs, disk indices, or the logical content of the answer.

Your ONLY job:
- lightly rewrite the English for clarity if needed,
- keep all math and logic identical,
- and return a single JSON object with two string fields:
  - "problem"
  - "answer"

Here is the raw data:

Problem:
{problem}

Answer:
{answer}
"""

    response = client.responses.create(
        model=GPT_MINI_MODEL,
        input=prompt.strip()
    )

    # responses API 결과에서 텍스트 추출
    out = response.output[0].content[0].text
    return out


# -----------------------------
# 5. 데모: 여러 문제 생성 및 출력
# -----------------------------

def demo_generate_many(num: int = 5, use_gpt_mini: bool = False) -> None:
    """
    num 개의 (문제, 정답) 페어를 생성해서 출력.
    use_gpt_mini = True 이면 gpt mini로 포맷팅까지 수행.
    """
    for idx in range(1, num + 1):
        raw_problem, raw_answer, meta = generate_raw_hanoi_problem()
        print(f"\n=== RAW PROBLEM #{idx} ===")
        print(raw_problem)
        print("\n--- RAW ANSWER ---")
        print(raw_answer)

        if use_gpt_mini:
            print("\n--- GPT MINI FORMATTED (JSON) ---")
            formatted = format_with_gpt_mini(raw_problem, raw_answer)
            print(formatted)
        print("\n" + "=" * 40)


def create_dataset_files(num_questions: int):
    """
    Hanoi 퍼즐 데이터셋 파일 생성
    
    Args:
        num_questions: 생성할 질문 수
    """
    import pandas as pd
    import json
    
    print(f"Hanoi 퍼즐 {num_questions}개 생성 중...")
    
    all_puzzles = []
    
    for i in range(num_questions):
        raw_problem, raw_answer, meta = generate_raw_hanoi_problem()
        puzzle_data = {
            'question': raw_problem,
            'answer': raw_answer,
            'n': meta['n'],
            'src': meta['src'],
            'aux': meta['aux'],
            'dst': meta['dst']
        }
        all_puzzles.append(puzzle_data)
    
    print(f"\n{len(all_puzzles)}개의 퍼즐 생성 완료")
    
    df = pd.DataFrame(all_puzzles)
    
    # 파일 저장
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # CSV
    csv_dir = PROJECT_ROOT / "data" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "hanoi.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 파일 생성: {csv_path}")
    
    # JSONL
    json_dir = PROJECT_ROOT / "data" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = json_dir / "hanoi.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in all_puzzles:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"JSONL 파일 생성: {jsonl_path}")
    
    return df, all_puzzles


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hanoi Puzzle Generator")
    parser.add_argument("--num", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_generate_many(num=5, use_gpt_mini=False)
    else:
        create_dataset_files(num_questions=args.num)
