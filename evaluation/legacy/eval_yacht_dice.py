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

# guess 디렉터리를 import 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'guess'))
from yacht_dice import YachtDiceConfig, solve_yacht_dice

load_dotenv()


def load_yacht_dice_data(jsonl_path):
    """YACHT_DICE.jsonl 파일을 로드하여 데이터프레임으로 변환"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    return df


def get_queries(model_name, config, add_chat_template):
    """
    평가용 쿼리 생성

    Args:
        model_name: 모델명
        config: YachtDiceConfig 인스턴스
        add_chat_template: 채팅 템플릿 적용 여부
    """
    if add_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = None

    yacht_dice_data = load_yacht_dice_data('data/json/YACHT_DICE_v1.jsonl')

    # config로부터 시스템 프롬프트 생성
    system_prompt = config.get_system_prompt()

    def prepare_queries(data, system_prompt, add_chat_template, tokenizer=None):
        queries = []
        for _, row in data.iterrows():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row['question']}
            ]
            if add_chat_template:
                messages = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            queries.append(messages)
        return queries

    yacht_qrys = prepare_queries(yacht_dice_data, system_prompt, add_chat_template, tokenizer)

    return yacht_qrys, yacht_dice_data


def get_response(qrys, model_name: str, engine: str):
    if not qrys:
        raise ValueError("Query list is empty. Provide at least one query.")

    if engine == 'vllm':
        try:
            tensor_parallel_size = torch.cuda.device_count()
        except AttributeError:
            raise RuntimeError("Torch is not properly configured for CUDA.")

        sampling_params = SamplingParams(temperature=0.0, max_tokens=4192)
        llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size)
        response = llm.generate(qrys, sampling_params)
        return response

    elif engine == 'litellm':
        uses_openai = any(prefix in model_name for prefix in ["gpt-", "o4", "chatgpt"]) or model_name.lower().startswith("openai/")
        if uses_openai and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다. export OPENAI_API_KEY=... 로 설정해주세요.")

        try:
            response = batch_completion(model=model_name, messages=qrys, temperature=0.0)
        except Exception as e:
            raise RuntimeError(f"LiteLLM batch_completion 호출 실패: {e}")

        if not isinstance(response, list):
            raise RuntimeError(f"예상치 못한 응답 타입: {type(response)}")

        contents = []
        for idx, item in enumerate(response):
            if not hasattr(item, "choices") or not item.choices:
                raise RuntimeError(f"인덱스 {idx}에서 유효하지 않은 응답을 수신했습니다: {item}")
            try:
                contents.append(item.choices[0].message.content)
            except Exception as e:
                raise RuntimeError(f"인덱스 {idx} 응답 파싱 실패: {e}")

        return contents

    else:
        raise ValueError(f"Engine '{engine}' is not supported. Choose 'vllm' or 'litellm'.")


def extract_final_answer_section(response: str) -> str:
    """
    응답에서 최종 정답 섹션만 추출

    중간 과정, 시행착오, 고민 과정을 제거하고 최종 정답만 반환
    """
    # 최종 정답을 나타내는 키워드들
    final_keywords = [
        r'최종[^\n]*정답',
        r'최종[^\n]*할당',
        r'최종[^\n]*배정',
        r'정답[:\s]*',
        r'결론[:\s]*',
        r'final[^\n]*answer',
        r'final[^\n]*assignment',
        r'답[:\s]*\n',
    ]

    # 키워드 이후 부분을 찾기
    for keyword in final_keywords:
        match = re.search(keyword, response, re.IGNORECASE)
        if match:
            return response[match.start():]

    # 키워드가 없으면 "총점" 이전까지의 마지막 부분 추출
    total_patterns = [
        r'총[\s]*점[:\s]*\d+',
        r'total[\s]*[:\s]*\d+',
    ]

    for pattern in total_patterns:
        matches = list(re.finditer(pattern, response, re.IGNORECASE))
        if matches:
            last_match = matches[-1]
            start_pos = max(0, last_match.start() - 500)
            return response[start_pos:]

    return response


def extract_total_score(response: str) -> Optional[int]:
    """
    LLM 응답에서 총점을 추출

    중간 과정의 점수를 무시하고 최종 총점만 추출
    """
    final_section = extract_final_answer_section(response)

    # 다양한 패턴으로 총점 찾기
    patterns = [
        r'총점[:\s]*[=\s]*(\d+)',
        r'총[^\d]*점수[:\s]*[=\s]*(\d+)',
        r'합계[:\s]*[=\s]*(\d+)',
        r'total[:\s]*[=\s]*(\d+)',
        r'최종[^\d]*점수[:\s]*[=\s]*(\d+)',
        r'전체[^\d]*점수[:\s]*[=\s]*(\d+)',
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, final_section, re.IGNORECASE))
        if matches:
            return int(matches[-1].group(1))

    # 마지막 시도: 응답 끝 부분에서 숫자 찾기
    lines = final_section.strip().split('\n')
    for line in reversed(lines[-5:]):
        if '총' in line or 'total' in line.lower() or '합' in line:
            numbers = re.findall(r'\d+', line)
            if numbers:
                return int(max(numbers, key=int))

    return None


def normalize_answer(answer_str):
    """답변을 정규화하여 비교 가능하게 만듭니다."""
    if isinstance(answer_str, str):
        # 숫자만 추출
        numbers = re.findall(r'\d+', answer_str)
        if numbers:
            return int(numbers[0])
    try:
        return int(answer_str)
    except (ValueError, TypeError):
        return None


def save_results_to_excel(results_df, model_name, accuracy, output_dir="results"):
    """평가 결과를 엑셀 파일로 저장"""

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{model_name.replace('/', '_')}_yacht_dice_v1_{timestamp}__{accuracy:.2f}.xlsx"
    filepath = os.path.join(output_dir, filename)

    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)

        summary_data = {
            'Model': [model_name],
            'Total_Questions': [len(results_df)],
            'Correct_Answers': [results_df['exact_match'].sum()],
            'Accuracy_Percentage': [accuracy],
            'Evaluation_Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\n평가 결과가 엑셀 파일로 저장되었습니다: {filepath}")
    return filepath


def evaluate_yacht_dice_model(model_name, config, save_to_excel, recalculate_answer=False):
    """
    Yacht Dice 모델 평가

    Args:
        model_name: 평가할 모델명
        config: YachtDiceConfig 인스턴스
        save_to_excel: 엑셀 저장 여부
        recalculate_answer: 정답을 다시 계산할지 여부 (커스텀 룰 사용 시 True)
    """
    yacht_qrys, yacht_data = get_queries(model_name, config, add_chat_template=False)

    print(f"총 {len(yacht_qrys)}개의 질문을 처리합니다...")
    print(f"설정: {config}")
    if recalculate_answer:
        print("⚠ 커스텀 설정으로 정답을 재계산합니다.")

    try:
        responses = get_response(yacht_qrys, model_name, engine='litellm')

        correct_cnt = 0
        total_cnt = len(yacht_data)

        results_data = []

        print("\n=== 평가 결과 ===")
        for i, (_, row) in enumerate(yacht_data.iterrows()):
            question = row['question']
            solution = row['solution']
            saved_answer = row['answer']
            model_answer = responses[i].strip()

            # 커스텀 룰 사용 시 정답 재계산
            if recalculate_answer and 'dice_results' in row:
                dice_results = row['dice_results']
                # 커스텀 config로 정답 재계산
                recalculated_score, _ = solve_yacht_dice(dice_results, config)
                correct_answer = str(recalculated_score)
                if i == 0:  # 첫 번째 문제에서만 출력
                    print(f"  정답 재계산: 원래={saved_answer}, 새로운 룰={correct_answer}")
            else:
                correct_answer = saved_answer

            # 모델 응답에서 총점 추출
            filtered_resps = extract_total_score(model_answer)

            # 정규화하여 비교
            normalized_correct = normalize_answer(correct_answer)
            normalized_model = filtered_resps

            is_correct = (normalized_model is not None and
                         normalized_correct is not None and
                         normalized_model == normalized_correct)

            if is_correct:
                correct_cnt += 1

            results_data.append({
                'id': i + 1,
                'question': question,
                'target': correct_answer,
                'solution': solution,
                'resps': model_answer,
                'filtered_resps': str(filtered_resps) if filtered_resps is not None else None,
                'normalized_target': normalized_correct,
                'normalized_model': normalized_model,
                'exact_match': is_correct,
                'model': model_name,
                'config': str(config)
            })

        results_df = pd.DataFrame(results_data)

        accuracy = correct_cnt / total_cnt * 100
        print(f"\n=== 최종 결과 ===")
        print(f"정확도: {accuracy:.2f}% ({correct_cnt}/{total_cnt})")

        if save_to_excel:
            excel_path = save_results_to_excel(results_df, model_name, accuracy)

        return accuracy, responses, results_df

    except Exception as e:
        print(f"오류 발생: {e}")
        return None, None, None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Yacht Dice 모델 평가")
    parser.add_argument("--model", type=str, default="gpt-4o", help="평가할 모델명")
    parser.add_argument("--save-excel", action="store_true", default=True, help="엑셀로 저장")
    parser.add_argument("--recalculate", action="store_true", help="커스텀 설정으로 정답 재계산")

    # 룰 변경 옵션
    parser.add_argument("--bonus-threshold", type=int, default=63, help="보너스 경계 점수")
    parser.add_argument("--bonus-points", type=int, default=35, help="보너스 점수")
    parser.add_argument("--full-house-points", type=int, default=25, help="Full House 점수")
    parser.add_argument("--small-straight-points", type=int, default=30, help="Small Straight 점수")
    parser.add_argument("--large-straight-points", type=int, default=40, help="Large Straight 점수")
    parser.add_argument("--yacht-points", type=int, default=50, help="Yacht 점수")
    parser.add_argument("--optimization-goal", type=str, default="maximize",
                       choices=["maximize", "minimize"], help="최적화 목표")

    args = parser.parse_args()

    # Config 생성
    config = YachtDiceConfig(
        bonus_threshold=args.bonus_threshold,
        bonus_points=args.bonus_points,
        full_house_points=args.full_house_points,
        small_straight_points=args.small_straight_points,
        large_straight_points=args.large_straight_points,
        yacht_points=args.yacht_points,
        optimization_goal=args.optimization_goal
    )

    evaluate_yacht_dice_model(args.model, config, args.save_excel, args.recalculate)
