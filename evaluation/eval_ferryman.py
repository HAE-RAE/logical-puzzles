import os
import json
import pandas as pd
from litellm import batch_completion
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def load_ferryman_data(jsonl_path):
    """FERRYMAN.jsonl 파일을 로드하여 데이터프레임으로 변환"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    return df

def get_queries(model_name, system_prompt, add_chat_template):
    
    if add_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = None
        
    ferryman_data = load_ferryman_data('data/json/FERRYMAN_v4.jsonl')

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

    ferryman_qrys = prepare_queries(ferryman_data, system_prompt, add_chat_template, tokenizer)

    return ferryman_qrys, ferryman_data

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
            # 에러 객체가 섞여 들어오는 경우 방어
            if not hasattr(item, "choices") or not item.choices:
                raise RuntimeError(f"인덱스 {idx}에서 유효하지 않은 응답을 수신했습니다: {item}")
            try:
                contents.append(item.choices[0].message.content)
            except Exception as e:
                raise RuntimeError(f"인덱스 {idx} 응답 파싱 실패: {e}")

        return contents

    else:
        raise ValueError(f"Engine '{engine}' is not supported. Choose 'vllm' or 'litellm'.")

def save_results_to_excel(results_df, model_name, accuracy, output_dir="results"):
    """평가 결과를 엑셀 파일로 저장"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{model_name.replace('/', '_')}_ferryman_v3_{timestamp}__{accuracy:.2f}.xlsx"
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

def evaluate_ferryman_model(model_name, save_to_excel):
    system_prompt = """당신은 뱃사공 운항 문제를 정확히 해결하는 전문가입니다.

### 규칙
1. 주어진 운항 규정을 모두 고려하여 단계별로 분석하세요.
2. 속도 제한, 의무 휴식, 화물 규정을 모두 적용하여 계산하세요.
3. 문제를 푼 후, 최종 답변을 다음과 같은 형식으로 작성하세요: $\\boxed{N시간 M분}$.
4. \\boxed{} 안에는 "X시간 Y분" 형식으로만 답하세요. 다른 단위나 설명은 포함하지 마세요.
"""
    
    ferryman_qrys, ferryman_data = get_queries(model_name, system_prompt, add_chat_template=False)
    
    print(f"총 {len(ferryman_qrys)}개의 질문을 처리합니다...")
    
    try:
        responses = get_response(ferryman_qrys, model_name, engine='litellm')
        
        correct_cnt = 0
        total_cnt = len(ferryman_data)
        
        results_data = []
        
        print("\n=== 평가 결과 ===")
        for i, (_, row) in enumerate(ferryman_data.iterrows()):
            question = row['question']
            solution = row['solution']
            correct_answer = row['answer']
            model_answer = responses[i].strip()
            
            def extract_boxed_answer(answer_str):
                """\\boxed{} 형태의 답변을 추출합니다."""
                import re
                if isinstance(answer_str, str):
                    boxed_pattern = r'\\boxed\{([^}]+)\}'
                    match = re.search(boxed_pattern, answer_str)
                    if match:
                        return match.group(1).strip()
                return answer_str
            
            def normalize_answer(answer_str):
                """답변을 정규화하여 비교 가능하게 만듭니다."""
                if isinstance(answer_str, str):
                    import re
                    time_pattern = r'(\d+)시간\s*(\d+)분'
                    match = re.search(time_pattern, answer_str)
                    if match:
                        hours = int(match.group(1))
                        minutes = int(match.group(2))
                        return f"{hours}시간 {minutes}분"
                    numbers = re.findall(r'\d+', answer_str)
                    if len(numbers) >= 2:
                        return f"{numbers[0]}시간 {numbers[1]}분"
                return str(answer_str)
            
            filtered_resps = extract_boxed_answer(model_answer)
            
            normalized_correct = normalize_answer(correct_answer)
            normalized_model = normalize_answer(filtered_resps)
            
            is_correct = normalized_model == normalized_correct
            
            if is_correct:
                correct_cnt += 1
            
            results_data.append({
                'id': i + 1,
                'question': question,
                'target': correct_answer,
                'solution': solution,
                'resps': model_answer,
                'filtered_resps': filtered_resps,
                'normalized_target': normalized_correct,
                'normalized_model': normalized_model,
                'exact_match': is_correct,
                'model': model_name
            })
            
            # print(f"\n질문 {i+1}: {question[:100]}...")
            # print(f"정답: {correct_answer} -> 정규화: {normalized_correct}")
            # print(f"모델 답변: {model_answer} -> 추출/정규화: {normalized_model}")
            # print(f"정확도: {'✓' if is_correct else '✗'}")
        
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
    model_name="gpt-4o"
    # model_name="gpt-4.1"
    # model_name="claude-3-5-sonnet-20240620"
    # model_name="claude-sonnet-4-20250514"
    # model_name="claude-opus-4-1-20250805"
    save_to_excel=True
    evaluate_ferryman_model(model_name, save_to_excel)
