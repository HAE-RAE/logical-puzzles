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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def load_kinship_data(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    return df

def get_queries(model_name, system_prompt, add_chat_template=True):
    
    if add_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = None
        
    jsonl_path = os.path.join(PROJECT_ROOT, 'data', 'json', 'KINSHIP_v1.jsonl')
    kinship_data = load_kinship_data(jsonl_path)

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

    kinship_qrys = prepare_queries(kinship_data, system_prompt, add_chat_template, tokenizer)

    return kinship_qrys, kinship_data

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
            raise RuntimeError("OPENAI_API_KEY environment variable is not set. Please set it using export OPENAI_API_KEY=...")

        try:
            response = batch_completion(model=model_name, messages=qrys, temperature=0.0)
        except Exception as e:
            raise RuntimeError(f"LiteLLM batch_completion call failed: {e}")

        if not isinstance(response, list):
            raise RuntimeError(f"Unexpected response type: {type(response)}")

        contents = []
        for idx, item in enumerate(response):
            # Error object defense
            if not hasattr(item, "choices") or not item.choices:
                raise RuntimeError(f"Invalid response received at index {idx}: {item}")
            try:
                contents.append(item.choices[0].message.content)
            except Exception as e:
                raise RuntimeError(f"Failed to parse response at index {idx}: {e}")

        return contents

    else:
        raise ValueError(f"Engine '{engine}' is not supported. Choose 'vllm' or 'litellm'.")

def save_results_to_excel(results_df, model_name, accuracy, output_dir="results"):
    """Save evaluation results to an Excel file"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename (including model name and timestamp)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{model_name.replace('/', '_')}_kinship_{timestamp}__{accuracy:.2f}.xlsx"
    filepath = os.path.join(output_dir, filename)
    
    # Save to Excel file
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Detailed results sheet
        results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        # Summary results sheet
        summary_data = {
            'Model': [model_name],
            'Total_Questions': [len(results_df)],
            'Correct_Answers': [results_df['exact_match'].sum()],
            'Accuracy_Percentage': [accuracy],
            'Evaluation_Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nEvaluation results saved to Excel file: {filepath}")
    return filepath

def evaluate_kinship_model(model_name, save_to_excel):
    system_prompt = """당신은 한국어 가족 관계 호칭을 정확히 파악하는 전문가입니다. 

### 규칙
1. 주어진 가족 관계를 단계별로 분석하여 최종 호칭을 도출하세요.
3. 답변은 띄어쓰기 없이 오직 호칭 단어 하나만 제시하세요.
4. 정답 외의 추가 설명(예: (아버지의 형제 중 형은 큰아버지) 등)이나 존칭 표현(예: -님)을 포함하지 마세요.

### 가능한 정답 목록
다음 호칭 중에서 가장 적절한 하나를 선택하세요: 고모, 고모부, 고모부님, 고모외할머니, 고모외할아버지, 고모할머니, 고모할아버지, 고종사촌, 내종형제, 대고모, 대고모부, 대이모, 대이모부, 도련님, 동서, 백모, 백부, 백조모, 백조부, 사촌, 삼촌, 서방, 서방님, 숙모, 숙부, 숙조모, 숙조부, 시누이, 시숙, 시아버지, 시어머니, 아가씨, 아버님, 아주머니, 아주버님, 어머님, 외대고모, 외대고모부, 외사촌, 외삼촌, 외삼촌할아버지, 외숙모, 외숙부, 외조모, 외조부, 외종조모, 외종조부, 외종형제, 외증조모, 외증조부, 외할머니, 외할아버지, 이모, 이모부, 이모할머니, 이모할아버지, 이종사촌, 이종형제, 작은아버지, 작은어머니, 작은외삼촌, 작은외숙모, 작은외할머니, 작은외할아버지, 작은이모, 작은이모부, 작은이모할머니, 작은할머니, 작은할아버지, 장모, 장모님, 장인, 장인어른, 제부, 종형제, 증조외할머니, 증조외할아버지, 증조할머니, 증조할아버지, 진외이모할머니, 진외이모할아버지, 진외종조부, 진외할아버지, 처남, 처남댁, 처제, 처형, 친할머니, 친할아버지, 큰아버지, 큰어머니, 큰외삼촌, 큰외숙모, 큰외할머니, 큰외할아버지, 큰이모, 큰이모부, 큰이모할머니, 큰할머니, 큰할아버지, 형님
"""

    print(f"Model: {model_name}")
    print(f"System prompt: {system_prompt}")
    print("=" * 50)
    
    kinship_qrys, kinship_data = get_queries(model_name, system_prompt, add_chat_template=False)
    
    print(f"Processing {len(kinship_qrys)} questions...")
    
    try:
        responses = get_response(kinship_qrys, model_name, engine='litellm')
        
        # Analyze results
        correct_count = 0
        total_count = len(kinship_data)
        
        results_data = []
        
        print("\n=== Evaluation Results ===")
        for i, (_, row) in enumerate(kinship_data.iterrows()):
            question = row['question']
            correct_answer = row['answer']
            model_answer = responses[i].strip()
            
            if isinstance(correct_answer, list):
                is_correct = model_answer in correct_answer
            else:
                # If answer is a string: handle multiple answers separated by commas
                # Example: "증조외할아버지, 외증조부" -> ["증조외할아버지", "외증조부"]
                answer_list = [ans.strip() for ans in str(correct_answer).split(',')]
                is_correct = model_answer in answer_list
            
            if is_correct:
                correct_count += 1
            
            results_data.append({
                'id': i + 1,
                'question': question,
                'target': correct_answer,
                'resps': model_answer,
                'exact_match': is_correct,
                'model': model_name
            })
            
            # print(f"\nQuestion {i+1}: {question}")
            # print(f"Correct answer: {correct_answer}")
            # print(f"Model answer: {model_answer}")
            # print(f"Accuracy: {'✓' if is_correct else '✗'}")
        
        results_df = pd.DataFrame(results_data)
        
        accuracy = correct_count / total_count * 100
        print(f"\n=== Final Results ===")
        print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
        
        if save_to_excel:
            excel_path = save_results_to_excel(results_df, model_name, accuracy)
        
        return accuracy, responses, results_df
        
    except Exception as e:
        print(f"Error occurred: {e}")
        return None, None, None

if __name__ == "__main__":
    model_name="gpt-4o"
    # model_name="gpt-5"
    # model_name="gpt-5-mini"
    
    save_to_excel=True

    evaluate_kinship_model(model_name, save_to_excel)
