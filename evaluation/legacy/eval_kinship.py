import os
import json
import re
import time
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
        
    jsonl_path = os.path.join(PROJECT_ROOT, 'data', 'json', 'kinship.jsonl')
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

        batch_size = 20
        delay_between_batches = 5
        
        contents = []
        total = len(qrys)
        total_batches = (total + batch_size - 1) // batch_size
        
        for batch_idx in range(0, total, batch_size):
            batch = qrys[batch_idx:batch_idx+batch_size]
            current_batch_num = batch_idx // batch_size + 1
            
            print(f"Processing batch {current_batch_num}/{total_batches} ({len(batch)} questions)...")
            
            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = batch_completion(model=model_name, messages=batch, temperature=0.0)
                    
                    if not isinstance(response, list):
                        raise RuntimeError(f"Unexpected response type: {type(response)}")
                    
                    for idx, item in enumerate(response):
                        if not hasattr(item, "choices") or not item.choices:
                            raise RuntimeError(f"Invalid response at index {batch_idx+idx}: {item}")
                        try:
                            contents.append(item.choices[0].message.content)
                        except Exception as e:
                            raise RuntimeError(f"Failed to parse response at index {batch_idx+idx}: {e}")
                    
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    if "RateLimitError" in error_str or "rate limit" in error_str.lower():
                        if retry < max_retries - 1:
                            wait_time = delay_between_batches * (retry + 2)
                            print(f"  Rate limit hit. Retrying in {wait_time}s... (attempt {retry+2}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            raise RuntimeError(f"Rate limit exceeded after {max_retries} retries: {e}")
                    else:
                        raise RuntimeError(f"LiteLLM batch_completion failed at batch {current_batch_num}: {e}")
            
            if current_batch_num < total_batches:
                time.sleep(delay_between_batches)

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
        # Detailed results sheet - select only specific columns
        export_columns = ['id', 'question', 'difficulty', 'correct_title', 'correct_letter', 
                         'model_response', 'model_letter', 'exact_match']
        results_df[export_columns].to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        # Overall summary sheet
        summary_data = {
            'Model': [model_name],
            'Total_Questions': [len(results_df)],
            'Correct_Answers': [results_df['exact_match'].sum()],
            'Accuracy_Percentage': [accuracy],
            'Evaluation_Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Difficulty-wise summary sheet
        if 'difficulty' in results_df.columns:
            difficulty_stats = []
            for diff in ['Easy', 'Medium', 'Hard']:
                diff_df = results_df[results_df['difficulty'] == diff]
                if len(diff_df) > 0:
                    correct = diff_df['exact_match'].sum()
                    total = len(diff_df)
                    acc = (correct / total * 100) if total > 0 else 0
                    num_choices = diff_df['num_choices'].iloc[0] if 'num_choices' in diff_df.columns else 'N/A'
                    difficulty_stats.append({
                        'Difficulty': diff,
                        'Num_Choices': num_choices,
                        'Total_Questions': total,
                        'Correct_Answers': correct,
                        'Accuracy_Percentage': f"{acc:.2f}%"
                    })
            
            if difficulty_stats:
                difficulty_df = pd.DataFrame(difficulty_stats)
                difficulty_df.to_excel(writer, sheet_name='By_Difficulty', index=False)
    
    print(f"\nEvaluation results saved to Excel file: {filepath}")
    return filepath

def evaluate_kinship_model(model_name, save_to_excel):
    system_prompt = """당신은 한국어 가족 관계 호칭 문제를 푸는 전문가입니다. 

### 규칙
1. 주어진 가족 관계를 단계별로 분석하여 올바른 호칭을 찾으세요.
2. 문제에 제시된 선택지 중 정답에 해당하는 알파벳(A, B, C, D, E)만 답하세요.
3. 추가 설명 없이 알파벳 하나만 출력하세요.

### 출력 형식
정답 알파벳만 출력하세요. 예: A

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
            correct_letter = row['answer']  # This is the correct letter (A, B, C, etc.)
            model_answer_raw = responses[i].strip()
            difficulty = row.get('difficulty', 'Unknown')
            
            choices = row.get('choices', {})
            num_choices = len(choices) if choices else 'N/A'
            
            if choices and correct_letter in choices:
                correct_title = choices[correct_letter]
            else:
                correct_title = "N/A"
            
            model_answer = model_answer_raw.upper().strip()
            
            letter_match = re.search(r'(?:^|[^A-Z])([A-E])(?:[^A-Z]|$)', model_answer)
            
            if not letter_match:
                letter_match = re.search(r'^([A-E])', model_answer)
            
            if not letter_match:
                letter_match = re.search(r'([A-E])(?:[^A-Z]|$)', model_answer)
            
            if letter_match:
                model_letter = letter_match.group(1)
            else:
                letter_match = re.search(r'([A-E])', model_answer)
                model_letter = letter_match.group(1) if letter_match else ""
            
            is_correct = (model_letter == correct_letter)
            
            if is_correct:
                correct_count += 1
            
            results_data.append({
                'id': i + 1,
                'question': question,
                'difficulty': difficulty,
                'correct_title': correct_title,
                'correct_letter': correct_letter,
                'model_response': model_answer_raw,
                'model_letter': model_letter,
                'exact_match': 1 if is_correct else 0,
                'num_choices': num_choices,  # Keep for internal use
                'model': model_name  # Keep for internal use
            })
            
            # print(f"\nQuestion {i+1}: {question[:100]}...")
            # print(f"Correct: {correct_letter} ({correct_title})")
            # print(f"Model: {model_letter} (raw: {model_answer_raw})")
            # print(f"Result: {'✓' if is_correct else '✗'}")
        
        results_df = pd.DataFrame(results_data)
        
        accuracy = correct_count / total_count * 100
        print(f"\n=== Overall Results ===")
        print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
        
        # Print difficulty-wise results
        if 'difficulty' in results_df.columns:
            print(f"\n=== Results by Difficulty ===")
            for diff in ['Easy', 'Medium', 'Hard']:
                diff_df = results_df[results_df['difficulty'] == diff]
                if len(diff_df) > 0:
                    diff_correct = diff_df['exact_match'].sum()
                    diff_total = len(diff_df)
                    diff_acc = (diff_correct / diff_total * 100) if diff_total > 0 else 0
                    num_choices = diff_df['num_choices'].iloc[0] if 'num_choices' in diff_df.columns else 'N/A'
                    print(f"{diff:7s} ({str(num_choices):>2s} choices): {diff_acc:5.2f}% ({diff_correct}/{diff_total})")
        
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
