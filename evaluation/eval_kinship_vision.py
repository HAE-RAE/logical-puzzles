import os
import json
import base64
import re
import time
import pandas as pd
from litellm import batch_completion
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def load_kinship_vision_data(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    return df

def extract_answer_letter(text):
    """Extract answer letter (A-E) from model response"""
    if not text:
        return ""
    
    text = text.strip().upper()
    
    match = re.search(r'(?:^|\s)([A-E])(?:\s|$|[.,!?])', text)
    if match:
        return match.group(1)
    
    match = re.search(r'^([A-E])', text)
    if match:
        return match.group(1)
    
    match = re.search(r'([A-E])(?:[^A-Z]|$)', text)
    if match:
        return match.group(1)
    
    match = re.search(r'[A-E]', text)
    if match:
        return match.group(0)
    
    return ""

def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_queries(system_prompt, image_base64, kinship_data):
    queries = []
    
    for _, row in kinship_data.iterrows():
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    },
                    {
                        "type": "text",
                        "text": row['question']
                    }
                ]
            }
        ]
        queries.append(messages)
    
    return queries

def get_response(qrys, model_name: str, batch_size=3, delay_between_batches=3):
    if not qrys:
        raise ValueError("Query list is empty. Provide at least one query.")
    
    uses_openai = any(prefix in model_name for prefix in ["gpt-", "o4", "chatgpt"]) or model_name.lower().startswith("openai/")
    if uses_openai and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set. Please set it with: export OPENAI_API_KEY=...")
    
    contents = []
    temperature = 1.0 if "gpt-5" in model_name else 0.0
    total = len(qrys)
    total_batches = (total + batch_size - 1) // batch_size
    
    for batch_idx in range(0, total, batch_size):
        batch = qrys[batch_idx:batch_idx+batch_size]
        current_batch_num = batch_idx // batch_size + 1
        
        print(f"Processing batch {current_batch_num}/{total_batches} ({len(batch)} questions)...")
        
        max_retries = 3
        for retry in range(max_retries):
            try:
                response = batch_completion(model=model_name, messages=batch, temperature=temperature)
                
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

def save_results_to_excel(results_df, model_name, accuracy, output_dir="results"):
    """Save evaluation results to Excel file"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{model_name.replace('/', '_')}_kinship_vision_{timestamp}__{accuracy:.2f}.xlsx"
    filepath = os.path.join(output_dir, filename)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        detailed_columns = ['id', 'question', 'difficulty', 'correct_letter', 'model_response', 'model_letter', 'exact_match']
        results_df[detailed_columns].to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        summary_data = {
            'Model': [model_name],
            'Total_Questions': [len(results_df)],
            'Correct_Answers': [results_df['exact_match'].sum()],
            'Accuracy_Percentage': [accuracy],
            'Evaluation_Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # By_Difficulty
        if 'difficulty' in results_df.columns:
            difficulty_stats = []
            for difficulty in ['Easy', 'Medium', 'Hard']:
                diff_data = results_df[results_df['difficulty'] == difficulty]
                if len(diff_data) > 0:
                    correct = diff_data['exact_match'].sum()
                    total = len(diff_data)
                    accuracy_pct = (correct / total * 100) if total > 0 else 0
                    difficulty_stats.append({
                        'Difficulty': difficulty,
                        'Total_Questions': total,
                        'Correct_Answers': correct,
                        'Accuracy_Percentage': accuracy_pct
                    })
            
            if difficulty_stats:
                difficulty_df = pd.DataFrame(difficulty_stats)
                difficulty_df.to_excel(writer, sheet_name='By_Difficulty', index=False)
    
    print(f"\nEvaluation results saved to Excel file: {filepath}")
    return filepath

def evaluate_kinship_vision_model(model_name, save_to_excel=True):
    system_prompt = """당신은 한국어 가족 관계 호칭 문제를 푸는 전문가입니다. 

### 규칙
1. 제공된 가족 사진을 참고하여, 주어진 대화에서 설명하는 인물을 찾으세요.
2. 대화를 단계별로 분석하여 각 인물 간의 관계를 추론하세요.
3. 문제에 제시된 선택지 중 정답에 해당하는 알파벳(A, B, C, D, E)만 답하세요.
4. 추가 설명 없이 알파벳 하나만 출력하세요.

### 출력 형식
정답 알파벳만 출력하세요. 예: A

"""

    print(f"Model: {model_name}")
    print(f"System Prompt: {system_prompt}")
    print("=" * 50)
    
    jsonl_path = os.path.join(PROJECT_ROOT, 'data', 'json', 'kinship_vision.jsonl')
    kinship_data = load_kinship_vision_data(jsonl_path)
    
    image_path = os.path.join(PROJECT_ROOT, 'evaluation_data', 'kinshop_vision', 'kinship.jpg')
    image_base64 = encode_image_to_base64(image_path)
    
    print(f"Processing {len(kinship_data)} questions...")
    print(f"Image: {image_path}")
    
    kinship_qrys = get_queries(system_prompt, image_base64, kinship_data)
    
    try:
        responses = get_response(kinship_qrys, model_name)
        
        correct_count = 0
        total_count = len(kinship_data)
        
        results_data = []
        
        print("\n=== Evaluation Results ===")
        for i, (_, row) in enumerate(kinship_data.iterrows()):
            question = row['question']
            correct_letter = row['answer']
            
            if 'choices' in row and isinstance(row['choices'], dict):
                choices = row['choices']
            else:
                choices = {}
            
            difficulty = row.get('difficulty', 'Unknown')
            
            model_answer_raw = responses[i].strip()
            model_letter = extract_answer_letter(model_answer_raw)
            
            is_correct = (model_letter == correct_letter)
            
            if is_correct:
                correct_count += 1
            
            results_data.append({
                'id': i + 1,
                'question': question,
                'difficulty': difficulty,
                'correct_letter': correct_letter,
                'model_response': model_answer_raw,
                'model_letter': model_letter,
                'exact_match': 1 if is_correct else 0,
                'model': model_name
            })
            
            print(f"\nQuestion {i+1} ({difficulty}): {question[:100]}...")
            print(f"Correct Answer: {correct_letter}")
            print(f"Model Answer: {model_letter} (Raw: {model_answer_raw})")
            print(f"Result: {'✓' if is_correct else '✗'}")
        
        results_df = pd.DataFrame(results_data)
        
        accuracy = correct_count / total_count * 100
        print(f"\n=== Final Results ===")
        print(f"Overall Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
        
        if 'difficulty' in results_df.columns:
            print(f"\n=== Accuracy by Difficulty ===")
            for difficulty in ['Easy', 'Medium', 'Hard']:
                diff_data = results_df[results_df['difficulty'] == difficulty]
                if len(diff_data) > 0:
                    diff_correct = diff_data['exact_match'].sum()
                    diff_total = len(diff_data)
                    diff_accuracy = (diff_correct / diff_total * 100) if diff_total > 0 else 0
                    print(f"{difficulty}: {diff_accuracy:.2f}% ({diff_correct}/{diff_total})")
        
        if save_to_excel:
            excel_path = save_results_to_excel(results_df, model_name, accuracy)
        
        return accuracy, responses, results_df
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    model_name = "gpt-4o"
    # model_name = "gpt-5"
    # model_name = "gpt-5-mini"
    # model_name = "gpt-5-nano"
    
    save_to_excel = True

    evaluate_kinship_vision_model(model_name, save_to_excel)

