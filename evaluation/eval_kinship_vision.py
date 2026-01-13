import os
import json
import base64
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

def get_response(qrys, model_name: str):
    if not qrys:
        raise ValueError("Query list is empty. Provide at least one query.")
    
    uses_openai = any(prefix in model_name for prefix in ["gpt-", "o4", "chatgpt"]) or model_name.lower().startswith("openai/")
    if uses_openai and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set. Please set it with: export OPENAI_API_KEY=...")
    
    try:
        temperature = 1.0 if "gpt-5" in model_name else 0.0
        response = batch_completion(model=model_name, messages=qrys, temperature=temperature)
    except Exception as e:
        raise RuntimeError(f"LiteLLM batch_completion call failed: {e}")
    
    if not isinstance(response, list):
        raise RuntimeError(f"Unexpected response type: {type(response)}")
    
    contents = []
    for idx, item in enumerate(response):
        if not hasattr(item, "choices") or not item.choices:
            raise RuntimeError(f"Invalid response received at index {idx}: {item}")
        try:
            contents.append(item.choices[0].message.content)
        except Exception as e:
            raise RuntimeError(f"Failed to parse response at index {idx}: {e}")
    
    return contents

def save_results_to_excel(results_df, model_name, accuracy, output_dir="results"):
    """Save evaluation results to Excel file"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"{model_name.replace('/', '_')}_kinship_vision_{timestamp}__{accuracy:.2f}.xlsx"
    filepath = os.path.join(output_dir, filename)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
        
        summary_data = {
            'Model': [model_name],
            'Total_Questions': [len(results_df)],
            'Correct_Answers': [results_df['Is_Correct'].sum()],
            'Accuracy_Percentage': [accuracy],
            'Evaluation_Date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nEvaluation results saved to Excel file: {filepath}")
    return filepath

def evaluate_kinship_vision_model(model_name, save_to_excel=True):
    system_prompt = """당신은 한국어 가족 관계 호칭을 정확히 파악하는 전문가입니다. 

### 규칙
1. 제공된 가족 사진을 참고하여, 주어진 대화에서 설명하는 인물을 찾으세요.
2. 대화를 단계별로 분석하여 각 인물 간의 관계를 추론하세요.
3. 최종 질문에 대한 답을 A, B, C, D 중에서 선택하세요.
4. 답변은 반드시 알파벳 한 글자만 출력하세요 (예: A 또는 B 또는 C 또는 D).
5. 추가 설명이나 다른 텍스트를 포함하지 마세요.

### Output Format
Output only a single letter: A, B, C, D
"""

    print(f"Model: {model_name}")
    print(f"System Prompt: {system_prompt}")
    print("=" * 50)
    
    jsonl_path = os.path.join(PROJECT_ROOT, 'data', 'json', 'KINSHIP_VISION_v5_hard.jsonl')
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
            correct_answer = row['answer']
            model_answer = responses[i].strip().upper()
            
            model_answer_clean = None
            for char in model_answer:
                if char in ['A', 'B', 'C', 'D']:
                    model_answer_clean = char
                    break
            
            if model_answer_clean is None:
                model_answer_clean = model_answer[:1] if model_answer else ""
            
            is_correct = model_answer_clean == correct_answer
            
            if is_correct:
                correct_count += 1
            
            results_data.append({
                'id': i + 1,
                'question': question,
                'target': correct_answer,
                'resps': model_answer_clean,
                'Is_Correct': is_correct,
                'model': model_name
            })
            
            print(f"\nQuestion {i+1}: {question[:100]}...")
            print(f"Correct Answer: {correct_answer}")
            print(f"Model Answer: {model_answer_clean} (Raw: {model_answer})")
            print(f"Result: {'✓' if is_correct else '✗'}")
        
        results_df = pd.DataFrame(results_data)
        
        accuracy = correct_count / total_count * 100
        print(f"\n=== Final Results ===")
        print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
        
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

