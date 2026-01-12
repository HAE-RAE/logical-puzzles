
import json
import sys

def summarized_status(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    model_name = file_path.split('_')[-1].replace('.json', '')
    print(f"--- {model_name} Performance by Difficulty ---")
    
    summary = {}
    for r in results:
        diff = r.get('difficulty', 'Unknown')
        if diff not in summary:
            summary[diff] = []
        summary[diff].append("OK" if r.get('correct') else "FAIL")
    
    order = ["TUTORIAL", "EASY", "INTERMEDIATE", "MEDIUM", "HARD", "VERY_HARD", "EXTREME"]
    for diff in order:
        if diff in summary:
            res = ", ".join(summary[diff])
            print(f"{diff:14} : {res}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/results/eval_results_cypher_gpt-5-mini.json"
    summarized_status(path)
