
import json

def report(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Total problems: {len(data)}")
    
    for i, r in enumerate(data):
        status = "OK" if r.get('correct') else "FAIL"
        diff = r.get('difficulty', '??')
        expected = r.get('expected', '??')
        predicted = r.get('predicted', '??')
        
        # Cipher logic used (approximate from difficulty)
        cipher = "Unknown"
        if diff == "EXTREME": cipher = "Playfair+Trans+Vig"
        elif diff == "VERY_HARD": cipher = "Playfair+Vig"
        elif diff == "HARD": cipher = "Vigenere+Rev"
        elif diff == "MEDIUM": cipher = "Vigenere"
        elif diff == "EASY" or diff == "TUTORIAL": cipher = "Substitution"

        # Check for spelling features
        unique_chars = len(set(expected))
        
        print(f"[{i+1}] {status} | {diff} | {cipher}")
        print(f"    Expected: {expected} (Uniq:{unique_chars}, Len:{len(expected)})")
        if status == "FAIL":
            print(f"    Predicted: {predicted}")
        print("-" * 30)

if __name__ == "__main__":
    import sys
    fname = sys.argv[1] if len(sys.argv) > 1 else "data/results/eval_results_cypher_gpt-4.1-mini.json"
    report(fname)
