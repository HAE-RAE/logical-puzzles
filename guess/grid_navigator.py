import random
import json
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import deque

class GridNavigator:
    def __init__(self, size: int):
        self.size = size
        self.grid = [[random.randint(0, 99) for _ in range(size)] for _ in range(size)]
        self.obstacles = []
        
    def is_prime(self, n: int) -> bool:
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    def solve(self, start: Tuple[int, int], end: Tuple[int, int], rules: List[str]) -> List[Tuple[int, int]]:
        """BFS to find a valid path satisfying rules"""
        queue = deque([(start, [start])])
        visited = set([start])
        
        while queue:
            (r, c), path = queue.popleft()
            if (r, c) == end:
                return path
            
            # Neighbors (Up, Down, Left, Right)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in visited:
                    val = self.grid[nr][nc]
                    
                    # Rule Check
                    valid = True
                    if (nr, nc) in self.obstacles: valid = False
                    
                    # Dynamic Rules Check
                    for rule in rules:
                        if "prime" in rule and self.is_prime(val): valid = False
                        if "even" in rule and val % 2 != 0: valid = False
                        if "multiple of 5" in rule and val % 5 != 0: valid = False
                        if "row_sum_even" in rule and (nr + nc) % 2 != 0: valid = False
                    
                    if valid:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), path + [(nr, nc)]))
        return None

    def render_grid(self) -> str:
        lines = []
        for r in range(self.size):
            row_str = " | ".join(f"{self.grid[r][c]:02}" if (r, c) not in self.obstacles else "XX" for r in range(self.size) for c in [r]) # Fixed indexing error in thought
            # Correction:
        lines = []
        for r in range(self.size):
            row_items = []
            for c in range(self.size):
                if (r, c) in self.obstacles:
                    row_items.append("XX")
                else:
                    row_items.append(f"{self.grid[r][c]:02}")
            lines.append(" | ".join(row_items))
        return "\n".join(lines)

def generate_grid_problem(difficulty: str, seed: int = None) -> Dict:
    rng = random.Random(seed)
    random.seed(seed)
    
    config = {
        "EASY": {"size": 4, "obstacles": 1, "rules": []},
        "MEDIUM": {"size": 5, "obstacles": 2, "rules": ["no_prime"]},
        "HARD": {"size": 6, "obstacles": 4, "rules": ["no_prime", "even_only"]},
        "VERY_HARD": {"size": 8, "obstacles": 6, "rules": ["no_prime", "even_only", "no_multiple_5"]},
        "EXTREME": {"size": 10, "obstacles": 10, "rules": ["no_prime", "even_only", "no_multiple_5", "row_sum_even"]}
    }
    
    c = config[difficulty]
    size = c["size"]
    nav = GridNavigator(size)
    
    # Place random obstacles
    nav.obstacles = random.sample([(r, c) for r in range(size) for c in range(size)], c["obstacles"])
    
    # Ensure start and end are not obstacles
    possible_points = [(r, c) for r in range(size) for c in range(size) if (r, c) not in nav.obstacles]
    start = possible_points[0]
    end = possible_points[-1]
    
    rule_texts = []
    for r in c["rules"]:
        if r == "no_prime": rule_texts.append("- Avoid cells containing prime numbers (2, 3, 5, 7, 11, etc.).")
        if r == "even_only": rule_texts.append("- You can only step on even numbers.")
        if r == "no_multiple_5": rule_texts.append("- Avoid cells containing multiples of 5.")
        if r == "row_sum_even": rule_texts.append("- The sum of coordinates (row + col) must be an even number at each position.")
        
    solution = nav.solve(start, end, rule_texts)
    
    # Retry if no solution
    if not solution:
        return generate_grid_problem(difficulty, seed + 1 if seed else None)
        
    problem_text = f"--- [GRID NAVIGATION MISSION] ---\n"
    problem_text += f"Target: Move from {start} to {end}.\n"
    problem_text += f"Grid Size: {size}x{size}\n\n"
    problem_text += "Rules:\n- Move only Up, Down, Left, or Right (no diagonals).\n"
    problem_text += "- 'XX' are impassable walls.\n"
    for rt in rule_texts:
        problem_text += rt + "\n"
    
    problem_text += "\nGrid Layout:\n"
    problem_text += nav.render_grid()
    problem_text += "\n\nProvide the path as a sequence of coordinates: (r,c) (r,c) ..."
    
    return {
        "question": problem_text,
        "answer": " ".join(f"({r},{c})" for r, c in solution),
        "difficulty": difficulty,
        "start": start,
        "end": end,
        "grid": nav.grid,
        "obstacles": nav.obstacles
    }

def create_grid_dataset(num_per_level: int = 5):
    all_problems = []
    levels = ["EASY", "MEDIUM", "HARD", "VERY_HARD", "EXTREME"]
    
    for level in levels:
        print(f"Generating {level} problems...")
        for i in range(num_per_level):
            prob = generate_grid_problem(level, seed=1000 + i + levels.index(level)*10)
            all_problems.append(prob)
            
    # Save to JSONL
    output_path = Path("data/json/GRID_NAVIGATOR.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in all_problems:
            f.write(json.dumps(p) + "\n")
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    create_grid_dataset()
