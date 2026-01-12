import random
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

class LogicGridSolver:
    """A constraint satisfaction solver for Logic Grid puzzles"""
    def __init__(self, entities: List[str], categories: List[List[str]]):
        self.entities = entities # e.g., ["Alice", "Bob", "Charlie"]
        self.categories = categories # e.g., [["Red", "Blue", "Green"], ["Dog", "Cat", "Bird"]]
        self.num_entities = len(entities)
        self.num_categories = len(categories)
        
        # State: grid[entity_idx][category_idx] = value_idx
        self.grid = [[-1 for _ in range(self.num_categories)] for _ in range(self.num_entities)]

    def solve(self, clues: List[Dict], find_all: bool = False) -> List[List[List[int]]]:
        self.solutions = []
        self._backtrack(0, 0, clues, find_all)
        return self.solutions

    def _is_valid(self, entity_idx: int, cat_idx: int, val_idx: int, clues: List[Dict]) -> bool:
        # Check uniqueness: no other entity in this category can have this value
        for i in range(entity_idx):
            if self.grid[i][cat_idx] == val_idx:
                return False
        
        # Check clues
        for clue in clues:
            if not self._check_clue(clue):
                return False
        return True

    def _check_clue(self, clue: Dict) -> bool:
        ctype = clue["type"]
        
        # Helper to get current value of an entity's category
        def get_val(ent_name, cat_idx):
            ent_idx = self.entities.index(ent_name)
            return self.grid[ent_idx][cat_idx]

        if ctype == "DIRECT":
            # "Alice has Red" -> Alice(cat0) == Red
            v = get_val(clue["ent"], clue["cat_idx"])
            if v != -1 and v != clue["val_idx"]: return False
            
        elif ctype == "NEGATIVE":
            # "Bob does not have Bird"
            v = get_val(clue["ent"], clue["cat_idx"])
            if v != -1 and v == clue["val_idx"]: return False
            
        elif ctype == "RELATIONAL":
            # "The person who has Red has the Dog"
            # If anyone has Red(cat A), they must have Dog(cat B)
            cat_a, val_a = clue["cat_a"], clue["val_a"]
            cat_b, val_b = clue["cat_b"], clue["val_b"]
            for i in range(self.num_entities):
                v_a = self.grid[i][cat_a]
                v_b = self.grid[i][cat_b]
                if v_a != -1 and v_b != -1:
                    if v_a == val_a and v_b != val_b: return False
                    if v_a != val_a and v_b == val_b: return False
                    
        return True

    def _backtrack(self, ent_idx: int, cat_idx: int, clues: List[Dict], find_all: bool):
        if ent_idx == self.num_entities:
            self.solutions.append([row[:] for row in self.grid])
            return

        next_ent = ent_idx + (cat_idx + 1) // self.num_categories
        next_cat = (cat_idx + 1) % self.num_categories

        for v in range(self.num_entities):
            if self._is_valid(ent_idx, cat_idx, v, clues):
                self.grid[ent_idx][cat_idx] = v
                self._backtrack(next_ent, next_cat, clues, find_all)
                if not find_all and len(self.solutions) > 0: return
                self.grid[ent_idx][cat_idx] = -1

def format_problem(data: Dict) -> str:
    text = "--- [LOGIC GRID DEDUCTION] ---\n"
    text += f"Entities: {', '.join(data['entities'])}\n"
    for i, cat in enumerate(data["categories"]):
        text += f"Category {i+1}: {', '.join(cat)}\n"
    text += "\nClues:\n"
    for i, clue in enumerate(data["clues"]):
        text += f"{i+1}. {clue['text']}\n"
    text += "\nGoal: For each entity, identify their choice in each category. Output as: Entity: Val1, Val2..."
    return text

class LogicGridGenerator:
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"]
        self.categories_pool = [
            ["Red", "Blue", "Green", "Yellow", "White"],
            ["Dog", "Cat", "Bird", "Fish", "Hamster"],
            ["Seoul", "Tokyo", "Paris", "London", "Berlin"],
            ["Doctor", "Artist", "Chef", "Pilot", "Lawyer"],
            ["Apple", "Banana", "Cherry", "Date", "Elderberry"]
        ]

    def generate(self, difficulty: str) -> Dict:
        config = {
            "EASY": {"ents": 3, "cats": 2, "neg": 0},
            "MEDIUM": {"ents": 3, "cats": 3, "neg": 1},
            "HARD": {"ents": 4, "cats": 3, "neg": 2},
            "VERY_HARD": {"ents": 4, "cats": 4, "neg": 3},
            "EXTREME": {"ents": 5, "cats": 5, "neg": 5}
        }
        c = config[difficulty]
        num_entities = c["ents"]
        num_categories = c["cats"]

        entities = self.rng.sample(self.names, num_entities)
        # Select categories and then select num_entities values for each
        cat_pools = self.rng.sample(self.categories_pool, num_categories)
        cats = [self.rng.sample(pool, num_entities) for pool in cat_pools]
        
        # 1. Create Ground Truth
        # row i is entity i, col j is category j's value index
        # To make it random, we shuffle the values in each category
        # But wait, to ensure unique solution, let's just create a random mapping.
        ground_truth = []
        for i in range(num_entities):
            ground_truth.append([-1] * num_categories)
            
        for cat_idx in range(num_categories):
            vals = list(range(num_entities))
            self.rng.shuffle(vals)
            for ent_idx in range(num_entities):
                ground_truth[ent_idx][cat_idx] = vals[ent_idx]
        
        # 2. Generate Clues
        clues = []
        max_clues = 20
        while len(clues) < max_clues:
            ctype = "DIRECT"
            if len(clues) >= num_entities: # Start adding variety
                ctype = self.rng.choice(["DIRECT", "NEGATIVE", "RELATIONAL"])
                if difficulty == "EASY" and ctype != "DIRECT": ctype = "DIRECT"
            
            new_clue = self._create_clue(entities, cats, ground_truth, ctype)
            if new_clue["text"] not in [c["text"] for c in clues]:
                clues.append(new_clue)
                
                # Check uniqueness
                solver = LogicGridSolver(entities, cats)
                solutions = solver.solve(clues, find_all=True)
                if len(solutions) == 1:
                    break
            
        # 3. Format Answer
        ans_parts = []
        for i, ent in enumerate(entities):
            ans_parts.append(f"{ent}: " + ", ".join([cats[j][ground_truth[i][j]] for j in range(num_categories)]))
        answer = " | ".join(ans_parts)

        return {
            "entities": entities,
            "categories": cats,
            "clues": clues,
            "difficulty": difficulty,
            "answer": answer,
            "problem": format_problem({"entities": entities, "categories": cats, "clues": clues})
        }

    def _create_clue(self, entities, cats, gt, ctype) -> Dict:
        ent_idx = self.rng.randrange(len(entities))
        cat_idx = self.rng.randrange(len(cats))
        val_idx = gt[ent_idx][cat_idx]
        
        if ctype == "DIRECT":
            return {"type": "DIRECT", "ent": entities[ent_idx], "cat_idx": cat_idx, "val_idx": val_idx, "text": f"{entities[ent_idx]} is associated with {cats[cat_idx][val_idx]}."}
        elif ctype == "NEGATIVE":
            wrong_val_idx = (val_idx + self.rng.randint(1, len(entities)-1)) % len(entities)
            return {"type": "NEGATIVE", "ent": entities[ent_idx], "cat_idx": cat_idx, "val_idx": wrong_val_idx, "text": f"{entities[ent_idx]} is NOT associated with {cats[cat_idx][wrong_val_idx]}."}
        else: # RELATIONAL
            cat_b = (cat_idx + 1) % len(cats)
            val_b = gt[ent_idx][cat_b]
            return {"type": "RELATIONAL", "cat_a": cat_idx, "val_a": val_idx, "cat_b": cat_b, "val_b": val_b, 
                    "text": f"The person associated with {cats[cat_idx][val_idx]} is also associated with {cats[cat_b][val_b]}."}

def create_logic_dataset(num_per_level: int = 5):
    all_problems = []
    generator = LogicGridGenerator(seed=2026)
    levels = ["EASY", "MEDIUM", "HARD", "VERY_HARD", "EXTREME"]
    
    for level in levels:
        print(f"Generating {level} problems...")
        for i in range(num_per_level):
            prob = generator.generate(level)
            all_problems.append({
                "question": prob["problem"],
                "answer": prob["answer"],
                "difficulty": level
            })
            
    output_path = Path("data/json/LOGIC_GRID.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in all_problems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    create_logic_dataset()
