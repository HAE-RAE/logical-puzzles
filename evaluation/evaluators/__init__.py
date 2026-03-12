from typing import Dict, Type

def _get_registry() -> Dict[str, Type]:
    """Dynamically create the evaluator registry."""
    from .kinship import KinshipEvaluator
    from .cipher import CipherEvaluator
    from .hanoi import HanoiEvaluator
    from .ferryman import FerrymanEvaluator
    from .array_formula import ArrayFormulaEvaluator
    from .causal_dag import CausalDAGEvaluator
    from .cryptarithmetic import CryptarithmeticEvaluator
    from .inequality import InequalityEvaluator
    from .logic_grid import LogicGridEvaluator
    from .minesweeper import MinesweeperEvaluator
    from .number_baseball import NumberBaseballEvaluator
    from .sat_puzzle import SATPuzzleEvaluator
    from .sudoku import SudokuEvaluator
    from .yacht_dice import YachtDiceEvaluator
    
    return {
        "array_formula": ArrayFormulaEvaluator,
        "array_formula_korean": ArrayFormulaEvaluator,  # The Korean version also uses the same evaluator
        "causal_dag": CausalDAGEvaluator,
        "causal_dag_korean": CausalDAGEvaluator,  # The Korean version also uses the same evaluator
        "cipher": CipherEvaluator,
        "cipher_korean": CipherEvaluator,
        "cryptarithmetic": CryptarithmeticEvaluator,
        "ferryman": FerrymanEvaluator,
        "ferryman_korean": FerrymanEvaluator,
        "hanoi": HanoiEvaluator,
        "hanoi_korean": HanoiEvaluator,
        "inequality": InequalityEvaluator,
        "kinship": KinshipEvaluator,
        "kinship_vision": KinshipEvaluator,
        "logic_grid": LogicGridEvaluator,
        "logic_grid_korean": LogicGridEvaluator,  # The Korean version also uses the same evaluator
        "minesweeper": MinesweeperEvaluator,
        "number_baseball": NumberBaseballEvaluator,
        "sat_puzzles": SATPuzzleEvaluator,
        "sat_puzzles_korean": SATPuzzleEvaluator,  # The Korean version also uses the same evaluator
        "sudoku": SudokuEvaluator,
        "yacht_dice": YachtDiceEvaluator,
    }


def get_evaluator(task_name: str):
    """Return an evaluator instance for the given task name."""
    registry = _get_registry()
    
    if task_name not in registry:
        available = ", ".join(registry.keys())
        raise ValueError(f"Unknown task: {task_name}. Available: {available}")
    
    return registry[task_name]()


def list_tasks() -> list:
    """Dynamically return the list of all registered tasks."""
    return list(_get_registry().keys())


__all__ = ["get_evaluator", "list_tasks"]
