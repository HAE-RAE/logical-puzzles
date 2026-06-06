"""Generate evaluation puzzle datasets (30 per difficulty per module).

Runs all 28 puzzle generators with --num tuned to produce 30 puzzles per
difficulty (easy/medium/hard) and stores outputs in dated directories so the
existing data/csv/ + data/jsonl/ files are preserved untouched.

Outputs:
    data/csv/{DATE}/{module}.csv
    data/jsonl/{DATE}/{module}.jsonl

Special handling:
    - hanoi_en/ko writes per-difficulty jsonl (hanoi_*_easy.jsonl, ...);
      we concatenate them into a single hanoi_*.jsonl matching the eval registry.
    - sat_puzzle_en/ko script outputs sat_puzzles_en/ko (registry uses 's').

Each generator runs in isolation: pre-existing files at the generator's
fixed output path are saved aside and restored after the move so the run does
not overwrite older data files.
"""
import sys
import shutil
import subprocess
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATE = date.today().isoformat()  # YYYY-MM-DD
CSV_DEST = PROJECT_ROOT / "data" / "csv" / DATE
JSONL_DEST = PROJECT_ROOT / "data" / "jsonl" / DATE
CSV_SRC_DIR = PROJECT_ROOT / "data" / "csv"
JSONL_SRC_DIR = PROJECT_ROOT / "data" / "jsonl"

# (eval_task_name, generator_script, num_arg, num_value, generator_output_basename)
# - num_arg: argparse flag name for the count
# - num_value: passed to that flag (per-diff vs total semantics already accounted for
#   so each module yields ~30 puzzles per difficulty)
# - generator_output_basename: filename stem the generator writes
#   (eval_task_name == output_basename in all cases except sat_puzzles_*)
GENERATORS = [
    # Per-difficulty semantics: --num is per-difficulty (30 -> 30/diff)
    ("array_formula_en",   "generation/array_formula_en.py",   "--num",         30,  "array_formula_en"),
    ("array_formula_ko",   "generation/array_formula_ko.py",   "--num",         30,  "array_formula_ko"),
    ("cipher_en",          "generation/cipher_en.py",          "--num",         30,  "cipher_en"),
    ("cipher_ko",          "generation/cipher_ko.py",          "--num",         30,  "cipher_ko"),
    ("ferryman_en",        "generation/ferryman_en.py",        "--num",         30,  "ferryman_en"),
    ("ferryman_ko",        "generation/ferryman_ko.py",        "--num",         30,  "ferryman_ko"),
    ("hanoi_en",           "generation/hanoi_en.py",           "--num",         30,  "hanoi_en"),
    ("hanoi_ko",           "generation/hanoi_ko.py",           "--num",         30,  "hanoi_ko"),
    ("kinship",            "generation/kinship.py",            "--num",         30,  "kinship"),
    ("kinship_vision",     "generation/kinship_vision.py",     "--num",         30,  "kinship_vision"),
    # Total semantics: --num is total (90 -> 30/diff x 3)
    ("causal_dag_en",      "generation/causal_dag_en.py",      "--num",         90,  "causal_dag_en"),
    ("causal_dag_ko",      "generation/causal_dag_ko.py",      "--num",         90,  "causal_dag_ko"),
    ("cryptarithmetic_en", "generation/cryptarithmetic_en.py", "--num",         90,  "cryptarithmetic_en"),
    ("cryptarithmetic_ko", "generation/cryptarithmetic_ko.py", "--num",         90,  "cryptarithmetic_ko"),
    ("inequality_en",      "generation/inequality_en.py",      "--num",         90,  "inequality_en"),
    ("inequality_ko",      "generation/inequality_ko.py",      "--num",         90,  "inequality_ko"),
    ("minesweeper_en",     "generation/minesweeper_en.py",     "--num",         90,  "minesweeper_en"),
    ("minesweeper_ko",     "generation/minesweeper_ko.py",     "--num",         90,  "minesweeper_ko"),
    ("number_baseball_en", "generation/number_baseball_en.py", "--num",         90,  "number_baseball_en"),
    ("number_baseball_ko", "generation/number_baseball_ko.py", "--num",         90,  "number_baseball_ko"),
    ("sudoku_en",          "generation/sudoku_en.py",          "--num",         90,  "sudoku_en"),
    ("sudoku_ko",          "generation/sudoku_ko.py",          "--num",         90,  "sudoku_ko"),
    ("yacht_dice_en",      "generation/yacht_dice_en.py",      "--num",         90,  "yacht_dice_en"),
    ("yacht_dice_ko",      "generation/yacht_dice_ko.py",      "--num",         90,  "yacht_dice_ko"),
    # --num-samples flag, total semantics
    ("logic_grid_en",      "generation/logic_grid_en.py",      "--num-samples", 90,  "logic_grid_en"),
    ("logic_grid_ko",      "generation/logic_grid_ko.py",      "--num-samples", 90,  "logic_grid_ko"),
    ("sat_puzzles_en",     "generation/sat_puzzle_en.py",      "--num-samples", 90,  "sat_puzzles_en"),
    ("sat_puzzles_ko",     "generation/sat_puzzle_ko.py",      "--num-samples", 90,  "sat_puzzles_ko"),
    ("saju_ko",            "generation/saju_ko.py",            "--num",         90,  "saju_ko"),
    ("jamo_ko",            "generation/jamo_ko.py",            "--num",         90,  "jamo_ko"),
]

# Generators that emit per-difficulty JSONL (<base>_{easy,medium,hard}.jsonl)
# plus a combined CSV (no single combined <base>.jsonl) — concatenate per-diff
# files when routing to the eval dir.
HANOI_MODULES = {"hanoi_en", "hanoi_ko", "saju_ko", "jamo_ko"}


def collect_protected_paths(out_basename: str) -> list[Path]:
    """Files the generator may overwrite — back them up before run."""
    paths = [
        CSV_SRC_DIR / f"{out_basename}.csv",
        JSONL_SRC_DIR / f"{out_basename}.jsonl",
    ]
    if out_basename in HANOI_MODULES:
        for diff in ("easy", "medium", "hard"):
            paths.append(JSONL_SRC_DIR / f"{out_basename}_{diff}.jsonl")
    return paths


def stash(paths: list[Path]) -> dict[Path, Path]:
    """Move existing files aside; return mapping for restore."""
    stashed = {}
    for p in paths:
        if p.exists():
            tmp = p.with_suffix(p.suffix + ".__stash__")
            shutil.move(str(p), str(tmp))
            stashed[p] = tmp
    return stashed


def restore(stashed: dict[Path, Path]) -> None:
    for orig, tmp in stashed.items():
        if tmp.exists():
            shutil.move(str(tmp), str(orig))


def move_outputs(task_name: str, out_basename: str) -> tuple[bool, str]:
    """Move generator outputs into dated dirs. Returns (ok, message)."""
    csv_src = CSV_SRC_DIR / f"{out_basename}.csv"
    jsonl_src = JSONL_SRC_DIR / f"{out_basename}.jsonl"
    csv_dst = CSV_DEST / f"{task_name}.csv"
    jsonl_dst = JSONL_DEST / f"{task_name}.jsonl"

    moved = []
    if csv_src.exists():
        shutil.move(str(csv_src), str(csv_dst))
        moved.append("csv")

    if out_basename in HANOI_MODULES:
        # Concatenate per-difficulty jsonls into one
        parts = []
        for diff in ("easy", "medium", "hard"):
            p = JSONL_SRC_DIR / f"{out_basename}_{diff}.jsonl"
            if p.exists():
                parts.append(p)
        if parts:
            with open(jsonl_dst, "w", encoding="utf-8") as out:
                for p in parts:
                    text = p.read_text(encoding="utf-8")
                    if text and not text.endswith("\n"):
                        text += "\n"
                    out.write(text)
            for p in parts:
                shutil.move(str(p), str(JSONL_DEST / p.name))
            moved.append(f"jsonl(concat {len(parts)})")
    elif jsonl_src.exists():
        shutil.move(str(jsonl_src), str(jsonl_dst))
        moved.append("jsonl")

    if not moved:
        return False, "no outputs found at expected paths"
    return True, ",".join(moved)


def run_generator(task_name: str, script: str, num_arg: str, num_val: int,
                  out_basename: str) -> tuple[bool, str]:
    """Run one generator, route output to dated dir, restore originals."""
    protected = collect_protected_paths(out_basename)
    stashed = stash(protected)

    import os as _os
    timeout_s = int(_os.environ.get("GEN_TIMEOUT", "1800"))
    cmd = ["uv", "run", "python", str(PROJECT_ROOT / script), num_arg, str(num_val)]
    print(f"  $ {' '.join(cmd)}  (timeout={timeout_s}s)")

    try:
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout_s
        )
        if proc.returncode != 0:
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
            return False, "generator failed:\n    " + "\n    ".join(tail)

        ok, msg = move_outputs(task_name, out_basename)
        return ok, msg
    except subprocess.TimeoutExpired:
        return False, f"timeout ({timeout_s}s)"
    except Exception as e:
        return False, f"exception: {e}"
    finally:
        restore(stashed)


def main():
    CSV_DEST.mkdir(parents=True, exist_ok=True)
    JSONL_DEST.mkdir(parents=True, exist_ok=True)
    print(f"Date: {DATE}")
    print(f"CSV dest:   {CSV_DEST}")
    print(f"JSONL dest: {JSONL_DEST}")
    print(f"Total modules: {len(GENERATORS)}")
    print("=" * 80)

    import os as _os
    skip_done = _os.environ.get("SKIP_DONE", "0") == "1"

    successes, failures, skipped = [], [], []
    for i, (task_name, script, num_arg, num_val, out_basename) in enumerate(GENERATORS, 1):
        if skip_done and (JSONL_DEST / f"{task_name}.jsonl").exists():
            print(f"\n[{i}/{len(GENERATORS)}] {task_name}  -- already done, skip")
            skipped.append(task_name)
            continue
        print(f"\n[{i}/{len(GENERATORS)}] {task_name}  ({num_arg} {num_val})")
        ok, msg = run_generator(task_name, script, num_arg, num_val, out_basename)
        if ok:
            print(f"  OK ({msg})")
            successes.append(task_name)
        else:
            print(f"  FAIL: {msg}")
            failures.append((task_name, msg))

    print("\n" + "=" * 80)
    print(f"Summary: {len(successes)} ok / {len(failures)} failed / {len(skipped)} skipped")
    if failures:
        print("\nFailed modules:")
        for name, msg in failures:
            short = msg.splitlines()[0][:120]
            print(f"  - {name}: {short}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
