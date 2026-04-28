# EN vs KO Cipher Calibration: What Had to Be Done Differently

Target model: `Qwen/Qwen3-VL-8B-Instruct` (4-bit nf4 via bitsandbytes).
Goal: land each difficulty in its target accuracy band on a 10-puzzle batch.

| Difficulty | Target band | EN result | KO result (same structure, letters swapped) |
| --- | --- | --- | --- |
| easy   | 7-9/10 | 9/10  | 10/10 |
| medium | 4-6/10 | 6/10  | 0/10  |
| hard   | 2-4/10 | 3/10  | 0/10  |

The "letters-only swap" lands EN exactly where intended but KO either ceilings (easy) or floors (medium/hard). This document records the structural reasons and the concrete knobs that need to change to bring KO into the same bands.

---

## 1. Why the same structure produces different difficulty

### 1.1 The "alphabet" is not symmetric across scripts

| Aspect | EN | KO |
| --- | --- | --- |
| Atomic unit the cipher operates on | letter (A-Z) | syllable (가-힣), with the cipher actually touching just the 초성 |
| Alphabet size | 26 | 19 (초성 row) — 14 if 쌍자음 are dropped |
| Boundary visibility | letters separated by no marker, but each letter is one token slice | each syllable is a single visible block — boundaries are "free" |
| Tokenizer behaviour | mostly 1 token per letter for 5-letter words | 1 token per syllable, but BPE merges common syllables into multi-char tokens; rarer syllables fragment into multi-byte sub-tokens |

Two consequences:

- **REVERSE is easier in KO** because the reverse operates on the visually-separated syllables. The model never has to decide where one unit ends and the next begins. In EN it has to scan 7-9 letters in a row and not skip any. → KO easy ceilings at 100%.
- **Caesar shift is harder in KO** because the cipher touches a *sub-component* of each syllable (only the 초성), not the whole syllable. The model has to decompose → look up an index in a 19-element list → add → recompose. In EN the operation is "letter -> letter", a single arithmetic step on a familiar 26-element ring.

### 1.2 The 19-자모 list is not internalised the way A-Z is

The model knows A-Z positions cold (A=0, M=12, Z=25). It does **not** know `ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ` cold. Inspecting the KO failure rolls:

- The model wrote down the 자모 list correctly when prompted but mis-counted past the 쌍자음 (ㄲ, ㄸ, ㅃ, ㅆ, ㅉ). Multiple traces double-count them or skip them.
- Latencies hit the 512-token max on every KO medium/hard puzzle (~36s) — the model burns its full budget reasoning, then emits an unrelated Korean phrase ("단위", "초성찾기", "그러나") instead of a decoded word.

### 1.3 Decompose/recompose adds a hidden step

EN Caesar: `'C' -> shift 3 -> 'F'`. One operation.

KO Caesar requires:
1. Decompose 가 → (초성=ㄱ, 중성=ㅏ, 종성=∅).
2. Locate ㄱ in the 19-element list (index 0).
3. Add shift mod 19.
4. Look up the new 초성.
5. Recompose into a syllable, preserving 중성/종성.

Steps 1, 4, and 5 require knowledge of Hangul syllable composition that the model treats as character-level pattern recall rather than rule application — so it makes mistakes a Python program never would.

---

## 2. Concrete knobs to recalibrate KO to the same bands

### 2.1 KO easy: bring 10/10 → 7-9/10

Goal: introduce just enough transcription friction that the model slips on 1-3 puzzles out of 10. Options, ordered by how much they bite:

1. **Switch the cipher unit from syllable to jamo.**
   "Reverse the jamo of each syllable then re-pack." Forces decomposition + recomposition on every syllable. Likely too hard — would push easy below the band.
2. **Use 4-5 syllable answers + drop the worked example.**
   Mirrors what worked for EN (longer words, single example). Expected to hit ~80%.
3. **Use answers with 받침 in every syllable** (e.g. 강물, 박물관, 손가락) and ask the model to write the answer with 받침 explicitly preserved on the final line. Hangul typing slips become more visible to the grader.

Recommended starting point: option 2 (low risk). Bank: 박물관, 도서관, 자전거, 비행기, 컴퓨터, 운동회, 지하철, 손가락, 송아지, 도토리묵, 솔방울, 거북이, 백과사전, 바람개비, 도화지... — keep 4-5 syllables, drop hint to 0 examples.

### 2.2 KO medium: bring 0/10 → 4-6/10

The model needs to *succeed sometimes*. The mathematically simpler operation has to land in the prompt:

1. **Shrink the alphabet from 19 → 14 자음 (exclude 쌍자음).**
   `ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ`. Removes the model's #1 documented error (mis-counting past doubled consonants). Restrict the answer bank to words whose every syllable's 초성 is in the 14-element list (most common Korean nouns satisfy this).
2. **Use small shifts {1, 2}** (mirroring what got EN medium iter2 to 9/10 before we backed off). Combined with #1, expect noticeable lift.
3. **Inline the explicit decryption mapping.**
   In EN we found that adding the per-letter table jumped medium 0→9. In KO, prepending a one-line `cipher_초성 -> plain_초성` table for the active shift removes the arithmetic step entirely. We *want* to use this knob in KO because without it the model can't even start; in EN we backed it off because it overshot.
4. **Constrain answers to 2 syllables** (사과, 시계, 의자, 강물, 식물 …). Halves the per-puzzle workload.
5. **Show a 2-syllable worked decryption with the full 4-step decompose / shift / recompose visible** — the EN version only showed letter substitutions; KO needs to show 음절 → (초성, 중성, 종성) → shift 초성 → 음절.

Suggested combo: 14-자모 alphabet + shifts {1, 2} + explicit decryption table + 2-syllable answers + decomposed worked example. If that ceilings, peel back the table first (parallel to what worked in EN iter3).

### 2.3 KO hard: bring 0/10 → 2-4/10

Hard adds a syllable-reverse on top of medium. The fix is the same as 2.2 plus:

6. **Keep the 5-step worked example, and add the reverse step explicitly written out** — `음절 [A][B] → reverse → [B][A] → shift each 초성 back → ...`.
7. **Cap shifts to {1, 2}**. The reverse step alone won't add much error if the shift step is solid.

Expected order to apply: 1 → 4 → 5 → 6 → re-measure → adjust 2/3 last.

---

## 3. Generic prompt knobs that matter more in KO than EN

| Knob | EN impact | KO impact | Recommendation for KO |
| --- | --- | --- | --- |
| Allow chain of thought ("show your work, then `원문:` line") | Lifts medium 0 → 6 | Already enabled, still 0/10 | Necessary but insufficient. |
| Worked example walkthrough | One example sufficient | One example insufficient — example needed FOR THE EXACT SHIFT | Pre-compute a worked example using the same `shift` value as the puzzle. |
| Explicit substitution table | Overshoots (9/10) | Probably required to enter the band at all | Include for KO medium; back off only if it ceilings. |
| `max_new_tokens` | 512 ample | KO traces routinely hit 512 mid-reasoning | Raise to 1024 for KO so the model can finish and emit the `원문:` line. |
| Answer length | 7-9 letters for the upper-band easy | 2 syllables (medium/hard), 4-5 syllables (easy) | Smaller for KO modular-arithmetic difficulties. |
| Alphabet size | 26 fixed | 19 → 14 | Drop 쌍자음 for KO modular-arithmetic difficulties. |

---

## 4. What we deliberately did NOT change between EN and KO

So the comparison is honest:

- Same number of puzzles per difficulty (10).
- Same overall prompt skeleton: algorithm description → ciphertext → decode rule → worked example → final-line format.
- Same model (`Qwen/Qwen3-VL-8B-Instruct` 4-bit), same generation settings (greedy, repetition_penalty 1.05, max_new_tokens 512).
- Same scoring (`CipherEvaluator._parse_answer` + `_check_answer`, which auto-detects KO vs EN from the gold).
- Same shift set for medium/hard ({3, 5, 7}), same number of worked examples (1), same word bank size (~30 entries).

The single variable changed was the alphabet/script. The result is that an isomorphic prompt structure produces highly **non-isomorphic** difficulty for an 8B-class instruction-tuned model.

---

## 5. Summary

- For **REVERSE**, KO is easier than EN → tighten KO easy by going longer/no-example.
- For any **modular arithmetic on the alphabet**, KO is dramatically harder than EN → loosen KO medium/hard by shrinking alphabet, shrinking shifts, shrinking words, and adding the explicit table.
- "Letters-only swap" is not a fair difficulty mirror across scripts for sub-50B models. To use cipher_*_simple as a cross-language benchmark with comparable difficulty bands, the per-language calibration above is required.
