"""Answer extraction and grading for JEEBench problem types.

Types (from the dataset): MCQ (single letter), MCQ(multiple) (letter set),
Integer (exact int), Numeric (float, |diff| <= 0.01).
"""
from __future__ import annotations

import re

_LETTER_RE = re.compile(r"\b([A-D])\b")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_letters(final_answer: str) -> list[str]:
    """Ordered unique option letters mentioned in the answer."""
    seen: list[str] = []
    for m in _LETTER_RE.finditer(final_answer):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def extract_number(final_answer: str) -> float | None:
    m = _NUMBER_RE.search(final_answer.replace(",", ""))
    return float(m.group(0)) if m else None


def grade(problem_type: str, gold: str, final_answer: str) -> dict:
    """Return {'correct': bool, 'partial': bool, 'predicted': str}."""
    gold = gold.strip()
    if problem_type == "MCQ":
        letters = extract_letters(final_answer)
        pred = letters[0] if letters else ""
        return {"correct": pred == gold, "partial": False, "predicted": pred}

    if problem_type == "MCQ(multiple)":
        pred_set = set(extract_letters(final_answer))
        gold_set = set(re.findall(r"[A-D]", gold))
        correct = pred_set == gold_set
        # JEE-style partial: a non-empty strict subset of the gold set
        partial = (not correct) and bool(pred_set) and pred_set < gold_set
        return {"correct": correct, "partial": partial,
                "predicted": "".join(sorted(pred_set))}

    if problem_type == "Integer":
        pred = extract_number(final_answer)
        try:
            gold_val = float(gold)
        except ValueError:
            return {"correct": False, "partial": False, "predicted": str(pred)}
        correct = pred is not None and float(pred) == gold_val
        return {"correct": correct, "partial": False, "predicted": str(pred)}

    if problem_type == "Numeric":
        pred = extract_number(final_answer)
        try:
            gold_val = float(gold)
        except ValueError:
            return {"correct": False, "partial": False, "predicted": str(pred)}
        correct = pred is not None and abs(pred - gold_val) <= 0.011
        return {"correct": correct, "partial": False, "predicted": str(pred)}

    return {"correct": False, "partial": False, "predicted": ""}


TYPE_INSTRUCTIONS = {
    "MCQ": "This is a single-correct multiple choice question. Exactly one of "
           "(A), (B), (C), (D) is correct. Begin final_answer with that letter.",
    "MCQ(multiple)": "This is a multiple choice question where ONE OR MORE options "
                     "are correct. Begin final_answer with all correct letters, "
                     "e.g. 'A, D'.",
    "Integer": "The final answer is a non-negative integer. Begin final_answer "
               "with that integer.",
    "Numeric": "The final answer is a number. Begin final_answer with the value "
               "correct to 2 decimal places.",
}
