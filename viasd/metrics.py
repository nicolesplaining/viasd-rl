"""GSM8K answer extraction and correctness."""
import re

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm(s: str):
    s = s.replace(",", "").rstrip(".").strip()
    try:
        f = float(s)
        return int(f) if f == int(f) else f
    except ValueError:
        return None


def extract_pred(text: str):
    """Prefer the number after 'answer is'; otherwise the last number in the text."""
    m = list(re.finditer(r"answer is\s*\$?(-?\d[\d,]*\.?\d*)", text, flags=re.IGNORECASE))
    if m:
        return _norm(m[-1].group(1))
    nums = _NUM.findall(text)
    return _norm(nums[-1]) if nums else None


def is_correct(pred_text: str, gold: str) -> bool:
    pred = extract_pred(pred_text)
    g = _norm(gold)
    return pred is not None and g is not None and pred == g
