"""
core/metrics.py — Metrik evaluasi (Bab III 3.6) dan uji statistik (3.7.4).
"""
import random
from math import comb


def precision_at_5(ranked_indices, relevant):
    """3.6.1: penyebut tetap 5; posisi kosong dihitung tidak relevan."""
    return sum(1 for i in ranked_indices[:5] if i in relevant) / 5


def reciprocal_rank(ranked_indices, relevant):
    """3.6.2: 1/rank hasil relevan pertama; 0 jika tidak ada."""
    for pos, i in enumerate(ranked_indices, 1):
        if i in relevant:
            return 1.0 / pos
    return 0.0


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def sd(xs):
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# ---------------------------------------------------------------------------
# Uji statistik (3.7.4)
# ---------------------------------------------------------------------------
def paired_t_test(a, b):
    """Paired t-test dua sisi (b - a). Mengembalikan dict t, df, p."""
    from scipy import stats
    if all(x == y for x, y in zip(a, b)):
        return {"t": 0.0, "df": len(a) - 1, "p": 1.0, "mean_diff": 0.0}
    res = stats.ttest_rel(b, a)
    return {"t": float(res.statistic), "df": len(a) - 1, "p": float(res.pvalue),
            "mean_diff": mean(y - x for x, y in zip(a, b))}


def paired_randomization_test(a, b, iterations=100_000, seed=42):
    """Uji randomization berpasangan (sign-flip) dua sisi atas selisih rata-rata."""
    diffs = [y - x for x, y in zip(a, b)]
    observed = abs(mean(diffs))
    if observed == 0:
        return {"p": 1.0, "observed_diff": 0.0, "iterations": iterations}
    rng = random.Random(seed)
    n, extreme = len(diffs), 0
    for _ in range(iterations):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(s) >= observed - 1e-12:
            extreme += 1
    return {"p": (extreme + 1) / (iterations + 1), "observed_diff": mean(diffs),
            "iterations": iterations}


def mcnemar_exact(a_correct, b_correct):
    """McNemar exact dua sisi. b = benar di A saja, c = benar di B saja."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"b": b, "c": c, "p": p}
