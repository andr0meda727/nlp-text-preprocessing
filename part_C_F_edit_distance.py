"""Part C - Levenshtein distance; Part F - comparison of four textdistance measures."""
import textdistance as td

from texts import RESULTS_DIR

output = []


def log(line: str = "") -> None:
    print(line)
    output.append(line)


# ---------------------------------------------------------------- Part C
def levenshtein_operations(a: str, b: str) -> list[str]:
    """Dynamic programming + backtracking: list of operations turning a into b."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1,                            # deletion
                          d[i][j - 1] + 1,                            # insertion
                          d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))   # substitution / match
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                ops.append(f"substitute '{a[i-1]}' -> '{b[j-1]}' (pos. {i})")
            i, j = i - 1, j - 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ops.append(f"insert '{b[j-1]}' (after pos. {i})")
            j -= 1
        else:
            ops.append(f"delete '{a[i-1]}' (pos. {i})")
            i -= 1
    return ops[::-1]


A, B = "Ala ma kota", "Ala posiada kota"
log("=== PART C ===")
log(f"'{A}' (len {len(A)})  vs  '{B}' (len {len(B)})")
log(f"Levenshtein distance (textdistance): {td.levenshtein.distance(A, B)}")
log(f"Normalized similarity: {td.levenshtein.normalized_similarity(A, B):.3f}")
ops = levenshtein_operations(A, B)
log(f"One of the optimal operation sequences ({len(ops)}):")
for op in ops:
    log(f"  - {op}")
log(f"Word-level distance (each word = 1 symbol): "
    f"{td.levenshtein.distance(A.split(), B.split())}")

# ---------------------------------------------------------------- Part F
MEASURES = {
    "Levenshtein": td.levenshtein,
    "Damerau-Lev.": td.damerau_levenshtein,
    "Jaro-Winkler": td.jaro_winkler,
    "Jaccard(bigr.)": td.Jaccard(qval=2),      # (multi)sets of character bigrams
}
PAIRS = [
    ("Ala ma kota", "Ala posiada kota", "word replaced with a synonym"),
    ("Wokulski", "Wokólski", "spelling typo"),
    ("Ala ma kota", "Ala am kota", "transposed letters"),
    ("zażółć gęślą jaźń", "zazolc gesla jazn", "diacritics only"),
    ("Ala ma kota", "kota ma Ala", "swapped words"),
    ("Izabela", "Izabeli", "inflection"),
]

log("\n=== PART F ===")
header = f"{'pair':42s}" + "".join(f"{name:>16s}" for name in MEASURES)
log("Distances (raw; for Jaro-Winkler and Jaccard: 1 - similarity):")
log(header)
for a, b, _ in PAIRS:
    row = f"{a + ' / ' + b:42s}"
    for m in MEASURES.values():
        dist = m.distance(a, b)
        row += f"{dist:>16.3f}" if isinstance(dist, float) else f"{dist:>16d}"
    log(row)

log("\nSimilarities normalized to [0, 1] (1 = identical):")
log(header + f"{'spread':>10s}")
spreads = []
for a, b, description in PAIRS:
    values = [m.normalized_similarity(a, b) for m in MEASURES.values()]
    spread = max(values) - min(values)
    spreads.append((spread, a, b, description))
    log(f"{a + ' / ' + b:42s}" + "".join(f"{v:>16.3f}" for v in values) + f"{spread:>10.3f}")

spreads.sort(reverse=True)
log("\nPairs ranked by spread (max - min similarity):")
for spread, a, b, description in spreads:
    log(f"  {spread:.3f}  '{a}' / '{b}'  - {description}")

# For comparison: Jaccard on word sets (qval=None) - blind to word order and to changes inside words
log("\nWord-level Jaccard (qval=None):")
for a, b, _ in PAIRS:
    log(f"  '{a}' / '{b}': {td.Jaccard(qval=None).normalized_similarity(a, b):.3f}")

(RESULTS_DIR / "C_F_edit_distance.txt").write_text("\n".join(output), encoding="utf-8")
