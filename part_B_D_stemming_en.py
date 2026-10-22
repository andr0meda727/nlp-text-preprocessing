import re
from collections import defaultdict

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

from texts import RESULTS_DIR, load_hamlet

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

hamlet = re.sub(r"\s+", " ", load_hamlet())
output = []


def log(line: str = "") -> None:
    print(line)
    output.append(line)


def wordnet_pos(tag: str) -> str:
    """Penn Treebank tag -> WordNet category (without POS the lemmatizer treats everything as a noun)."""
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)


def lemma(word: str, tag: str) -> str:
    return lemmatizer.lemmatize(word.lower(), wordnet_pos(tag))


# ---------------------------------------------------------------- Part B
SENTENCES = [
    "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer "
    "The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles, "
    "And, by opposing end them?",
    "Let me not think on't,--Frailty, thy name is Woman!",
    "Neither a borrower nor a lender be:",
    "There are more things in heaven and earth, Horatio, Than are dreamt of in your philosophy.",
    "Though this be madness, yet there's method in it.",
]
for s in SENTENCES:  # make sure the sentences really come from the downloaded file
    assert s in hamlet, s

log("=== PART B: token | stem (Porter) | lemma (WordNet + POS) ===")
differences_b = []
for i, sentence in enumerate(SENTENCES, 1):
    log(f"\nSentence {i}: {sentence}")
    tokens = [t for t in word_tokenize(sentence) if re.search(r"[A-Za-z]", t)]
    log(f"{'token':14s} {'POS':5s} {'stem':12s} {'lemma':12s}")
    for tok, tag in pos_tag(tokens):
        st, lm = porter.stem(tok), lemma(tok, tag)
        marker = "  <-- different" if st != lm else ""
        if st != lm:
            differences_b.append((tok, st, lm))
        log(f"{tok:14s} {tag:5s} {st:12s} {lm:12s}{marker}")

log(f"\nWords for which stemming and lemmatization differ ({len(differences_b)}):")
for tok, st, lm in differences_b:
    log(f"  {tok:12s} stem={st:10s} lemma={lm}")

# ---------------------------------------------------------------- Part D
log("\n\n=== PART D: stemmer comparison on the vocabulary of the whole play ===")
vocab = sorted({w.lower() for w in re.findall(r"[A-Za-z]+", hamlet) if len(w) > 2})
log(f"Unique words (length > 2): {len(vocab)}")

stemmers = {"Porter": porter.stem, "Lancaster": lancaster.stem, "Snowball": snowball.stem}
stems = {name: {w: f(w) for w in vocab} for name, f in stemmers.items()}
for name in stemmers:
    log(f"  {name:10s} -> {len(set(stems[name].values()))} unique stems")

diff_pl = [w for w in vocab if stems["Porter"][w] != stems["Lancaster"][w]]
diff_ps = [w for w in vocab if stems["Porter"][w] != stems["Snowball"][w]]
log(f"\nPorter != Lancaster: {len(diff_pl)} words ({len(diff_pl)/len(vocab):.1%})")
log(f"Porter != Snowball:  {len(diff_ps)} words ({len(diff_ps)/len(vocab):.1%})")

SELECTED = ["generous", "general", "university", "conscience", "suffer", "nobler", "fortune",
            "outrageous", "philosophy", "heaven", "madness", "question", "something", "doing",
            "fairly", "lying", "sweetly", "organization"]
log(f"\n{'word':14s} {'Porter':12s} {'Lancaster':12s} {'Snowball':12s}")
for w in SELECTED:
    log(f"{w:14s} {porter.stem(w):12s} {lancaster.stem(w):12s} {snowball.stem(w):12s}")

log("\nExamples from Hamlet where Porter != Snowball:")
for w in diff_ps[:25]:
    log(f"  {w:14s} Porter={stems['Porter'][w]:12s} Snowball={stems['Snowball'][w]}")


# Overstemming: different lemmas -> the same stem
def group_by_stem(name: str) -> dict:
    groups = defaultdict(set)
    for w in vocab:
        groups[stems[name][w]].add(w)
    return groups


def any_lemma(word: str) -> str:
    """Context-free lemma: try verb, adjective, noun."""
    for pos in ("v", "a", "n"):
        lm = lemmatizer.lemmatize(word, pos)
        if lm != word:
            return lm
    return word


log("\nOverstemming - Lancaster merges words with different meanings (selected groups):")
lan_groups = group_by_stem("Lancaster")
candidates = [(st, ws) for st, ws in lan_groups.items() if len({any_lemma(w) for w in ws}) >= 3]
candidates.sort(key=lambda x: -len(x[1]))
for st, ws in candidates[:15]:
    log(f"  {st:8s} <- {', '.join(sorted(ws))}")

log("\nOverstemming - Porter (same stem, different lemmas):")
por_groups = group_by_stem("Porter")
candidates_p = [(st, ws) for st, ws in por_groups.items() if len({any_lemma(w) for w in ws}) >= 2]
candidates_p.sort(key=lambda x: -len(x[1]))
for st, ws in candidates_p[:15]:
    log(f"  {st:8s} <- {', '.join(sorted(ws))}")

# Understemming: same lemma -> different stems
log("\nUnderstemming - forms of one lexeme with different stems (Porter):")
by_lemma = defaultdict(set)
for w in vocab:
    by_lemma[any_lemma(w)].add(w)
shown = 0
for lm, ws in sorted(by_lemma.items()):
    sts = {porter.stem(w) for w in ws}
    if len(ws) > 1 and len(sts) > 1:
        log(f"  lemma {lm:10s}: " + ", ".join(f"{w}->{porter.stem(w)}" for w in sorted(ws)))
        shown += 1
        if shown >= 20:
            break

log("\nClassic control examples:")
for pair in [("universe", "university"), ("general", "generous"), ("policy", "police"),
             ("organ", "organization"), ("think", "thought"), ("speak", "spoke"),
             ("mouse", "mice"), ("absorb", "absorption")]:
    log("  " + " | ".join(f"{w}: P={porter.stem(w)}, L={lancaster.stem(w)}, S={snowball.stem(w)}"
                          for w in pair))

(RESULTS_DIR / "B_D_stemming_en.txt").write_text("\n".join(output), encoding="utf-8")
