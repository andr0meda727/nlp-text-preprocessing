import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk import ngrams
from nltk.tokenize import sent_tokenize

from texts import RESULTS_DIR, load_lalka

raw = load_lalka()
print(f"Characters: {len(raw):,}")


# Cleaning: lowercase, remove special characters, collapse whitespace
def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-ząćęłńóśźż\s]", " ", text)   # keep letters only (incl. Polish ones)
    return re.sub(r"\s+", " ", text).strip()


# Tokenization: sentence splitting first (it needs punctuation, so it runs on
# the raw text), then each sentence is cleaned and split into words.
sentences = sent_tokenize(re.sub(r"\s+", " ", raw), language="polish")
sentence_words = [clean(s).split() for s in sentences]
sentence_words = [s for s in sentence_words if s]
words = [w for s in sentence_words for w in s]

print(f"Sentences:    {len(sentence_words):,}")
print(f"Words:        {len(words):,}")
print(f"Unique words: {len(set(words)):,}")


# n-grams counted within sentences (the end of one sentence is not joined with the next)
def count_ngrams(n: int) -> Counter:
    return Counter(g for s in sentence_words for g in ngrams(s, n))


unigrams, bigrams, trigrams = count_ngrams(1), count_ngrams(2), count_ngrams(3)

report = []
for name, counter in [("UNIGRAMS", unigrams), ("BIGRAMS", bigrams), ("TRIGRAMS", trigrams)]:
    report.append(f"\n{name}: {sum(counter.values()):,} occurrences, {len(counter):,} unique")
    for g, c in counter.most_common(20):
        report.append(f"  {' '.join(g):35s} {c}")
    # share of all occurrences covered by the 10 most frequent n-grams
    top10 = sum(c for _, c in counter.most_common(10)) / sum(counter.values())
    report.append(f"  top-10 share: {top10:.2%}")
    hapax = sum(1 for c in counter.values() if c == 1) / len(counter)
    report.append(f"  n-grams occurring only once (hapax): {hapax:.2%}")

print("\n".join(report))
(RESULTS_DIR / "A_ngrams.txt").write_text(
    f"Sentences: {len(sentence_words)}\nWords: {len(words)}\nUnique words: {len(set(words))}\n"
    + "\n".join(report), encoding="utf-8")


# Histograms of the 10 most frequent bigrams and trigrams
def plot_top10(counter: Counter, title: str, filename: str, color: str) -> None:
    top = counter.most_common(10)
    labels = [" ".join(g) for g, _ in top][::-1]
    values = [c for _, c in top][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, values, color=color)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("liczba wystąpień")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


plot_top10(bigrams, "10 najczęstszych bigramów — „Lalka”, t. I", "A_bigrams.png", "#3b6ea8")
plot_top10(trigrams, "10 najczęstszych trigramów — „Lalka”, t. I", "A_trigrams.png", "#c0603a")



# 1. Jakie bigramy dominują w analizowanym tekście?
#    Na pierwszym miejscu jest „panna izabela” (285), a w pierwszej dziesiątce są też „pan ignacy” (150)
#    i „pan tomasz” (130), czyli tytuł grzecznościowy + imię głównych postaci. Pozostałe miejsca zajmują
#    połączenia zaimka zwrotnego „się” z przyimkiem lub spójnikiem („się z”, „się w”, „się do”, „się na”,
#    „się że”) oraz „mu się”, „w tej”. Te drugie to szkielet gramatyczny polszczyzny (czasowniki zwrotne,
#    konstrukcje „zdawało mu się, że…”), więc o treści nie mówią prawie nic. Bigramy z imionami pokazują
#    za to bohaterów i XIX-wieczną etykietę: o Izabeli prawie zawsze pisze się „panna Izabela”.
#    Wyżej na liście są też „tysięcy rubli” (110) i „tej chwili” (99), czyli pieniądze i tempo narracji.

# 2. Czy trigramy pokazują powtarzalne zwroty charakterystyczne dla autora?
#    Tak. Najczęstszy trigram „w tej chwili” (96) jest typowym dla Prusa zwrotem narracyjnym. Często wracają
#    też formuły mowy pozornie zależnej i domysłu: „mu się że”, „mi się że”, „zdawało mu się”, „zdaje mi się”,
#    „zdawało się że”. Pasuje to do psychologicznej narracji „Lalki”, w której dużo miejsca zajmują myśli
#    Wokulskiego i Rzeckiego. Inne charakterystyczne zwroty to „w takim razie”, „w każdym razie”,
#    „a nade wszystko”, „od czasu do (czasu)”, „cóż to za”. „Trzydzieści tysięcy rubli” wiąże się z wątkiem
#    majątku Wokulskiego. Trigram „a a a” pochodzi z zapisu wykrzyknień i zająknięć w dialogach (np. „A… a… a…”).

# 3. Jak różni się rozkład unigramów od rozkładu bigramów?
#    - Unigramów jest 22 074 typów na 116 876 wystąpień, a bigramów 78 606 typów na 109 357 wystąpień.
#      Bigramów jest więc ok. 3,5 raza więcej typów, a przeciętny bigram występuje dużo rzadziej.
#    - 10 najczęstszych unigramów to 17,7% całego tekstu (same słowa funkcyjne: i, się, w, nie, na…).
#      10 najczęstszych bigramów to tylko 1,5%, a trigramów 0,3%.
#    - Unigramy na szczycie listy to prawie wyłącznie słowa funkcyjne. Wśród najczęstszych bigramów pojawiają
#      się już jednostki znaczeniowe (imiona z tytułem, „tysięcy rubli”), bo połączenie dwóch słów niesie więcej
#      informacji o treści niż pojedyncze słowo funkcyjne.

