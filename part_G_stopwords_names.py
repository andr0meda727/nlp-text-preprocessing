import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk import ngrams
from nltk.tokenize import sent_tokenize
from stop_words import get_stop_words

from texts import RESULTS_DIR, load_lalka


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-ząćęłńóśźż\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def count_ngrams(sentence_words: list[list[str]], n: int) -> Counter:
    return Counter(g for sentence in sentence_words for g in ngrams(sentence, n))


def plot_top10(counter: Counter, title: str, filename: str, color: str) -> None:
    top = counter.most_common(10)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([" ".join(g) for g, _ in top], [c for _, c in top], color=color)
    for i, (_, count) in enumerate(top):
        ax.text(count, i, f" {count}", va="center", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("liczba wystąpień")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


WORD_RE = re.compile(r"\b[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+\b")
NAME_RE = re.compile(r"[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+")


def find_names(sentences: list[str], stop_words: set[str]) -> Counter:
    names = Counter()
    for sentence in sentences:
        words = list(WORD_RE.finditer(sentence))
        for match in words[1:]:
            word = match.group()
            if NAME_RE.fullmatch(word) and word.lower() not in stop_words:
                names[word] += 1
    return names


def main() -> None:
    raw = load_lalka()
    sentences = sent_tokenize(re.sub(r"\s+", " ", raw), language="polish")
    sentence_words = [clean(s).split() for s in sentences]
    sentence_words = [s for s in sentence_words if s]
    stop_words = {w.lower() for w in get_stop_words("polish")}
    filtered_sentences = [[w for w in s if w not in stop_words] for s in sentence_words]

    before = {n: count_ngrams(sentence_words, n) for n in (1, 2, 3)}
    after = {n: count_ngrams(filtered_sentences, n) for n in (1, 2, 3)}
    words_before = sum(before[1].values())
    words_after = sum(after[1].values())
    removed = words_before - words_after
    removed_share = removed / words_before if words_before else 0.0

    output = []

    def log(line: str = "") -> None:
        print(line)
        output.append(line)

    log("=== CZĘŚĆ G ===")
    log(f"Stop words: {len(stop_words)}; lista: stop-words/polish")
    log(f"Zdania z wyrazami przed filtrowaniem: {len(sentence_words)}")
    log(f"Zdania z wyrazami po filtrowaniu: {sum(bool(s) for s in filtered_sentences)}")
    log(f"Wyrazy przed: {words_before}; po: {words_after}")
    log(f"Usunięto {removed} wystąpień ({removed_share:.2%}).")

    labels = {1: "UNIGRAMY", 2: "BIGRAMY", 3: "TRIGRAMY"}
    for title, counters in [("PRZED USUNIĘCIEM STOP WORDS", before),
                            ("PO USUNIĘCIU STOP WORDS", after)]:
        log(f"\n=== {title} ===")
        for n, counter in counters.items():
            total = sum(counter.values())
            top10_share = sum(c for _, c in counter.most_common(10)) / total if total else 0.0
            hapax_share = sum(c == 1 for c in counter.values()) / len(counter) if counter else 0.0
            log(f"\n{labels[n]}: {total} wystąpień, {len(counter)} typów")
            for gram, count in counter.most_common(10):
                log(f"  {' '.join(gram):40s} {count}")
            log(f"  Udział top 10 we wszystkich wystąpieniach: {top10_share:.2%}")
            log(f"  Udział typów występujących tylko raz: {hapax_share:.2%}")

    log("\n=== PORÓWNANIE WARIANTÓW ===")
    log(f"{'Jednostka':12s} {'Wyst. przed':>12s} {'Wyst. po':>12s} "
        f"{'Typy przed':>12s} {'Typy po':>12s} {'Wspólne top10':>15s}")
    for n in (1, 2, 3):
        top_before = {g for g, _ in before[n].most_common(10)}
        top_after = {g for g, _ in after[n].most_common(10)}
        overlap = len(top_before & top_after)
        log(f"{labels[n]:12s} {sum(before[n].values()):12d} {sum(after[n].values()):12d} "
            f"{len(before[n]):12d} {len(after[n]):12d} {overlap:12d}/10")

    plot_top10(after[2], "10 najczęstszych bigramów bez stop words — „Lalka”, t. I",
               "G_bigrams.png", "#3b6ea8")
    plot_top10(after[3], "10 najczęstszych trigramów bez stop words — „Lalka”, t. I",
               "G_trigrams.png", "#c0603a")

    names = find_names(sentences, stop_words)
    log("\n=== NAZWY WŁASNE ===")
    log(f"Wzorzec słów: {WORD_RE.pattern}")
    log(f"Wzorzec kandydatów (fullmatch): {NAME_RE.pattern}")
    log(f"Znaleziono {sum(names.values())} wystąpień i {len(names)} różnych kandydatów.")
    log("\n10 najczęstszych kandydatów na nazwy własne:")
    for i, (name, count) in enumerate(names.most_common(10), 1):
        log(f"  {i:2d}. {name:24s} {count}")

    (RESULTS_DIR / "G_stopwords_names.txt").write_text("\n".join(output) + "\n", encoding="utf-8")
    (RESULTS_DIR / "G_stopwords_pl.txt").write_text(
        "\n".join(sorted(stop_words)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
