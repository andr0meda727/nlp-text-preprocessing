import csv
import json
import re
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer

from texts import BASE_DIR, RESULTS_DIR, load_lalka


VOCAB_SIZES = (1000, 5000, 30000)
WORDS = ("nieprzetłumaczalny", "hiperparametryzacja", "kotkowi", "Wokulskiego")
MODEL_IDS = {
    "GPT-2": "gpt2",
    "HerBERT": "allegro/herbert-base-cased",
    "Bielik": "speakleash/Bielik-1.5B-v3",
}
SENTENCES = {
    "PL": [
        "Wokulski rozmawiał z Izabelą o przyszłości sklepu.",
        "Rano kupiec otworzył sklep i przywitał pierwszych klientów.",
        "Na dworcu czekał pociąg do Warszawy.",
        "Nieprzetłumaczalny żart utrudnił rozmowę podczas spotkania.",
        "Hiperparametryzacja modelu wymaga dokładnych eksperymentów.",
    ],
    "EN": [
        "Wokulski talked with Izabela about the future of the shop.",
        "In the morning, the merchant opened the shop and greeted the first customers.",
        "A train to Warsaw was waiting at the station.",
        "An untranslatable joke made the conversation difficult during the meeting.",
        "Hyperparameter tuning of the model requires careful experiments.",
    ],
}


def train_bpe(text: str, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement=" ", prepend_scheme="never")
    tokenizer.decoder = decoders.Metaspace(replacement=" ", prepend_scheme="never")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["[UNK]"],
        initial_alphabet=[chr(code) for code in range(32, 127)] + list("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"),
        show_progress=False,
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    return tokenizer


def token_ids(tokenizer, text: str) -> list[int]:
    if isinstance(tokenizer, Tokenizer):
        return tokenizer.encode(text, add_special_tokens=False).ids
    return tokenizer.encode(text, add_special_tokens=False, truncation=False)


def write_csv(filename: str, rows: list[dict]) -> None:
    with (RESULTS_DIR / filename).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list]) -> list[str]:
    def cell(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    return ["| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |"] + [
        "| " + " | ".join(map(cell, row)) + " |" for row in rows
    ]


def plot_means(averages: dict) -> None:
    labels = list(averages["PL"])
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for language, offset, color in [("PL", -0.2, "#3b6ea8"), ("EN", 0.2, "#c0603a")]:
        values = [averages[language][label] for label in labels]
        bars = ax.bar([position + offset for position in x], values, width=0.4,
                      label=language, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("średnia liczba tokenów na zdanie")
    ax.set_title("Porównanie tokenizatorów — 5 zdań PL i 5 zdań EN")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Język")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "H_tokenizer_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    text = re.sub(r"\s+", " ", load_lalka()).strip()
    own = {}
    word_rows = []
    vocabulary_rows = []
    for size in VOCAB_SIZES:
        print(f"Trening BPE: {size}...", flush=True)
        tokenizer = train_bpe(text, size)
        own[size] = tokenizer
        path = BASE_DIR / "tokenizer_pl.json" if size == 5000 else RESULTS_DIR / f"tokenizer_pl_{size}.json"
        tokenizer.save(str(path))
        vocabulary_rows.append({
            "target_vocab": size,
            "actual_vocab": tokenizer.get_vocab_size(),
            "corpus_tokens": len(token_ids(tokenizer, text)),
            "mean_PL": mean(len(token_ids(tokenizer, s)) for s in SENTENCES["PL"]),
            "mean_EN": mean(len(token_ids(tokenizer, s)) for s in SENTENCES["EN"]),
        })
        for word in WORDS:
            ids = token_ids(tokenizer, word)
            fragments = [tokenizer.decode([token_id]) for token_id in ids]
            word_rows.append({
                "word": word,
                "target_vocab": size,
                "actual_vocab": tokenizer.get_vocab_size(),
                "token_count": len(ids),
                "tokens": json.dumps(fragments, ensure_ascii=False),
            })
        print(f"Zapisano {path.name}; słownik: {tokenizer.get_vocab_size()}.", flush=True)

    write_csv("H_word_splits.csv", word_rows)
    write_csv("H_vocab_sizes.csv", vocabulary_rows)

    ready = {}
    for name, model_id in MODEL_IDS.items():
        print(f"Wczytywanie tokenizatora: {model_id}...", flush=True)
        local_bielik = BASE_DIR / "bielik_tokenizer"
        use_local = name == "Bielik" and (local_bielik / "tokenizer.json").is_file()
        try:
            ready[name] = AutoTokenizer.from_pretrained(
                str(local_bielik) if use_local else model_id,
                cache_dir=str(BASE_DIR / ".cache" / "huggingface"),
                use_fast=True,
                trust_remote_code=False,
                local_files_only=use_local,
            )
        except OSError as exc:
            if name == "Bielik":
                raise RuntimeError(
                    "Bielik: pobierz pliki do bielik_tokenizer "
                    "lub zaloguj się: .\\venv\\Scripts\\hf.exe auth login."
                ) from exc
            raise

    comparison = {"BPE 5000": own[5000], **ready}
    all_tokenizers = {f"BPE {size}": tokenizer for size, tokenizer in own.items()} | ready
    counts = []
    for language, sentences in SENTENCES.items():
        for index, sentence in enumerate(sentences, 1):
            counts.append({
                "id": f"{language}{index}", "language": language, "sentence": sentence,
                **{name: len(token_ids(tokenizer, sentence)) for name, tokenizer in all_tokenizers.items()},
            })

    averages = {
        language: {
            name: mean(row[name] for row in counts if row["language"] == language)
            for name in comparison
        } for language in SENTENCES
    }
    write_csv("H_sentence_counts.csv", counts)
    plot_means(averages)

    report = [
        "# Część H", "",
        "Korpus: „Lalka”, t. I. Znakowy BPE. Liczenie bez tokenów specjalnych.", "",
        "## Trzy rozmiary słownika", "",
    ]
    report += markdown_table(
        ["Docelowy słownik", "Uzyskany słownik", "Tokeny w korpusie", "Średnia PL", "Średnia EN"],
        [[r["target_vocab"], r["actual_vocab"], r["corpus_tokens"],
          f'{r["mean_PL"]:.2f}', f'{r["mean_EN"]:.2f}'] for r in vocabulary_rows],
    )
    report += ["", "## Podział słów", ""]
    report += markdown_table(
        ["Słowo", *[str(size) for size in VOCAB_SIZES]],
        [[word, *[next(r["tokens"] for r in word_rows
                       if r["word"] == word and r["target_vocab"] == size)
                   for size in VOCAB_SIZES]] for word in WORDS],
    )
    report += ["", "## Zdania testowe", ""]
    report += markdown_table(["ID", "Zdanie"], [[r["id"], r["sentence"]] for r in counts])
    report += ["", "## Liczba tokenów", ""]
    report += markdown_table(
        ["ID", *comparison], [[r["id"], *[r[name] for name in comparison]] for r in counts])
    report += ["", "## Średnie", ""]
    report += markdown_table(
        ["Tokenizer", "PL", "EN", "PL/EN"],
        [[name, f'{averages["PL"][name]:.2f}', f'{averages["EN"][name]:.2f}',
          f'{averages["PL"][name] / averages["EN"][name]:.3f}'] for name in comparison],
    )
    report += ["", "![Średnie liczby tokenów](H_tokenizer_comparison.png)", ""]
    (RESULTS_DIR / "H_tokenizers.md").write_text("\n".join(report), encoding="utf-8")
    print("\nPorównanie ukończone. Wyniki: results/H_tokenizers.md", flush=True)
    for language, values in averages.items():
        print(language + ": " + "; ".join(f"{name}={value:.2f}" for name, value in values.items()))


if __name__ == "__main__":
    main()
