import json
import re

import spacy
from stop_words import get_stop_words
from tokenizers import Tokenizer

from texts import BASE_DIR, RESULTS_DIR, load_lalka


def main() -> None:
    paragraphs = [p.strip() for p in load_lalka().split("\n\n") if len(p.strip()) > 200]
    excerpt = re.sub(r"\s+", " ", " ".join(paragraphs[:2])).strip()
    nlp = spacy.load("pl_core_news_sm")
    tokens = [token for token in nlp(excerpt) if token.is_alpha]
    words = [token.text.lower() for token in tokens]
    lemmas = [token.lemma_.lower() for token in tokens]
    stop_words = set(get_stop_words("polish"))
    filtered = [word for word in words if word not in stop_words]

    tokenizer = Tokenizer.from_file(str(BASE_DIR / "tokenizer_pl.json"))
    encoded = tokenizer.encode(" ".join(words), add_special_tokens=False)
    variants = {
        "Wyrazy": words,
        "Lematy": lemmas,
        "Bez stop words": filtered,
        "BPE 5000": [tokenizer.decode([token_id]) for token_id in encoded.ids],
    }

    report = [
        "# Część I", "",
        "Baza: wyrazy spaCy/is_alpha, małe litery. Lematy z kontekstu fragmentu.", "",
        "| Wariant | Liczba jednostek | Liczba typów |",
        "| --- | ---: | ---: |",
    ]
    for name, units in variants.items():
        types = len(set(encoded.ids)) if name == "BPE 5000" else len(set(units))
        report.append(f"| {name} | {len(units)} | {types} |")

    result = "\n".join(report) + "\n"
    print(result)
    (RESULTS_DIR / "I_comparison.md").write_text(result, encoding="utf-8")
    (RESULTS_DIR / "I_units.json").write_text(
        json.dumps({"fragment": excerpt, **variants}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
