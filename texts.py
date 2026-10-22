"""Shared helpers: loading the texts and stripping publisher headers/footers."""
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PL_FILE = BASE_DIR / "lalka-tom-pierwszy.txt"
EN_FILE = BASE_DIR / "hamlet.txt"


def load_lalka() -> str:
    """Text of "Lalka" (vol. I) without the leading metadata and the Wolne Lektury footer."""
    text = PL_FILE.read_text(encoding="utf-8")
    text = text.split("\n-----\n")[0]          # Wolne Lektury license footer
    start = text.find("Tom I")                  # skip author, title, ISBN
    return text[start + len("Tom I"):].strip()


def load_hamlet() -> str:
    """Text of "Hamlet" without Gutenberg header/footer, footnotes and speaker tags."""
    text = EN_FILE.read_text(encoding="utf-8")
    text = text.split("*** START OF THE PROJECT GUTENBERG EBOOK HAMLET ***")[1]
    text = text.split("*** END OF THE PROJECT GUTENBERG EBOOK HAMLET ***")[0]
    text = text[text.find("ACT I."):]                                  # skip editor's preface
    text = re.sub(r"\[Footnote.*?\]\s*\n\s*\n", "\n", text, flags=re.S)  # footnote blocks
    text = re.sub(r"\[\d+\]", "", text)                                # footnote refs like [8]
    text = re.sub(r"_[A-Z][a-z]*\._", "", text)                        # speakers: _Ham._
    text = text.replace("_", "")
    return text
