"""Part E - Polish needs different tools.
Tokenization and lemmatization: spaCy (pl_core_news_sm); stemming: Stempel (pystempel package)."""
import spacy
from nltk.stem import SnowballStemmer
from pystempel import Stemmer

from texts import RESULTS_DIR, load_lalka

output = []


def log(line: str = "") -> None:
    print(line)
    output.append(line)


nlp = spacy.load("pl_core_news_sm")
stempel = Stemmer.polimorf()                 # Stempel with the table trained on the PoliMorf dictionary
snowball_en = SnowballStemmer("english")


def stem_pl(word: str) -> str:
    return stempel(word.lower()) or word.lower()   # Stempel returns None for unknown words


# Excerpt: the first two paragraphs of chapter I of "Lalka"
text = load_lalka()
paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 200]
excerpt = " ".join(paragraphs[:2])
log("=== PART E.1: tokenization, stemming and lemmatization of Polish text ===")
log(f"Excerpt ({len(excerpt)} characters):\n{excerpt}\n")

doc = nlp(excerpt)
log(f"{'token':18s} {'POS':6s} {'stem (Stempel)':18s} {'lemma (spaCy)':18s} {'Snowball EN':16s}")
tokens = [t for t in doc if t.is_alpha]
for t in tokens:
    log(f"{t.text:18s} {t.pos_:6s} {stem_pl(t.text):18s} {t.lemma_.lower():18s} "
        f"{snowball_en.stem(t.text):16s}")

lemmas = {t.lemma_.lower() for t in tokens}
stems = {stem_pl(t.text) for t in tokens}
forms = {t.text.lower() for t in tokens}
log(f"\nTokens (words): {len(tokens)}, unique forms: {len(forms)}, "
    f"unique stems: {len(stems)}, unique lemmas: {len(lemmas)}")
log(f"Sentences (spaCy): {len(list(doc.sents))}")

# Stem vs lemma differences
log("\nWords where the Stempel stem differs from the spaCy lemma:")
for t in tokens:
    if stem_pl(t.text) != t.lemma_.lower():
        log(f"  {t.text:16s} stem={stem_pl(t.text):16s} lemma={t.lemma_.lower()}")

# Paradigm of a single word - why lemmatization matters for an inflected language
log("\nInflection of 'sklep' and 'człowiek' - stem vs lemma vs English Snowball:")
for w in ["sklep", "sklepu", "sklepowi", "sklepem", "sklepie", "sklepy", "sklepów",
          "człowiek", "człowieka", "ludzie", "ludźmi", "panna", "pannie", "panną"]:
    log(f"  {w:12s} Stempel={stem_pl(w):12s} spaCy={nlp(w)[0].lemma_:12s} "
        f"Snowball-EN={snowball_en.stem(w)}")

# ---------------------------------------------------------------- E.2
log("\n=== PART E.2: English Snowball applied to Polish words ===")
WORDS = ["kotkowi", "pięknymi", "zbliżyłabym", "sklepach", "rozmawiałeś", "najlepszego",
         "ubierając", "wspaniałości", "chodzimy", "ludzie", "panny", "tysięcy"]
log(f"{'word':14s} {'Snowball EN':14s} {'Stempel':14s} {'spaCy lemma':14s}")
for w in WORDS:
    log(f"{w:14s} {snowball_en.stem(w):14s} {stem_pl(w):14s} {nlp(w)[0].lemma_:14s}")

# ---------------------------------------------------------------- E.3
log("\n=== PART E.3: languages supported by SnowballStemmer ===")
log(", ".join(SnowballStemmer.languages))
log(f"Count: {len(SnowballStemmer.languages)}; Polish supported: "
    f"{'polish' in SnowballStemmer.languages}")

(RESULTS_DIR / "E_polish.txt").write_text("\n".join(output), encoding="utf-8")
