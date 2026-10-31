from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
UPLOADS = DATA / "documents"
PROCESSED = DATA / "processed"
SAMPLE = DATA / "sample"
DB = DATA / "index.json"

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ABSTAIN_THRESHOLD = 0.30         # BM25 / LSA / TF-IDF scores (small corpus: 4 chunks → scores ~0.08–0.94)
NEURAL_ABSTAIN_THRESHOLD = 0.74  # neural cosine scores (multilingual-e5-small range ~0.74–0.83 on these docs)
BM25_K1 = 2.2
BM25_B = 0.75
CHUNK_MAX = 1100
CHUNK_SPLIT = 950

STOP_WORDS = frozenset(
    "a an and are as at be by for from has have in into is it of on or "
    "that the this to was were will with what who when where which how".split()
)

# Legal-domain stop words: terms that appear in virtually every property/civil
# document and carry no discriminative retrieval signal.  Kept separate from
# the general stop-word list so the distinction is explicit and auditable.
# "parties" removed: it is discriminative for party-name queries (TC-03).
LEGAL_STOP_WORDS = frozenset(
    "property deed agreement document party clause section".split()
)

ALL_STOP_WORDS = STOP_WORDS | LEGAL_STOP_WORDS
