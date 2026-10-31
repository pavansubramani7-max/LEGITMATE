"""Local, dependency-light NLP demonstrations for the LEGITMATE NLP Studio."""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
import math
import re
import unicodedata
from typing import Any

from app.nlp.language import detect_language
from app.store import load
from app.extraction.entities import extract_entities
from app.extraction.ner_model import get_nlp_for_language, spacy_model_name
from app.nlp.sequence_demo import model_status as sequence_model_status
from app.config import ROOT

_TOKEN_RE = re.compile(r"\w+(?:['’.-]\w+)*|[^\w\s]", re.UNICODE)
_IRREGULAR = {"parties": "party", "properties": "property", "went": "go", "was": "be", "were": "be", "is": "be", "are": "be", "has": "have", "had": "have", "did": "do", "children": "child", "men": "man", "women": "woman", "better": "good"}
_STOP = frozenset("a an the and or but if of to in on at by for from with as is are was were be been being this that it they we you he she".split())
_AUX = frozenset("am is are was were be been being has have had do does did can could will would shall should may might must".split())
_PRON = frozenset("i me my we us our you your he him his she her it they them their who whom which that".split())
_DET = frozenset("a an the this that these those each any some no every".split())
_PREP = frozenset("about above across after against among around at before behind below beneath beside between by down during for from in into near of off on over through to under until up upon with within without".split())
_VERBS = frozenset("agree execute sign pay own lease sell buy transfer convey file submit deliver contain include state provide require make give hold occupy terminate commence".split())


def _lemma(token: str) -> str:
    word = token.casefold()
    if not word.isalpha():
        return word
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _pos(token: str, previous: str = "") -> str:
    word = token.casefold()
    if not any(c.isalnum() for c in token): return "PUNCT"
    if word.isdigit(): return "NUM"
    if word in _AUX: return "AUX"
    if word in _PRON: return "PRON"
    if word in _DET: return "DET"
    if word in _PREP: return "ADP"
    if word in {"and", "or", "but", "nor", "yet"}: return "CCONJ"
    if word.endswith("ly"): return "ADV"
    if word.endswith(("ous", "ful", "ive", "able", "ible", "al", "less")): return "ADJ"
    if word in _VERBS or word.endswith(("ing", "ed", "ize", "ise", "fy")): return "VERB"
    if token[:1].isupper() and previous in {"", ".", "!", "?"}: return "PROPN"
    if word.isalpha(): return "NOUN"
    return "X"


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    words = [t.casefold() for t in tokens if any(ch.isalnum() for ch in t)]
    return [tuple(words[i:i+n]) for i in range(max(0, len(words)-n+1))]


def _perplexity(tokens: list[str]) -> dict[str, float | int]:
    words = [t.casefold() for t in tokens if any(ch.isalnum() for ch in t)]
    if not words: return {"unigram": 0.0, "bigram": 0.0, "tokens": 0}
    vocab = max(1, len(set(words)))
    uni = Counter(words); bi = Counter(zip(words, words[1:])); prefix = Counter(words[:-1])
    alpha = 1.0
    uni_log = sum(math.log((uni[w] + alpha) / (len(words) + alpha * vocab)) for w in words) / len(words)
    if len(words) > 1:
        bi_log = sum(math.log((bi[(a,b)] + alpha) / (prefix[a] + alpha * vocab)) for a,b in zip(words, words[1:])) / (len(words)-1)
        bi_pp = math.exp(min(30, -bi_log))
    else: bi_pp = float(vocab)
    return {"unigram": round(math.exp(min(30, -uni_log)), 3), "bigram": round(bi_pp, 3), "tokens": len(words)}


def analyze_text(text: str, *, corpus_chunks: list[str] | None = None) -> dict[str, Any]:
    sample = text.strip()[:10000]
    language = detect_language(sample)
    language_code = str(language.get("code", "und"))
    language_confidence = float(language.get("confidence", 0) or 0)
    spacy_model = spacy_model_name(language_code) if language_confidence >= 0.55 else None
    spacy_nlp = get_nlp_for_language(language_code) if spacy_model else None
    tagged = []
    if spacy_nlp is not None:
        spacy_doc = spacy_nlp(sample)
        tokens = [token.text for token in spacy_doc]
        for token in spacy_doc:
            if len(tagged) >= 500:
                break
            lemma = (token.lemma_ or token.text).strip()
            tagged.append({"text": token.text, "lemma": lemma, "pos": token.pos_ or "X"})
        linguistic_pipeline = f"spaCy · {spacy_model}"
        linguistic_note = "Trained local spaCy pipeline selected from the automatic language estimate. Review language estimate and model coverage; entity/fact output still needs human review."
    else:
        tokens = _TOKEN_RE.findall(sample)
        previous = ""
        for token in tokens[:500]:
            tag = _pos(token, previous)
            tagged.append({"text": token, "lemma": _lemma(token), "pos": tag})
            previous = token
        linguistic_pipeline = "LEGITMATE transparent rules"
        if not spacy_model:
            linguistic_note = "No compatible spaCy model is selected for this language estimate; POS and lemma values are rule-based teaching approximations."
        else:
            linguistic_note = f"{spacy_model} is not installed or could not load; POS and lemma values are rule-based teaching approximations."
    terms = [t.casefold() for t in tokens if any(c.isalnum() for c in t)]
    counts = Counter(terms)
    if corpus_chunks is None:
        data = load()
        corpus = [str(c.get("text", "")) for c in data.get("chunks", [])[:300] if str(c.get("text", "")).strip()]
    else:
        corpus = [str(c) for c in corpus_chunks[:300] if str(c).strip()]
    docs = [sample] + corpus
    vocab = set(terms)
    df = Counter()
    for doc in docs:
        df.update(set(t.casefold() for t in _TOKEN_RE.findall(doc) if any(c.isalnum() for c in t)))
    tfidf = []
    max_tf = max(counts.values(), default=1)
    for term, count in counts.most_common(30):
        if term in _STOP: continue
        idf = math.log((len(docs) + 1) / (df[term] + 1)) + 1
        tfidf.append({"term": term, "tf": count, "tfidf": round((count / max_tf) * idf, 4)})
    grams = Counter(_ngrams(tokens, 2))
    # Reuse the project's existing legal entity extraction; results are explicitly candidates.
    entities = extract_entities("nlp-lab", "Text entered in NLP Studio", 1, sample)
    scripts: Counter[str] = Counter()
    for char in sample:
        if char.isalpha():
            name = unicodedata.name(char, "UNKNOWN").split()[0]
            scripts[name] += 1
    return {
        "language": language,
        "token_count": len(tokens),
        "unique_terms": len(vocab),
        "tokens": tagged,
        "bow": [{"term": k, "count": v} for k,v in counts.most_common(24)],
        "tfidf": tfidf,
        "bigrams": [{"text": " ".join(k), "count": v} for k,v in grams.most_common(12)],
        "perplexity": _perplexity(tokens),
        "entities": [{"kind": i["kind"], "value": i["value"], "evidence": i["context"]} for i in entities[:30]],
        "scripts": [{"name": k, "count": v} for k,v in scripts.most_common(8)],
        "methods": {
            "tokenization": "spaCy tokenizer" if spacy_nlp is not None else "Unicode-aware rule tokenizer",
            "pos": "spaCy trained POS pipeline" if spacy_nlp is not None else "Lightweight rule-based POS estimate",
            "lemmatization": "spaCy trained lemmatizer" if spacy_nlp is not None else "Rule-based lemma approximation",
            "linguistic_pipeline": linguistic_pipeline,
            "linguistic_note": linguistic_note,
            "tfidf_corpus_docs": len(docs),
        },
        "disclaimer": "The active linguistic pipeline is reported per analysis. Rule output is a teaching approximation; spaCy output remains model-derived and needs human review. Extracted names and claims are candidates, not verified findings.",
    }


def _tokens(text: str) -> list[str]:
    return [t.casefold() for t in _TOKEN_RE.findall(text) if any(c.isalnum() for c in t)][:300]


def _lcs_length(a: list[str], b: list[str]) -> int:
    row = [0] * (len(b) + 1)
    for x in a:
        new = [0]
        for j, y in enumerate(b, 1):
            new.append(row[j-1] + 1 if x == y else max(row[j], new[-1]))
        row = new
    return row[-1]


def evaluate_text(reference: str, candidate: str) -> dict[str, Any]:
    ref, cand = _tokens(reference), _tokens(candidate)
    if not ref or not cand:
        return {"bleu": 0.0, "rouge_l": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "reference_tokens": len(ref), "candidate_tokens": len(cand)}
    precisions = []
    for n in range(1, 5):
        r = Counter(tuple(ref[i:i+n]) for i in range(max(0, len(ref)-n+1)))
        c = Counter(tuple(cand[i:i+n]) for i in range(max(0, len(cand)-n+1)))
        total = sum(c.values())
        precisions.append((sum(min(v, r[g]) for g,v in c.items()) + 1) / (total + 1))
    bp = min(1.0, math.exp(1 - len(ref) / len(cand)))
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / 4)
    common = _lcs_length(ref, cand)
    precision = common / len(cand); recall = common / len(ref)
    rouge = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"bleu": round(bleu, 4), "rouge_l": round(rouge, 4), "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(rouge, 4), "reference_tokens": len(ref), "candidate_tokens": len(cand), "note": "Sentence-level BLEU-4 with add-one smoothing; ROUGE-L uses token-level longest common subsequence. These scores compare wording, not legal correctness."}


def capability_status() -> dict[str, Any]:
    def available(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError, AttributeError):
            return False

    spacy_model = spacy_model_name("en")
    spacy_runtime = available("spacy")
    spacy_pipeline = get_nlp_for_language("en") if spacy_runtime else None
    sequence_models = sequence_model_status()
    sequence_task_active = bool(sequence_models["task_connected"])
    report_path = ROOT / "data" / "processed" / "kaggle_legal" / "dataset_report.json"
    report: dict[str, Any] = {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        report = {}
    qa = report.get("indian_legal_qa", {})
    cuad_stats_path = ROOT / "data" / "processed" / "cuad" / "stats.json"
    try:
        cuad_stats = json.loads(cuad_stats_path.read_text(encoding="utf-8")) if cuad_stats_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        cuad_stats = {}
    iclec = report.get("iclec", {})
    open_nyaya = report.get("opennyaya", {})
    processed_cuad = ROOT / "data" / "processed" / "cuad"
    raw_qa = ROOT / "data" / "raw" / "indqa" / "legal_qa.csv"
    prepared_qa = ROOT / "data" / "processed" / "kaggle_legal" / "indian_legal_qa"
    raw_opennyaya = ROOT / "data" / "raw" / "opennyaya" / "nyai-supreme-court-clean"
    raw_iclec = ROOT / "data" / "raw" / "iclec" / "indian_supreme_court_judgments.csv"
    prepared_iclec = ROOT / "data" / "processed" / "kaggle_legal" / "iclec" / "judgments.jsonl.gz"
    raw_cuad = ROOT / "data" / "raw" / "cuad" / "legalbenchrag_cuad.parquet"

    def split_files(folder, names=("train", "val", "test")) -> list[str]:
        return [name for name in names if (folder / f"{name}.jsonl").is_file()]

    cuad_splits = split_files(processed_cuad)
    qa_splits = split_files(prepared_qa)
    research_datasets = [
        {"name": "LegalBenchRAG · CUAD", "records": int(cuad_stats.get("total_pairs", 0)), "unit": "query/evidence pairs", "license": "CC BY 4.0", "local_state": "Prepared splits available" if len(cuad_splits) == 3 else "Raw source available · prepare splits" if raw_cuad.is_file() else "Source/splits not found", "splits": cuad_splits, "setup_command": "python data/pipeline/prepare_cuad.py" if raw_cuad.is_file() and len(cuad_splits) != 3 else "", "detail": "Contract retrieval benchmark; query-level train/validation/test splits are preserved."},
        {"name": "Indian Legal QA 10K", "records": int(qa.get("prepared_pairs", 0)), "unit": "prepared Q/A pairs", "license": "CC BY 4.0", "local_state": "Prepared splits available" if len(qa_splits) == 3 else "Raw source available · prepare splits" if raw_qa.is_file() else "Source/splits not found", "splits": qa_splits, "setup_command": "python data/pipeline/prepare_kaggle_legal.py" if raw_qa.is_file() and len(qa_splits) != 3 else "", "detail": "Case-grouped split. Answers are weak labels, not verified evidence passages."},
        {"name": "OpenNyaya judgments", "records": int(open_nyaya.get("judgment_files", 0)), "unit": "judgment files", "license": "CC BY 4.0", "local_state": "Raw research corpus available" if raw_opennyaya.is_dir() else "Source not found", "splits": [], "detail": "Separate legal-language research corpus; OCR artifacts are possible and files are not injected into document answers."},
        {"name": "ICLEC", "records": int(iclec.get("judgment_rows", 0)), "unit": "judgment rows", "license": "CC0-1.0", "local_state": "Prepared corpus available" if prepared_iclec.is_file() else "Raw source available · prepare splits" if raw_iclec.is_file() else "Source/splits not found", "splits": [], "setup_command": "python data/pipeline/prepare_kaggle_legal.py" if raw_iclec.is_file() and not prepared_iclec.is_file() else "", "detail": "Case-grouped research corpus; optional metadata fields are sparse."},
    ]
    syllabus_modules = [
        {
            "id": "module-1", "title": "Foundations of NLP & Embeddings", "coverage": "Partial",
            "topics": [
                {"name": "Tokenization, POS, lemmatization", "status": "Optional trained pipeline + transparent fallback", "detail": "The analyzer auto-detects language and uses a compatible installed spaCy pipeline where available; otherwise it labels its rule-based teaching estimates."},
                {"name": "NER", "status": "Optional spaCy + legal regex fallback", "detail": "spaCy entities are used when the compatible English pipeline loads; property identifiers and the fallback extractor use legal-specific rules. Results remain candidates."},
                {"name": "N-grams and perplexity", "status": "Demo available", "detail": "Unigram/bigram counts and smoothed perplexity run on the entered sample, not a held-out language corpus."},
                {"name": "BoW and TF-IDF", "status": "Demo available", "detail": "Calculated locally for the entered text and a bounded set of indexed passages."},
                {"name": "Word2Vec, GloVe, FastText", "status": "Not implemented", "detail": "No trained or pretrained model for these embedding families is connected."},
            ],
        },
        {
            "id": "module-2", "title": "Deep Learning for Text Processing", "coverage": "Partial",
            "topics": [
                {"name": "GRU, CNN, vanilla RNN, and LSTM sequence classifiers", "status": "Demo available", "detail": "Selectable CPU-trained architecture teaching demos classify six broad topic cues using the same synthetic split; they are not legal classifiers."},
                {"name": "Attention pooling", "status": "Demo available", "detail": "The sequence demos expose normalized pooling weights as model signals, not explanations or evidence."},
                {"name": "CNN, vanilla RNN, LSTM", "status": "Trained teaching demos" if sequence_task_active else "Training needed", "detail": "The Studio offers selectable CNN, vanilla RNN, LSTM, and GRU classifiers trained on the included synthetic classroom split. They are demonstrations, not legal classifiers."},
                {"name": "BLEU, ROUGE, precision, recall, F1", "status": "Demo available", "detail": "Text-overlap metrics run when a user supplies a reference and candidate answer; they do not measure legal correctness."},
            ],
        },
        {
            "id": "module-3", "title": "Contextual Embeddings & Transformers", "coverage": "Partial",
            "topics": [
                {"name": "Pretrained multilingual E5", "status": "Used by retrieval", "detail": "The cached multilingual E5 sentence-transformer supplies contextual semantic retrieval when its neural backend is active."},
                {"name": "ELMo, BERT, RoBERTa, ALBERT, GPT, T5", "status": "Not implemented", "detail": "These named model families are not integrated as separate NLP Studio tasks."},
                {"name": "Fine-tuning and transfer learning", "status": "Incomplete experiment", "detail": "A training pipeline exists, but no validated fine-tuned checkpoint is used by the app."},
                {"name": "Hugging Face Transformers", "status": "Runtime only", "detail": "Package availability does not mean these model tasks or checkpoints are integrated."},
            ],
        },
        {
            "id": "module-4", "title": "Advanced NLP Applications", "coverage": "Partial",
            "topics": [
                {"name": "Closed-domain question answering", "status": "Working feature", "detail": "Evidence-grounded retrieval over uploaded documents, citations, conflicts, and optional answer drafting."},
                {"name": "Open-domain conversational AI", "status": "Optional provider", "detail": "Live chat uses the configured external AI provider; it is not a local Rasa or Dialogflow deployment."},
                {"name": "Extractive legal case brief", "status": "Working feature", "detail": "Local TF-IDF sentence ranking produces source-linked summaries, candidate facts, chronology, and both sides of open conflicts."},
                {"name": "Dialogflow, Rasa, abstractive summary, translation", "status": "Not implemented", "detail": "No Dialogflow/Rasa connector, abstractive summarization model, or OpenNMT/MarianMT translation task is integrated."},
                {"name": "Legal NLP case study", "status": "Prototype", "detail": "Property/civil document review is the application domain; all extracted findings require human verification."},
            ],
        },
        {
            "id": "module-5", "title": "Speech & Multimodal NLP", "coverage": "Not implemented",
            "topics": [
                {"name": "Whisper and Wav2Vec 2.0", "status": "Not available", "detail": "No speech upload/transcription workflow or ASR checkpoint is installed."},
                {"name": "CLIP multimodal NLP", "status": "Runtime only", "detail": "PyTorch/Transformers are present, but no verified CLIP checkpoint or image-text task is connected."},
                {"name": "IndicNLP and low-resource languages", "status": "Partial", "detail": "Unicode script counting and multilingual E5 retrieval are available; IndicNLP-specific processing is not installed."},
            ],
        },
    ]
    return {
        "working_now": [
            {"name": "Tokenization + legal entity extraction", "status": "ready", "detail": "Runs locally on text you enter; extracted entities are candidate suggestions."},
            {"name": "BoW, TF-IDF, N-grams + perplexity", "status": "ready", "detail": "Calculated locally; TF-IDF uses the text and up to 300 indexed passages."},
            {"name": "BLEU, ROUGE-L, Precision, Recall, F1", "status": "ready", "detail": "Compare a candidate answer with a reference answer."},
            {"name": "CNN / RNN / GRU / LSTM comparison", "status": "ready" if sequence_task_active else "training_needed", "detail": "Select a locally trained sequence architecture to classify a short teaching sentence and inspect pooling weights. Tiny synthetic models, not legal evidence."},
            {"name": "Source-linked extractive case brief", "status": "ready", "detail": "Ranks exact sentences from the local document index, attaches file/page citations, lists candidate facts and dates, and shows open conflicts without choosing a winner. No AI provider call."},
            {"name": "Indic script profile", "status": "ready", "detail": "Counts Unicode writing systems and estimates common script; not a full IndicNLP pipeline."},
        ],
        "optional_models": [
            {"name": "spaCy POS + lemmatization", "state": "active_pipeline" if spacy_pipeline is not None else "runtime_only" if spacy_runtime else "not_installed", "runtime": spacy_runtime, "checkpoint": bool(spacy_pipeline), "task_connected": bool(spacy_pipeline), "model_name": spacy_model, "detail": "English pipeline can run POS, lemmatization, and tokenization on detected English text." if spacy_pipeline is not None else "A spaCy package alone is insufficient; install the language model to activate trained POS and lemma output." if spacy_runtime else "Install spaCy and its English model to activate the trained pipeline.", "setup_command": "python -m pip install spacy\npython -m spacy download en_core_web_sm" if spacy_pipeline is None else ""},
            {"name": "Word2Vec / GloVe / FastText", "state": "runtime_only" if available("gensim") else "not_installed", "runtime": available("gensim"), "checkpoint": False, "task_connected": False, "detail": "No trained/pretrained vector checkpoint or nearest-neighbour exercise is connected; gensim installation alone does not activate embeddings.", "setup_command": "python -m pip install gensim\n# Then train on an approved research corpus or provide a licensed pretrained vector file." if not available("gensim") else "# No connected model or exercise yet; the installed package alone is not an embedding model."},
            {"name": "CNN / vanilla RNN / LSTM", "state": "active_pipeline" if sequence_task_active else "runtime_only" if available("torch") else "not_installed", "runtime": available("torch"), "checkpoint": all(model["trained_weights"] for model in sequence_models["architectures"] if model["architecture"] != "gru") if sequence_task_active else False, "task_connected": sequence_task_active, "detail": "Selectable CNN, vanilla RNN, LSTM, and GRU classifiers are connected in the NLP Studio. Checkpoints use only a tiny synthetic teaching set; they are not legal classifiers." if sequence_task_active else "PyTorch runtime is present; run the local teaching-model training script to create the architecture checkpoints.", "setup_command": "" if sequence_task_active else ".venv\\Scripts\\python.exe data\\pipeline\\train_gru_sequence_demo.py"},
            {"name": "Whisper speech recognition", "state": "runtime_only" if available("faster_whisper") or available("whisper") else "not_installed", "runtime": available("faster_whisper") or available("whisper"), "checkpoint": False, "task_connected": False, "detail": "No audio upload/transcription route or downloaded speech checkpoint is connected. Audio stays out of legal document processing.", "setup_command": "python -m pip install faster-whisper\n# A compatible speech model must be downloaded before transcription; no ASR task is connected yet." if not available("faster_whisper") else "# Runtime may exist; a local model and a reviewed transcription workflow are still absent."},
            {"name": "CLIP image-text", "state": "runtime_only" if available("transformers") and available("torch") else "not_installed", "runtime": available("transformers") and available("torch"), "checkpoint": False, "task_connected": False, "detail": "Transformers and PyTorch are runtime dependencies only; no CLIP checkpoint or document-image matching task is connected.", "setup_command": "python -m pip install transformers\n# Add a compatible CLIP checkpoint and an image-text evaluation before enabling the task." if not (available("transformers") and available("torch")) else "# Runtime exists; no compatible local CLIP checkpoint or image-text task is connected."},
            {"name": "IndicNLP tools", "state": "runtime_only" if available("indicnlp") else "not_installed", "runtime": available("indicnlp"), "checkpoint": False, "task_connected": False, "detail": "Automatic script profiling is active. IndicNLP normalization/tokenization is not connected and requires its package plus language resources.", "setup_command": "python -m pip install indic-nlp-library\n# Add Indic language resources and a reviewed normalization/tokenization exercise." if not available("indicnlp") else "# Toolkit runtime exists; IndicNLP-specific processing is not connected."},
        ],
        "research_datasets": research_datasets,
        "syllabus_modules": syllabus_modules,
    }
