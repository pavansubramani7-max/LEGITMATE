# LEGITMATE NLP Studio and CS3250 map

## Open the Studio

1. Start LEGITMATE and sign in.
2. Choose **NLP Studio** in the main navigation.
3. Use **Analyze a text** for tokenization and linguistic/statistical demos.
4. Use **Compare answer quality** to compare a reference and candidate answer.
5. Use **GRU sequence classifier + attention** to classify one short teaching sentence and inspect its attention weights.

The entered text for these two local demonstrations is processed by the LEGITMATE server. These tools do not call Gemini/Groq and do not modify uploaded documents or `data/index.json`.

## What the local demos mean

- **Tokenization:** splits Unicode text into word/number tokens and punctuation.
- **POS and lemma:** the analyzer estimates language automatically. When the matching local spaCy package and model are installed, the output uses that trained tokenizer/POS/lemmatizer pipeline and names the model. Otherwise it falls back to LEGITMATE's transparent rule estimates and marks that fallback in the result. Current optional spaCy model mapping covers English, Spanish, French, German, Portuguese, and Italian; an uncertain/unsupported language uses the rule path. No model is downloaded during analysis.
- **NER:** reuses LEGITMATE's existing rule/spaCy-optional entity extraction. Entities are candidates, not verified facts.
- **Bag-of-Words:** counts terms in the entered sample.
- **TF-IDF:** calculates sample term weights using the sample plus up to 300 indexed passages as the reference corpus. A small corpus makes weights unstable.
- **N-grams and perplexity:** displays frequent bigrams and smoothed unigram/bigram perplexity. Because the sample is both the training text and scoring text, these numbers are a demonstration only; they are not a held-out language-model evaluation.
- **BLEU-4 and ROUGE-L:** sentence-level, smoothed BLEU and token-level longest-common-subsequence ROUGE-L. The score measures textual overlap, not legal correctness or evidence quality.
- **GRU sequence classifier + additive attention:** classifies one short sentence into one of six broad teaching topics (party, date, amount, obligation, property, or other) and displays the model's token-attention weights. It was CPU-trained on 72 hand-labeled synthetic snippets and evaluated on 12 separately held-out synthetic examples. The current recorded validation accuracy is 0.750. This set is too small to establish reliable generalization; model scores and attention weights are not legal confidence, evidence, or human-readable explanations. The model is not connected to uploaded-document answers or retrieval.
- **Indic script profile:** counts writing-system characters and uses LEGITMATE's lightweight language estimate. This is not full IndicNLP processing or reliable translation.

### Optional local linguistic pipeline

The Model and Toolkit Readiness cards distinguish the package runtime from an installed model checkpoint and from a task connected in LEGITMATE. To add the English spaCy pipeline to the active virtual environment, run these commands from the project folder in PowerShell:

```powershell
python -m pip install spacy
python -m spacy download en_core_web_sm
```

Restart LEGITMATE afterward and refresh NLP Studio. Analysis reports the selected spaCy model or explicitly reports the rule fallback. The optional installation does not download models automatically, change uploaded files, or promote extracted entities to verified facts.

Other cards provide copyable setup/next-step commands. Installing gensim, PyTorch, Whisper, Transformers, or IndicNLP does not by itself create a connected model task. The app reports missing checkpoints and tasks separately; no architecture is presented as working just because its package is installed.

To reproducibly retrain the teaching checkpoint with the fixed seed, run:

```powershell
.venv\Scripts\python.exe data\pipeline\train_gru_sequence_demo.py
```

The script saves `models/gru-sequence-demo.pt`. It uses only the synthetic training split in `data/evaluation/sequence_demo_cases_v1.json`; the validation examples are held out from gradient updates. Training accuracy is not a substitute for a property/civil evaluation set.

## CS3250 coverage status

| Syllabus area | Current LEGITMATE status |
| --- | --- |
| Module 1: tokenization and NER | Demonstrated in the Studio and existing document extraction. |
| Module 1: POS and lemmatization | Optional installed spaCy model for supported detected languages; transparent rule fallback when the model is missing or language is unsupported. |
| Module 1: N-grams, perplexity, BoW, TF-IDF | Local demonstrations available; meaningful perplexity needs a separate training/test corpus. |
| Module 1: Word2Vec, GloVe, FastText | No connected vector checkpoint or similarity task. The Studio distinguishes an optional gensim runtime from actual embeddings. |
| Module 2: GRU and attention | A small CPU-trained synthetic teaching model is available in NLP Studio, with token-attention weights. It is not a legal classifier. |
| Module 2: CNN, vanilla RNN, LSTM | Not implemented. PyTorch being installed is only a runtime prerequisite. |
| Module 2: BLEU, ROUGE, precision, recall, F1 | Text-overlap demo available. Production evaluation still needs a curated reference set and task-specific metrics. |
| Module 3: ELMo/BERT/RoBERTa/ALBERT/GPT/T5 | The project has an optional multilingual E5 retrieval encoder and Gemini API drafting. These are not demonstrations of all listed architectures; fine-tuning is not complete. |
| Module 4: closed-domain QA and conversational AI | Implemented as local cited retrieval/abstention and optional Gemini/Groq chat/drafting. Open-domain QA, Rasa/Dialogflow, summarization, and translation are not complete. |
| Module 5: Whisper/Wav2Vec, CLIP, IndicNLP | Script profiling is a lightweight local demo. ASR, CLIP, and full IndicNLP integrations require their libraries, compatible model weights, and validation; they are not currently active. |

## Next implementation boundary

The synthetic property/civil retrieval regression set v2 is available at `data/evaluation/property_civil_cases_v2.json` (50 questions over sample PDFs); v1 is preserved. It supports baseline comparisons but is not a reviewed real-world validation set. Expand it with reviewed, appropriately licensed property/civil documents before training a legal classifier or fine-tuning retrieval embeddings. Speech recognition and CLIP should be optional modules with explicit model-download/storage controls; they are not dependencies for the legal document assistant. Never report an architecture as implemented only because its Python package is installed.
