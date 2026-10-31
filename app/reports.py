"""Self-contained HTML report for document NLP and evidence review."""
from __future__ import annotations

from html import escape
from typing import Any


def _e(value: Any) -> str:
    return escape(str(value if value is not None else ""), quote=True)


def _table(headers: list[str], rows: list[list[Any]], empty: str = "No results in this analysis sample.") -> str:
    if not rows:
        return f'<p class="empty">{_e(empty)}</p>'
    head = "".join(f"<th>{_e(value)}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{_e(value)}</td>" for value in row) + "</tr>" for row in rows)
    return f"<div class=table-wrap><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def render_document_report(report: dict[str, Any]) -> str:
    analysis = report.get("nlp_analysis") or {}
    scorecard = report.get("coverage_scorecard") or {}
    counts = report.get("counts") or {}
    pp = analysis.get("perplexity") or {}
    topics = "".join(f'<span class=tag>{_e(topic)}</span>' for topic in report.get("topics", [])) or '<span class=empty>No topic cues matched.</span>'
    metrics = [
        ("Pages / sections", report.get("pages_or_sections", 0)),
        ("Readable sections", report.get("readable_pages_or_sections", 0)),
        ("Sections scored", f"{scorecard.get('sections_scored', 0)} / {scorecard.get('sections_total', 0)}"),
        ("Full-document tokens", sum(int(row.get("tokens", 0) or 0) for row in scorecard.get("per_section", []))),
        ("Sample unique terms", analysis.get("unique_terms", 0)),
        ("Candidate statements", counts.get("claim", 0)),
        ("Dates / amounts", int(counts.get("date", 0)) + int(counts.get("amount", 0))),
    ]
    metric_html = "".join(f'<div class=metric><strong>{_e(value)}</strong><span>{_e(label)}</span></div>' for label, value in metrics)

    tokens = analysis.get("tokens") or []
    token_rows = [[item.get("text"), item.get("pos"), item.get("lemma")] for item in tokens[:160]]
    bow_rows = [[item.get("term"), item.get("count")] for item in (analysis.get("bow") or [])]
    tfidf_rows = [[item.get("term"), item.get("tf"), item.get("tfidf")] for item in (analysis.get("tfidf") or [])]
    bigram_rows = [[item.get("text"), item.get("count")] for item in (analysis.get("bigrams") or [])]
    entity_rows = [[item.get("kind"), item.get("value"), item.get("evidence")] for item in (analysis.get("entities") or [])]
    script_rows = [[item.get("name"), item.get("count")] for item in (analysis.get("scripts") or [])]
    candidate_rows = [[item.get("kind"), item.get("value"), f"{item.get('citation_unit', 'section')} {item.get('page', '')}", item.get("excerpt")] for item in (report.get("candidate_highlights") or [])]
    topic_score_rows = [[item.get("topic"), item.get("sections"), f"{item.get('coverage_pct', 0)}%", item.get("cue_occurrences")] for item in scorecard.get("topics", [])]
    section_score_rows = [[item.get("section"), item.get("status"), item.get("tokens"), item.get("unique_terms"), f"{item.get('lexical_diversity_pct', 0)}%", f"{item.get('topic_coverage_pct', 0)}%", ", ".join(item.get("topic_cues", [])) or "—", item.get("candidates")] for item in scorecard.get("per_section", [])]
    ocr = report.get("ocr") or {}
    ocr_detection = ocr.get("language_detection") or {}
    ocr_rows = [[row.get("page"), row.get("recognized_words"), row.get("recognized_characters"), f"{round(float(row.get('alphabetic_share', 0)) * 100)}%", ", ".join(row.get("flags", [])) or "No basic text warning"] for row in ocr.get("quality_signals", [])]
    warnings = "".join(f'<li>{_e(item)}</li>' for item in report.get("warnings", []))
    warnings_html = f'<section class=notice><strong>Review notes</strong><ul>{warnings}</ul></section>' if warnings else ""
    language = report.get("language") or {}
    methods = analysis.get("methods") or {}
    sample_note = report.get("analysis_scope") or "NLP diagnostics use a bounded sample from the document; legal candidate extraction scans all readable pages/sections."

    sections = [
        ("Tokenization, POS and approximate lemmas", _table(["Token", "POS estimate", "Lemma estimate"], token_rows, "No tokens were found.")),
        ("Bag of Words", _table(["Term", "Count"], bow_rows)),
        ("TF-IDF", _table(["Term", "Term count", "TF-IDF score"], tfidf_rows)),
        ("Frequent bigrams", _table(["Word pair", "Count"], bigram_rows)),
        ("Language model perplexity", f'<div class=statline><span>Unigram: <b>{_e(pp.get("unigram", "—"))}</b></span><span>Bigram: <b>{_e(pp.get("bigram", "—"))}</b></span><span>Tokens: <b>{_e(pp.get("tokens", 0))}</b></span></div><p class=empty>Computed on the sampled document text with add-one smoothing. This descriptive teaching metric is not a legal accuracy score.</p>'),
        ("Entity candidates", _table(["Type", "Candidate", "Text evidence"], entity_rows)),
        ("Writing-system profile", _table(["Unicode script group", "Character count"], script_rows)),
        ("Full-document topic cue scores", f'<p class=empty>{_e(scorecard.get("note", "Coverage metrics summarize observable text signals; they are not legal accuracy scores."))}</p>{_table(["Topic cue", "Sections matched", "Section coverage", "Cue occurrences"], topic_score_rows, "No readable sections were available to score.")}'),
        ("Every page / section · full-document scorecard", _table(["Source section", "Status", "Tokens", "Unique terms", "Lexical diversity", "Topic cue score", "Topic cues", "Candidates"], section_score_rows, "No page/section records were available.")),
        ("Evidence-linked extracted candidates", _table(["Type", "Candidate text", "Source", "Context"], candidate_rows)),
    ]
    if ocr_rows:
        auto_language = (f'<p class=empty>Automatic language selection: {_e(ocr_detection.get("detected_label", "language uncertain"))} → local pack {_e(ocr_detection.get("language", ocr.get("language", "unknown")))} via {_e(ocr_detection.get("method", "local detection"))}.'
                         + (" A local fallback pack was used because the detected language data is not installed." if ocr_detection.get("fallback") else "")
                         + "</p>") if ocr_detection else ""
        sections.insert(0, ("Local OCR page review signals", f'<p class=empty>Language pack: {_e(ocr.get("language", "unknown"))}. These simple text diagnostics are warning indicators only, not OCR confidence scores. Compare every finding with the original page image.</p>{auto_language}{_table(["Page", "Words", "Characters", "Alphabetic share", "Review signals"], ocr_rows)}'))
    sections_html = "".join(f'<section class=card><h2>{_e(title)}</h2>{body}</section>' for title, body in sections)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LEGITMATE · {_e(report.get("filename", "Document"))} · NLP Report</title>
<style>
:root{{--plum:#761344;--gold:#b38a46;--ink:#202637;--muted:#687386;--line:#e8e0d9;--paper:#faf8f5}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 "Times New Roman",serif}}
.page{{max-width:1100px;margin:0 auto;padding:40px 24px 64px}}.masthead{{padding:30px;border-radius:18px;color:white;background:linear-gradient(120deg,#5f123b,#981b5a);box-shadow:0 16px 38px #32101f22}}
.kicker{{color:#f1d69e;font-size:12px;font-weight:bold;letter-spacing:.15em;text-transform:uppercase}}h1{{margin:8px 0;font-size:clamp(30px,5vw,46px);font-weight:500;line-height:1.08}}.meta{{color:#f2dfe8;font-size:15px;overflow-wrap:anywhere}}
.metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}}.metric,.card{{border:1px solid var(--line);border-radius:13px;background:white;box-shadow:0 4px 16px #1e29390b}}.metric{{padding:14px 16px}}.metric strong{{display:block;color:var(--plum);font-size:26px}}.metric span,.label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.07em}}
.topics{{display:flex;flex-wrap:wrap;gap:8px;margin:15px 0}}.tag{{padding:6px 11px;border:1px solid #e6d6b2;border-radius:999px;background:#fbf5e8;color:#69471d;font-size:14px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.card{{min-width:0;padding:18px;break-inside:avoid}}.card h2{{margin:0 0 12px;color:var(--plum);font-size:21px;font-weight:600}}.table-wrap{{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:9px}}table{{width:100%;border-collapse:collapse;font-size:15px}}th,td{{padding:10px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{position:sticky;top:0;color:#fff;background:var(--plum);font-size:14px}}td{{max-width:420px}}.notice{{margin:16px 0;padding:14px 17px;border-left:4px solid var(--gold);border-radius:0 10px 10px 0;background:#fff8e8}}.notice ul{{margin:5px 0 0;padding-left:20px}}.empty,.scope,.foot{{color:var(--muted);font-size:15px}}.scope{{padding:14px 16px;border:1px solid var(--line);border-radius:10px;background:white;margin:16px 0}}.statline{{display:flex;flex-wrap:wrap;gap:9px}}.statline span{{padding:10px 12px;border-radius:8px;background:#f8f4ed}}.foot{{margin-top:20px;padding-top:14px;border-top:1px solid var(--line)}}
@media(max-width:700px){{.grid{{grid-template-columns:1fr}}.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}.page{{padding:18px 12px 38px}}.masthead{{padding:22px}}}}
@media print{{body{{background:white}}.page{{max-width:none;padding:0}}.masthead{{box-shadow:none}}.card,.metric{{box-shadow:none}}.grid{{gap:8px}}.table-wrap{{max-height:none;overflow:visible}}th{{position:static}}}}
</style></head><body><main class=page>
<header class=masthead><div class=kicker>LEGITMATE · Document intelligence</div><h1>NLP analysis report</h1><div class=meta>{_e(report.get("filename", "Document"))} · {_e(report.get("file_type", "file").upper())} · {int(report.get("size_bytes",0))/(1024*1024):.2f} MB · Generated {_e(report.get("generated_at", ""))}</div></header>
<div class=metrics>{metric_html}</div><div class=label>Likely topic cues</div><div class=topics>{topics}</div>
<div class=scope><b>Language estimate:</b> {_e(language.get("label", "Undetermined"))} · {_e(language.get("script", "Unknown"))} · confidence estimate {_e(language.get("confidence", 0))}<br><b>Analysis scope:</b> {_e(sample_note)}<br><b>Methods:</b> {_e(methods.get("tokenization", ""))}; {_e(methods.get("pos", ""))}; {_e(methods.get("lemmatization", ""))}; TF-IDF corpus passages: {_e(methods.get("tfidf_corpus_docs", 0))}.</div>
{warnings_html}<div class=grid>{sections_html}</div>
<section class=card style="margin-top:14px"><h2>Answer comparison scores</h2><p class=empty>BLEU-4, ROUGE-L, precision, recall and F1 are now available in NLP Studio whenever you provide both a reference answer and a candidate answer. They score wording overlap, not factual correctness; this document scorecard instead measures coverage and observable text signals across every readable section.</p><p class=empty>Showing up to 100 source-linked candidates in this report ({_e(report.get("total_candidates", 0))} found overall). Open Review Facts in LEGITMATE for the complete indexed list.</p></section>
<footer class=foot>{_e(report.get("notice", "Automated extraction is a set of review candidates, not verified facts or legal advice."))} All extracted content and NLP outputs require human review against the original file.</footer>
</main></body></html>'''
