#!/usr/bin/env python3
"""CASPER: character-anchored salient passage extraction over full novels.
Given a novel and a target character, locate the character (alias regex,
first-person heuristic), extract event/attribute/relationship anchors by
syntactic gating, window each anchor (-4/+1), score spans by entities +
transitive verbs + centroid-cosine centrality, order coverage-first, pack
to a token budget, then synthesize the profile in one generative call.
Everything before synthesis is non-generative.
"""
import re, csv, json, threading
from pathlib import Path
import numpy as np
import spacy, tiktoken
from sentence_transformers import SentenceTransformer
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[0]

# configuration (prior-fixed, not tuned)
GEN_MODEL = "gpt-4o-mini-2024-07-18"                    # the only generative model in the pipeline
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WIN = (4, 1)                                            # sentences before/after each anchor
MAX_SPAN = 10                                           # max sentences per merged span
ARC_THIRDS, REL_TARGET, ATTR_TARGET = 3, 3, 2
SAL_W = {"ne": 1.0, "trans": 1.0, "centrality": 2.0}
WORD_LIMIT = 1200                                       # word budget passed to the synthesis prompt

enc = tiktoken.get_encoding("o200k_base")
def ntok(s): return len(enc.encode(s or ""))
def getenv(k):
    for l in open(ROOT / ".env"):
        if l.startswith(k + "="): return l.split("=", 1)[1].strip()
oai = OpenAI(api_key=getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1", timeout=180)

# data wiring: titles, character names, match terms, prompt template
def norm(s): return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")
_items = [json.loads(l) for l in open(ROOT / "data/mr_items.jsonl")]
id2title = {it["book_id"]: it["title"] for it in _items}
id2char = {it["book_id"]: it["character"] for it in _items}
MT = {r["book_id"]: [x.strip() for x in (r.get("match_terms") or "").split(";") if x.strip()]
      for r in csv.DictReader(open(ROOT / "data/character_match_terms.csv"))}
PROFILE_PROMPT = (ROOT / "prompts/original_character_profiling/profile_once_prompt.txt").read_text()

# NLP setup
print("loading spaCy + MiniLM ...", flush=True)
emb = SentenceTransformer(EMB_MODEL)                                 # shared (torch inference thread-safe)
seg = spacy.blank("en"); seg.add_pipe("sentencizer"); seg.max_length = 6_000_000  # rule-based sentencizer
_tls = threading.local()
def get_nlp():
    if not hasattr(_tls, "nlp"):
        _tls.nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"]); _tls.nlp.max_length = 6_000_000
    return _tls.nlp

SUBJ = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
PRON3 = {"he", "she", "they", "him", "her", "them"}
COMPL = {"dobj", "obj", "dative", "attr", "oprd", "ccomp", "xcomp", "advcl", "prep", "acomp", "agent"}
COP = {"be", "is", "am", "are", "was", "were", "been", "being"}

def split_sents(t):
    return [(s.start_char, s.text.strip()) for s in seg(t).sents if 3 <= len(s.text.strip()) <= 600]
def matches_char(t, terms):
    # full-name match only (no surname over-match)
    tl = t.lower(); return any(tl == x.lower() for x in terms)
def vocative(tok):
    nbrs = [tok.nbor(k).text for k in (-1, 1) if 0 <= tok.i + k < len(tok.doc)]
    return ("," in nbrs) and any(q in tok.doc.text for q in ('"', "'", "“"))

# anchor detection
def parse_once(book):
    """One expensive spaCy parse per book; returns materials reused across window settings.
       Localizes the protagonist (alias regex + first-person heuristic) and extracts
       event / attribute / relationship anchors by syntactic gating."""
    text = (ROOT / f"data/books/{book}.txt").read_text(errors="replace")
    terms = MT.get(book, [id2char[book]]); first = id2char[book].split()[0]
    term_pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)) + r")\b", re.I)
    sents = split_sents(text); n = len(sents)
    mention_idx = [i for i, (_, s) in enumerate(sents) if term_pat.search(s)]
    fp_pat = re.compile(r"\b(I|my|me|myself)\b"); fp_idx = [i for i, (_, s) in enumerate(sents) if fp_pat.search(s)]
    first_person = len(fp_idx) > max(40, 5 * len(mention_idx)); FP = {"i"} if first_person else set()
    cand = sorted(set(mention_idx) | (set(fp_idx) if first_person else {i + 1 for i in mention_idx if i + 1 < n}))
    if len(cand) > 3000:
        step = len(cand) / 3000; cand = [cand[int(k * step)] for k in range(3000)]
    named = set(mention_idx); proto = lambda i: (i in named) or (first_person and fp_pat.search(sents[i][1]) is not None)
    nlp = get_nlp(); feat = {}; ev_anchor, attr_anchor = set(), set(); rel_anchor = {}
    for i, doc in zip(cand, nlp.pipe([sents[i][1] for i in cand], batch_size=64)):
        others = [e.text for e in doc.ents if e.label_ == "PERSON" and not matches_char(e.text.split()[0], terms) and e.text.lower() != first.lower()]
        feat[i] = {"ne": sum(1 for e in doc.ents if e.label_ in ("PERSON", "GPE", "ORG", "FAC", "LOC", "EVENT", "NORP")),
                   "trans": sum(1 for t in doc if t.pos_ == "VERB" and any(c.dep_ in ("dobj", "obj") for c in t.children))}
        hn = i in named
        for tok in doc:
            # event anchor: protagonist as subject with a complement child
            if tok.dep_ in SUBJ and tok.head.pos_ in ("VERB", "AUX"):
                sn = matches_char(tok.text, terms); sf = tok.text.lower() in FP; sp = tok.text.lower() in PRON3
                if not (sn or sf or (sp and hn and not others)): continue
                if sn and vocative(tok): continue
                if not any(c.dep_ in COMPL for c in tok.head.children): continue
                ev_anchor.add(i); break
            # attribute anchor: copula subject, or appositive on the protagonist
            sb = matches_char(tok.text, terms) or (tok.text.lower() in FP)
            if sb and tok.dep_ in SUBJ and tok.head.lemma_.lower() in COP: attr_anchor.add(i)
            if matches_char(tok.text, terms) and any(c.dep_ == "appos" for c in tok.children): attr_anchor.add(i)
        # relationship anchor: protagonist sentence co-occurring with other PERSON entities
        if proto(i) and others: rel_anchor[i] = set(others)
    return {"sents": sents, "n": n, "first_person": first_person, "ev": ev_anchor, "attr": attr_anchor, "rel": rel_anchor, "feat": feat}

# centroid-cosine centrality
def centrality_centroid(svec):
    """No-graph centrality: cosine of each window vector to the mean window vector.
       svec is L2-normalized (emb.encode(normalize_embeddings=True)) so dot == cosine."""
    n = int(svec.shape[0]) if hasattr(svec, "shape") else len(svec)
    if n == 0: return {0: 0.0}
    centroid = svec.mean(axis=0, keepdims=True)
    cos = (svec @ centroid.T).ravel()
    return {i: float(cos[i]) for i in range(n)}

# windows, salience, coverage-first ordering
def build_ordered(P, window):
    """Windows around anchors -> merged spans -> salience scoring -> coverage-first ordering."""
    sents, n = P["sents"], P["n"]; WL, WR = window
    anchors = sorted(P["ev"] | set(P["rel"]) | P["attr"])
    # window -WL/+WR around each anchor, merge adjacent intervals, chunk to <= MAX_SPAN sentences
    intervals = sorted([(max(0, i - WL), min(n - 1, i + WR)) for i in anchors]); merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1] + 1: merged[-1][1] = max(merged[-1][1], hi)
        else: merged.append([lo, hi])
    spans = []
    for lo, hi in merged:
        k = lo
        while k <= hi: spans.append([k, min(k + MAX_SPAN - 1, hi)]); k += MAX_SPAN
    sp = []
    for lo, hi in spans:
        idxs = list(range(lo, hi + 1)); persons = set().union(*[P["rel"].get(i, set()) for i in idxs]) if any(i in P["rel"] for i in idxs) else set()
        txt = " ".join(sents[i][1] for i in idxs)
        sp.append({"mid": (lo + hi) / 2, "text": txt, "pos": sents[lo][0], "tokens": ntok(txt),
                   "has_ev": any(i in P["ev"] for i in idxs), "has_attr": any(i in P["attr"] for i in idxs), "persons": persons,
                   "ne": sum(P["feat"].get(i, {}).get("ne", 0) for i in idxs), "trans": sum(P["feat"].get(i, {}).get("trans", 0) for i in idxs)})
    if not sp: return []
    # salience = 1*z(entity) + 1*z(transitive-verb) + 2*z(centroid centrality)
    svec = emb.encode([s["text"] for s in sp], normalize_embeddings=True, batch_size=128)
    cen = centrality_centroid(svec)
    def z(a):
        a = np.array(a, float); return (a - a.min()) / (np.ptp(a) + 1e-9)
    nz, tz, pz = z([s["ne"] for s in sp]), z([s["trans"] for s in sp]), z([cen.get(i, 0.0) for i in range(len(sp))])
    for k, s in enumerate(sp): s["sal"] = SAL_W["ne"] * nz[k] + SAL_W["trans"] * tz[k] + SAL_W["centrality"] * pz[k]
    # coverage-first ordering: arc thirds (event) -> relationship persons -> attributes -> rest by salience
    order_by_sal = sorted(range(len(sp)), key=lambda k: -sp[k]["sal"]); picked, ordered = set(), []
    def push(k):
        if k not in picked: picked.add(k); ordered.append(k)
    for th in range(ARC_THIRDS):
        lo, hi = th / ARC_THIRDS, (th + 1) / ARC_THIRDS
        for k in order_by_sal:
            if lo <= sp[k]["mid"] / max(n, 1) < hi and sp[k]["has_ev"]: push(k); break
    cov = set()
    for k in order_by_sal:
        if len(cov) >= REL_TARGET: break
        if sp[k]["persons"] - cov: push(k); cov |= sp[k]["persons"]
    na = 0
    for k in order_by_sal:
        if na >= ATTR_TARGET: break
        if sp[k]["has_attr"]: push(k); na += 1
    for k in order_by_sal: push(k)
    return [sp[k] for k in ordered]

# budget truncation
def evidence_at(ordered, budget):
    """Greedily pack coverage-first spans up to the token budget; emit in reading (position) order."""
    chosen, spent = [], 0
    for s in ordered:
        if spent + s["tokens"] > budget: continue
        chosen.append(s); spent += s["tokens"]
        if spent >= budget - 30: break
    return "\n\n".join(s["text"] for s in sorted(chosen, key=lambda s: s["pos"])), spent

# generative synthesis
def synthesize(book, evidence):
    """The single generative call: fixed prompt template, temperature 0."""
    r = oai.chat.completions.create(model=GEN_MODEL, temperature=0,
        messages=[{"role": "user", "content": PROFILE_PROMPT.format(evidence, id2char[book], WORD_LIMIT)}], max_tokens=1800)
    return r.choices[0].message.content or ""

# end-to-end
def casper_profile(book, budget=35000, window=WIN):
    """Run the full pipeline for one book: parse -> order spans -> pack evidence -> synthesize.
       Returns (profile_text, evidence_text, used_tokens)."""
    P = parse_once(book)
    ordered = build_ordered(P, window)
    evidence, used = evidence_at(ordered, budget)
    profile = synthesize(book, evidence) if evidence.strip() else ""
    return profile, evidence, used

if __name__ == "__main__":
    import sys
    book = sys.argv[1] if len(sys.argv) > 1 else next(iter(id2char))
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 35000
    prof, ev, used = casper_profile(book, budget)
    print(f"\n[{book}] budget={budget} evidence_tokens={used}\n{'='*60}\n{prof}", flush=True)
