#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate.py - seeded simulation of the EXPECTED results of the planned study
"Reliable information retrieval for Kazakh-language legal question answering"
(Results section, "expected results" option of the assignment).

This script generates EXPECTED values for the Results section of the article;
it does not run retrieval or generation models. Effect sizes are assumptions
calibrated on the literature cited in the article.

Every number written by this script is an expected value from a simulation.
It must be reported as expected / simulated, never as a measurement.

How the simulation works
------------------------
1. Randomness. A master numpy SeedSequence(2026) is spawned into independent
   child streams (questions, corpus, retrieval, latency, generation, judge,
   bootstrap, randomization); every stream feeds a Generator(PCG64). All random
   draws are made up front in a fixed order. The stream mapping is written to
   data/sim_config.json. No other seed is ever tried.
2. Per-question outcomes. Each of the 500 benchmark questions gets a type, a
   dev/test split (100/400, stratified by type), a number of gold chunks and a
   latent difficulty. For every planned configuration the script simulates
   per-question outcomes:
     retrieval  - the rank of every gold chunk (log-rank model with shared
                  latent components, so configurations are correlated per
                  question as in a real paired evaluation), from which
                  Recall@5, Recall@10, MRR, P@1 and nDCG@10 are computed,
                  plus a per-query latency;
     generation - abstention, legal correctness (0 / 0.5 / 1), hallucination,
                  support, citation accuracy, exact match and token F1 from a
                  latent answer-quality model with terms for model, retrieval
                  on/off, gold chunk in context, prompt language, question
                  type and difficulty.
3. Calibration. The assumed expected values ("targets" in CONFIG, set from the
   literature reviewed in the article and from our earlier work on
   structure-aware hybrid legal RAG) are turned into effect parameters by
   bisection on the fixed draws: the effect parameter of a configuration is the
   value for which the simulated sample mean equals the assumed value (up to
   the resolution of the metric). Noise-structure parameters (correlations
   between configurations) are fixed in CONFIG; they were set once so that the
   designed significance pattern of the specification holds.
4. Analysis. The planned analysis is run on the simulated per-question scores
   exactly as it will be run on real scores: 95% percentile bootstrap CIs
   (10,000 resamples of questions), paired bootstrap CIs of differences,
   two-sided paired approximate randomization (10,000 sign flips) for
   continuous scores, exact McNemar for binary outcomes, Holm-Bonferroni within
   the planned families, alpha = 0.05.
5. Output. CSV and JSON files (data/), vector PDF figures (figures/, PNG
   previews in figures/preview/) and LaTeX tables (tables/).

Usage:   python simulate.py [--seed 2026] [--out <results dir>]
Needs:   Python 3, NumPy, SciPy, pandas, matplotlib.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Constants of the planned analysis
# ---------------------------------------------------------------------------
MASTER_SEED = 2026
N_BOOT = 10_000          # bootstrap resamples (percentile CIs)
N_PERM = 10_000          # sign flips in the approximate randomization test
SIG_LEVEL = 0.05         # family-wise alpha after Holm-Bonferroni
DEPTH = 100              # retrieval depth: a gold chunk below rank 100 counts as not retrieved

STREAMS = ["questions", "corpus", "retrieval", "latency", "generation",
           "judge", "bootstrap", "randomization"]

TYPES = ["norm", "interp", "applic", "proc"]
TYPE_NAMES = {"norm": "Norm identification",
              "interp": "Interpretation of provisions",
              "applic": "Applicability conditions",
              "proc": "Procedural requirements"}

MODELS = ["gpt", "kaz", "llama", "qwen"]
MODEL_NAMES = {"gpt": "GPT-6.1", "kaz": "KazLLM-1.0-8B",
               "llama": "Llama-3.1-8B", "qwen": "Qwen3.5-9B"}

RET_NAMES = {"sentence": "Sentence-level", "fixed": "Fixed-size 200",
             "sliding": "Sliding window 200/100", "sa": "Structure-aware",
             "tfidf": "TF-IDF", "bm25": "BM25", "labse": "LaBSE",
             "e5": "multilingual-E5-large", "bge": "BGE-M3 (dense)",
             "lin": "Hybrid, linear (alpha = 0.4)", "rrf": "Hybrid, RRF (k = 60)",
             "rr": "Hybrid, linear + reranker"}
RET_METRICS = ["R5", "R10", "MRR", "P1", "nDCG10"]
GEN_METRICS = ["em", "f1", "cit", "sup", "hal", "cor"]
GEN_METRIC_NAMES = {"em": "EM", "f1": "F1", "cit": "Citation accuracy",
                    "sup": "Support", "hal": "Hallucination", "cor": "Legal correctness"}
ERR_CATS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
ERR_LABELS = {"E1": "E1 Retrieval miss",
              "E2": "E2 Citation missing",
              "E3": "E3 Wrong/invented provision",
              "E4": "E4 Incomplete answer",
              "E5": "E5 Misinterpretation",
              "E6": "E6 Cross-ref. / repealed norm",
              "E7": "E7 Language error",
              "E8": "E8 Unjustified abstention"}

# RAG variants simulated per model: prompt language x number of chunks k
RAG_VARIANTS = ["kk5", "ru5", "kk1", "kk3", "kk10"]
RUN_SEEDS = [13, 42]     # extra generation runs (2026 is the main run)

# ---------------------------------------------------------------------------
# Configuration: design values, assumed expected values (targets) and the
# fixed noise-structure parameters. Written to data/sim_config.json.
# ---------------------------------------------------------------------------
CONFIG = {
    "master_seed": MASTER_SEED,
    "n_bootstrap": N_BOOT,
    "n_permutations": N_PERM,
    "alpha": SIG_LEVEL,
    "design": {
        "types": TYPES,
        "n_all": [160, 130, 110, 100],
        "n_dev": [32, 26, 22, 20],
        "no_gold_test": [4, 6, 5, 4],      # 19 test questions without linkable gold
        "no_gold_dev": [1, 1, 1, 1],       # 4 dev questions without linkable gold
        "gold_count_values": [1, 2, 3],
        "gold_count_probs": [0.55, 0.30, 0.15],
    },
    "corpus": {
        # reported facts (KazakhLawCorpus data paper)
        "raw_records": 223245, "clean_records": 215889,
        # simulated quantities
        "p_removed_no_title": 0.953,
        "p_flagged": 0.0426,
        "p_eligible": 0.19,
        "acts_planned": 1200,
        "act_pool": 3000, "act_zipf_s": 1.12, "p_second_act": 0.15,
        "p_parse_fail_distractor": 14 / 986,
        "words_total_expected": 9.4e6, "words_sigma": 1.25,
        "words_per_article": 205.0,
        "sent_median": 14.0, "sent_sigma": 0.62,
        "unit_median": 112.0, "unit_sigma": 0.95,
        "short_frag_rate": 0.09, "short_frag_median": 9.0,
        "max_chunk_words": 450, "short_words": 20, "p_merge_short": 0.72,
        "boiler_frags_per_act": 0.4, "boiler_median": 9.0,
        "removed_token_share_mean": 0.035,
        "p_act_latin": 0.37, "latin_word_share_in_affected": 0.024,
    },
    "retrieval": {
        "rho_grid": [0.8, 0.85, 0.9, 0.95],  # candidate within-question correlations of gold chunks
        "w_difficulty": 0.40,           # share of question-level variance from difficulty
        "families": {"tfidf": ["lex", 0.35], "bm25": ["lex", 0.40],
                     "labse": ["den", 0.20], "e5": ["den", 0.30], "bge": ["den", 0.40]},
        "chunk_shared": 0.5,            # share of chunk-level noise common to retrievers
        "hybrid_own_h0": 0.35,          # weight of hybrid-specific noise at alpha = 0.5
        "alpha_curve_gamma": 1.6,       # shape of the alpha curve between anchors
        "rho_rrf": 0.93, "rho_rr": 0.80, "reranker_candidates": 50,
        "rho_chunking": {"sentence": 0.55, "fixed": 0.65, "sliding": 0.75},
        "ablation_affected_prob": {"translit": 0.37, "boilerplate": 0.55, "merging": 0.30},
        "ablation_sigma": 0.6,
        "neutral_alpha_rq1": 0.5,
        "selected_alpha_expected": 0.4,
        "alpha_grid": [round(0.1 * i, 1) for i in range(11)],
        # assumed expected values (%; MRR and nDCG x100) on the 381 test questions with gold
        "targets": {
            "sentence": [50.5, 58.8, 40.6, 30.7, 44.9],
            "fixed":    [57.6, 65.4, 47.2, 36.5, 51.3],
            "sliding":  [62.3, 69.8, 53.1, 42.0, 56.4],
            "sa":       [66.4, 74.9, 54.6, 43.3, 57.9],
            "tfidf":    [46.1, 54.0, 38.2, 29.4, 42.3],
            "bm25":     [52.7, 60.6, 44.1, 34.4, 48.2],
            "labse":    [47.0, 55.9, 38.5, 28.6, 42.6],
            "e5":       [56.5, 64.8, 47.6, 36.2, 51.5],
            "bge":      [60.4, 68.9, 52.8, 41.5, 56.1],
            "lin":      [66.9, 75.5, 54.4, 43.0, 58.6],
            "rrf":      [66.3, 75.0, 54.0, 42.5, 58.1],
            "rr":       [68.2, 76.1, 56.9, 45.9, 60.3],
        },
        "type_targets_R10": {"bm25": [71.0, 53.1, 54.2, 61.0],
                             "lin": [83.1, 70.6, 69.4, 75.8]},
        "ablation_targets_R10": {"bm25": {"translit": 56.4, "boilerplate": 58.0, "merging": 59.9},
                                 "lin": {"translit": 74.3, "boilerplate": 74.0, "merging": 74.8}},
        "ablation_targets_MRR_reference": {"bm25": {"translit": 41.0, "boilerplate": 42.6, "merging": 43.5},
                                           "lin": {"translit": 53.7, "boilerplate": 52.6, "merging": 52.9}},
        "latency_ms": {"sentence": 78, "fixed": 47, "sliding": 58, "sa": 49,
                       "tfidf": 9, "bm25": 14, "labse": 31, "e5": 37, "bge": 36,
                       "fusion": 1.0, "reranker_extra": 361},
        "latency_sigma": 0.30, "reranker_latency_sigma": 0.25,
    },
    "generation": {
        "rho_difficulty": 0.5,          # correlation of generation and retrieval difficulty
        "var_components": {"G": 0.30, "U": 0.20, "V": 0.30, "W": 0.20},
        "psi_hal_redraw": 0.35,         # share of hallucination draws that differ between RAG variants
        "hal_logit_corr0": 2.2, "hal_logit_corr05": 1.0,
        "sup_logit_per_corr": 1.6,
        "cit_logit_per_corr": 1.2, "cit_logit_hit": 1.0,
        "cit_count_probs": [0.55, 0.30, 0.15],
        "f1_slope_corr": 1.5, "f1_sigma": 0.7, "f1_abstain_logit": -3.2,
        "no_citation_prob": {"gpt": {"llm": 0.40, "rag": 0.05}, "kaz": {"llm": 0.72, "rag": 0.18},
                             "llama": {"llm": 0.80, "rag": 0.84}, "qwen": {"llm": 0.68, "rag": 0.15}},
        "abstain_rate": {"gpt": {"llm": 0.005, "rag": 0.053}, "kaz": {"llm": 0.008, "rag": 0.030},
                         "llama": {"llm": 0.008, "rag": 0.018}, "qwen": {"llm": 0.010, "rag": 0.035}},
        "abstain_logit_miss": {"gpt": 2.6, "kaz": 1.3, "llama": 2.0, "qwen": 1.7},
        "run_redraw_share": {"gpt": 0.30, "kaz": 0.05, "llama": 0.08, "qwen": 0.015},
        # assumed expected values (%), Kazakh prompt, k = 5, n = 400
        "targets": {
            "gpt":   {"llm": [3.8, 33.1, 16.9, 46.3, 34.0, 55.4], "rag": [7.0, 47.2, 66.1, 76.5, 13.8, 73.6]},
            "kaz":   {"llm": [1.8, 24.6, 5.7, 24.8, 56.3, 34.9], "rag": [4.0, 38.4, 47.9, 59.3, 24.5, 58.1]},
            "llama": {"llm": [1.0, 19.2, 4.6, 18.5, 64.8, 27.0], "rag": [2.5, 32.6, 7.2, 48.0, 28.8, 46.6]},
            "qwen":  {"llm": [2.3, 25.7, 7.9, 28.5, 52.0, 38.8], "rag": [4.3, 39.6, 51.8, 61.0, 22.8, 57.4]},
        },
        "full_correct_share_rag": {"gpt": 0.62, "kaz": 0.42, "llama": 0.32, "qwen": 0.43},
        "full_correct_share_given_miss": {"gpt": 0.235, "kaz": 0.17, "llama": 0.105, "qwen": 0.17},
        "type_targets_cor": {"gpt": [82.4, 68.1, 66.2, 74.0], "qwen": [66.0, 52.3, 49.7, 58.8]},
        "ru_targets": {"gpt": [73.1, 14.3], "kaz": [52.6, 27.8], "llama": [48.4, 27.1], "qwen": [58.6, 22.0]},
        "topk_targets": {"gpt": {"1": [63.2, 18.9], "3": [71.0, 15.0], "10": [74.2, 13.5]},
                         "qwen": {"1": [49.0, 27.5], "3": [55.6, 23.6], "10": [55.9, 24.1]}},
        "error_probs_E2_E7": {"gpt": [14, 18, 31, 16, 9, 1], "kaz": [34, 38, 43, 33, 10, 4],
                              "llama": [70, 44, 36, 30, 8, 14], "qwen": [30, 34, 40, 30, 9, 17]},
    },
    "judge": {"n_sample": 100, "per_model": 25,
              "kappa_hal_target": 0.71, "kappa_cor_target": 0.64, "kappa_annot_target": 0.78},
}

# Designed significance pattern of the specification ("sig" / "ns");
# "sig*" = designed significant, a flip to n.s. is accepted by the specification.
DESIGNED = {
    ("F1", "SA - Sentence"): {"R10": "sig", "MRR": "sig", "nDCG10": "sig"},
    ("F1", "SA - Fixed"): {"R10": "sig", "MRR": "sig", "nDCG10": "sig"},
    ("F1", "SA - Sliding"): {"R10": "sig", "MRR": "ns", "nDCG10": "ns"},
    ("F2", "Hybrid linear - BM25"): {"R10": "sig", "MRR": "sig", "nDCG10": "sig"},
    ("F2", "Hybrid linear - BGE-M3"): {"R10": "sig", "MRR": "ns", "nDCG10": "sig*"},
    ("F2", "RRF - linear"): {"R10": "ns", "MRR": "ns", "nDCG10": "ns"},
    ("F2", "Reranker - none"): {"R10": "ns", "MRR": "ns", "nDCG10": "ns"},
    ("F2s", "Reranker - none"): {"P1": "ns"},
    ("F3", "GPT-6.1: RAG - LLM-only"): {"cit": "sig", "sup": "sig", "hal": "sig", "cor": "sig"},
    ("F3", "KazLLM-1.0-8B: RAG - LLM-only"): {"cit": "sig", "sup": "sig", "hal": "sig", "cor": "sig"},
    ("F3", "Llama-3.1-8B: RAG - LLM-only"): {"cit": "ns", "sup": "sig", "hal": "sig", "cor": "sig"},
    ("F3", "Qwen3.5-9B: RAG - LLM-only"): {"cit": "sig", "sup": "sig", "hal": "sig", "cor": "sig"},
    ("F4", "GPT-6.1: Kazakh - Russian"): {"hal": "ns", "cor": "ns"},
    ("F4", "KazLLM-1.0-8B: Kazakh - Russian"): {"hal": "ns", "cor": "sig"},
    ("F4", "Llama-3.1-8B: Kazakh - Russian"): {"hal": "ns", "cor": "ns"},
    ("F4", "Qwen3.5-9B: Kazakh - Russian"): {"hal": "ns", "cor": "ns"},
    ("F5", "BM25: w/o transliteration - full"): {"R10": "sig"},
    ("F5", "BM25: w/o boilerplate removal - full"): {"R10": "sig"},
    ("F5", "BM25: w/o short-fragment merging - full"): {"R10": "ns"},
    ("F5", "Hybrid: w/o transliteration - full"): {"R10": "ns"},
    ("F5", "Hybrid: w/o boilerplate removal - full"): {"R10": "ns"},
    ("F5", "Hybrid: w/o short-fragment merging - full"): {"R10": "ns"},
    ("F6", "GPT-6.1: k=3 - k=5"): {"cor": "ns"},
    ("F6", "GPT-6.1: k=10 - k=5"): {"cor": "ns"},
    ("F6", "Qwen3.5-9B: k=3 - k=5"): {"cor": "ns"},
    ("F6", "Qwen3.5-9B: k=10 - k=5"): {"cor": "ns"},
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bisect(fun, target, lo, hi, increasing=True, iters=45):
    """Find x in [lo, hi] with fun(x) closest to target for a monotone (possibly
    step-wise) function. Returns the end point whose value is closest."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (fun(mid) < target) == increasing:
            lo = mid
        else:
            hi = mid
    return lo if abs(fun(lo) - target) <= abs(fun(hi) - target) else hi


def make_streams(seed):
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.Generator(np.random.PCG64(ch)) for name, ch in zip(STREAMS, children)}


# ---------------------------------------------------------------------------
# 1. Benchmark questions
# ---------------------------------------------------------------------------
def simulate_questions(rng, cfg):
    d = cfg["design"]
    rows, k = [], 0
    for ti, t in enumerate(TYPES):
        n_all, n_dev = d["n_all"][ti], d["n_dev"][ti]
        is_dev = rng.permutation(n_all) < n_dev          # stratified random split
        test_pos, dev_pos = np.flatnonzero(~is_dev), np.flatnonzero(is_dev)
        no_gold = np.concatenate([rng.choice(test_pos, d["no_gold_test"][ti], replace=False),
                                  rng.choice(dev_pos, d["no_gold_dev"][ti], replace=False)])
        n_gold = rng.choice(d["gold_count_values"], size=n_all, p=d["gold_count_probs"])
        n_gold[no_gold] = 0
        for i in range(n_all):
            k += 1
            rows.append({"qid": f"Q{k:03d}", "type": t, "type_idx": ti,
                         "split": "dev" if is_dev[i] else "test", "n_gold": int(n_gold[i])})
    q = pd.DataFrame(rows)
    q["difficulty"] = rng.standard_normal(len(q))
    return q


# ---------------------------------------------------------------------------
# 2. Corpus statistics (Table 1, Panels A and B)
# ---------------------------------------------------------------------------
def _length_stats(lengths, short):
    lengths = np.asarray(lengths)
    return {"chunks": int(lengths.size), "median": float(np.median(lengths)),
            "p5": float(np.percentile(lengths, 5)), "p95": float(np.percentile(lengths, 95)),
            "share_short": float(100 * np.mean(lengths < short)), "words": int(lengths.sum())}


def simulate_corpus(rng, cfg, n_questions):
    c = cfg["corpus"]
    out = {}
    removed = c["raw_records"] - c["clean_records"]                 # reported: 7,356
    no_title = int(rng.binomial(removed, c["p_removed_no_title"]))
    out["removed_total"] = removed
    out["removed_no_title"] = no_title
    out["removed_duplicate_id"] = removed - no_title
    out["flagged_kept"] = int(rng.binomial(c["clean_records"], c["p_flagged"]))
    out["eligible_pool"] = int(rng.binomial(c["clean_records"], c["p_eligible"]))

    # acts referenced by the benchmark: Zipf-like popularity of acts (codes are cited often)
    ranks = np.arange(1, c["act_pool"] + 1)
    w = ranks ** (-c["act_zipf_s"])
    n_refs = n_questions + int(rng.binomial(n_questions, c["p_second_act"]))
    acts = rng.choice(ranks, size=n_refs, p=w / w.sum())
    referenced = int(np.unique(acts).size)
    distractors = c["acts_planned"] - referenced
    failed = int(rng.binomial(distractors, c["p_parse_fail_distractor"]))
    out.update(acts_referenced=referenced, acts_distractors=distractors,
               acts_parse_failed=failed, acts_parsed=c["acts_planned"] - failed)
    n_acts = out["acts_parsed"]

    # words per act after cleaning (log-normal, heavy right tail)
    mu = np.log(c["words_total_expected"] / n_acts) - c["words_sigma"] ** 2 / 2
    words = np.maximum(40, np.round(rng.lognormal(mu, c["words_sigma"], n_acts))).astype(np.int64)
    W = int(words.sum())
    act_end = np.cumsum(words)
    n_art = 1 + rng.poisson(words / c["words_per_article"])
    out.update(words_total=W, articles_total=int(n_art.sum()))

    # sentence-level chunks: sentence boundaries merged with act boundaries
    sent_mean = c["sent_median"] * np.exp(c["sent_sigma"] ** 2 / 2)
    ns = int(W / sent_mean * 1.15) + 1000
    sl = np.maximum(1, np.round(rng.lognormal(np.log(c["sent_median"]), c["sent_sigma"], ns))).astype(np.int64)
    cs = np.cumsum(sl)
    pos = np.union1d(cs[cs < W], act_end)
    sent_len = np.diff(np.concatenate([[0], pos]))

    # fixed-size 200 words without overlap (restarts at each act)
    k_fix = np.ceil(words / 200).astype(np.int64)
    fix_len = np.full(int(k_fix.sum()), 200, dtype=np.int64)
    fix_len[np.cumsum(k_fix) - 1] = words - 200 * (k_fix - 1)

    # sliding window 200 words, stride 100 (restarts at each act)
    k_sl = np.where(words <= 200, 1, np.ceil((words - 200) / 100).astype(np.int64) + 1)
    sld_len = np.full(int(k_sl.sum()), 200, dtype=np.int64)
    start_last = np.where(words <= 200, 0, (k_sl - 1) * 100)
    sld_len[np.cumsum(k_sl) - 1] = np.minimum(200, words - start_last)

    # structure-aware: article / clause units (log-normal lengths) cut at act ends,
    # units longer than 450 words split into equal parts at clause level; short
    # heading or one-line fragments (< 20 words) are mostly merged with the next unit
    nf = int(W / (c["unit_median"] * np.exp(c["unit_sigma"] ** 2 / 2)) * 1.3) + 1000
    fl = np.maximum(1, np.round(rng.lognormal(np.log(c["unit_median"]), c["unit_sigma"], nf))).astype(np.int64)
    fcs = np.cumsum(fl)
    pos = np.union1d(fcs[fcs < W], act_end)
    piece = np.diff(np.concatenate([[0], pos]))
    piece_act = np.searchsorted(act_end, pos, side="left")
    nsplit = np.ceil(piece / c["max_chunk_words"]).astype(np.int64)
    rep_len = np.repeat(piece // nsplit, nsplit)
    rem = np.repeat(piece % nsplit, nsplit)
    within = np.arange(rep_len.size) - np.repeat(np.cumsum(nsplit) - nsplit, nsplit)
    units = rep_len + (within < rem)
    unit_act = np.repeat(piece_act, nsplit)
    # short fragments are inserted before randomly chosen units of the same act
    # (their words are taken from that unit, so every strategy covers the same words)
    eligible = np.flatnonzero(units >= 60)
    n_short = int(rng.binomial(units.size, c["short_frag_rate"]))
    at = np.sort(rng.choice(eligible, n_short, replace=False))
    short_len = np.clip(np.round(rng.lognormal(np.log(c["short_frag_median"]), 0.5, n_short)), 2, 19).astype(np.int64)
    units = units.copy()
    units[at] -= short_len
    pre_len = np.insert(units, at, short_len)
    pre_act = np.insert(unit_act, at, unit_act[at])
    boiler = np.maximum(2, np.round(rng.lognormal(np.log(c["boiler_median"]), 0.5,
                                                  rng.poisson(c["boiler_frags_per_act"] * n_acts)))).astype(np.int64)
    before_clean = np.concatenate([pre_len, boiler])
    u_merge = rng.random(pre_len.size)
    sa_len = pre_len.copy()
    keep = np.ones(sa_len.size, dtype=bool)
    for i in np.flatnonzero(pre_len < c["short_words"]):
        j = i + 1
        if (j < sa_len.size and pre_act[j] == pre_act[i] and u_merge[i] < c["p_merge_short"]
                and sa_len[i] + sa_len[j] <= c["max_chunk_words"]):
            sa_len[j] += sa_len[i]
            keep[i] = False
    sa_len = sa_len[keep]

    short = c["short_words"]
    out["chunking"] = {"sentence": _length_stats(sent_len, short),
                       "fixed": _length_stats(fix_len, short),
                       "sliding": _length_stats(sld_len, short),
                       "sa": _length_stats(sa_len, short),
                       "sa_before_cleaning": _length_stats(before_clean, short)}

    # second-level cleaning: removed tokens and transliterated Latin look-alikes
    a_b = 20.0
    removed_share = rng.beta(a_b * c["removed_token_share_mean"], a_b * (1 - c["removed_token_share_mean"]), n_acts)
    raw_words = words / (1 - removed_share)
    out["removed_token_share"] = float(100 * (raw_words - words).sum() / raw_words.sum())
    latin_act = rng.random(n_acts) < c["p_act_latin"]
    m_l = c["latin_word_share_in_affected"]
    kappa_b = 30.0                                                 # Beta concentration
    latin_share = np.where(latin_act, rng.beta(kappa_b * m_l, kappa_b * (1 - m_l), n_acts), 0.0)
    out["latin_word_share"] = float(100 * (latin_share * words).sum() / W)
    out["latin_act_share"] = float(100 * latin_act.mean())
    return out


# ---------------------------------------------------------------------------
# 3. Retrieval: log-rank model of every gold chunk
# ---------------------------------------------------------------------------
# A gold chunk j of question q under configuration c gets a latent
#     L_qcj = m_c,type(q) + T_c(Z_qcj),   T_c(z) = s_lo*(z - z0) (z < z0), s_hi*(z - z0) (z >= z0)
#     Z_qcj = sqrt(rho_c) * zq_qc + sqrt(1 - rho_c) * ze_qcj   (question / chunk-level parts)
# and the rank ceil(exp(L)) (at least 1). Ranks of the gold chunks of one
# question are made distinct, ranks beyond DEPTH count as not retrieved.
# Z_qcj is a standard-normal combination of shared components (question
# difficulty, lexical / dense family components, chunk-level components), so
# configurations are correlated per question as in a real paired evaluation.
IDCG = np.array([np.nan] + [sum(1.0 / np.log2(i + 2) for i in range(k)) for k in (1, 2, 3)])


def raw_ranks(L, mask):
    raw = np.maximum(1.0, np.ceil(np.exp(np.minimum(L, 9.0))))
    return np.where(mask, raw, np.inf)


def resolve_ranks(raw):
    r = np.sort(raw, axis=1)
    for j in (1, 2):
        r[:, j] = np.where(np.isfinite(r[:, j]), np.maximum(r[:, j], r[:, j - 1] + 1), np.inf)
    return np.where(r > DEPTH, np.inf, r)


def per_question_metrics(r, ng):
    return {"R5": (r <= 5).sum(1) / ng,
            "R10": (r <= 10).sum(1) / ng,
            "MRR": 1.0 / r[:, 0],
            "P1": (r[:, 0] == 1).astype(float),
            "nDCG10": np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0).sum(1) / IDCG[ng]}


class RetrCase:
    """Gold-chunk latents of a set of questions for one retrieval configuration.
    The latent is mapped to log-rank by a monotone two-piece linear transform with
    slope s_lo below the break point z0 and s_hi above it."""

    def __init__(self, Z, ng, tidx, post=None, z0=0.0):
        self.Z, self.ng, self.tidx, self.post, self.z0 = Z, ng, tidx, post, z0
        self.mask = np.arange(3)[None, :] < ng[:, None]
        self.neg = Z < z0
        self.Zc = Z - z0

    def subset(self, sel):
        return RetrCase(self.Z[sel], self.ng[sel], self.tidx[sel], z0=self.z0)

    def raw(self, mq, slo, shi, add=0.0):
        L = mq[:, None] + np.where(self.neg, slo, shi) * self.Zc + add
        raw = raw_ranks(L, self.mask)
        return self.post(raw) if self.post is not None else raw

    def ranks(self, mq, slo, shi, add=0.0):
        return resolve_ranks(self.raw(mq, slo, shi, add))

    def r10(self, mq, slo, shi):
        r = self.ranks(mq, slo, shi)
        return 100 * ((r <= 10).sum(1) / self.ng).mean()

    def means(self, mq, slo, shi):
        m = per_question_metrics(self.ranks(mq, slo, shi), self.ng)
        return np.array([100 * m[k].mean() for k in RET_METRICS])


def calibrate_retriever(case_fn, target, delta=None, type_target=None, rho_grid=(0.9,),
                        z0_grid=(-0.6, -0.3, 0.3, 0.6, 0.9), tol=0.9):
    """Find (within-question correlation rho, m per type, s_lo, s_hi). m by bisection so
    that Recall@10 equals the assumed value (per question type if type targets are
    given); rho and (s_lo, s_hi) by grid search so that R@5, MRR, P@1 and nDCG@10 are
    as close as possible to the assumed values."""
    target = np.asarray(target, float)
    others = [0, 2, 3, 4]
    best = None
    combos = [(rho, 0.0) for rho in rho_grid]
    k = 0
    while k < len(combos):
        rho, top = combos[k]                          # top = break point z0
        k += 1
        case = case_fn(rho, top)
        subs = [case.subset(case.tidx == t) for t in range(4)] if type_target is not None else None

        def solve_m(slo, shi):
            if type_target is None:
                x = bisect(lambda m: case.r10(m + delta[case.tidx], slo, shi), target[1], -4, 7,
                           increasing=False, iters=22)
                return x + delta
            mt = np.zeros(4)
            for t in range(4):
                s = subs[t]
                mt[t] = bisect(lambda m: s.r10(np.full(s.ng.size, m), slo, shi), type_target[t], -4, 7,
                               increasing=False, iters=22)
            return mt

        def score(slo, shi):
            mt = solve_m(slo, shi)
            e = case.means(mt[case.tidx], slo, shi) - target
            return np.abs(e[others]).max() + 1e-3 * np.sum(e ** 2), mt

        b_rho = None
        for step, span in ((None, None), (0.03, 0.15), (0.006, 0.03)):
            if step is None:
                grid_lo, grid_hi = np.arange(0.6, 3.61, 0.15), np.arange(1.0, 7.01, 0.3)
            else:
                grid_lo = np.arange(b_rho[1] - span, b_rho[1] + span + 1e-9, step)
                grid_hi = np.arange(b_rho[2] - span, b_rho[2] + span + 1e-9, step)
            for slo in grid_lo:
                for shi in grid_hi:
                    if slo <= 0.05 or shi <= 0.05:
                        continue
                    sc, mt = score(slo, shi)
                    if b_rho is None or sc < b_rho[0]:
                        b_rho = (sc, slo, shi, mt)
        if best is None or b_rho[0] < best[0]:
            best = b_rho + (rho, top)
        if k == len(rho_grid) and best[0] > tol:      # move the break point only if the fit is poor
            combos += [(r, t) for r in rho_grid for t in z0_grid]
    _, slo, shi, mt, rho, top = best
    return {"mt": mt, "s_lo": float(slo), "s_hi": float(shi), "rho": float(rho), "z0": float(top)}


# A latent is kept as a pair (question-level part zq [n], chunk-level part ze [n x 3]);
# the gold-chunk latent is Z = sqrt(rho) * zq + sqrt(1 - rho) * ze, where rho is the
# within-question correlation of the gold chunks (calibrated per configuration).
def mixZ(pair, rho, idx=None):
    zq, ze = pair
    if idx is not None:
        zq, ze = zq[idx], ze[idx]
    return np.sqrt(rho) * zq[:, None] + np.sqrt(1 - rho) * ze


def combine(pairs, weights, corr_q, corr_e):
    """Weighted sum of latent pairs, rescaled to unit variance per part.
    corr_q / corr_e: correlation of the first two parts (other parts independent)."""
    out = []
    for part, cor in ((0, corr_q), (1, corr_e)):
        s = sum(w * p[part] for w, p in zip(weights, pairs))
        var = sum(w ** 2 for w in weights) + 2 * weights[0] * weights[1] * cor
        out.append(s / np.sqrt(var))
    return tuple(out)


def build_retrieval_latents(rng, q, cfg):
    rc = cfg["retrieval"]
    nq = len(q)
    names_q = ["lex", "den", "tfidf", "bm25", "labse", "e5", "bge", "H", "rrf", "rr",
               "sentence", "fixed", "sliding"]
    names_g = ["shared", "tfidf", "bm25", "labse", "e5", "bge", "H", "rrf", "rr",
               "sentence", "fixed", "sliding", "abl_translit", "abl_boilerplate", "abl_merging"]
    Qd = {k: rng.standard_normal(nq) for k in names_q}
    Gd = {k: rng.standard_normal((nq, 3)) for k in names_g}
    aff_u = {v: rng.random(nq) for v in ("translit", "boilerplate", "merging")}
    D = q["difficulty"].to_numpy()
    wd, cs = rc["w_difficulty"], rc["chunk_shared"]
    Zp = {}
    for c, (fam, wx) in rc["families"].items():
        zq = np.sqrt(wd) * D + np.sqrt(wx) * Qd[fam] + np.sqrt(1 - wd - wx) * Qd[c]
        ze = np.sqrt(cs) * Gd["shared"] + np.sqrt(1 - cs) * Gd[c]
        Zp[c] = (zq, ze)

    def own(name, wdif):
        return (np.sqrt(wdif) * D + np.sqrt(1 - wdif) * Qd[name], Gd[name])

    Zp["H"] = own("H", 0.0)

    def hyb(a):
        """Linear hybrid with BM25 weight a (0 = BGE-M3 only, 1 = BM25 only)."""
        h = rc["hybrid_own_h0"] * 4 * a * (1 - a)
        return combine([Zp["bm25"], Zp["bge"], Zp["H"]], [a, 1 - a, h], wd, cs)

    def mix_pair(p, o, r):
        return (r * p[0] + np.sqrt(1 - r ** 2) * o[0], r * p[1] + np.sqrt(1 - r ** 2) * o[1])

    Zp["lin"] = hyb(rc["selected_alpha_expected"])
    Zp["sa"] = hyb(rc["neutral_alpha_rq1"])       # RQ1: hybrid with alpha = 0.5 on structure-aware chunks
    Zp["rrf"] = mix_pair(Zp["sa"], own("rrf", 0.0), rc["rho_rrf"])
    Zp["rr"] = mix_pair(Zp["lin"], own("rr", 0.0), rc["rho_rr"])
    for ch, r in rc["rho_chunking"].items():
        Zp[ch] = mix_pair(Zp["sa"], own(ch, wd), r)
    return Zp, hyb, Gd, aff_u


def alpha_params(a, P, gamma):
    """Effect parameters of the linear hybrid at BM25 weight a, interpolated between
    the calibrated anchors 0 (BGE-M3), 0.4 (selected), 0.5 (RQ1) and 1 (BM25)."""
    def mix(p, q_, w):
        return {k: p[k] + (q_[k] - p[k]) * w for k in ("mt", "s_lo", "s_hi", "rho", "z0")}
    if a <= 0.4:
        return mix(P["lin"], P["bge"], ((0.4 - a) / 0.4) ** gamma)
    if a <= 0.5:
        return mix(P["lin"], P["sa"], (a - 0.4) / 0.1)
    return mix(P["sa"], P["bm25"], ((a - 0.5) / 0.5) ** gamma)


def simulate_retrieval(Zp, hyb, Gd, aff_u, q, cfg):
    rc = cfg["retrieval"]
    T = rc["targets"]
    rho_grid = tuple(rc["rho_grid"])
    idx_t = np.flatnonzero(((q["split"] == "test") & (q["n_gold"] > 0)).to_numpy())
    idx_d = np.flatnonzero(((q["split"] == "dev") & (q["n_gold"] > 0)).to_numpy())
    ng_t, tt = q["n_gold"].to_numpy()[idx_t], q["type_idx"].to_numpy()[idx_t]
    ng_d, td = q["n_gold"].to_numpy()[idx_d], q["type_idx"].to_numpy()[idx_d]
    w_t = np.bincount(tt, minlength=4) / tt.size

    def case_fn(c, post=None):
        return lambda rho, z0=0.0: RetrCase(mixZ(Zp[c], rho, idx_t), ng_t, tt, post, z0)

    P = {}
    for c in ("bm25", "lin"):                      # per-type assumed Recall@10 (Table 4, Panel C)
        P[c] = calibrate_retriever(case_fn(c), T[c], type_target=rc["type_targets_R10"][c], rho_grid=rho_grid)
    d_lin = P["lin"]["mt"] - w_t @ P["lin"]["mt"]
    d_bm25 = P["bm25"]["mt"] - w_t @ P["bm25"]["mt"]
    d_avg = 0.5 * (d_lin + d_bm25)
    for c in ("tfidf", "labse", "e5", "bge"):
        P[c] = calibrate_retriever(case_fn(c), T[c], delta=d_avg, rho_grid=rho_grid)
    for c in ("sa", "rrf", "sentence", "fixed", "sliding"):
        P[c] = calibrate_retriever(case_fn(c), T[c], delta=d_lin, rho_grid=rho_grid)
    # reranker: re-scores the top-50 candidates of the linear hybrid only
    raw_lin = case_fn("lin")(P["lin"]["rho"], P["lin"]["z0"]).raw(P["lin"]["mt"][tt], P["lin"]["s_lo"], P["lin"]["s_hi"])
    K = rc["reranker_candidates"]

    def rr_post(raw):
        return np.where(raw_lin <= K, np.minimum(raw, K), raw_lin)
    P["rr"] = calibrate_retriever(case_fn("rr", rr_post), T["rr"], delta=d_lin, rho_grid=rho_grid)

    ranks, pq, cases = {}, {}, {}
    for c in T:
        cases[c] = case_fn(c, rr_post if c == "rr" else None)(P[c]["rho"], P[c]["z0"])
        ranks[c] = cases[c].ranks(P[c]["mt"][tt], P[c]["s_lo"], P[c]["s_hi"])
        pq[c] = per_question_metrics(ranks[c], ng_t)

    # cleaning ablation (Table 4, Panel A): affected questions get a worse latent
    abl = {}
    sig = rc["ablation_sigma"]
    for c in ("bm25", "lin"):
        cs_ = cases[c]
        mq = P[c]["mt"][tt]
        for v, p_aff in rc["ablation_affected_prob"].items():
            aff = (aff_u[v][idx_t] < p_aff)[:, None]
            G = Gd["abl_" + v][idx_t]

            def r10(mu):
                r = cs_.ranks(mq, P[c]["s_lo"], P[c]["s_hi"], add=aff * (mu + sig * G))
                return 100 * ((r <= 10).sum(1) / ng_t).mean()
            mu = bisect(r10, rc["ablation_targets_R10"][c][v], -1.0, 4.0, increasing=False, iters=40)
            r = cs_.ranks(mq, P[c]["s_lo"], P[c]["s_hi"], add=aff * (mu + sig * G))
            abl[(c, v)] = {"mu": float(mu), "affected_share": float(aff.mean()),
                           "pq": per_question_metrics(r, ng_t)}

    # alpha sensitivity on the development set (Fig. 1)
    gamma = rc["alpha_curve_gamma"]
    dev = {}
    for a in rc["alpha_grid"]:
        pa = alpha_params(a, P, gamma)
        cd = RetrCase(mixZ(hyb(a), pa["rho"], idx_d), ng_d, td, z0=pa["z0"])
        dev[a] = per_question_metrics(cd.ranks(pa["mt"][td], pa["s_lo"], pa["s_hi"]), ng_d)
    cd = RetrCase(mixZ(Zp["rrf"], P["rrf"]["rho"], idx_d), ng_d, td, z0=P["rrf"]["z0"])
    dev["rrf"] = per_question_metrics(cd.ranks(P["rrf"]["mt"][td], P["rrf"]["s_lo"], P["rrf"]["s_hi"]), ng_d)

    # hits of the linear hybrid used as generator context (k = 1, 3, 5, 10)
    hits = {k: (ranks["lin"] <= k).any(axis=1) for k in (1, 3, 5, 10)}
    return {"idx_test": idx_t, "idx_dev": idx_d, "params": P, "ranks": ranks, "pq": pq,
            "ablation": abl, "dev": dev, "hits": hits, "type_test": tt, "type_dev": td,
            "delta_lin": d_lin, "delta_avg": d_avg}


def simulate_latency(rng, n, cfg):
    rc = cfg["retrieval"]
    L, s = rc["latency_ms"], rc["latency_sigma"]

    def ln(mean, sig):
        return mean * np.exp(sig * rng.standard_normal(n) - sig ** 2 / 2)
    lat = {}
    for c in ("tfidf", "bm25", "labse", "e5", "bge"):
        lat[c] = ln(L[c], s)
    lat["lin"] = lat["bm25"] + lat["bge"] + ln(L["fusion"], s)           # both retrievers + fusion
    lat["rrf"] = lat["bm25"] + lat["bge"] + ln(L["fusion"], s)
    lat["rr"] = lat["lin"] + ln(L["reranker_extra"], rc["reranker_latency_sigma"])
    for c in ("sentence", "fixed", "sliding", "sa"):                      # separate timing run for RQ1
        lat[c] = ln(L[c], s)
    return lat


# ---------------------------------------------------------------------------
# 4. Answer generation: latent answer-quality model
# ---------------------------------------------------------------------------
# For model m, setting s (LLM-only / RAG) and variant v (prompt language, k):
#   abstain  ~ Bernoulli(sigmoid(a0 + a_miss * [gold chunk not in context]))      (RAG)
#   eta      = theta_m,v + delta_m,type + beta_m * [gold chunk in context] - noise
#   noise    = sqrt(.3) G_q + sqrt(.2) U_qm + sqrt(.3) V_qms + sqrt(.2) W_qmv
#              (G: question difficulty, U: model, V: model x setting, W: variant; the
#               shared parts make paired comparisons correlated as in a real evaluation)
#   legal correctness = 1 if eta > 0, 0.5 if -tau_m < eta <= 0, else 0; 0 for abstentions
#   hallucinated ~ Bernoulli(sigmoid(h + k0 [corr = 0] + k05 [corr = 0.5]))   (not for abstentions)
#   supported    ~ Bernoulli(sigmoid(s + k_s corr)) among non-hallucinated answers
#   citation accuracy = share of correct cited provisions (0 if no citation)
#   EM only for fully correct answers; token F1 logit-normal around the correctness level
def draw_generation(rng, n):
    g = {"G_noise": rng.standard_normal(n), "hit_u": rng.random(n)}
    for m in MODELS:
        d = {"U": rng.standard_normal(n), "Uem": rng.random(n), "Zf_shared": rng.standard_normal(n)}
        for s in ("llm", "rag"):
            d["V_" + s] = rng.standard_normal(n)
            d["Ua_" + s] = rng.random(n)
            d["Uh_" + s] = rng.random(n)
            d["Us_" + s] = rng.random(n)
            d["Uc0_" + s] = rng.random(n)
            d["Un_" + s] = rng.random(n)
            d["Up_" + s] = rng.random((n, 3))
            d["Zf_" + s] = rng.standard_normal(n)
        for v in ["llm"] + RAG_VARIANTS:
            d["W_" + v] = rng.standard_normal(n)
            d["Bh_" + v] = rng.random(n)
            d["Uh2_" + v] = rng.random(n)
        for sd in RUN_SEEDS:
            d[f"Bseed_{sd}"] = rng.random(n)
            d[f"Wseed_{sd}"] = rng.standard_normal(n)
        g[m] = d
    return g


def simulate_generation(rng, q, ret, cfg):
    gc = cfg["generation"]
    test = (q["split"] == "test").to_numpy()
    idx = np.flatnonzero(test)
    n = idx.size
    tt = q["type_idx"].to_numpy()[idx]
    has_gold = q["n_gold"].to_numpy()[idx] > 0
    D = q["difficulty"].to_numpy()[idx]
    w_t = np.bincount(tt, minlength=4) / n
    g = draw_generation(rng, n)

    # context hits: gold questions from the simulated hybrid ranks; questions without
    # linked gold get a context-quality draw at the observed hit rate (never E1)
    pos_gold = np.searchsorted(idx, ret["idx_test"])
    hits = {}
    for k, h in ret["hits"].items():
        hk = np.zeros(n, dtype=bool)
        hk[pos_gold] = h
        hk[~has_gold] = g["hit_u"][~has_gold] < h.mean()
        hits[k] = hk
    miss5 = has_gold & ~hits[5]

    vc = gc["var_components"]
    rd = gc["rho_difficulty"]
    Gq = rd * D + np.sqrt(1 - rd ** 2) * g["G_noise"]
    k0, k05 = gc["hal_logit_corr0"], gc["hal_logit_corr05"]

    def corr_of(theta_q, tau, beta, hit, nz, ab):
        eta = theta_q + beta * hit - nz
        c = np.where(eta > 0, 1.0, np.where(eta > -tau, 0.5, 0.0))
        return np.where(ab, 0.0, c)

    out, params = {}, {}
    order = ["gpt", "qwen", "kaz", "llama"]           # type effects of kaz/llama use gpt/qwen
    deltas = {}
    for m in order:
        d = g[m]
        P = {}
        o = {}

        def noise(setting, v, W=None):
            W = d["W_" + v] if W is None else W
            return (np.sqrt(vc["G"]) * Gq + np.sqrt(vc["U"]) * d["U"]
                    + np.sqrt(vc["V"]) * d["V_" + setting] + np.sqrt(vc["W"]) * W)

        def abstain(a0, setting, hit):
            if setting == "llm":
                p = np.full(n, sigmoid(a0))
            else:
                p = sigmoid(a0 + gc["abstain_logit_miss"][m] * (~hit))
            return d["Ua_" + setting] < p

        a0 = {}
        for s in ("llm", "rag"):
            a0[s] = bisect(lambda a: abstain(a, s, hits[5]).mean(), gc["abstain_rate"][m][s], -14, 4)
        P["a0"] = a0

        # --- main RAG configuration (Kazakh prompt, k = 5): theta per type, tau, beta
        tg = gc["targets"][m]
        nz5 = noise("rag", "kk5")
        ab5 = abstain(a0["rag"], "rag", hits[5])
        typed = m in gc["type_targets_cor"]
        if typed:
            tgt_t = np.array(gc["type_targets_cor"][m]) / 100
        else:
            delta_m = 0.5 * (deltas["gpt"] + deltas["qwen"])

        def solve_theta(tau, beta):
            if typed:
                th = np.zeros(4)
                for t in range(4):
                    s_ = tt == t
                    th[t] = bisect(lambda x: corr_of(x, tau, beta, hits[5][s_], nz5[s_], ab5[s_]).mean(),
                                   tgt_t[t], -8, 8, iters=30)
                return th
            x = bisect(lambda x: corr_of(x + delta_m[tt], tau, beta, hits[5], nz5, ab5).mean(),
                       tg["rag"][5] / 100, -8, 8, iters=30)
            return x + delta_m

        def full_share(th, tau, beta, sel=None):
            c = corr_of(th[tt], tau, beta, hits[5], nz5, ab5)
            return np.mean(c[sel] == 1.0) if sel is not None else np.mean(c == 1.0)

        tau, beta = 1.0, 1.0
        for _ in range(8):
            beta = bisect(lambda b: full_share(solve_theta(tau, b), tau, b, miss5),
                          gc["full_correct_share_given_miss"][m], 0.0, 6.0, increasing=False, iters=22)
            tau = bisect(lambda u: full_share(solve_theta(u, beta), u, beta),
                         gc["full_correct_share_rag"][m], 0.02, 6.0, increasing=False, iters=22)
        th = solve_theta(tau, beta)
        deltas[m] = th - w_t @ th
        P.update(tau=float(tau), beta=float(beta), theta_type_kk5=th.tolist(), delta_type=deltas[m].tolist())

        def run_variant(setting, v, theta, hit, W=None):
            ab = abstain(a0[setting], setting, hit)
            nz = noise(setting, v, W)
            return ab, corr_of(theta + deltas[m][tt], tau, beta if setting == "rag" else 0.0, hit, nz, ab)

        def calib_theta(setting, v, hit, target):
            return bisect(lambda x: run_variant(setting, v, x, hit)[1].mean(), target / 100, -8, 8, iters=40)

        thetas = {"kk5": float(w_t @ th)}
        thetas["llm"] = calib_theta("llm", "llm", np.zeros(n, bool), tg["llm"][5])
        thetas["ru5"] = calib_theta("rag", "ru5", hits[5], gc["ru_targets"][m][0])
        if m in gc["topk_targets"]:
            for k in (1, 3, 10):
                thetas[f"kk{k}"] = calib_theta("rag", f"kk{k}", hits[k], gc["topk_targets"][m][str(k)][0])
        P["theta"] = thetas

        def hal_uniform(v):
            if v == "llm":
                return d["Uh_llm"]
            return np.where(d["Bh_" + v] < gc["psi_hal_redraw"], d["Uh2_" + v], d["Uh_rag"])

        def hal_of(h, corr, ab, U):
            return (U < sigmoid(h + k0 * (corr == 0.0) + k05 * (corr == 0.5))) & ~ab

        hal_targets = {"llm": tg["llm"][4], "kk5": tg["rag"][4], "ru5": gc["ru_targets"][m][1]}
        if m in gc["topk_targets"]:
            for k in (1, 3, 10):
                hal_targets[f"kk{k}"] = gc["topk_targets"][m][str(k)][1]
        P["h"] = {}
        for v in thetas:
            setting = "llm" if v == "llm" else "rag"
            k = 5 if v in ("llm", "kk5", "ru5") else int(v[2:])
            hit = np.zeros(n, bool) if setting == "llm" else hits[k]
            ab, corr = run_variant(setting, v, thetas[v], hit)
            U = hal_uniform(v)
            h = bisect(lambda x: hal_of(x, corr, ab, U).mean(), hal_targets[v] / 100, -12, 8)
            P["h"][v] = h
            o[v] = {"abst": ab, "cor": corr, "hal": hal_of(h, corr, ab, U), "hit": hit}

        # --- support, citation accuracy, EM and F1 for the two main settings
        P["s"], P["c"], P["e"], P["f"] = {}, {}, {}, {}
        for v, setting, ti in (("llm", "llm", 0), ("kk5", "rag", 1)):
            ab, corr, hal, hit = o[v]["abst"], o[v]["cor"], o[v]["hal"], o[v]["hit"]
            Us = d["Us_" + setting]

            def sup_of(x):
                return (Us < sigmoid(x + gc["sup_logit_per_corr"] * corr)) & ~ab & ~hal
            sx = bisect(lambda x: sup_of(x).mean(), tg[setting][3] / 100, -12, 12)
            pi0 = gc["no_citation_prob"][m][setting]
            cites = (d["Uc0_" + setting] >= pi0) & ~ab
            p1, p2, _ = gc["cit_count_probs"]
            ncit = 1 + (d["Un_" + setting] > p1).astype(int) + (d["Un_" + setting] > p1 + p2).astype(int)

            def cit_of(x):
                p = sigmoid(x + gc["cit_logit_per_corr"] * corr + gc["cit_logit_hit"] * hit)
                ok = (d["Up_" + setting] < p[:, None]) & (np.arange(3)[None, :] < ncit[:, None])
                return np.where(cites, ok.sum(1) / ncit, 0.0)
            cx = bisect(lambda x: cit_of(x).mean(), tg[setting][2] / 100, -12, 12)

            def em_of(x):
                return (corr == 1.0) & ~ab & (d["Uem"] < x)
            ex = bisect(lambda x: em_of(x).mean(), tg[setting][0] / 100, 0.0, 1.0)
            em = em_of(ex)
            Zf = np.sqrt(0.5) * d["Zf_shared"] + np.sqrt(0.5) * d["Zf_" + setting]

            def f1_of(x):
                f = sigmoid(x + gc["f1_slope_corr"] * corr + gc["f1_sigma"] * Zf)
                f = np.where(em, 1.0, f)
                return np.where(ab, sigmoid(gc["f1_abstain_logit"] + 0.4 * Zf), f)
            fx = bisect(lambda x: f1_of(x).mean(), tg[setting][1] / 100, -12, 12)
            P["s"][v], P["c"][v], P["e"][v], P["f"][v] = sx, cx, ex, fx
            o[v].update(sup=sup_of(sx), cit=cit_of(cx), em=em.astype(float), f1=f1_of(fx))
            o[v]["unver"] = ~o[v]["abst"] & ~o[v]["hal"] & ~o[v]["sup"]

        # --- run-to-run variation (generation seeds 13 and 42; 2026 is the main run)
        runs = {2026: float(100 * o["kk5"]["cor"].mean())}
        f = gc["run_redraw_share"][m]
        for sd in RUN_SEEDS:
            W = np.where(d[f"Bseed_{sd}"] < f, d[f"Wseed_{sd}"], d["W_kk5"])
            _, corr = run_variant("rag", "kk5", thetas["kk5"], hits[5], W)
            runs[sd] = float(100 * corr.mean())
        P["runs"] = runs
        out[m], params[m] = o, P
    return {"idx": idx, "type": tt, "has_gold": has_gold, "hits": hits, "miss5": miss5,
            "out": out, "params": params}


# ---------------------------------------------------------------------------
# 5. Error taxonomy (Fig. 3) and judge agreement
# ---------------------------------------------------------------------------
def simulate_errors(rng, gen, cfg):
    probs = cfg["generation"]["error_probs_E2_E7"]
    res = {}
    U = {m: rng.random(gen["idx"].size) for m in MODELS}
    for m in MODELS:
        o = gen["out"][m]["kk5"]
        failed = o["cor"] < 1.0
        e1 = failed & gen["miss5"]
        e8 = failed & ~e1 & o["abst"]
        p = np.array(probs[m], float)
        cum = np.cumsum(p / p.sum())
        other = np.searchsorted(cum, U[m], side="right")              # 0..5 -> E2..E7
        cat = np.full(failed.size, "", dtype=object)
        cat[failed] = np.array(["E2", "E3", "E4", "E5", "E6", "E7"])[np.minimum(other, 5)][failed]
        cat[e1] = "E1"
        cat[e8] = "E8"
        res[m] = {"failed": int(failed.sum()), "counts": {c: int((cat == c).sum()) for c in ERR_CATS},
                  "category": cat}
    return res


def cohen_kappa_counts(M, k, weighted):
    """Cohen's kappa from (B, k*k) confusion counts; linear weights if weighted."""
    M = M.reshape(-1, k, k)
    M = M / M.sum(axis=(1, 2), keepdims=True)
    i, j = np.indices((k, k))
    Wt = 1 - np.abs(i - j) / (k - 1) if weighted else (i == j).astype(float)
    po = (M * Wt).sum(axis=(1, 2))
    pe = (np.einsum("bi,bj->bij", M.sum(2), M.sum(1)) * Wt).sum(axis=(1, 2))
    return (po - pe) / (1 - pe)


def kappa(a, b, k, weighted, C=None):
    codes = (a * k + b).astype(int)
    onehot = np.eye(k * k)[codes]
    if C is None:
        return float(cohen_kappa_counts(onehot.sum(0)[None, :], k, weighted)[0])
    return cohen_kappa_counts(C @ onehot, k, weighted)


def simulate_judge(rng, gen, cfg, boot_counts):
    jc = cfg["judge"]
    n = gen["idx"].size
    pick = np.concatenate([rng.choice(n, jc["per_model"], replace=False) for _ in MODELS])
    hal = np.concatenate([gen["out"][m]["kk5"]["hal"][pick[i * jc["per_model"]:(i + 1) * jc["per_model"]]]
                          for i, m in enumerate(MODELS)]).astype(int)
    cor = np.concatenate([gen["out"][m]["kk5"]["cor"][pick[i * jc["per_model"]:(i + 1) * jc["per_model"]]]
                          for i, m in enumerate(MODELS)])
    cor3 = np.round(cor * 2).astype(int)                 # 0, 1, 2 for 0, 0.5, 1
    U = {k: rng.random(hal.size) for k in ("jh", "jc", "jd", "ah", "ac", "ad", "bh", "bc", "bd")}

    def flip_bin(x, u, p):
        return np.where(u < p, 1 - x, x)

    def flip_ord(x, u, ud, p):
        step = np.where(x == 0, 1, np.where(x == 2, -1, np.where(ud < 0.5, -1, 1)))
        return np.where(u < p, x + step, x)

    p_jh = bisect(lambda p: kappa(hal, flip_bin(hal, U["jh"], p), 2, False), jc["kappa_hal_target"],
                  0.0, 0.5, increasing=False)
    p_jc = bisect(lambda p: kappa(cor3, flip_ord(cor3, U["jc"], U["jd"], p), 3, True), jc["kappa_cor_target"],
                  0.0, 0.8, increasing=False)

    def annot_kappa(p):
        a = flip_ord(cor3, U["ac"], U["ad"], p)
        b = flip_ord(cor3, U["bc"], U["bd"], p)
        return kappa(a, b, 3, True)
    p_a = bisect(annot_kappa, jc["kappa_annot_target"], 0.0, 0.5, increasing=False)
    p_ah = bisect(lambda p: kappa(flip_bin(hal, U["ah"], p), flip_bin(hal, U["bh"], p), 2, False),
                  jc["kappa_annot_target"], 0.0, 0.5, increasing=False)
    j_hal, j_cor = flip_bin(hal, U["jh"], p_jh), flip_ord(cor3, U["jc"], U["jd"], p_jc)
    a_cor, b_cor = flip_ord(cor3, U["ac"], U["ad"], p_a), flip_ord(cor3, U["bc"], U["bd"], p_a)
    a_hal, b_hal = flip_bin(hal, U["ah"], p_ah), flip_bin(hal, U["bh"], p_ah)
    C = boot_counts
    res = {}
    for name, (x, y, k, w) in {"judge_vs_human_hallucination": (j_hal, hal, 2, False),
                               "judge_vs_human_legal_correctness_weighted": (j_cor, cor3, 3, True),
                               "annotator_A_vs_B_legal_correctness_weighted": (a_cor, b_cor, 3, True),
                               "annotator_A_vs_B_hallucination": (a_hal, b_hal, 2, False)}.items():
        bk = kappa(x, y, k, w, C)
        res[name] = {"kappa": kappa(x, y, k, w), "ci_lo": float(np.nanpercentile(bk, 2.5)),
                     "ci_hi": float(np.nanpercentile(bk, 97.5)), "n": int(x.size)}
    res["_flip_probs"] = {"judge_hal": p_jh, "judge_cor": p_jc, "annotators_cor": p_a, "annotators_hal": p_ah}
    return res


# ---------------------------------------------------------------------------
# 6. Planned statistical analysis
# ---------------------------------------------------------------------------
class Stats:
    """Bootstrap count matrices and sign-flip matrices, created once per sample
    in a fixed order (so results do not depend on the order of later calls)."""

    def __init__(self, rng_boot, rng_perm, boot_sizes, perm_sizes):
        self.C, self.S, self.n = {}, {}, {}
        for key, n in boot_sizes:
            idx = rng_boot.integers(0, n, size=(N_BOOT, n))
            off = np.arange(N_BOOT)[:, None] * n
            self.C[key] = np.bincount((idx + off).ravel(), minlength=N_BOOT * n).reshape(N_BOOT, n).astype(float)
            self.n[key] = n
        for key, n in perm_sizes:
            self.S[key] = rng_perm.choice(np.array([-1.0, 1.0]), size=(N_PERM, n))

    def ci(self, key, v, scale=100.0):
        v = np.asarray(v, float)
        bm = self.C[key] @ v / v.size
        lo, hi = np.percentile(bm, [2.5, 97.5])
        return scale * v.mean(), scale * lo, scale * hi

    def ar_test(self, key, d):
        """Two-sided paired approximate randomization test (random sign flips)."""
        d = np.asarray(d, float)
        if np.all(d == 0):
            return 1.0
        obs = abs(d.sum())
        perm = np.abs(self.S[key] @ d)
        return float((1 + np.sum(perm >= obs - 1e-9)) / (N_PERM + 1))


def mcnemar_exact(x, y):
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    b, c = int(np.sum(x & ~y)), int(np.sum(~x & y))
    if b + c == 0:
        return 1.0, b, c
    return float(stats.binomtest(b, b + c, 0.5).pvalue), b, c


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, (p.size - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# ---------------------------------------------------------------------------
# 7. Formatting helpers (decimal convention of the specification)
# ---------------------------------------------------------------------------
def f1d(x):
    s = f"{x:.1f}"
    return "0.0" if s in ("-0.0",) else s


def sgn(x):
    s = f1d(x)
    return s if s.startswith("-") or s == "0.0" else "+" + s


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def tex(s):
    """Typeset minus signs for LaTeX."""
    return s.replace("-", "$-$") if isinstance(s, str) else s


def ci_str(v, lo, hi):
    return f"{f1d(v)} [{f1d(lo)}, {f1d(hi)}]"


def rnd(x, k=6):
    if isinstance(x, dict):
        return {str(a): rnd(b, k) for a, b in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(b, k) for b in x]
    if isinstance(x, (np.floating, float)):
        return round(float(x), k)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return rnd(x.tolist(), k)
    return x


# ---------------------------------------------------------------------------
# 8. Figures (vector PDF, Okabe-Ito palette, no in-plot titles)
# ---------------------------------------------------------------------------
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "purple": "#CC79A7",
         "sky": "#56B4E9", "vermillion": "#D55E00", "yellow": "#F0E442", "grey": "#999999"}


def set_style():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
        "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "hatch.linewidth": 0.5, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def sim_label(ax, x=0.99, y=0.02, ha="right", va="bottom"):
    ax.text(x, y, "expected (simulation)", transform=ax.transAxes, ha=ha, va=va,
            fontsize=7, color="0.5", style="italic")


def save(fig, figdir, names):
    for nm in names:
        fig.savefig(figdir / f"{nm}.pdf", format="pdf", metadata={"CreationDate": None})
    fig.savefig(figdir / "preview" / f"{names[0]}.png", dpi=220)
    plt.close(fig)


def fig_alpha(fig1, figdir):
    """Fig. 1: Recall@10 and MRR of the linear hybrid vs. BM25 weight alpha (dev set)."""
    a = fig1["alpha"].to_numpy()
    fig, ax = plt.subplots(figsize=(4.9, 2.3), layout="constrained")
    for met, col, mk, lab in (("R10", OKABE["blue"], "o", "Recall@10"), ("MRR", OKABE["vermillion"], "s", "MRR ×100")):
        ax.fill_between(a, fig1[f"{met}_lo"], fig1[f"{met}_hi"], color=col, alpha=0.16, lw=0)
        ax.plot(a, fig1[met], color=col, marker=mk, ms=3.6, lw=1.4, label=f"{lab} (linear)")
        ax.axhline(fig1.attrs[f"rrf_{met}"], color=col, ls="--", lw=1.0, label=f"{lab} (RRF)")
    ax.axvline(fig1.attrs["selected_alpha"], color="0.3", ls=":", lw=1.1,
               label=f"selected α = {fig1.attrs['selected_alpha']:.1f}")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(np.round(np.arange(0, 1.01, 0.1), 1))
    ax.set_ylim(30, 92)
    ax.set_xlabel("BM25 weight α in the linear fusion (–)")
    ax.set_ylabel("Score on the dev set (%)")
    h, l = ax.get_legend_handles_labels()
    order = [0, 2, 1, 3, 4]
    fig.legend([h[i] for i in order], [l[i] for i in order], ncol=3, loc="outside upper center",
               frameon=False, columnspacing=1.0, handlelength=2.0)
    sim_label(ax, 0.01, 0.03, ha="left")
    save(fig, figdir, ["fig1_alpha", "fig_alpha"])


def fig_prompt(fig2, figdir):
    """Fig. 2: legal correctness and hallucination, Kazakh vs. Russian system prompt (RAG)."""
    labels = ["GPT-\n6.1", "KazLLM-\n1.0-8B", "Llama-\n3.1-8B", "Qwen3.5-\n9B"]
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 2.45), layout="constrained")
    x = np.arange(4)
    for ax, met, ylab in ((axes[0], "cor", "Legal correctness (%)"), (axes[1], "hal", "Hallucination rate (%)")):
        top = []
        for j, (lang, col, mk, lab) in enumerate((("kk", OKABE["blue"], "o", "Kazakh system prompt"),
                                                   ("ru", OKABE["orange"], "^", "Russian system prompt"))):
            sub = fig2[(fig2["metric"] == met) & (fig2["prompt"] == lang)].set_index("model").loc[MODELS]
            v, lo, hi = sub["value"].to_numpy(), sub["ci_lo"].to_numpy(), sub["ci_hi"].to_numpy()
            ax.errorbar(x + (j - 0.5) * 0.28, v, yerr=[v - lo, hi - v], fmt=mk, color=col, ms=4.2,
                        capsize=2, elinewidth=1.0, capthick=0.8, label=lab)
            top.append(hi)
        top = np.max(top, axis=0)
        sig = fig2[(fig2["metric"] == met) & (fig2["prompt"] == "kk")].set_index("model").loc[MODELS, "significant"]
        for i, s in enumerate(sig.to_numpy()):
            if s:
                ax.text(x[i], top[i] + 1.0, "*", ha="center", va="bottom", fontsize=11)
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylabel(ylab)
    axes[0].set_ylim(35, 88)
    axes[1].set_ylim(0, 40)
    sim_label(axes[1], 0.99, 0.98, va="top")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="outside upper center", frameon=False)
    fig.supxlabel("Generator model (RAG, k = 5)", fontsize=8.5)
    save(fig, figdir, ["fig2_prompt_language", "fig_prompt"])


def fig_errors(fig3, figdir):
    """Fig. 3: primary error category of every not fully correct answer (RAG, Kazakh prompt)."""
    cols = [OKABE[k] for k in ("vermillion", "orange", "purple", "blue", "sky", "green", "yellow", "grey")]
    hatches = ["", "////", "", "\\\\\\\\", "....", "", "xxx", "----"]
    fig, ax = plt.subplots(figsize=(4.9, 2.55), layout="constrained")
    y = np.arange(4)[::-1]
    left = np.zeros(4)
    tab = fig3.set_index("model").loc[MODELS]
    for c, col, ht in zip(ERR_CATS, cols, hatches):
        vals = tab[c].to_numpy(float)
        ax.barh(y, vals, left=left, color=col, edgecolor="0.2", linewidth=0.35, hatch=ht,
                height=0.64, label=ERR_LABELS[c])
        left += vals
    for yi, tot in zip(y, tab["failed"].to_numpy()):
        ax.text(tot + 4, yi, f"{tot}", va="center", fontsize=8)
    ax.set_yticks(y, [MODEL_NAMES[m] for m in MODELS])
    ax.set_xlim(0, 300)
    ax.set_xlabel("Failed answers (count)")
    fig.legend(ncol=2, loc="outside lower center", frameon=False, handlelength=1.6,
               handleheight=0.9, labelspacing=0.3, columnspacing=1.5)
    sim_label(ax, 0.99, 0.98, va="top")
    save(fig, figdir, ["fig3_error_taxonomy", "fig_errors"])


# ---------------------------------------------------------------------------
# 9. LaTeX tables (booktabs / sn-jnl rules); every table says values are expected
# ---------------------------------------------------------------------------
NOTE = "Expected values from a seeded simulation (not measured)."


def cell2(v, lo, hi, bold=False):
    a = tex(f1d(v))
    return (r"\textbf{" + a + "}" if bold else a), r"{\scriptsize[" + tex(f1d(lo)) + ", " + tex(f1d(hi)) + "]}"


def write_table2(t2, path):
    L = [r"% Table 2 (tab:main). Generated by simulate.py - " + NOTE,
         r"\begin{table}[t]", r"\caption{Main results on the test set. Panel A (RQ1): chunking strategies with the same hybrid retriever (BM25 + BGE-M3, $\alpha=0.5$, no reranker). Panel B (RQ2): retrievers on structure-aware chunks. Panel C (RQ3): generators without (LLM-only) and with retrieval (RAG, top 5 chunks from the linear hybrid), Kazakh system prompt. " + NOTE + r" Metrics in \%, MRR and nDCG@10 $\times 100$, 95\% bootstrap CIs (10{,}000 resamples) in brackets, latency in ms per query. Bold marks the best value in a panel column (highest; lowest for hallucination) and does not imply significance; tests are in Table~\ref{tab:tests}. $n=381$ (A, B) and $400$ (C).}\label{tab:main}",
         r"\footnotesize", r"\setlength{\tabcolsep}{3.2pt}", r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule"]
    for panel, title, cfgs in (("A", "Panel A: RQ1, chunking (hybrid, $\\alpha=0.5$)", ["sentence", "fixed", "sliding", "sa"]),
                               ("B", "Panel B: RQ2, retrievers (structure-aware chunks)",
                                ["tfidf", "bm25", "labse", "e5", "bge", "lin", "rrf", "rr"])):
        L.append(r"\multicolumn{8}{@{}l}{\textit{" + title + r"}}\\")
        L.append(r"\multicolumn{2}{@{}l}{Configuration} & R@5 & R@10 & MRR & P@1 & nDCG@10 & Lat.\ (ms)\\")
        L.append(r"\midrule")
        sub = t2[t2["panel"] == panel]
        best = {mt: sub[sub["metric"] == mt]["value"].round(1).max() for mt in RET_METRICS}
        for c in cfgs:
            r1, r2 = [], []
            for mt in RET_METRICS:
                row = sub[(sub["config"] == c) & (sub["metric"] == mt)].iloc[0]
                a, b = cell2(row["value"], row["ci_lo"], row["ci_hi"], round(row["value"], 1) == best[mt])
                r1.append(a)
                r2.append(b)
            lat = sub[(sub["config"] == c) & (sub["metric"] == "latency_ms")].iloc[0]["value"]
            name = RET_NAMES[c].replace("alpha = 0.4", r"$\alpha=0.4$").replace("k = 60", r"$k=60$")
            L.append(r"\multicolumn{2}{@{}l}{" + name + "} & " + " & ".join(r1) + f" & {int(round(lat))}" + r"\\")
            L.append(r"\multicolumn{2}{@{}l}{} & " + " & ".join(r2) + r" & \\")
        L.append(r"\midrule")
    L.append(r"\multicolumn{8}{@{}l}{\textit{Panel C: RQ3, answer generation (Kazakh system prompt)}}\\")
    L.append(r"Model & Setting & EM & F1 & Cit.\ acc. & Support & Halluc. & Legal corr.\\")
    L.append(r"\midrule")
    sub = t2[t2["panel"] == "C"]
    best = {}
    for mt in GEN_METRICS:
        vals = sub[sub["metric"] == mt]["value"].round(1)
        best[mt] = vals.min() if mt == "hal" else vals.max()
    for m in MODELS:
        for s, sname in (("llm", "LLM-only"), ("rag", "RAG")):
            r1, r2 = [], []
            for mt in GEN_METRICS:
                row = sub[(sub["config"] == f"{m}_{s}") & (sub["metric"] == mt)].iloc[0]
                a, b = cell2(row["value"], row["ci_lo"], row["ci_hi"], round(row["value"], 1) == best[mt])
                r1.append(a)
                r2.append(b)
            L.append((MODEL_NAMES[m] if s == "llm" else "") + f" & {sname} & " + " & ".join(r1) + r"\\")
            L.append(r" & & " + " & ".join(r2) + r"\\")
    L += [r"\botrule", r"\end{tabular}", r"\footnotetext{" + NOTE + " " + t2.attrs["sd_note"] + "}",
          r"\end{table}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_table3(tests, path):
    def cell(r):
        d = tex(sgn(r["delta"]))
        ci = r"{\scriptsize[" + tex(f1d(r["ci_lo"])) + ", " + tex(f1d(r["ci_hi"])) + "]}"
        p = r"{\scriptsize $p_{\mathrm{Holm}}$" + ("$<$0.001" if r["p_holm"] < 0.001 else f"={r['p_holm']:.3f}") + "}"
        if r["significant"]:
            d = r"\textbf{" + d + "}"
        return d + " " + ci, p

    L = [r"% Table 3 (tab:tests). Generated by simulate.py - " + NOTE,
         r"\begin{table}[t]", r"\caption{Planned pairwise comparisons on the test set: difference $\Delta$ in percentage points with 95\% paired-bootstrap CI and Holm-adjusted $p$-value. Continuous scores (Recall@10, MRR, nDCG@10, citation accuracy, legal correctness): two-sided paired approximate randomization with 10{,}000 sign flips; binary outcomes (support, hallucination): exact McNemar test. Holm--Bonferroni correction within the families F1 (RQ1, 9 tests), F2 (RQ2, 12 tests), F3 (RAG vs.\ LLM-only, 16 tests) and F4 (prompt language, 8 tests); bold: $p_{\mathrm{Holm}}<0.05$. SA: structure-aware chunking; Linear and RRF: hybrid BM25 + BGE-M3 with linear fusion ($\alpha=0.4$) or reciprocal rank fusion; Reranker: linear hybrid with cross-encoder reranker. " + NOTE + r" $n=381$ (Panel A) and $400$ (Panel B).}\label{tab:tests}",
         r"\footnotesize", r"\setlength{\tabcolsep}{4pt}"]
    L += [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
          r"\multicolumn{4}{@{}l}{\textit{Panel A: retrieval (families F1 and F2)}}\\",
          r"Comparison & R@10 & MRR & nDCG@10\\", r"\midrule"]
    rowsA = [("F1", "SA - Sentence", "SA $-$ sentence"),
             ("F1", "SA - Fixed", "SA $-$ fixed-size"),
             ("F1", "SA - Sliding", "SA $-$ sliding window"),
             ("F2", "Hybrid linear - BM25", "Linear $-$ BM25"),
             ("F2", "Hybrid linear - BGE-M3", "Linear $-$ BGE-M3"),
             ("F2", "RRF - linear", "RRF $-$ linear"),
             ("F2", "Reranker - none", "Reranker $-$ none")]
    for fam, comp, name in rowsA:
        c1, c2 = [], []
        for mt in ("R10", "MRR", "nDCG10"):
            r = tests[(tests["family"] == fam) & (tests["comparison"] == comp) & (tests["metric"] == mt)].iloc[0]
            a, b = cell(r)
            c1.append(a)
            c2.append(b)
        L.append(f"{name} ({fam}) & " + " & ".join(c1) + r"\\")
        L.append(" & " + " & ".join(c2) + r"\\[2pt]")
    L += [r"\botrule", r"\end{tabular}", r"", r"\vspace{6pt}", r"",
          r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
          r"\multicolumn{5}{@{}l}{\textit{Panel B: generation (families F3 and F4)}}\\",
          r"Comparison & Cit.\ acc. & Support & Halluc. & Legal corr.\\", r"\midrule"]
    for fam, suffix, lab in (("F3", "RAG - LLM-only", "RAG $-$ LLM-only"), ("F4", "Kazakh - Russian", "Kazakh $-$ Russian system prompt, RAG")):
        L.append(r"\multicolumn{5}{@{}l}{" + lab + f" ({fam})" + r"}\\")
        for m in MODELS:
            comp = f"{MODEL_NAMES[m]}: {suffix}"
            c1, c2 = [], []
            for mt in ("cit", "sup", "hal", "cor"):
                sel = tests[(tests["family"] == fam) & (tests["comparison"] == comp) & (tests["metric"] == mt)]
                if sel.empty:
                    c1.append("--")
                    c2.append("")
                else:
                    a, b = cell(sel.iloc[0])
                    c1.append(a)
                    c2.append(b)
            L.append(r"\quad " + MODEL_NAMES[m] + " & " + " & ".join(c1) + r"\\")
            L.append(" & " + " & ".join(c2) + r"\\[2pt]")
    rp = tests[(tests["family"] == "F2s")].iloc[0]
    L += [r"\botrule", r"\end{tabular}",
          r"\footnotetext{" + NOTE + r" Reranker $-$ no reranker on P@1 (exact McNemar, single supplementary test): $\Delta=" + tex(sgn(rp["delta"])).replace("$", "") + r"$~pp, 95\% CI [" + tex(f1d(rp["ci_lo"])) + ", " + tex(f1d(rp["ci_hi"])) + r"], $p=" + fp(rp["p_raw"]).replace("<", "<") + r"$. The full list of planned tests, including families F5 and F6, with raw and adjusted $p$-values is in the repository (\texttt{results/data/tests\_all.csv}).}",
          r"\end{table}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_table4(t4, tests, path):
    def g(panel, row, col):
        r = t4[(t4["panel"] == panel) & (t4["row"] == row) & (t4["column"] == col)].iloc[0]
        return r

    L = [r"% Table 4 (tab:additional). Generated by simulate.py - " + NOTE,
         r"\begin{table}[t]", r"\caption{Additional analyses on the test set. Panel A: cleaning ablation, Recall@10 and MRR ($\times 100$) of BM25 and the linear hybrid ($n=381$); bold: Recall@10 differs from full cleaning with $p_{\mathrm{Holm}}<0.05$ (approximate randomization, family F5, 6 tests). Panel B: number of retrieved chunks $k$ given to the generator, legal correctness and hallucination rate (RAG, Kazakh prompt, $n=400$); $k=3$ and $k=10$ are tested against $k=5$ on legal correctness (family F6, 4 tests). Panel C: results per question type (Recall@10: $n=124/98/83/76$; legal correctness: $n=128/104/88/80$), descriptive only. " + NOTE + r" Values in \%, 95\% bootstrap CIs in brackets.}\label{tab:additional}",
         r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
         r"\multicolumn{5}{@{}l}{\textit{Panel A: cleaning ablation ($n=381$)}}\\",
         r" & \multicolumn{2}{c}{BM25} & \multicolumn{2}{c}{Hybrid (linear)}\\", r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
         r"Variant & R@10 & MRR & R@10 & MRR\\", r"\midrule"]
    for row, name in (("full", "Full cleaning"), ("translit", "without transliteration"),
                      ("boilerplate", "without boilerplate removal"), ("merging", "without short-fragment merging")):
        c1, c2 = [], []
        for col in ("bm25_R10", "bm25_MRR", "lin_R10", "lin_MRR"):
            r = g("A", row, col)
            bold = bool(r.get("significant", False)) if isinstance(r.get("significant", False), (bool, np.bool_)) else False
            a, b = cell2(r["value"], r["ci_lo"], r["ci_hi"], bold)
            c1.append(a)
            c2.append(b)
        L.append(name + " & " + " & ".join(c1) + r"\\")
        L.append(" & " + " & ".join(c2) + r"\\")
    L += [r"\midrule", r"\multicolumn{5}{@{}l}{\textit{Panel B: chunks $k$ given to the generator ($n=400$)}}\\",
          r" & \multicolumn{2}{c}{GPT-6.1} & \multicolumn{2}{c}{Qwen3.5-9B}\\", r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
          r"$k$ & Legal corr. & Halluc. & Legal corr. & Halluc.\\", r"\midrule"]
    for k in (1, 3, 5, 10):
        c1, c2 = [], []
        for col in ("gpt_cor", "gpt_hal", "qwen_cor", "qwen_hal"):
            r = g("B", str(k), col)
            a, b = cell2(r["value"], r["ci_lo"], r["ci_hi"])
            c1.append(a)
            c2.append(b)
        L.append((f"{k} (main)" if k == 5 else str(k)) + " & " + " & ".join(c1) + r"\\")
        L.append(" & " + " & ".join(c2) + r"\\")
    L += [r"\midrule", r"\multicolumn{5}{@{}l}{\textit{Panel C: per question type}}\\",
          r" & \multicolumn{2}{c}{Recall@10} & \multicolumn{2}{c}{Legal corr. (RAG)}\\", r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
          r"Question type & BM25 & Hybrid & GPT-6.1 & Qwen3.5-9B\\", r"\midrule"]
    for t in TYPES:
        c1, c2 = [], []
        for col in ("bm25_R10", "lin_R10", "gpt_cor", "qwen_cor"):
            r = g("C", t, col)
            a, b = cell2(r["value"], r["ci_lo"], r["ci_hi"])
            c1.append(a)
            c2.append(b)
        nm = TYPE_NAMES[t].replace("Interpretation of provisions", "Interpretation").replace("Applicability conditions", "Applicability")
        L.append(nm + " & " + " & ".join(c1) + r"\\")
        L.append(" & " + " & ".join(c2) + r"\\")
    f6 = tests[tests["family"] == "F6"]
    f6txt = "; ".join(f"{r['comparison']}: $\\Delta={tex(sgn(r['delta'])).replace('$', '')}$~pp, $p_{{\\mathrm{{Holm}}}}={fp(r['p_holm'])}$"
                      for _, r in f6.iterrows())
    L += [r"\botrule", r"\end{tabular}",
          r"\footnotetext{" + NOTE + r" Panel B tests (legal correctness, F6): " + f6txt.replace("k=", "$k$=") + r". $k=5$ was fixed in the Methods before the analysis; $k=1$ is descriptive.}",
          r"\end{table}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def write_table1(t1, corpus, qtab, path):
    def v(item):
        return t1[t1["item"] == item].iloc[0]["value"]

    def num(x):
        return f"{int(x):,}".replace(",", "{,}")
    ch = corpus["chunking"]
    L = [r"% Table 1 (tab:data). Generated by simulate.py - " + NOTE,
         r"\begin{table}[t]", r"\caption{Data used in the study. Panel A: metadata corpus and selected acts; Panel B: chunks per chunking strategy after cleaning; Panel C: benchmark questions by type and split. Rows marked (r) are reported in the KazakhLawCorpus data paper \citep{tleubayeva2026kazakhlawcorpus}; all other values are expected values from a seeded simulation (not measured).}\label{tab:data}",
         r"\footnotesize", r"\begin{tabular}{@{}p{5.6cm}rp{4.4cm}@{}}", r"\toprule",
         r"\multicolumn{3}{@{}l}{\textit{Panel A: corpus and selected acts}}\\", r"Item & Value & Note\\", r"\midrule",
         r"Source & -- & KazakhLawCorpus raw release (Hugging Face), NUBA Solutions export of Adilet.zan.kz and other public sources (r)\\",
         r"Period of adoption & 1947--2026 & (r)\\",
         r"Records in raw release & " + num(corpus_val(corpus, "raw")) + r" & (r)\\",
         r"Records removed & " + num(corpus["removed_total"]) + r" & (r); no usable Kazakh title or description " + num(corpus["removed_no_title"]) + ", duplicated identifier " + num(corpus["removed_duplicate_id"]) + r" (expected)\\",
         r"Records after cleaning & " + num(corpus_val(corpus, "clean")) + r" & (r); our re-cleaning is expected to reproduce it\\",
         r"Records flagged but kept & " + num(corpus["flagged_kept"]) + r" & expected; " + f1d(100 * corpus["flagged_kept"] / corpus_val(corpus, "clean")) + r"\% of records\\",
         r"In-force acts with Kazakh text and normative form & " + num(corpus["eligible_pool"]) + r" & expected eligible pool\\",
         r"Acts selected for full-text download & " + num(1200) + r" & planned: " + num(corpus["acts_referenced"]) + " referenced by the benchmark + " + num(corpus["acts_distractors"]) + r" random distractors (expected)\\",
         r"Acts parsed successfully & " + num(corpus["acts_parsed"]) + r" & expected; " + str(corpus["acts_parse_failed"]) + r" distractor pages fail, not replaced\\",
         r"Words after cleaning & " + f"{corpus['words_total'] / 1e6:.1f}" + r"\,M & expected; " + num(corpus["articles_total"]) + r" articles\\",
         r"Text removed by second-level cleaning & " + f1d(corpus["removed_token_share"]) + r"\% & expected share of tokens (headers, footers, pagination, identifiers)\\",
         r"Latin look-alike letters transliterated & " + f1d(corpus["latin_word_share"]) + r"\% & expected share of words; in " + f"{corpus['latin_act_share']:.0f}" + r"\% of acts\\",
         r"\midrule", r"\end{tabular}", r"",
         r"\begin{tabular}{@{}lrrrr@{}}",
         r"\multicolumn{5}{@{}l}{\textit{Panel B: chunks per strategy after cleaning (expected)}}\\",
         r"Strategy & Chunks & Median words & P5--P95 words & $<$20 words (\%)\\", r"\midrule"]
    for key, name in (("sentence", "Sentence-level"), ("fixed", "Fixed-size (200 w, no overlap)"),
                      ("sliding", "Sliding window (200 w, stride 100)"), ("sa", "Structure-aware ($\\leq$450 w)")):
        s = ch[key]
        L.append(f"{name} & {num(s['chunks'])} & {s['median']:.0f} & {s['p5']:.0f}--{s['p95']:.0f} & {f1d(s['share_short'])}" + r"\\")
    sb = ch["sa_before_cleaning"]
    L.append(r"\quad before cleaning and merging & " + num(sb["chunks"]) + f" & {sb['median']:.0f} & {sb['p5']:.0f}--{sb['p95']:.0f} & {f1d(sb['share_short'])}" + r"\\")
    L += [r"\midrule", r"\end{tabular}", r"",
          r"\begin{tabular}{@{}lrrrr@{}}",
          r"\multicolumn{5}{@{}l}{\textit{Panel C: benchmark questions}}\\",
          r"Question type & All & Dev & Test & Test with gold\\", r"\midrule"]
    for _, r in qtab.iterrows():
        L.append(f"{r['type_name']} & {r['all']} & {r['dev']} & {r['test']} & {r['test_with_gold']}" + r"\\")
    L += [r"\botrule", r"\end{tabular}",
          r"\footnotetext{(r) reported in \citep{tleubayeva2026kazakhlawcorpus}. Retrieval metrics use the 381 test questions with at least one linked gold chunk; generation metrics use all 400. Type counts and split sizes are planned design values; ``with gold'' counts and all other values without (r) are expected values from a seeded simulation (not measured). Dev questions with gold: " + str(int(qtab.attrs["dev_with_gold"])) + "; mean gold provisions per question: " + f"{qtab.attrs['mean_gold']:.2f}" + r" (expected).}",
          r"\end{table}"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def corpus_val(corpus, which):
    return CONFIG["corpus"]["raw_records"] if which == "raw" else CONFIG["corpus"]["clean_records"]


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seed", type=int, default=MASTER_SEED, help="master seed (the study uses 2026)")
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent),
                    help="results directory (data/, figures/, tables/ are written inside)")
    args = ap.parse_args()
    t_wall, t_cpu = time.perf_counter(), time.process_time()

    out = Path(args.out)
    ddir, fdir, tdir = out / "data", out / "figures", out / "tables"
    for p in (ddir, fdir, fdir / "preview", tdir):
        p.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(CONFIG))
    cfg["master_seed"] = args.seed
    rngs = make_streams(args.seed)

    # ---- simulation --------------------------------------------------------
    q = simulate_questions(rngs["questions"], cfg)
    corpus = simulate_corpus(rngs["corpus"], cfg, len(q))
    Zp, hyb, Gd, aff_u = build_retrieval_latents(rngs["retrieval"], q, cfg)
    ret = simulate_retrieval(Zp, hyb, Gd, aff_u, q, cfg)
    lat = simulate_latency(rngs["latency"], ret["idx_test"].size, cfg)
    gen = simulate_generation(rngs["generation"], q, ret, cfg)
    errs = simulate_errors(rngs["judge"], gen, cfg)

    tt, tg = ret["type_test"], gen["type"]
    boot_sizes = [("test_gold", tt.size), ("test", tg.size), ("dev_gold", ret["idx_dev"].size)]
    boot_sizes += [(f"test_gold_{t}", int(np.sum(tt == i))) for i, t in enumerate(TYPES)]
    boot_sizes += [(f"test_{t}", int(np.sum(tg == i))) for i, t in enumerate(TYPES)]
    boot_sizes += [("judge", cfg["judge"]["n_sample"])]
    st = Stats(rngs["bootstrap"], rngs["randomization"], boot_sizes,
               [("test_gold", tt.size), ("test", tg.size)])
    judge = simulate_judge(rngs["judge"], gen, cfg, st.C["judge"])
    pq, go = ret["pq"], gen["out"]

    # ---- Table 1 -----------------------------------------------------------
    qtab = []
    for i, t in enumerate(TYPES):
        s = q[q["type_idx"] == i]
        qtab.append({"type": t, "type_name": TYPE_NAMES[t], "all": len(s), "dev": int((s["split"] == "dev").sum()),
                     "test": int((s["split"] == "test").sum()),
                     "test_with_gold": int(((s["split"] == "test") & (s["n_gold"] > 0)).sum())})
    qtab = pd.DataFrame(qtab)
    qtab.loc[len(qtab)] = {"type": "total", "type_name": "Total", "all": int(qtab["all"].sum()),
                           "dev": int(qtab["dev"].sum()), "test": int(qtab["test"].sum()),
                           "test_with_gold": int(qtab["test_with_gold"].sum())}
    qtab.attrs["dev_with_gold"] = int(((q["split"] == "dev") & (q["n_gold"] > 0)).sum())
    qtab.attrs["mean_gold"] = float(q.loc[q["n_gold"] > 0, "n_gold"].mean())
    t1 = [("A", "raw_records", cfg["corpus"]["raw_records"], "reported"),
          ("A", "removed_total", corpus["removed_total"], "reported"),
          ("A", "removed_no_title", corpus["removed_no_title"], "expected"),
          ("A", "removed_duplicate_id", corpus["removed_duplicate_id"], "expected"),
          ("A", "clean_records", cfg["corpus"]["clean_records"], "reported"),
          ("A", "flagged_kept", corpus["flagged_kept"], "expected"),
          ("A", "eligible_pool", corpus["eligible_pool"], "expected"),
          ("A", "acts_planned", cfg["corpus"]["acts_planned"], "planned"),
          ("A", "acts_referenced", corpus["acts_referenced"], "expected"),
          ("A", "acts_distractors", corpus["acts_distractors"], "expected"),
          ("A", "acts_parse_failed", corpus["acts_parse_failed"], "expected"),
          ("A", "acts_parsed", corpus["acts_parsed"], "expected"),
          ("A", "words_total", corpus["words_total"], "expected"),
          ("A", "articles_total", corpus["articles_total"], "expected"),
          ("A", "removed_token_share_pct", corpus["removed_token_share"], "expected"),
          ("A", "latin_word_share_pct", corpus["latin_word_share"], "expected"),
          ("A", "latin_act_share_pct", corpus["latin_act_share"], "expected")]
    for k, s in corpus["chunking"].items():
        for kk, vv in s.items():
            t1.append(("B", f"{k}_{kk}", vv, "expected"))
    for _, r in qtab.iterrows():
        for col in ("all", "dev", "test", "test_with_gold"):
            t1.append(("C", f"{r['type']}_{col}", r[col], "planned" if col != "test_with_gold" else "expected"))
    t1.append(("C", "dev_with_gold", qtab.attrs["dev_with_gold"], "expected"))
    t1.append(("C", "mean_gold_per_question", qtab.attrs["mean_gold"], "expected"))
    t1 = pd.DataFrame(t1, columns=["panel", "item", "value", "status"])

    # ---- Table 2 -----------------------------------------------------------
    rows = []
    for panel, cfgs in (("A", ["sentence", "fixed", "sliding", "sa"]),
                        ("B", ["tfidf", "bm25", "labse", "e5", "bge", "lin", "rrf", "rr"])):
        for c in cfgs:
            for mt in RET_METRICS:
                v, lo, hi = st.ci("test_gold", pq[c][mt])
                rows.append({"panel": panel, "config": c, "name": RET_NAMES[c], "metric": mt,
                             "value": v, "ci_lo": lo, "ci_hi": hi, "n": tt.size})
            rows.append({"panel": panel, "config": c, "name": RET_NAMES[c], "metric": "latency_ms",
                         "value": float(lat[c].mean()), "ci_lo": np.nan, "ci_hi": np.nan, "n": tt.size})
    for m in MODELS:
        for s, v_ in (("llm", "llm"), ("rag", "kk5")):
            o = go[m][v_]
            for mt in GEN_METRICS + ["abst", "unver"]:
                v, lo, hi = st.ci("test", o[mt].astype(float))
                rows.append({"panel": "C" if mt in GEN_METRICS else "C-supp", "config": f"{m}_{s}",
                             "name": f"{MODEL_NAMES[m]} {'LLM-only' if s == 'llm' else 'RAG'}",
                             "metric": mt, "value": v, "ci_lo": lo, "ci_hi": hi, "n": tg.size})
            v, lo, hi = st.ci("test", (o["cor"] == 1.0).astype(float))
            rows.append({"panel": "C-supp", "config": f"{m}_{s}", "name": f"{MODEL_NAMES[m]} {s}",
                         "metric": "fully_correct", "value": v, "ci_lo": lo, "ci_hi": hi, "n": tg.size})
    t2 = pd.DataFrame(rows)

    # ---- run-to-run SD -----------------------------------------------------
    run_rows = []
    for m in MODELS:
        runs = gen["params"][m]["runs"]
        sd = float(np.std([runs[2026]] + [runs[s] for s in RUN_SEEDS], ddof=1))
        for s in [13, 42, 2026]:
            run_rows.append({"model": MODEL_NAMES[m], "seed": s, "legal_correctness": runs[s], "sd_pp": sd})
    run_sd = pd.DataFrame(run_rows)
    sd_txt = ", ".join(f"{MODEL_NAMES[m]} {f1d(run_sd[run_sd['model'] == MODEL_NAMES[m]]['sd_pp'].iloc[0])}"
                       for m in MODELS)
    t2.attrs["sd_note"] = ("Run-to-run SD of legal correctness over seeds 13, 42 and 2026 (RAG, Kazakh prompt), "
                           f"in pp: {sd_txt} (expected).")

    # ---- planned tests -----------------------------------------------------
    T = []

    def add(fam, comp, metric, x, y, key, kind):
        x, y = np.asarray(x, float), np.asarray(y, float)
        d = x - y
        delta, lo, hi = st.ci(key, d)
        if kind == "AR":
            p, b, c = st.ar_test(key, d), None, None
            test = "approximate randomization (10,000 sign flips)"
        else:
            p, b, c = mcnemar_exact(x > 0.5, y > 0.5)
            test = "exact McNemar"
        T.append({"family": fam, "comparison": comp, "metric": metric, "delta": delta, "ci_lo": lo, "ci_hi": hi,
                  "test": test, "p_raw": p, "discordant_b": b, "discordant_c": c, "n": d.size,
                  "value_a": 100 * x.mean(), "value_b": 100 * y.mean()})

    for comp, a, b in (("SA - Sentence", "sa", "sentence"), ("SA - Fixed", "sa", "fixed"), ("SA - Sliding", "sa", "sliding")):
        for mt in ("R10", "MRR", "nDCG10"):
            add("F1", comp, mt, pq[a][mt], pq[b][mt], "test_gold", "AR")
    for comp, a, b in (("Hybrid linear - BM25", "lin", "bm25"), ("Hybrid linear - BGE-M3", "lin", "bge"),
                       ("RRF - linear", "rrf", "lin"), ("Reranker - none", "rr", "lin")):
        for mt in ("R10", "MRR", "nDCG10"):
            add("F2", comp, mt, pq[a][mt], pq[b][mt], "test_gold", "AR")
    add("F2s", "Reranker - none", "P1", pq["rr"]["P1"], pq["lin"]["P1"], "test_gold", "McNemar")
    for m in MODELS:
        for mt, kind in (("cit", "AR"), ("sup", "McNemar"), ("hal", "McNemar"), ("cor", "AR")):
            add("F3", f"{MODEL_NAMES[m]}: RAG - LLM-only", mt, go[m]["kk5"][mt], go[m]["llm"][mt], "test", kind)
    for m in MODELS:
        for mt, kind in (("hal", "McNemar"), ("cor", "AR")):
            add("F4", f"{MODEL_NAMES[m]}: Kazakh - Russian", mt, go[m]["kk5"][mt], go[m]["ru5"][mt], "test", kind)
    abl_names = {"translit": "w/o transliteration", "boilerplate": "w/o boilerplate removal",
                 "merging": "w/o short-fragment merging"}
    for c, cn in (("bm25", "BM25"), ("lin", "Hybrid")):
        for v in ("translit", "boilerplate", "merging"):
            add("F5", f"{cn}: {abl_names[v]} - full", "R10", ret["ablation"][(c, v)]["pq"]["R10"], pq[c]["R10"],
                "test_gold", "AR")
    for m in ("gpt", "qwen"):
        for k in (3, 10):
            add("F6", f"{MODEL_NAMES[m]}: k={k} - k=5", "cor", go[m][f"kk{k}"]["cor"], go[m]["kk5"]["cor"], "test", "AR")
    tests = pd.DataFrame(T)
    tests["p_holm"] = np.nan
    for fam, idx in tests.groupby("family").groups.items():
        tests.loc[idx, "p_holm"] = holm(tests.loc[idx, "p_raw"].to_numpy())
    tests["significant"] = tests["p_holm"] < SIG_LEVEL
    tests["designed"] = [DESIGNED.get((r["family"], r["comparison"]), {}).get(r["metric"], "")
                         for _, r in tests.iterrows()]
    tests["matches_design"] = [(d_ == "") or (d_ == "sig*") or ((d_ == "sig") == s_)
                               for d_, s_ in zip(tests["designed"], tests["significant"])]

    # ---- Table 4 -----------------------------------------------------------
    rows = []
    sigF5 = {r["comparison"]: r["significant"] for _, r in tests[tests["family"] == "F5"].iterrows()}
    for c, cn in (("bm25", "BM25"), ("lin", "Hybrid")):
        for row in ("full", "translit", "boilerplate", "merging"):
            src = pq[c] if row == "full" else ret["ablation"][(c, row)]["pq"]
            for mt in ("R10", "MRR"):
                v, lo, hi = st.ci("test_gold", src[mt])
                sig = bool(sigF5.get(f"{cn}: {abl_names[row]} - full", False)) if (row != "full" and mt == "R10") else False
                rows.append({"panel": "A", "row": row, "column": f"{c}_{mt}", "value": v, "ci_lo": lo,
                             "ci_hi": hi, "n": tt.size, "significant": sig})
    for m in ("gpt", "qwen"):
        for k in (1, 3, 5, 10):
            for mt in ("cor", "hal"):
                v, lo, hi = st.ci("test", go[m][f"kk{k}"][mt].astype(float))
                rows.append({"panel": "B", "row": str(k), "column": f"{m}_{mt}", "value": v, "ci_lo": lo,
                             "ci_hi": hi, "n": tg.size, "significant": False})
    for i, t in enumerate(TYPES):
        for c in ("bm25", "lin"):
            v, lo, hi = st.ci(f"test_gold_{t}", pq[c]["R10"][tt == i])
            rows.append({"panel": "C", "row": t, "column": f"{c}_R10", "value": v, "ci_lo": lo, "ci_hi": hi,
                         "n": int(np.sum(tt == i)), "significant": False})
        for m in ("gpt", "qwen"):
            v, lo, hi = st.ci(f"test_{t}", go[m]["kk5"]["cor"][tg == i])
            rows.append({"panel": "C", "row": t, "column": f"{m}_cor", "value": v, "ci_lo": lo, "ci_hi": hi,
                         "n": int(np.sum(tg == i)), "significant": False})
    t4 = pd.DataFrame(rows)

    # ---- Figure 1: alpha sensitivity on dev --------------------------------
    rows = []
    for a in cfg["retrieval"]["alpha_grid"]:
        r = {"alpha": a, "n": ret["idx_dev"].size}
        for mt in ("R10", "MRR", "nDCG10"):
            v, lo, hi = st.ci("dev_gold", ret["dev"][a][mt])
            r.update({mt: v, f"{mt}_lo": lo, f"{mt}_hi": hi})
        rows.append(r)
    fig1 = pd.DataFrame(rows)
    rrf_dev = {mt: 100 * ret["dev"]["rrf"][mt].mean() for mt in ("R10", "MRR", "nDCG10")}
    best_alpha = float(fig1.loc[fig1["nDCG10"].idxmax(), "alpha"])
    fig1.attrs.update(selected_alpha=best_alpha, rrf_R10=rrf_dev["R10"], rrf_MRR=rrf_dev["MRR"])
    fig1_out = fig1.copy()
    fig1_out.loc[len(fig1_out)] = {"alpha": "RRF", "n": ret["idx_dev"].size,
                                   **{mt: rrf_dev[mt] for mt in ("R10", "MRR", "nDCG10")}}

    # ---- Figure 2: prompt language -----------------------------------------
    rows = []
    for m in MODELS:
        for lang, v_ in (("kk", "kk5"), ("ru", "ru5")):
            for mt in ("cor", "hal"):
                v, lo, hi = st.ci("test", go[m][v_][mt].astype(float))
                tr = tests[(tests["family"] == "F4") & (tests["comparison"] == f"{MODEL_NAMES[m]}: Kazakh - Russian")
                           & (tests["metric"] == mt)].iloc[0]
                rows.append({"model": m, "model_name": MODEL_NAMES[m], "prompt": lang, "metric": mt, "value": v,
                             "ci_lo": lo, "ci_hi": hi, "n": tg.size, "delta_kk_minus_ru": tr["delta"],
                             "p_holm": tr["p_holm"], "significant": bool(tr["significant"])})
    fig2 = pd.DataFrame(rows)

    # ---- Figure 3 and retrieval-answer 2x2 ---------------------------------
    fig3 = pd.DataFrame([{"model": m, "model_name": MODEL_NAMES[m], "failed": errs[m]["failed"], **errs[m]["counts"]}
                         for m in MODELS])
    rows = []
    hg = gen["has_gold"]
    for m in MODELS:
        full = go[m]["kk5"]["cor"] == 1.0
        hit = gen["hits"][5]
        for grp, sel in (("gold_in_top5", hg & hit), ("gold_not_in_top5", hg & ~hit), ("no_linked_gold", ~hg)):
            rows.append({"model": MODEL_NAMES[m], "group": grp, "questions": int(sel.sum()),
                         "fully_correct": int((full & sel).sum()), "not_fully_correct": int((~full & sel).sum()),
                         "share_fully_correct_pct": 100 * float(full[sel].mean()) if sel.any() else np.nan})
    rx2 = pd.DataFrame(rows)
    judge_df = pd.DataFrame([{"comparison": k, **v} for k, v in judge.items() if not k.startswith("_")])

    # ---- consistency checks -------------------------------------------------
    def val(panel, c, mt):
        return float(t2[(t2["panel"] == panel) & (t2["config"] == c) & (t2["metric"] == mt)]["value"].iloc[0])
    checks = {}
    for panel, cfgs in (("A", ["sentence", "fixed", "sliding", "sa"]),
                        ("B", ["tfidf", "bm25", "labse", "e5", "bge", "lin", "rrf", "rr"])):
        for c in cfgs:
            v = {mt: val(panel, c, mt) for mt in RET_METRICS}
            checks[f"order_P1<MRR<R5<R10_{c}"] = v["P1"] < v["MRR"] < v["R5"] < v["R10"]
            checks[f"nDCG_between_MRR_R10_{c}"] = v["MRR"] < v["nDCG10"] < v["R10"]
            tgt = cfg["retrieval"]["targets"][c]
            checks[f"within_1pp_{c}"] = bool(np.all(np.abs(np.array([v[mt] for mt in RET_METRICS]) - tgt) <= 1.0))
    checks["sentence_worst_chunking"] = all(val("A", "sentence", mt) < val("A", c, mt)
                                            for c in ("fixed", "sliding", "sa") for mt in RET_METRICS)
    checks["hybrid_latency_gt_single"] = all(lat["lin"].mean() > lat[c].mean() for c in ("tfidf", "bm25", "labse", "e5", "bge"))
    checks["reranker_latency_300_500"] = 300 <= lat["rr"].mean() <= 500
    for m in MODELS:
        for s in ("llm", "rag"):
            v = {mt: val("C", f"{m}_{s}", mt) for mt in GEN_METRICS}
            tgt = cfg["generation"]["targets"][m][s]
            checks[f"within_1pp_{m}_{s}"] = bool(np.all(np.abs(np.array([v[mt] for mt in GEN_METRICS]) - tgt) <= 1.0))
            o = go[m]["kk5" if s == "rag" else "llm"]
            checks[f"categories_sum_100_{m}_{s}"] = bool(np.all(o["abst"].astype(int) + o["hal"] + o["sup"] + o["unver"] == 1))
        for mt in ("cit", "sup", "cor", "f1", "em"):
            checks[f"rag_better_{m}_{mt}"] = val("C", f"{m}_rag", mt) > val("C", f"{m}_llm", mt)
        checks[f"rag_less_hallucination_{m}"] = val("C", f"{m}_rag", "hal") < val("C", f"{m}_llm", "hal")
    for s in ("llm", "rag"):
        for mt in GEN_METRICS:
            g_ = val("C", f"gpt_{s}", mt)
            others = [val("C", f"{m}_{s}", mt) for m in MODELS[1:]]
            checks[f"gpt_best_{s}_{mt}"] = g_ < min(others) if mt == "hal" else g_ > max(others)
    checks["kaz_vs_qwen_legal_diff_lt_1.5"] = abs(val("C", "kaz_rag", "cor") - val("C", "qwen_rag", "cor")) < 1.5
    checks["em_below_8_everywhere"] = all(val("C", f"{m}_{s}", "em") < 8 for m in MODELS for s in ("llm", "rag"))
    checks["selected_alpha_is_0.4"] = best_alpha == 0.4
    checks["designed_significance_pattern"] = bool(tests["matches_design"].all())
    checks["error_counts_sum"] = all(sum(errs[m]["counts"].values()) == errs[m]["failed"] for m in MODELS)
    t4c = t4[t4["panel"] == "C"]
    for col, ref in (("bm25_R10", val("B", "bm25", "R10")), ("lin_R10", val("B", "lin", "R10")),
                     ("gpt_cor", val("C", "gpt_rag", "cor")), ("qwen_cor", val("C", "qwen_rag", "cor"))):
        sub = t4c[t4c["column"] == col]
        wm = float((sub["value"] * sub["n"]).sum() / sub["n"].sum())
        checks[f"per_type_weighted_mean_{col}"] = abs(wm - ref) <= 0.3 + 1e-9

    # ---- write data ---------------------------------------------------------
    def to_csv(df, name):
        df.to_csv(ddir / name, index=False, float_format="%.6f", lineterminator="\n")
    to_csv(t1, "table1_data.csv")
    to_csv(qtab, "table1_questions.csv")
    to_csv(t2, "table2_main.csv")
    to_csv(tests, "tests_all.csv")
    to_csv(t4, "table4_additional.csv")
    to_csv(fig1_out, "fig1_alpha.csv")
    to_csv(fig2, "fig2_prompt.csv")
    to_csv(fig3, "fig3_errors.csv")
    to_csv(run_sd, "run_sd.csv")
    to_csv(judge_df, "judge_agreement.csv")
    to_csv(rx2, "retrieval_answer_2x2.csv")
    # per-question simulated scores (the unit of all analyses)
    qt = q.iloc[ret["idx_test"]][["qid", "type", "n_gold"]].reset_index(drop=True)
    pr = pd.concat([qt] + [pd.DataFrame({f"{c}__{mt}": pq[c][mt] for mt in RET_METRICS}) for c in pq]
                   + [pd.DataFrame({f"{c}__latency_ms": lat[c]}) for c in lat], axis=1)
    to_csv(pr, "per_question_retrieval.csv")
    qg = q.iloc[gen["idx"]][["qid", "type", "n_gold"]].reset_index(drop=True)
    cols = {"gold_in_top5": gen["hits"][5].astype(int)}
    for m in MODELS:
        for v_, o in go[m].items():
            for mt in ("abst", "cor", "hal", "sup", "cit", "em", "f1"):
                if mt in o:
                    cols[f"{m}__{v_}__{mt}"] = np.asarray(o[mt], float)
        cols[f"{m}__kk5__error_category"] = errs[m]["category"]
    to_csv(pd.concat([qg, pd.DataFrame(cols)], axis=1), "per_question_generation.csv")

    gp = {m: {k: v for k, v in gen["params"][m].items()} for m in MODELS}
    rp = {c: {"m_by_type": p["mt"], "s_lo": p["s_lo"], "s_hi": p["s_hi"], "rho": p["rho"], "z0": p["z0"]} for c, p in ret["params"].items()}
    sim_cfg = dict(cfg)
    sim_cfg["streams"] = {name: f"SeedSequence({args.seed}).spawn({len(STREAMS)})[{i}] -> Generator(PCG64)"
                          for i, name in enumerate(STREAMS)}
    sim_cfg["stream_use"] = {"questions": "type split, dev/test split, gold counts, difficulty",
                             "corpus": "record removal breakdown, flags, act selection, act lengths, chunk lengths",
                             "retrieval": "latent components of gold-chunk ranks, ablation shifts",
                             "latency": "per-query latencies",
                             "generation": "answer-quality latents, abstention, hallucination, support, citations, EM, F1, run seeds",
                             "judge": "error categories, judge-agreement sample and label noise",
                             "bootstrap": "10,000 resamples per sample (count matrices)",
                             "randomization": "10,000 sign flips per sample"}
    sim_cfg["calibrated_parameters"] = rnd({
        "retrieval": rp,
        "ablation_shift": {f"{c}_{v}": a["mu"] for (c, v), a in ret["ablation"].items()},
        "generation": gp,
        "judge_flip_probabilities": judge["_flip_probs"]})
    (ddir / "sim_config.json").write_text(json.dumps(rnd(sim_cfg), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "label": "EXPECTED values from a seeded simulation (not measured)",
        "master_seed": args.seed,
        "table1": {"corpus": corpus, "questions": qtab.to_dict(orient="records"),
                   "dev_with_gold": qtab.attrs["dev_with_gold"], "mean_gold": qtab.attrs["mean_gold"]},
        "table2": t2.to_dict(orient="records"),
        "run_sd": run_sd.to_dict(orient="records"),
        "tests": tests.to_dict(orient="records"),
        "table4": t4.to_dict(orient="records"),
        "fig1_alpha": {"rows": fig1.to_dict(orient="records"), "rrf": rrf_dev, "selected_alpha": best_alpha},
        "fig2_prompt": fig2.to_dict(orient="records"),
        "fig3_errors": fig3.to_dict(orient="records"),
        "retrieval_answer_2x2": rx2.to_dict(orient="records"),
        "judge_agreement": judge_df.to_dict(orient="records"),
        "latency_ms": {c: float(v.mean()) for c, v in lat.items()},
        "ablation_affected_share": {f"{c}_{v}": a["affected_share"] for (c, v), a in ret["ablation"].items()},
        "checks": checks,
    }
    summary = rnd(summary)
    js = json.dumps(summary, indent=1, ensure_ascii=False, allow_nan=True) + "\n"
    (ddir / "results_summary.json").write_text(js, encoding="utf-8")
    digest = hashlib.sha256(js.encode("utf-8")).hexdigest()

    # ---- figures and tables -------------------------------------------------
    set_style()
    fig_alpha(fig1, fdir)
    fig_prompt(fig2, fdir)
    fig_errors(fig3, fdir)
    write_table1(t1, corpus, qtab, tdir / "table1_data.tex")
    write_table2(t2, tdir / "table2_main.tex")
    write_table3(tests, tdir / "table3_tests.tex")
    write_table4(t4, tests, tdir / "table4_additional.tex")

    # ---- environment and runtime -------------------------------------------
    cpu = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    versions = (f"python {platform.python_version()}\nnumpy {np.__version__}\nscipy {scipy.__version__}\n"
                f"pandas {pd.__version__}\nmatplotlib {matplotlib.__version__}\n"
                f"platform {platform.platform()}\ncpu {cpu} ({os.cpu_count()} logical cores)\n")
    (ddir / "versions.txt").write_text(versions, encoding="utf-8")
    wall, cpu_t = time.perf_counter() - t_wall, time.process_time() - t_cpu
    (ddir / "runtime.txt").write_text(
        f"wall_clock_seconds {wall:.1f}\ncpu_seconds {cpu_t:.1f}\ncpu {cpu} ({os.cpu_count()} logical cores)\n"
        f"master_seed {args.seed}\nresults_summary_sha256 {digest}\n", encoding="utf-8")
    failed = [k for k, v in checks.items() if not v]
    print(f"done in {wall:.1f} s wall / {cpu_t:.1f} s CPU; results_summary.json sha256 {digest}")
    print("failed checks:", failed if failed else "none")
    bad = tests[~tests["matches_design"]]
    if len(bad):
        print(bad[["family", "comparison", "metric", "delta", "p_raw", "p_holm", "designed"]].to_string())


if __name__ == "__main__":
    main()
