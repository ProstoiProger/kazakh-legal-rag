# Targets and their sources

**No value in this repository is a measurement.** Every mean reported in the article was first written down by the authors as a target. The simulation (`src/simulate.py`) reproduces these targets by calibration, to within about ±1 point. The targets are stored in `outputs/data/sim_config.json` and in `CONFIG` in `src/simulate.py`.

The pattern of significant and non-significant comparisons was also fixed in advance, in `DESIGNED` in `src/simulate.py`. The noise correlations were chosen so that this pattern holds; see the column `matches_design` in `outputs/data/tests_all.csv`.

This file states, for every group of targets, where the number comes from.

- **"From [Prototype]"** means the target is close to a value measured in our earlier prototype study [Prototype].
- **"Direction from …"** means a cited study supports the direction of the effect but not its size.
- **"Authors' assumption"** means no cited study gives the number. Where a target goes against the measured results of [Prototype], we say so.

## References used below (cited by short name; see the article's reference list)

- [Prototype] Tleubayeva, Abilkhaiyr, Orazgali, Mansurova, Shomanov, Makhambetova. *Structure-Aware Hybrid Legal RAG for Grounded Question Answering in Kazakh*. Presented at AMLDS 2026, Osaka; proceedings in press (`tleubayeva2026structure`).
  - It measured, on a different and smaller setting: four chunking strategies; BM25, BGE-M3 and a hybrid; a reranker; and GPT-4.1, KazLLM-1.0-8B, Hermes-3-Llama-3.1-8B and Qwen2.5-7B, with and without RAG.
- [Mansurova 2025] Mansurova, Tleubayeva, Nugumanova, Shomanov, Seker. *A Systematic Evaluation of Large Language Models and Retrieval-Augmented Generation for the Task of Kazakh Question Answering*. Information 16(11), 943 (2025) (`mansurova2025systematic`).
  - Qualitative findings: RAG narrows the gap between small open models and large proprietary models for Kazakh QA; good retrieval does not guarantee a correct answer.
- [Kalzhanov 2026] Kalzhanov, Artykbay, Kalzhan. *Development of the Retrieval-Augmented Generation (RAG) System for the Kazakh Language Using Hybrid Information Methods*. Journal of Problems in Computer Science and Information Technologies 4(1) (2026) (`kalzhanov2026hybrid`).
  - A RAG system for Kazakh, tested on legal documents of Kazakhstan. A hybrid of BM25 and vector search gives more accurate answers than either alone: 82.2% answer accuracy, +3.3 and +5.5 points over the single methods.
  - This is answer accuracy, not a retrieval metric.
- Data paper: Tleubayeva, Mansurova, Shomanov, Makhambetova, Boluk. *KazakhLawCorpus: A Large-Scale Metadata Corpus for Kazakh Legislation*. Data 11(9), 217 (2026) (`tleubayeva2026kazakhlawcorpus`). It is the source of the real corpus counts.

Target values are listed in the same order as the columns of `CONFIG`.

## Retrieval: chunking (Panel A, `retrieval.targets.sentence/fixed/sliding/sa`)

| Target | Values (R@5, R@10, MRR, P@1, nDCG@10) | Source | Comment |
|---|---|---|---|
| Structure-aware, R@10 | 74.9 | **From [Prototype]** | [Prototype] measured R@10 = 0.74 for structure-aware chunks |
| Sentence-level, R@10 | 58.8 | **From [Prototype]** | [Prototype]: 0.58 |
| Fixed-size and sliding-window, R@10 | 65.4, 69.8 | **Authors' assumption, different from [Prototype]** | [Prototype] measured fixed-size 0.71 > sliding window 0.62. Our targets reverse this order and put both below structure-aware |
| All MRR, P@1, nDCG@10 and R@5 values | see `sim_config.json` | **Authors' assumption** | They are set below the values of [Prototype] (structure-aware MRR 0.646, nDCG@10 0.665) and keep the order P@1 < MRR < nDCG@10 < R@10 |
| Size of the structure-aware gain (+16 points over sentence-level) | — | **From [Prototype] for R@10; optimistic** | This gain is larger than the measured effects of structural chunking on German and Thai legal texts cited in the article |
| Latency 47–78 ms | — | **Authors' assumption** | [Prototype] reports 854–4,379 ms in another setup. The targets assume retrieval time only, with batch size 1 on a local index |

## Retrieval: retrievers on structure-aware chunks (Panel B)

| Target | Values | Source | Comment |
|---|---|---|---|
| Hybrid better than BM25 and than BGE-M3 | linear hybrid R@10 75.5 vs BGE-M3 68.9 and BM25 60.6 | **Direction from [Kalzhanov 2026]; size is the authors' assumption; different from [Prototype]** | [Kalzhanov 2026] reports that the hybrid is better on answer accuracy. [Prototype] measured BM25 R@10 0.78, BGE-M3 0.75 and hybrid 0.75, i.e. BM25 best. Our targets assume Kazakh stemming for BM25 and a better-tuned fusion, but they do not follow [Prototype] |
| TF-IDF, LaBSE, multilingual-E5-large | 54.0, 55.9, 64.8 (R@10) | **Authors' assumption** | No cited study measures them on Kazakh legal text. LaBSE below BM25 is an assumption |
| RRF ≈ linear fusion | 75.0 vs 75.5 (R@10) | **Authors' assumption** | Only general literature on fusion functions informs it; there is no Kazakh legal measurement |
| Reranker: small, non-significant gain at +361 ms | 76.1 R@10, 56.9 MRR | **Authors' assumption, more favorable than [Prototype]** | [Prototype] measured that reranking lowered P@1 (0.58 → 0.54) and MRR (0.655 → 0.622) and added 387–842 ms |
| Linear hybrid at α = 0.4 chosen on dev | `selected_alpha_expected` = 0.4 | **Authors' assumption** | The dev curve is interpolated between calibrated anchors |
| Recall@10 per question type | `type_targets_R10` | **Authors' assumption** | — |
| Cleaning ablation (homoglyph normalization, boilerplate, merging) | `ablation_targets_R10` | **Authors' assumption** | The direction (BM25 more sensitive than the hybrid) is a plausibility argument, not a measurement |

## Generation (Panel C, `generation.targets`; EM, F1, citation accuracy, support, hallucination, legal correctness)

| Target | Values | Source | Comment |
|---|---|---|---|
| RAG better than LLM-only on every grounding metric | all models | **Direction from [Prototype] and [Mansurova 2025]** | Both report large gains from retrieval |
| LLM-only levels (hallucination 34–65%, legal correctness 27–55) | see `targets.*.llm` | **Authors' assumption, much milder than [Prototype]** | [Prototype] measured LLM-only hallucination of 0.965–0.999, citation accuracy 0–0.03, support 0.002–0.04 and legal correctness 0–0.15. Our targets also count unverifiable-only answers and abstentions separately |
| GPT-6.1 RAG: citation accuracy 66.1, support 76.5, hallucination 13.8 | `targets.gpt.rag` | **Close to [Prototype] for the proprietary model** | [Prototype] measured GPT-4.1, RAG with Kazakh prompt: citation accuracy 0.72, support 0.78, hallucination 0.20. GPT-6.1 itself is not measured anywhere |
| KazLLM-1.0-8B RAG | 47.9, 59.3, 24.5 (cit., support, halluc.) | **Authors' assumption, informed by [Prototype]** | [Prototype] measured 0.28, 0.66, 0.32. Our citation accuracy target is higher |
| Llama-3.1-8B RAG: citation accuracy stays low (4.6 → 7.2) | `targets.llama` | **Authors' assumption** | [Prototype] measured 0.19 for the Hermes-3 variant of Llama-3.1-8B. The low value is what makes success criterion C3 fail; it was chosen, not observed |
| Qwen3.5-9B RAG | 51.8, 61.0, 22.8 | **Authors' assumption** | [Prototype] measured the older Qwen2.5-7B (0.15, 0.49, 0.48); no measurement of Qwen3.5-9B exists |
| Small models close to each other, gap to GPT-6.1 smaller with RAG | — | **Direction from [Mansurova 2025]** | Size is the authors' assumption |
| EM below 8% | `targets.*[0]` | **Authors' assumption, different from [Prototype]** | [Prototype] reports EM 0.25–0.67 for RAG. Our targets assume strict exact match on free-form answers |
| Legal correctness (mean of 0/0.5/1) and full-correct shares | `targets.*[5]`, `full_correct_share_rag`, `full_correct_share_given_miss` | **Authors' assumption** | Direction from [Mansurova 2025]: some retrieval misses are still answered correctly |
| Abstention rates | `abstain_rate` | **Authors' assumption** | — |
| Prompt language (only KazLLM legal correctness differs, +5.5) | `ru_targets` | **Authors' assumption** | [Prototype] compared Russian and Kazakh RAG for GPT-4.1 only, and changed the language of the context as well. It found the Kazakh setting better (support 0.78 vs 0.65). Our targets assume no difference for GPT-6.1 |
| Number of chunks k = 1, 3, 10 | `topk_targets` | **Authors' assumption** | — |
| Legal correctness per question type | `type_targets_cor` | **Authors' assumption** | — |
| Error categories E2–E7 (mix per model) | `error_probs_E2_E7` | **Authors' assumption** | E1 and E8 follow from the simulated retrieval misses and abstentions |
| Run-to-run variation | `run_redraw_share` | **Authors' assumption** | — |
| Judge agreement κ (0.71, 0.64, 0.78) | `judge.*_target` | **Authors' assumption** | — |

## Statistical pattern

| Item | Source | Comment |
|---|---|---|
| Which comparisons are significant (`DESIGNED`) | **Authors' pre-specification** | It is chosen together with the targets, so the tests illustrate the reporting format and carry no evidence |
| Noise correlations between configurations (`rho_*`, `var_components`, `psi_hal_redraw`) | **Authors' choice** | They were set so that `DESIGNED` holds with seed 2026 (no other seed was tried) |

## Corpus and benchmark

| Item | Source |
|---|---|
| 223,245 raw records, 7,356 removed, 215,889 clean records, period 1947–2026 | **Data paper (real)** |
| 1,200 selected acts; question counts per type and the 100/400 split | **Planned design values** |
| All other corpus statistics (removal breakdown, flagged records, words, chunk counts and lengths, 1,188 parsed acts, 381 questions with linked gold) | **Authors' assumption, simulated**; supplementary table `outputs/tables/tableS1_data_full.tex` only |
