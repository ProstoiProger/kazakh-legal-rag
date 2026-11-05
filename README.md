# Kazakh legal RAG: simulation code and assumed results

Code and outputs for the article

> A. Orazgali, Y. Bolatkhan, A. Amirbekov. *Development and Research of Reliable Information Retrieval Methods for Kazakh-Language Question Answering Systems Based on External Knowledge Sources.* Astana IT University, 2026 (manuscript).

## Disclosure

> **No experiment has been run. The repository contains no empirical information.**
>
> - Every reported mean is an author-specified target (`CONFIG['retrieval']['targets']`, `CONFIG['generation']['targets']` and the other `*_targets` entries in `src/simulate.py`, also written to `outputs/data/sim_config.json`). The simulation reaches each target by calibration.
> - The pattern of significant and non-significant tests is pre-specified in `DESIGNED` (`src/simulate.py`). The noise correlations were chosen so that this pattern holds; see the column `matches_design` in `outputs/data/tests_all.csv`.
> - The outputs illustrate the planned analysis and contain no empirical information. Confidence intervals and p-values describe resampling noise within one simulated data set. They are not evidence for or against any hypothesis.
> - [`TARGETS_SOURCES.md`](TARGETS_SOURCES.md) maps every group of targets to its source. It states plainly which targets are the authors' own assumptions, and where they differ from the measured results of our earlier prototype study.
> - The full retrieval and generation pipeline will be added to this repository when the experiments are run. It covers corpus download and cleaning, chunking, indexing, retrieval, generation and LLM-as-a-judge evaluation, together with the prompts.

## Contents

```
.
├── README.md
├── TARGETS_SOURCES.md       source or assumption behind every target
├── LICENSE                  MIT
├── CITATION.cff             how to cite this repository
├── requirements.txt         exact package versions used
├── environment.yml          conda alternative (same versions)
├── sim/
│   └── simulate_results.py  entry point: python sim/simulate_results.py --seed 2026
├── src/
│   ├── simulate.py          targets, calibration, simulation and the planned statistical analysis
│   ├── make_paper_figures.py  draws the figures at their printed size (figures/)
│   ├── make_compact_tables.py writes Tables 1 and 2 of the article (outputs/tables/)
│   └── verify_outputs.py    compares files with outputs/SHA256SUMS
├── outputs/
│   ├── SHA256SUMS           checksums of all reproducible outputs
│   ├── data/                CSV/JSON outputs, configuration, environment, runtime
│   └── tables/              Tables 1 and 2 of the article and supplementary full tables (S1–S4)
└── figures/                 Fig. 2 of the article and supplementary figures S1–S3 (vector PDF)
```

## Installation

Python 3.13 is required. The published outputs were made with Python 3.13.17.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You can use conda instead: `conda env create -f environment.yml && conda activate kazakh-legal-rag-sim`.

## How to reproduce

Run all commands from the repository root:

```bash
python sim/simulate_results.py --seed 2026   # simulation and analysis -> outputs/ (about 100 s)
python src/make_paper_figures.py             # figures -> figures/ (previews -> outputs/figures/preview/)
python src/make_compact_tables.py            # Tables 1 and 2 of the article -> outputs/tables/
python src/verify_outputs.py                 # compare with the published checksums
```

- **Seed.** The study uses one master seed, 2026. `numpy.random.SeedSequence(2026)` is split into eight independent `Generator(PCG64)` streams: questions, corpus, retrieval, latency, generation, judge, bootstrap and randomization. No other seed was tried. Calibration only changes documented effect parameters so that the targets are reached. All parameters, and the use of each stream, are written to `outputs/data/sim_config.json`.
- **Runtime.** The run that produced the current outputs took 99.7 s wall-clock (99.2 s CPU) on an Intel Xeon processor with 2 logical cores (`outputs/data/runtime.txt`). Most of the time goes to the calibration grid searches and to 10,000 bootstrap resamples and 10,000 sign flips per test.
- **Reproducibility check.** With the pinned versions, a new run gives byte-identical files for all files listed in `outputs/SHA256SUMS`: every CSV, `results_summary.json`, `sim_config.json`, the LaTeX tables and the PDF figures.
  - We checked this by regenerating everything from an empty `outputs/` and `figures/`.
  - Two files describe the machine and are expected to differ between computers: `runtime.txt` (wall time) and `versions.txt` (platform string).
- **Other versions.** Other NumPy, SciPy or pandas versions should give the same numbers in most cases, but this is not guaranteed. PDF bytes depend on the matplotlib and fonttools versions.

`simulate.py` also writes working figures and full LaTeX tables with 95% CIs. The working figures go to `outputs/figures/`, which is git-ignored. `make_paper_figures.py` redraws the figures from the CSV files at their printed size. `fig_errors.pdf` is 10.4 × 6.4 cm with 7 pt text, for inclusion at `0.8\linewidth`. `make_compact_tables.py` writes Tables 1 and 2 of the article from `table1_questions.csv`, `table2_main.csv`, `tests_all.csv` and `run_sd.csv`.

## Output files

All values are assumed targets reproduced by the simulation, not measurements. Metrics are ×100. CIs are 95% percentile bootstrap intervals from 10,000 resamples of questions.

| File | Content |
|---|---|
| `outputs/data/table1_data.csv` | Corpus and chunking statistics (item, value, status). Status is `reported` (data paper), `planned` or `expected` (simulated) |
| `outputs/data/table1_questions.csv` | Benchmark questions by type and split |
| `outputs/data/table2_main.csv` | Main results, one row per configuration × metric, with value, CI and n. Panel A: chunking. Panel B: retrievers. Panel C: generators, LLM-only vs. RAG. `C-supp`: abstention, unverifiable-only and fully-correct shares |
| `outputs/data/tests_all.csv` | All 56 planned paired tests (families F1–F6, plus the supplementary test F2s). For each test: Δ in pp with paired-bootstrap CI, raw and Holm-adjusted p (6 significant digits), significance, McNemar discordant pairs, the designed outcome (`designed`) and `matches_design` |
| `outputs/data/power_mde.csv` | Minimum detectable difference of every planned test at 80% power, under the assumed per-question variance: MDE = (z₁₋α′/2 + z₀.₈)·SD(paired differences)/√n, α′ = 0.05/family size |
| `outputs/data/halluc_answered.csv` | Hallucination rate among answered (non-abstained) questions for every model, setting and prompt language |
| `outputs/data/table4_additional.csv` | Cleaning ablation, number of chunks k, per question type |
| `outputs/data/fig1_alpha.csv` | Recall@10, MRR and nDCG@10 of the linear hybrid on the dev set for α = 0, 0.1, …, 1, and RRF |
| `outputs/data/fig2_prompt.csv` | Legal correctness and hallucination with a Kazakh vs. a Russian system prompt (RAG), with F4 test results |
| `outputs/data/fig3_errors.csv` | Error taxonomy: failed answers per model and primary category E1–E8 |
| `outputs/data/run_sd.csv` | Legal correctness of the main RAG configurations for generation seeds 13, 42 and 2026, and their SD |
| `outputs/data/judge_agreement.csv` | Cohen's κ (judge vs. human consensus, inter-annotator) on 100 verdicts |
| `outputs/data/retrieval_answer_2x2.csv` | Gold provision in the top 5 (hit@5) vs. fully correct answer, per model |
| `outputs/data/per_question_retrieval.csv`, `per_question_generation.csv` | Simulated per-question scores: the unit of all analyses |
| `outputs/data/results_summary.json` | All main outputs in one file, plus the internal consistency checks (`checks`) |
| `outputs/data/sim_config.json` | Design values, **targets**, noise parameters, calibrated parameters and seed-stream mapping |
| `outputs/data/versions.txt`, `runtime.txt` | Python and package versions, platform, CPU, wall/CPU time, SHA-256 of `results_summary.json` |
| `outputs/tables/table_data.tex`, `table_main.tex` | **Identical to Tables 1 and 2 of the article.** Table 1 contains no simulated values. In Table 2, bold marks the best value per panel column among the configurations kept by the decision rules, and † marks a Holm-significant difference from the panel's reference, taken from `tests_all.csv` |
| `outputs/tables/tableS1_data_full.tex` … `tableS4_additional_full.tex` | Supplementary full tables with 95% CIs: data, main results, all F1–F4 tests, additional analyses |
| `figures/fig_errors.pdf` | **Fig. 2 of the article** (error taxonomy) |
| `figures/figS1_alpha.pdf`, `figS2_prompt_language.pdf`, `figS3_error_taxonomy_full.pdf` | Supplementary figures: α sensitivity on the dev set, prompt language, and a larger version of the error taxonomy |

### Mapping to the article

| Article | Source in this repository |
|---|---|
| Table 1 (`tab:data`), data and planned settings | `outputs/tables/table_data.tex` (counts reported in the KazakhLawCorpus data paper and planned design values; no simulated values) |
| Table 2 (`tab:main`), main results for chunking, retrieval and generation | `outputs/tables/table_main.tex` ← `table2_main.csv` (values; the 95% CIs are in the same file) and `tests_all.csv`, families F1–F3 († markers) |
| Fig. 1, pipeline | drawn in LaTeX (TikZ); not produced by this code |
| Fig. 2 (`fig:errors`), error taxonomy | `figures/fig_errors.pdf` ← `fig3_errors.csv` |
| Numbers in the Results text: Δ and p-values, α choice, RRF, reranker, prompt language, cleaning ablation, top-k, per type, run-to-run SD, judge κ, hit@5, minimum detectable differences, hallucination among answered questions | `tests_all.csv` (all 56 tests), `fig1_alpha.csv`, `fig2_prompt.csv`, `table4_additional.csv`, `run_sd.csv`, `judge_agreement.csv`, `retrieval_answer_2x2.csv`, `power_mde.csv`, `halluc_answered.csv` |
| Supplementary, not printed in the article | `figures/figS1_alpha.pdf`, `figS2_prompt_language.pdf`, `figS3_error_taxonomy_full.pdf`; `outputs/tables/tableS1`–`S4` |

All values in the article's tables and text are copied from these files without change. Differences quoted in the article text are computed from the rounded table values.

## Data

- **KazakhLawCorpus (metadata).**
  - Hugging Face: <https://huggingface.co/datasets/Arailym-tleubayeva/KazakhLawCorpus>.
  - Data paper: Tleubayeva, A., Mansurova, A., Shomanov, A., Makhambetova, Zh., Boluk, P. S. (2026). KazakhLawCorpus: A Large-Scale Metadata Corpus for Kazakh Legislation. *Data*, 11(9), 217. <https://doi.org/10.3390/data11090217>.
  - The data paper reports a raw release of 223,245 records (one per legal act, about 48 columns, with Kazakh and Russian fields) and a clean release of 215,889 records with 25 Kazakh-language fields.
  - The data come from a NUBA Solutions LLP export of mainly Adilet.zan.kz and cover acts adopted in 1947–2026.
  - The license is CC BY 4.0, as stated in the data paper.
  - The corpus is not redistributed here.
- **Full texts of legal acts.**
  - The planned study will download the full texts of the selected acts (about 1,200) from the public Adilet legal information system of the Republic of Kazakhstan (<https://adilet.zan.kz>), following the site's terms of use.
  - It will use the document addresses stored in the corpus, take each act in its current consolidated version on the download date, and record that date.
  - The texts are not part of this repository, and the simulation does not use them.
- **Benchmark (500 questions).**
  - The benchmark has 500 Kazakh legal questions in four types (norm identification, interpretation, applicability conditions, procedural requirements), with reference answers and linked gold provisions.
  - It is not included in this repository; its release conditions are listed under "Before publishing" below.
  - The simulation uses only its design values: the number of questions per type and the 100/400 dev/test split.
- **Simulated outputs.** All files in `outputs/` and `figures/` are released under the repository license.

## Citation

If you use this repository, please cite the article (full reference to be added after publication) and the repository (see `CITATION.cff`):

```bibtex
@misc{orazgali2026kazakhlegalrag_code,
  author       = {Orazgali, Azamat and Bolatkhan, Yerassyl and Amirbekov, Adilzhan},
  title        = {Code and assumed results for: Development and Research of Reliable Information Retrieval Methods for {K}azakh-Language Question Answering Systems Based on External Knowledge Sources},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/REPLACE-ME/kazakh-legal-rag},
  note         = {Version 0.1.0. Seeded simulation (seed 2026) of the planned study; no measured results}
}
```

Please also cite KazakhLawCorpus (the data paper above) when you use the corpus.

## License

The code and the generated outputs are released under the MIT license (`LICENSE`). Copyright (c) 2026 Azamat Orazgali, Yerassyl Bolatkhan, Adilzhan Amirbekov. Third-party data (KazakhLawCorpus, Adilet texts) keep their own licenses and terms.

## Release

These steps create the v0.1.0 release:

```bash
git init && git add . && git commit -m "v0.1.0: simulation and expected outputs" \
  && git tag -a v0.1.0 -m "v0.1.0" && git push --follow-tags
```

Before the push, add the GitHub remote with `git remote add origin <url>` and `git branch -M main`. Then:

1. Enable the Zenodo GitHub integration for the repository, and publish a GitHub release from the tag `v0.1.0` so that Zenodo archives it.
2. Add the Zenodo DOI under `identifiers` in `CITATION.cff`, in this README and in the Code availability statement of the article.

## Before publishing (checklist for the authors)

- [ ] Replace `https://github.com/REPLACE-ME/kazakh-legal-rag` in this README and in `CITATION.cff` with the real URL. It must be the same URL as `\repourl` in the article.
- [ ] Confirm the MIT license choice. The same license must be named in the Code availability statement.
- [ ] Re-check the KazakhLawCorpus dataset card (license and row count). It could not be opened from the authors' build environment.
- [ ] Decide on the benchmark release: its origin, who may access it, and its license or access conditions.
- [ ] Create the v0.1.0 tag and the Zenodo DOI (see "Release").
