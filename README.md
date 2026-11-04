# Kazakh legal RAG: simulation code and expected results

Code and outputs for the article

> A. Orazgali, Y. Bolatkhan, A. Amirbekov. *Development and Research of Reliable Information Retrieval Methods for Kazakh-Language Question Answering Systems Based on External Knowledge Sources.* Astana IT University, 2026 (manuscript).

> **Important: the current results are EXPECTED values from a seeded simulation, not measurements.**
> No retrieval or generation experiment has been run yet. The script in this repository simulates per-question outcomes of the planned study. Its effect sizes are assumptions taken from the literature cited in the article. It then applies the planned statistical analysis to these simulated scores. The tables and figures have the structure that the final results will have. Every number in them is an expected value, and the statistical tests are not evidence for the hypotheses.
> The full retrieval and generation pipeline (corpus download and cleaning, chunking, indexing, retrieval, generation, LLM-as-a-judge evaluation) and the prompts will be added to this repository when the experiments are run.

## Contents

```
.
├── README.md
├── LICENSE                  MIT (see "Licence")
├── CITATION.cff             how to cite this repository
├── requirements.txt         exact package versions used
├── environment.yml          conda alternative (same versions)
├── sim/
│   └── simulate_results.py  entry point: python sim/simulate_results.py --seed 2026
├── src/
│   ├── simulate.py          the simulation and the planned statistical analysis
│   ├── make_paper_figures.py  draws the figures at their printed size (figures/)
│   ├── make_compact_tables.py writes Tables 1 and 2 of the article (outputs/tables/)
│   └── verify_outputs.py    compares files with outputs/SHA256SUMS
├── outputs/
│   ├── SHA256SUMS           checksums of all reproducible outputs
│   ├── data/                CSV/JSON outputs, configuration, environment, runtime
│   └── tables/              LaTeX tables: compact tables of the article and full tables with 95% CIs
└── figures/                 figures of the article and supplementary figures (vector PDF)
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
python src/make_paper_figures.py             # paper figures -> figures/
python src/make_compact_tables.py            # Tables 1 and 2 of the article -> outputs/tables/
python src/verify_outputs.py                 # compare with the published checksums
```

- **Seed.** The study uses one master seed, 2026. `numpy.random.SeedSequence(2026)` is split into eight independent `Generator(PCG64)` streams: questions, corpus, retrieval, latency, generation, judge, bootstrap and randomization. No other seed was tried. Calibration only changes documented effect parameters. All parameters, and the use of each stream, are written to `outputs/data/sim_config.json`.
- **Runtime.** The published run took 99 s wall-clock (99 s CPU) on an Intel Xeon processor with 2 logical cores (`outputs/data/runtime.txt`). Most of the time goes to the calibration grid searches and to 10,000 bootstrap resamples and 10,000 sign flips per test.
- **Reproducibility check.** With the pinned versions, a new run gives byte-identical files for all files listed in `outputs/SHA256SUMS`: every CSV, `results_summary.json`, `sim_config.json`, the LaTeX tables and the PDF figures. We checked this by running the code from this repository layout and comparing with the outputs used in the article. Two files describe the machine and are expected to differ between computers: `runtime.txt` (wall time) and `versions.txt` (platform string). A run overwrites them in `outputs/data/`. The copies in the repository are the ones from the run reported in the article.
- **Other versions.** Other NumPy, SciPy or pandas versions should give the same numbers in most cases, but this is not guaranteed. PDF bytes depend on the matplotlib and fonttools versions.

`simulate.py` also writes working versions of the figures to `outputs/figures/`, with PNG previews. This folder is git-ignored. `make_paper_figures.py` redraws the figures from the CSV files at their printed size. `fig_errors.pdf` is 13.0 × 4.6 cm with 7 pt text, for inclusion at `\linewidth`. The script also copies `fig1_alpha.pdf` unchanged and writes PNG previews to `figures/preview/`, which is git-ignored. `make_compact_tables.py` writes the two tables of the article from `table1_data.csv`, `table1_questions.csv`, `table2_main.csv` and `tests_all.csv`.

## Output files

All values are expected values from the simulation. Metrics are in %, and MRR and nDCG@10 are multiplied by 100. CIs are 95% percentile bootstrap intervals from 10,000 resamples of questions.

| File | Content |
|---|---|
| `outputs/data/table1_data.csv` | Corpus and chunking statistics (item, value, status: `reported` / `planned` / `expected`) |
| `outputs/data/table1_questions.csv` | Benchmark questions by type and split (all, dev, test, test with linked gold evidence) |
| `outputs/data/table2_main.csv` | Main results: one row per configuration × metric with value, CI and n. Panel A: chunking (RQ1). Panel B: retrievers (RQ2). Panel C: generators LLM-only vs. RAG (RQ3). `C-supp`: abstention, unverifiable-only and fully-correct shares |
| `outputs/data/tests_all.csv` | All 56 planned paired tests. For each: family (F1–F6, plus F2s), comparison, metric, Δ in pp with paired-bootstrap CI, test, raw and Holm-adjusted p, significance, discordant pairs for McNemar |
| `outputs/data/table4_additional.csv` | Additional analyses: cleaning ablation (A), number of chunks k (B), per question type (C) |
| `outputs/data/fig1_alpha.csv` | Recall@10, MRR and nDCG@10 of the linear hybrid on the dev set for α = 0, 0.1, …, 1, and RRF |
| `outputs/data/fig2_prompt.csv` | Legal correctness and hallucination with Kazakh vs. Russian system prompt (RAG), with CIs and F4 test results |
| `outputs/data/fig3_errors.csv` | Error taxonomy: failed answers per model and primary category E1–E8 |
| `outputs/data/run_sd.csv` | Legal correctness of the main RAG configurations for generation seeds 13, 42, 2026, and their SD |
| `outputs/data/judge_agreement.csv` | Cohen's κ (judge vs. human consensus, inter-annotator) on 100 verdicts, with bootstrap CIs |
| `outputs/data/retrieval_answer_2x2.csv` | Gold chunk in top 5 vs. fully correct answer, per model |
| `outputs/data/per_question_retrieval.csv` | Simulated per-question retrieval scores and latencies (381 test questions with gold); the unit of all retrieval analyses |
| `outputs/data/per_question_generation.csv` | Simulated per-question generation outcomes (400 test questions) per model and variant, including the error category |
| `outputs/data/results_summary.json` | All of the above in one file, plus the internal consistency checks (`checks`) |
| `outputs/data/sim_config.json` | Design values, assumed expected values (targets), noise parameters, calibrated parameters, seed-stream mapping |
| `outputs/data/versions.txt`, `runtime.txt` | Python and package versions, platform, CPU, wall/CPU time, SHA-256 of `results_summary.json` |
| `outputs/tables/table_data.tex`, `table_main.tex` | Tables 1 and 2 of the article. Point values only. In Table 2, bold marks the best value per panel column and † marks a Holm-significant difference from the panel's reference configuration, taken from `tests_all.csv` |
| `outputs/tables/table1_data.tex` … `table4_additional.tex` | Full LaTeX versions of the data, main, tests and additional-analyses tables (booktabs, with 95% CIs) written by `simulate.py`; supplementary |
| `figures/fig_errors.pdf` | Error-taxonomy figure of the article |
| `figures/fig1_alpha.pdf`, `fig2_prompt_language.pdf`, `fig3_error_taxonomy.pdf` | Supplementary figures: α sensitivity, prompt language, and the larger error-taxonomy version |

### Mapping to the article

| Article | Source in this repository |
|---|---|
| Table 1 (`tab:data`), data used | `outputs/tables/table_data.tex` ← `table1_data.csv`, `table1_questions.csv` (status r = reported in the KazakhLawCorpus data paper, p = planned, e = expected) |
| Table 2 (`tab:main`), main results for chunking, retrieval and generation | `outputs/tables/table_main.tex` ← `table2_main.csv` (values; the 95% CIs are in the same file) and `tests_all.csv`, families F1–F3 († markers) |
| Fig. 2 (`fig:errors`), error taxonomy | `figures/fig_errors.pdf` ← `fig3_errors.csv` |
| Numbers in the Results text: Δ and p-values, α choice, RRF, reranker, prompt language, cleaning ablation, top-k, per type, run-to-run SD, judge κ, retrieval–answer 2×2 | `tests_all.csv` (all 56 tests, F1–F6), `fig1_alpha.csv`, `fig2_prompt.csv`, `table4_additional.csv`, `run_sd.csv`, `judge_agreement.csv`, `retrieval_answer_2x2.csv` |
| Supplementary, not printed in the shortened article | `figures/fig1_alpha.pdf` (α sensitivity), `figures/fig2_prompt_language.pdf`, `figures/fig3_error_taxonomy.pdf`, `outputs/tables/table1_data.tex` … `table4_additional.tex` (full tables with CIs, including the pairwise tests and the additional analyses) |

Fig. 1 of the article (pipeline diagram) is drawn in LaTeX and is not produced by this code. All values in the article's tables and text are copied from the CSV files without change.

## Data

- **KazakhLawCorpus (metadata).** Hugging Face: <https://huggingface.co/datasets/Arailym-tleubayeva/KazakhLawCorpus>. Data paper: Tleubayeva, A., Mansurova, A., Shomanov, A., Makhambetova, Zh., Boluk, P. S. (2026). KazakhLawCorpus: A Large-Scale Metadata Corpus for Kazakh Legislation. *Data*, 11(9), 217. <https://doi.org/10.3390/data11090217>.
  - As reported in the data paper, the raw release has 223,245 records (one per legal act, about 48 columns, Kazakh and Russian fields). The clean release has 215,889 records and 25 Kazakh-language fields.
  - The data come from a NUBA Solutions LLP export of mainly Adilet.zan.kz and cover acts adopted in 1947–2026.
  - The licence is CC BY 4.0, as stated in the data paper. *We could not open the dataset card from our working environment to re-check the licence and row count. TEAM: confirm on the dataset card.*
  - The corpus is not redistributed here. Download it from Hugging Face.
- **Full texts of legal acts.** The planned study will download the full texts of the selected acts (about 1,200) from the public Adilet legal information system of the Republic of Kazakhstan (<https://adilet.zan.kz>), following the site's terms of use. It will use the document addresses stored in the corpus and record the download date. The texts are not part of this repository. The simulation does not use them.
- **Benchmark (500 questions).** The benchmark has 500 Kazakh legal questions in four types (norm identification, interpretation, applicability conditions, procedural requirements) with reference answers and linked gold provisions. It is not included in this repository. *TEAM: confirm the origin of the questions, who may access them, and under which licence or conditions they can be shared (for example, release with the experiment code, or on request).* The simulation uses only its design values: the number of questions per type, the 100/400 dev/test split and the number of questions with linked gold evidence.
- **Simulated outputs.** All files in `outputs/` and `figures/` are released under the repository licence.

## Citation

If you use this repository, please cite the article (full reference to be added after publication) and the repository (see `CITATION.cff`):

```bibtex
@misc{orazgali2026kazakhlegalrag_code,
  author       = {Orazgali, Azamat and Bolatkhan, Yerassyl and Amirbekov, Adilzhan},
  title        = {Code and expected results for: Development and Research of Reliable Information Retrieval Methods for {K}azakh-Language Question Answering Systems Based on External Knowledge Sources},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/REPLACE-ME/kazakh-legal-rag},
  note         = {Version 0.1.0. Expected values from a seeded simulation (seed 2026)}
}
```

Please also cite KazakhLawCorpus (the data paper above) when you use the corpus.

## Licence

The code and the generated outputs are released under the MIT licence (`LICENSE`). Copyright (c) 2026 Azamat Orazgali, Yerassyl Bolatkhan, Adilzhan Amirbekov.

> **TEAM: confirm the licence choice (MIT) before the first public push.** The same licence must be named in the Code Availability statement of the article. Third-party data (KazakhLawCorpus, Adilet texts) keep their own licences and terms.
