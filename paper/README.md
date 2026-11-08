# IEEE conference paper: ML rankers vs. calendar-month seasonality

**Draft title:** *Do Machine Learning Rankers Rescue Calendar-Month Seasonality? A Point-in-Time Walk-Forward Evaluation of Seasonal Stock Selection*

Repository: https://github.com/smvijaykumar/seasonal_options_system

Target: an IEEE conference such as IEEE CIFEr (Computational Intelligence for Financial Engineering & Economics), in IEEEtran conference format, about 6 pages.

## Layout

```
experiments.py        all numbers in the paper (≈30 s)
figures.py            the 4 figures -> latex/figures/*.pdf
data/universe.csv     the 210 stocks + SPY studied
data/monthly_closes.pkl  the exact monthly prices behind the published results (2 MB)
results/              CSV/JSON outputs that every table is copied from
latex/main.tex        the paper
latex/references.bib
requirements.txt      library versions used
```

## Reproduce

From a fresh clone of the repository (Python 3.11):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r paper/requirements.txt
python3 paper/experiments.py
python3 paper/figures.py
```

This uses `data/monthly_closes.pkl` and reproduces `results/` exactly. To rebuild the data from scratch instead, run `python3 paper/experiments.py --refresh --out /tmp/fresh_results`. It re-downloads every ticker in `data/universe.csv` from Yahoo Finance, so `--out` keeps the published results untouched. Yahoo revises adjusted prices and drops delisted tickers (WBD had no history as of 2026-11-07), so a fresh download matches the method but not every decimal. In a test on 2026-11-07, the main strategies' excess returns over the benchmark moved by less than 0.02 %/month. The noisier "+Filter" variants and the November-only GBDT figure moved by up to about 0.2 %/month, but no result became statistically significant, and every conclusion in the paper held.

`monthly_closes.pkl` is a Python pickle. Only load pickles from a source you trust.

Compile on Overleaf by uploading the `latex/` folder (a New Project, then Upload Project with a zip of it), or build locally with `~/.local/bin/tectonic latex/main.tex` (Tectonic 0.17.0, a single binary; delete that file to uninstall). The latest local build is `ieee_paper_draft.pdf`. Compile the final submission with pdfLaTeX on Overleaf so IEEEtran uses true Times; local XeTeX builds substitute the Times clone TeX Gyre Termes, and the title falls back to Latin Modern.

## Before submitting: your checklist

1. **Author block**: put your name, affiliation, and email in `main.tex`. Check whether the venue uses double-blind review. If it does, anonymize the author block and the acknowledgment.
2. **AI disclosure**: IEEE requires AI-generated content to be disclosed in the acknowledgments. A draft disclosure is there; edit it to match exactly how you used AI.
3. **References**: every entry in `references.bib` is a real paper, but check the volume, pages, and DOI against the publisher's site, and add DOIs.
4. **Venue rules**: page limit, deadline, and template version come from the specific call for papers.
5. **Survivorship bias** is the biggest weakness a reviewer will raise: the universe is 210 names chosen on 2026-10-06. If you have access to CRSP or point-in-time S&P 500 membership through a university library, re-running on that would make the paper much stronger.
6. **Read every sentence yourself**: you are the author and responsible for every claim.
