# IEEE conference paper: ML rankers vs. calendar-month seasonality

**Draft title:** *Do Machine Learning Rankers Rescue Calendar-Month Seasonality? A Point-in-Time Walk-Forward Evaluation of Seasonal Stock Selection*

Target: an IEEE conference such as IEEE CIFEr (Computational Intelligence for Financial Engineering & Economics), in IEEEtran conference format, about 6 pages.

## Layout

```
experiments.py   all numbers in the paper (≈30 s, reads ../output/backtest_monthly_cache.pkl)
figures.py       the 4 figures -> latex/figures/*.pdf
results/         CSV/JSON outputs that every table is copied from
latex/main.tex   the paper
latex/references.bib
```

## Reproduce

```bash
cd seasonal_options_system
venv/bin/python3 paper/experiments.py
venv/bin/python3 paper/figures.py
```

Compile on Overleaf by uploading the `latex/` folder (a New Project, then Upload Project with a zip of it), or build locally with `~/.local/bin/tectonic latex/main.tex` (Tectonic 0.17.0, a single binary; delete that file to uninstall). The latest local build is `ieee_paper_draft.pdf`. Compile the final submission with pdfLaTeX on Overleaf so IEEEtran uses true Times; local XeTeX builds substitute the Times clone TeX Gyre Termes, and the title falls back to Latin Modern.

## Before submitting: your checklist

1. **Author block**: put your name, affiliation, and email in `main.tex`. Check whether the venue uses double-blind review. If it does, anonymize the author block and the acknowledgment.
2. **AI disclosure**: IEEE requires AI-generated content to be disclosed in the acknowledgments. A draft disclosure is there; edit it to match exactly how you used AI.
3. **References**: every entry in `references.bib` is a real paper, but check the volume, pages, and DOI against the publisher's site, and add DOIs.
4. **Venue rules**: page limit, deadline, and template version come from the specific call for papers.
5. **Survivorship bias** is the biggest weakness a reviewer will raise: the universe is 210 names chosen on 2026-10-06. If you have access to CRSP or point-in-time S&P 500 membership through a university library, re-running on that would make the paper much stronger.
6. **Read every sentence yourself**: you are the author and responsible for every claim.
