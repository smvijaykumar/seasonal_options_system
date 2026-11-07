#!/usr/bin/env python3
"""Experiments for the IEEE paper.

Walk-forward, point-in-time evaluation of seasonal cross-sectional stock
selection over every calendar month 2005-01 .. 2025-12, comparing:

  * benchmarks: SPY, equal-weight eligible universe (EW), 12-1 momentum top-N
  * rule-based: seasonal filter (all names passing the hard filters),
                z-score composite (the production system's Stage 2+3 ranking)
  * learned:    logistic regression, gradient-boosted trees, MLP -- trained on
                an expanding window of strictly earlier target months,
                retrained every January, ranking the whole eligible universe
                and (variant) only the filter-passing names.

All features at target month t use only data available at the close of t-1.
Outputs go to paper/results/.
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
CACHE = HERE.parent / "output" / "backtest_monthly_cache.pkl"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

TOP_N = 10
MIN_YEARS = 15
COST_BPS = 15.0
TRAIN_START = pd.Timestamp("1995-01-01")
TEST_START = pd.Timestamp("2005-01-01")
TEST_END = pd.Timestamp("2025-12-01")
SEED = 7

# Production-system weights (config.yaml), RSI term dropped (no historical RSI).
W = {"xs_wr": 1.0, "xs_avg": 1.0, "m1_avg": 0.5, "m3_avg": 0.5, "neglogp": 0.5}

FEATURES = ["m2_avg", "m2_wr", "m2_std", "m2_n", "m1_avg", "m3_avg", "neglogp",
            "xs_wr", "xs_avg", "rev_1", "mom_12_1", "vol_12"]


# ---------------------------------------------------------------- data
def load_returns():
    monthly = pickle.load(open(CACHE, "rb"))
    closes = pd.DataFrame(monthly).sort_index()
    closes.index = pd.to_datetime(closes.index)
    closes = closes[closes.index <= TEST_END]
    rets = closes.pct_change(fill_method=None) * 100.0
    return closes, rets


def seasonal_stats(rets):
    """For each (year, month, ticker): stats of that calendar month's returns
    over years strictly before `year`. Returns dict month -> dict of frames
    indexed by year."""
    out = {}
    for m in range(1, 13):
        rm = rets[rets.index.month == m].copy()
        rm.index = rm.index.year
        pos = (rm > 0).astype(float).where(rm.notna())
        out[m] = {
            "avg": rm.expanding().mean().shift(1),
            "wr": pos.expanding().mean().shift(1) * 100,
            "std": rm.expanding().std().shift(1),
            "n": rm.notna().cumsum().shift(1),
        }
    return out


def label(avg, wr, n):
    lab = np.where((avg >= 1.5) & (wr >= 55), "Strong",
                   np.where((avg <= -1.5) & (wr <= 45), "Weak", "Neutral"))
    return np.where(n >= 5, lab, "No data")


def build_panel(closes, rets):
    st = seasonal_stats(rets)
    dates = rets.index[(rets.index >= TRAIN_START) & (rets.index <= TEST_END)]
    rows = []
    for t in dates:
        y, m = t.year, t.month
        m1 = 12 if m == 1 else m - 1
        m3 = 1 if m == 12 else m + 1
        i = rets.index.get_loc(t)
        if i < 13:
            continue
        s2 = {k: st[m][k].loc[y] for k in st[m]} if y in st[m]["avg"].index else None
        if s2 is None:
            continue
        s1 = {k: st[m1][k].loc[y] for k in st[m1]} if y in st[m1]["avg"].index else None
        s3 = {k: st[m3][k].loc[y] for k in st[m3]} if y in st[m3]["avg"].index else None
        df = pd.DataFrame({
            "m2_avg": s2["avg"], "m2_wr": s2["wr"], "m2_std": s2["std"], "m2_n": s2["n"],
            "m1_avg": s1["avg"] if s1 else np.nan, "m1_wr": s1["wr"] if s1 else np.nan,
            "m1_n": s1["n"] if s1 else np.nan,
            "m3_avg": s3["avg"] if s3 else np.nan, "m3_wr": s3["wr"] if s3 else np.nan,
            "m3_n": s3["n"] if s3 else np.nan,
        })
        # non-seasonal controls, known at close of t-1
        df["rev_1"] = rets.iloc[i - 1]
        df["mom_12_1"] = (closes.iloc[i - 2] / closes.iloc[i - 13] - 1) * 100
        df["vol_12"] = rets.iloc[i - 12:i].std()
        df["fwd_ret"] = rets.iloc[i]
        df = df[(df["m2_n"] >= MIN_YEARS) & df["fwd_ret"].notna()
                & df[["rev_1", "mom_12_1", "vol_12"]].notna().all(axis=1)]
        df = df[df.index != "SPY"]
        if len(df) < 2 * TOP_N:
            continue
        df["date"] = t
        rows.append(df.rename_axis("ticker").reset_index())
    p = pd.concat(rows, ignore_index=True)

    # point-in-time derived features (cross-sectional, within month)
    g = p.groupby("date")
    p["xs_wr"] = p["m2_wr"] - g["m2_wr"].transform("median")
    p["xs_avg"] = p["m2_avg"] - g["m2_avg"].transform("median")
    k = np.round(p["m2_wr"] / 100 * p["m2_n"]).astype(int)
    pval = binom.sf(k - 1, p["m2_n"].astype(int), 0.5)
    p["neglogp"] = -np.log10(np.clip(pval, 1e-12, 1))
    p["m1_label"] = label(p["m1_avg"], p["m1_wr"], p["m1_n"].fillna(0))
    p["m2_label"] = label(p["m2_avg"], p["m2_wr"], p["m2_n"])
    p["m3_label"] = label(p["m3_avg"], p["m3_wr"], p["m3_n"].fillna(0))
    p["passes_filter"] = ((p["m2_label"] == "Strong") & (p["m1_label"] != "Weak")
                          & (p["m3_label"] != "Weak") & (p["xs_wr"] > 0))
    p["y"] = (p["fwd_ret"] > g["fwd_ret"].transform("median")).astype(int)
    for c in ["m1_avg", "m3_avg"]:
        p[c] = p[c].fillna(g[c].transform("median"))
    return p


# ---------------------------------------------------------------- scorers
def z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0


def zscore_composite(d):
    return (W["xs_wr"] * z(d["xs_wr"]) + W["xs_avg"] * z(d["xs_avg"])
            + W["m1_avg"] * z(d["m1_avg"]) + W["m3_avg"] * z(d["m3_avg"])
            + W["neglogp"] * z(d["neglogp"]))


def xs_rank(p, cols):
    """Cross-sectional percentile rank of each feature within its month --
    removes level shifts between regimes and makes features comparable."""
    return p.groupby("date")[cols].rank(pct=True)


def make_models():
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        "GBDT": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                               min_samples_leaf=200, l2_regularization=1.0,
                                               random_state=SEED),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3,
                                                            max_iter=300, early_stopping=True,
                                                            random_state=SEED)),
    }


def walk_forward_ml(p):
    X = xs_rank(p, FEATURES).values
    y = p["y"].values
    preds = {name: np.full(len(p), np.nan) for name in make_models()}
    for year in range(TEST_START.year, TEST_END.year + 1):
        tr = (p["date"] < pd.Timestamp(f"{year}-01-01")).values
        te = (p["date"].dt.year == year).values
        for name, model in make_models().items():
            model.fit(X[tr], y[tr])
            preds[name][te] = model.predict_proba(X[te])[:, 1]
    for name, v in preds.items():
        p[f"score_{name}"] = v
    return p, list(preds)


# ---------------------------------------------------------------- portfolios
def portfolios(p, ml_names, spy):
    test = p[p["date"] >= TEST_START]
    recs, ics, pooled = [], [], {"ZScore": [], **{n: [] for n in ml_names}}
    cost = COST_BPS / 100.0
    for t, d in test.groupby("date"):
        r = {"date": t, "SPY": spy.get(t, np.nan), "EW": d["fwd_ret"].mean()}
        r["Momentum"] = d.nlargest(TOP_N, "mom_12_1")["fwd_ret"].mean() - cost
        f = d[d["passes_filter"]].copy()
        r["n_filter"] = len(f)
        r["SeasonalFilter"] = f["fwd_ret"].mean() - cost if len(f) else np.nan
        if len(f):
            f["zs"] = zscore_composite(f)
            r["ZScore"] = f.nlargest(TOP_N, "zs")["fwd_ret"].mean() - cost
            pooled["ZScore"] += list(zip(f["zs"], f["fwd_ret"] - cost))
        else:
            r["ZScore"] = np.nan
        ic = {"date": t}
        for n in ml_names:
            col = f"score_{n}"
            r[n] = d.nlargest(TOP_N, col)["fwd_ret"].mean() - cost
            r[f"{n}+Filter"] = (f.nlargest(TOP_N, col)["fwd_ret"].mean() - cost) if len(f) else np.nan
            ic[n] = spearmanr(d[col], d["fwd_ret"]).statistic
            pooled[n] += list(zip(d[col], d["fwd_ret"] - cost))
        dz = d.copy()
        dz["zs_all"] = zscore_composite(dz)
        ic["ZScore"] = spearmanr(dz["zs_all"], dz["fwd_ret"]).statistic
        ic["Momentum"] = spearmanr(d["mom_12_1"], d["fwd_ret"]).statistic
        recs.append(r)
        ics.append(ic)
    port = pd.DataFrame(recs).set_index("date")
    # months with no filter-passing names: strategy holds cash (0%) -- stated in paper
    for c in ["SeasonalFilter", "ZScore"] + [f"{n}+Filter" for n in ml_names]:
        port[f"{c}_invested"] = port[c].notna()
        port[c] = port[c].fillna(0.0)
    return port, pd.DataFrame(ics).set_index("date"), pooled


def nw_tstat(x, lags=3):
    x = pd.Series(x).dropna()
    res = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.params[0]), float(res.tvalues[0])


def bootstrap_ci(x, n=5000, block=6, seed=SEED):
    """Stationary-ish moving-block bootstrap CI for the mean."""
    x = np.asarray(pd.Series(x).dropna())
    rng = np.random.default_rng(seed)
    T = len(x)
    means = []
    for _ in range(n):
        idx = []
        while len(idx) < T:
            s = rng.integers(0, T - block + 1)
            idx.extend(range(s, s + block))
        means.append(x[idx[:T]].mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def metrics(port, cols, bench="EW"):
    rows = []
    for c in cols:
        r = port[c] / 100
        cum = (1 + r).cumprod()
        dd = (cum / cum.cummax() - 1).min()
        ex = port[c] - port[bench]
        m_ex, t_ex = nw_tstat(ex)
        lo, hi = bootstrap_ci(ex)
        rows.append({
            "strategy": c,
            "mean_pct": port[c].mean(),
            "ann_ret_pct": (cum.iloc[-1] ** (12 / len(r)) - 1) * 100,
            "ann_vol_pct": r.std() * np.sqrt(12) * 100,
            "sharpe": r.mean() / r.std() * np.sqrt(12),
            "max_dd_pct": dd * 100,
            "hit_rate": (port[c] > 0).mean(),
            "beat_bench": (ex > 0).mean() if c != bench else np.nan,
            "excess_pct": m_ex if c != bench else np.nan,
            "excess_t_nw": t_ex if c != bench else np.nan,
            "excess_ci_lo": lo if c != bench else np.nan,
            "excess_ci_hi": hi if c != bench else np.nan,
        })
    return pd.DataFrame(rows)


def calibration(pooled, name, bins=10):
    s, r = map(np.array, zip(*pooled[name]))
    w = (r > 0).astype(float)
    q = pd.qcut(s, bins, labels=False, duplicates="drop")
    tab = pd.DataFrame({"q": q, "s": s, "w": w, "r": r}).groupby("q").agg(
        score=("s", "mean"), win=("w", "mean"), ret=("r", "mean"), n=("w", "size"))
    iso = IsotonicRegression(out_of_bounds="clip").fit(s, w)
    return tab, iso


# ---------------------------------------------------------------- main
def main():
    closes, rets = load_returns()
    spy = rets["SPY"]
    print(f"universe: {rets.shape[1] - 1} tickers, months {rets.index[0].date()}..{rets.index[-1].date()}")
    p = build_panel(closes, rets)
    print(f"panel rows: {len(p):,}  months: {p['date'].nunique()}  "
          f"test months: {(p.drop_duplicates('date')['date'] >= TEST_START).sum()}")
    p, ml = walk_forward_ml(p)
    port, ic, pooled = portfolios(p, ml, spy)

    strat = ["SPY", "EW", "Momentum", "SeasonalFilter", "ZScore"] + ml + [f"{n}+Filter" for n in ml]
    allm = metrics(port, strat)
    print("\n=== All months 2005-2025 ===")
    print(allm.round(3).to_string(index=False))

    nov = port[port.index.month == 11]
    novm = metrics(nov, strat)
    print("\n=== Novembers only (reproduces production backtest setting) ===")
    print(novm[["strategy", "mean_pct", "hit_rate", "excess_pct", "excess_t_nw"]].round(3).to_string(index=False))

    icsum = pd.DataFrame({
        "mean_ic": ic.mean(), "ic_t": ic.mean() / ic.std() * np.sqrt(ic.count()),
        "ic_pos": (ic > 0).mean()})
    print("\n=== Rank IC (whole eligible universe) ===")
    print(icsum.round(4).to_string())

    # sub-period robustness
    sub = []
    for a, b in [(2005, 2010), (2011, 2015), (2016, 2020), (2021, 2025)]:
        s = port[(port.index.year >= a) & (port.index.year <= b)]
        for c in ["ZScore", "SeasonalFilter", "GBDT", "LogReg", "MLP", "Momentum"]:
            sub.append({"period": f"{a}-{b}", "strategy": c,
                        "excess_pct": (s[c] - s["EW"]).mean(),
                        "beat": ((s[c] - s["EW"]) > 0).mean()})
    sub = pd.DataFrame(sub).pivot(index="strategy", columns="period", values="excess_pct")
    print("\n=== Sub-period mean monthly excess vs EW ===")
    print(sub.round(3).to_string())

    # per calendar month excess of ZScore and GBDT
    bym = pd.DataFrame({
        c: (port[c] - port["EW"]).groupby(port.index.month).mean() for c in ["ZScore", "GBDT", "LogReg"]})
    print("\n=== Mean excess vs EW by calendar month ===")
    print(bym.round(3).to_string())

    # GBDT permutation importance proxy: LR coefficients on rank features (interpretable)
    X = xs_rank(p, FEATURES)
    trn = p["date"] < TEST_END
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(X[trn], p["y"][trn])
    coefs = pd.Series(lr[-1].coef_[0], index=FEATURES).sort_values()
    print("\n=== LogReg coefficients (full-sample fit, descriptive only) ===")
    print(coefs.round(4).to_string())

    calib = {}
    for n in ["ZScore", "GBDT", "LogReg"]:
        tab, _ = calibration(pooled, n)
        calib[n] = tab
        print(f"\n=== Calibration deciles: {n} ===")
        print(tab.round(3).to_string())

    # sensitivity: top_n for GBDT / ZScore / filter months invested
    print("\nmonths with >=1 filter-passing name:", int(port["ZScore_invested"].sum()), "of", len(port))
    print("median filter-passing names per month:", float(port["n_filter"].median()))

    # persist
    port.to_csv(OUT / "monthly_returns.csv")
    ic.to_csv(OUT / "rank_ic.csv")
    allm.to_csv(OUT / "metrics_all_months.csv", index=False)
    novm.to_csv(OUT / "metrics_november.csv", index=False)
    icsum.to_csv(OUT / "ic_summary.csv")
    sub.to_csv(OUT / "subperiod_excess.csv")
    bym.to_csv(OUT / "excess_by_month.csv")
    coefs.to_csv(OUT / "logreg_coefs.csv")
    for n, tab in calib.items():
        tab.to_csv(OUT / f"calibration_{n}.csv")
    json.dump({"n_tickers": int(rets.shape[1] - 1), "panel_rows": int(len(p)),
               "test_months": int(len(port)),
               "test_rows": int((p["date"] >= TEST_START).sum()),
               "months_invested_filter": int(port["ZScore_invested"].sum()),
               "median_filter_names": float(port["n_filter"].median())},
              open(OUT / "meta.json", "w"), indent=2)
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
