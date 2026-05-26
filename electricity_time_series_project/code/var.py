"""
VAR / VECM and Granger causality analysis on the daily DE-AT-LU panel.

Runs ADF/KPSS, Johansen, an optional VECM, a first-difference VAR with lag
selection, residual diagnostics, the Granger causality matrix, and orthogonal
IRF/FEVD with wind -> solar -> load -> price Cholesky ordering.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_coint_rank


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "dataset"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"

RAW_COLS = {
    "load": "DE_load_actual_entsoe_transparency",
    "wind": "DE_wind_generation_actual",
    "solar": "DE_solar_generation_actual",
    "price": "DE_price_day_ahead",
}

ORDER = ["wind", "solar", "load", "price"]


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def find_file(filename: str) -> Path:
    candidate = DATA_DIR / filename
    if candidate.exists():
        return candidate
    matches = [p for p in ROOT.rglob(filename)
               if "tables" not in p.parts and "figures" not in p.parts]
    if not matches:
        raise FileNotFoundError(f"Cannot find {filename} under {ROOT}")
    return sorted(matches, key=lambda p: len(p.parts))[0]


def load_daily_panel() -> pd.DataFrame:
    data_path = find_file("de_panel_daily.csv")
    df = pd.read_csv(data_path, parse_dates=["ts"]).set_index("ts")
    out = df[list(RAW_COLS.values())].rename(columns={v: k for k, v in RAW_COLS.items()})
    out = out.dropna().sort_index()
    out.to_csv(TABLE_DIR / "table_var_input_daily.csv", encoding="utf-8-sig")
    return out


def transform_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    # log(x+1) for non-negative variables; shifted log for price (can be negative)
    shift = max(0.0, 1.0 - float(raw["price"].min()))
    levels = pd.DataFrame(index=raw.index)
    levels["wind"] = np.log(raw["wind"] + 1.0)
    levels["solar"] = np.log(raw["solar"] + 1.0)
    levels["load"] = np.log(raw["load"] + 1.0)
    levels["price"] = np.log(raw["price"] + shift)
    levels = levels[ORDER].replace([np.inf, -np.inf], np.nan).dropna()
    diffs = levels.diff().dropna()
    return levels, diffs, shift


def stationarity_table(levels: pd.DataFrame, diffs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, series in levels.items():
        for transform, s in [("level", series), ("diff", diffs[name])]:
            s = s.dropna()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
                try:
                    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
                except Exception:
                    kpss_stat, kpss_p = np.nan, np.nan
            rows.append({
                "variable": name,
                "transform": transform,
                "adf_stat": adf_stat,
                "adf_p": adf_p,
                "kpss_stat": kpss_stat,
                "kpss_p": kpss_p,
                "interpretation": (
                    "stationary"
                    if adf_p < 0.05 and (pd.isna(kpss_p) or kpss_p > 0.05)
                    else "nonstationary_or_mixed"
                ),
            })
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "table_stationarity_var42.csv",
                  index=False, encoding="utf-8-sig")
    return result


def johansen_table(levels: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # k_ar_diff=1 corresponds to a VAR(2) in levels
    joh = coint_johansen(levels, det_order=0, k_ar_diff=1)
    rows = []
    for i in range(len(joh.lr1)):
        rows.append({
            "null_rank": f"r <= {i}",
            "trace_stat": joh.lr1[i],
            "trace_crit_90": joh.cvt[i, 0],
            "trace_crit_95": joh.cvt[i, 1],
            "trace_crit_99": joh.cvt[i, 2],
            "reject_trace_95": bool(joh.lr1[i] > joh.cvt[i, 1]),
            "maxeig_stat": joh.lr2[i],
            "maxeig_crit_90": joh.cvm[i, 0],
            "maxeig_crit_95": joh.cvm[i, 1],
            "maxeig_crit_99": joh.cvm[i, 2],
            "reject_maxeig_95": bool(joh.lr2[i] > joh.cvm[i, 1]),
        })
    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "table_johansen_var42.csv",
                 index=False, encoding="utf-8-sig")

    rank_res = select_coint_rank(levels, det_order=0, k_ar_diff=1,
                                 method="trace", signif=0.05)
    return table, int(rank_res.rank)


def fit_vecm(levels: pd.DataFrame, rank: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    stale_files = [
        TABLE_DIR / "table_vecm_alpha_var42.csv",
        TABLE_DIR / "table_vecm_beta_var42.csv",
        TABLE_DIR / "vecm_summary_var42.txt",
    ]
    if rank <= 0 or rank >= len(levels.columns):
        for file in stale_files:
            if file.exists():
                file.unlink()
        note = (
            f"Selected Johansen rank is {rank}. A reduced-rank VECM is only "
            f"estimated when 0 < rank < {len(levels.columns)}. The short-run "
            "analysis uses a stationary VAR on first differences.\n"
        )
        (TABLE_DIR / "vecm_not_estimated_var42.txt").write_text(note, encoding="utf-8")
        return None, None
    res = VECM(levels, k_ar_diff=1, coint_rank=rank, deterministic="co").fit()
    alpha = pd.DataFrame(res.alpha, index=levels.columns,
                         columns=[f"ec{i + 1}" for i in range(rank)])
    beta = pd.DataFrame(res.beta, index=levels.columns,
                        columns=[f"beta{i + 1}" for i in range(rank)])
    alpha.to_csv(TABLE_DIR / "table_vecm_alpha_var42.csv", encoding="utf-8-sig")
    beta.to_csv(TABLE_DIR / "table_vecm_beta_var42.csv", encoding="utf-8-sig")
    (TABLE_DIR / "vecm_summary_var42.txt").write_text(str(res.summary()), encoding="utf-8")
    return alpha, beta


def fit_var(diffs: pd.DataFrame) -> tuple[object, pd.DataFrame, int]:
    model = VAR(diffs)
    selected = model.select_order(maxlags=14)
    order_table = pd.DataFrame({
        "lag": list(range(0, 15)),
        "aic": selected.ics["aic"],
        "bic": selected.ics["bic"],
        "hqic": selected.ics["hqic"],
        "fpe": selected.ics["fpe"],
    })
    order_table.to_csv(TABLE_DIR / "table_var_lag_selection_var42.csv",
                       index=False, encoding="utf-8-sig")
    lag = int(selected.selected_orders.get("aic")
              or selected.selected_orders.get("bic") or 1)
    lag = max(1, lag)
    res = model.fit(lag)
    (TABLE_DIR / "var_summary_var42.txt").write_text(str(res.summary()), encoding="utf-8")
    return res, order_table, lag


def diagnostics(var_res, lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    roots = pd.DataFrame({
        "root": var_res.roots,
        "modulus": np.abs(var_res.roots),
        "inverse_modulus": 1 / np.abs(var_res.roots),
    })
    roots["stable_condition_modulus_gt_1"] = roots["modulus"] > 1
    roots.to_csv(TABLE_DIR / "table_var_roots_var42.csv",
                 index=False, encoding="utf-8-sig")

    resid = pd.DataFrame(var_res.resid, columns=var_res.names)
    rows = []
    lb_lag = min(12, max(4, len(resid) // 10))
    for col in resid.columns:
        lb = acorr_ljungbox(resid[col], lags=[lb_lag], return_df=True)
        rows.append({
            "variable": col, "ljung_box_lag": lb_lag,
            "lb_stat": lb["lb_stat"].iloc[0], "lb_p": lb["lb_pvalue"].iloc[0],
        })
    diag = pd.DataFrame(rows)
    diag.to_csv(TABLE_DIR / "table_var_residual_diagnostics_var42.csv",
                index=False, encoding="utf-8-sig")
    return roots, diag


def granger_matrix(diffs: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    rows = []
    pmat = pd.DataFrame(np.nan, index=ORDER, columns=ORDER)
    fmat = pd.DataFrame(np.nan, index=ORDER, columns=ORDER)
    max_lag = max(1, min(max_lag, 14))
    for cause in ORDER:
        for effect in ORDER:
            if cause == effect:
                continue
            data = diffs[[effect, cause]].dropna()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            candidates = []
            for lag_i, payload in test.items():
                f_stat, p_value, df_denom, df_num = payload[0]["ssr_ftest"]
                candidates.append((p_value, f_stat, lag_i, df_num, df_denom))
            p_value, f_stat, best_lag, df_num, df_denom = sorted(candidates, key=lambda x: x[0])[0]
            pmat.loc[cause, effect] = p_value
            fmat.loc[cause, effect] = f_stat
            rows.append({
                "cause": cause, "effect": effect,
                "best_lag_by_min_p": best_lag,
                "f_stat": f_stat, "p_value": p_value,
                "df_num": df_num, "df_denom": df_denom,
                "significant_5pct": p_value < 0.05,
            })
    detail = pd.DataFrame(rows).sort_values(["effect", "p_value"])
    pmat.to_csv(TABLE_DIR / "table_granger_pvalues_var42.csv", encoding="utf-8-sig")
    fmat.to_csv(TABLE_DIR / "table_granger_fstats_var42.csv", encoding="utf-8-sig")
    detail.to_csv(TABLE_DIR / "table_granger_detail_var42.csv",
                  index=False, encoding="utf-8-sig")
    return pmat


def plot_granger_heatmap(pmat: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        pmat.astype(float),
        annot=True, fmt=".3f", cmap="viridis_r",
        vmin=0, vmax=0.2, mask=pmat.isna(),
        cbar_kws={"label": "p-value"},
    )
    plt.title("Granger Causality P-values\n(row causes column)")
    plt.xlabel("Effect")
    plt.ylabel("Cause")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_granger_heatmap_var42.png", dpi=220)
    plt.close()


def irf_and_fevd(var_res, periods: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    irf = var_res.irf(periods)
    fig = irf.plot(orth=True, signif=0.05)
    fig.set_size_inches(13, 10)
    fig.suptitle("Orthogonal Impulse Response Functions", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_irf_var42.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    responses = []
    price_idx = ORDER.index("price")
    for impulse in ORDER:
        imp_idx = ORDER.index(impulse)
        for h in range(periods + 1):
            responses.append({
                "horizon_days": h,
                "impulse": impulse,
                "response": "price",
                "orth_irf": irf.orth_irfs[h, price_idx, imp_idx],
            })
    price_irf = pd.DataFrame(responses)
    price_irf.to_csv(TABLE_DIR / "table_irf_price_responses_var42.csv",
                     index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    for impulse in ORDER:
        subset = price_irf[price_irf["impulse"] == impulse]
        plt.plot(subset["horizon_days"], subset["orth_irf"],
                 marker="o", linewidth=1.4, label=impulse)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Price Responses to One-standard-deviation Shocks")
    plt.xlabel("Horizon (days)")
    plt.ylabel("Response in transformed price difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_price_irf_var42.png", dpi=220)
    plt.close()

    fevd = var_res.fevd(periods)
    fig = fevd.plot(figsize=(12, 8))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_fevd_var42.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    decomp = fevd.decomp
    fevd_rows = []
    for variable_idx, variable in enumerate(ORDER):
        for h in range(periods):
            for shock_idx, shock in enumerate(ORDER):
                fevd_rows.append({
                    "variable": variable,
                    "horizon_days": h + 1,
                    "shock": shock,
                    "share": decomp[variable_idx, h, shock_idx],
                })
    fevd_table = pd.DataFrame(fevd_rows)
    fevd_table.to_csv(TABLE_DIR / "table_fevd_all_var42.csv",
                      index=False, encoding="utf-8-sig")
    price_fevd = (fevd_table[fevd_table["variable"] == "price"]
                  .pivot(index="horizon_days", columns="shock", values="share"))
    price_fevd.to_csv(TABLE_DIR / "table_fevd_price_var42.csv", encoding="utf-8-sig")
    return price_irf, price_fevd


def fmt_p(x: float) -> str:
    if pd.isna(x):
        return ""
    if x < 0.001:
        return "<0.001"
    return f"{x:.3f}"


def write_result_notes(
    raw: pd.DataFrame,
    shift: float,
    rank: int,
    lag: int,
    roots: pd.DataFrame,
    pmat: pd.DataFrame,
    price_fevd: pd.DataFrame,
) -> None:
    stable = bool((roots["modulus"] > 1).all())
    inv_max = roots["inverse_modulus"].max()
    fevd20 = price_fevd.iloc[-1].reindex(ORDER)

    lines = [
        f"VAR/VECM run on {raw.index.min().date()}--{raw.index.max().date()} "
        f"({len(raw)} daily obs).",
        f"Price shift used in log transform: {shift:.4f} "
        f"({int((raw['price'] <= 0).sum())} non-positive price obs).",
        f"Johansen rank: {rank}. VAR lag: {lag}. "
        f"Stable (|root|>1 for all): {stable}, max inverse-root modulus: {inv_max:.3f}.",
        "",
        "Granger p-values toward price:",
    ]
    for cause in ["wind", "solar", "load"]:
        lines.append(f"  {cause} -> price: {fmt_p(pmat.loc[cause, 'price'])}")
    lines.append("Reverse direction:")
    for effect in ["wind", "solar", "load"]:
        lines.append(f"  price -> {effect}: {fmt_p(pmat.loc['price', effect])}")
    lines += [
        "",
        "20-day FEVD shares for price:",
        f"  own:  {fevd20['price'] * 100:.1f}%",
        f"  wind: {fevd20['wind'] * 100:.1f}%",
        f"  solar:{fevd20['solar'] * 100:.1f}%",
        f"  load: {fevd20['load'] * 100:.1f}%",
    ]
    (TABLE_DIR / "var_result_notes.txt").write_text("\n".join(lines) + "\n",
                                                    encoding="utf-8")


def main() -> None:
    ensure_dirs()
    raw = load_daily_panel()
    levels, diffs, shift = transform_data(raw)
    levels.to_csv(TABLE_DIR / "table_transformed_levels_var42.csv", encoding="utf-8-sig")
    diffs.to_csv(TABLE_DIR / "table_transformed_diffs_var42.csv", encoding="utf-8-sig")

    stationarity_table(levels, diffs)
    _, rank = johansen_table(levels)
    fit_vecm(levels, rank)
    var_res, _, lag = fit_var(diffs)
    roots, _ = diagnostics(var_res, lag)
    pmat = granger_matrix(diffs, lag)
    plot_granger_heatmap(pmat)
    _, price_fevd = irf_and_fevd(var_res, periods=20)

    write_result_notes(raw, shift, rank, lag, roots, pmat, price_fevd)
    print(f"Done. Outputs in {ROOT}")


if __name__ == "__main__":
    main()
