"""
SARIMA / SARIMAX hourly load forecasting for the DE-AT-LU market.

Estimates a SARIMA model, a SARIMAX with wind/solar/calendar exogs, and an
enhanced SARIMAX with day-of-week dummies and weekly Fourier terms. Compares
forecasts on the 2018 test set against ETS and seasonal-naive baselines and
writes tables and figures.
"""
from __future__ import annotations

import re
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "dataset"
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"

TARGET = "DE_load_actual_entsoe_transparency"
BASE_EXOG = ["DE_wind_generation_actual", "DE_solar_generation_actual", "is_weekend"]
ENHANCED_BASE_EXOG = ["DE_wind_generation_actual", "DE_solar_generation_actual"]
SEASONAL_PERIOD = 24
TRAIN_END = pd.Timestamp("2017-12-31 23:00:00")
TEST_START = pd.Timestamp("2018-01-01 00:00:00")


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


def load_hourly_panel() -> pd.DataFrame:
    path = find_file("de_panel_hourly.csv")
    df = pd.read_csv(path, parse_dates=["ts"]).set_index("ts").sort_index()
    cols = [TARGET, "DE_wind_generation_actual", "DE_solar_generation_actual",
            "is_weekend", "month"]
    df = df[cols].copy().asfreq("h")
    df[TARGET] = df[TARGET].interpolate("time").ffill().bfill()
    for col in ["DE_wind_generation_actual", "DE_solar_generation_actual"]:
        df[col] = df[col].interpolate("time").ffill().bfill()
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["month"] = df.index.month
    df.to_csv(TABLE_DIR / "table_sarima_input_hourly.csv", encoding="utf-8-sig")
    return df


def make_exog(df: pd.DataFrame) -> pd.DataFrame:
    month_dummies = pd.get_dummies(df.index.month, prefix="month",
                                   drop_first=True, dtype=float)
    month_dummies.index = df.index
    return pd.concat([df[BASE_EXOG].astype(float), month_dummies], axis=1)


def fourier_terms(index: pd.DatetimeIndex, period: float, order: int, prefix: str) -> pd.DataFrame:
    t = np.arange(len(index), dtype=float)
    data = {}
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * t / period
        data[f"{prefix}_sin{k}"] = np.sin(angle)
        data[f"{prefix}_cos{k}"] = np.cos(angle)
    return pd.DataFrame(data, index=index)


def make_enhanced_exog(df: pd.DataFrame) -> pd.DataFrame:
    # day-of-week dummies + weekly Fourier replace the single is_weekend flag
    month_dummies = pd.get_dummies(df.index.month, prefix="month",
                                   drop_first=True, dtype=float)
    month_dummies.index = df.index
    dow_dummies = pd.get_dummies(df.index.dayofweek, prefix="dow",
                                 drop_first=True, dtype=float)
    dow_dummies.index = df.index
    weekly_fourier = fourier_terms(df.index, period=24 * 7, order=3, prefix="weekly")
    return pd.concat(
        [df[ENHANCED_BASE_EXOG].astype(float), month_dummies, dow_dummies, weekly_fourier],
        axis=1,
    )


def split_data(df: pd.DataFrame, exog: pd.DataFrame):
    y = df[TARGET].astype(float)
    y_train = y.loc[y.index <= TRAIN_END]
    y_test = y.loc[y.index >= TEST_START]
    return y_train, y_test, exog.loc[y_train.index], exog.loc[y_test.index]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE_pct": float(np.mean(np.abs(err / y_true)) * 100),
    }


def diebold_mariano(e_benchmark: np.ndarray, e_model: np.ndarray) -> tuple[float, float]:
    # Two-sided test on squared loss; positive stat => benchmark has larger loss.
    d = np.asarray(e_benchmark) ** 2 - np.asarray(e_model) ** 2
    d = d[np.isfinite(d)]
    n = len(d)
    gamma0 = np.var(d, ddof=1)
    if gamma0 <= 0:
        return np.nan, np.nan
    dm = np.mean(d) / np.sqrt(gamma0 / n)
    p = 2 * (1 - norm.cdf(abs(dm)))
    return float(dm), float(p)


def fit_model(y_train, order, seasonal_order, exog=None, maxiter: int = 80):
    model = SARIMAX(
        y_train, exog=exog,
        order=order, seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False, enforce_invertibility=False,
        concentrate_scale=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=maxiter, method="lbfgs")


def fit_candidate_table(y_train: pd.Series) -> pd.DataFrame:
    # Small grid on the 2015-2016 subset; a full auto_arima is too slow.
    sample = y_train.loc[: "2016-12-31 23:00:00"]
    candidates = [
        ((1, 0, 1), (1, 1, 0, 24)),
        ((2, 0, 1), (1, 1, 0, 24)),
        ((2, 1, 1), (1, 1, 0, 24)),
        ((2, 0, 2), (1, 1, 0, 24)),
        ((3, 0, 1), (1, 1, 0, 24)),
    ]
    rows = []
    for order, seasonal_order in candidates:
        t0 = time.perf_counter()
        try:
            res = fit_model(sample, order, seasonal_order, maxiter=50)
            rows.append({
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "aic": float(res.aic),
                "bic": float(res.bic),
                "hqic": float(res.hqic),
                "converged": bool(res.mle_retvals.get("converged", False)),
                "runtime_sec": time.perf_counter() - t0,
            })
        except Exception as exc:
            rows.append({
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "aic": np.nan, "bic": np.nan, "hqic": np.nan,
                "converged": False,
                "runtime_sec": time.perf_counter() - t0,
                "error": str(exc)[:160],
            })
    table = pd.DataFrame(rows).sort_values("aic", na_position="last")
    table.to_csv(TABLE_DIR / "table_sarima_order_candidates.csv",
                 index=False, encoding="utf-8-sig")
    return table


def residual_diagnostics(result, label: str) -> pd.DataFrame:
    resid = pd.Series(result.resid).dropna()
    rows = []
    for lag in [10, 20, 24, 48]:
        lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
        rows.append({"model": label, "lag": lag,
                     "lb_stat": lb["lb_stat"].iloc[0],
                     "lb_p": lb["lb_pvalue"].iloc[0]})
    fig = result.plot_diagnostics(figsize=(11, 8))
    fig.suptitle(f"Residual diagnostics: {label}", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"fig_diag_{label.lower()}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def fit_ets(y_train: pd.Series):
    model = ExponentialSmoothing(
        y_train, trend="add", seasonal="add",
        seasonal_periods=24, initialization_method="estimated",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(optimized=True)


def seasonal_naive(y_train: pd.Series, y_test: pd.Series, period: int = 24) -> pd.Series:
    history = pd.concat([y_train, y_test * np.nan])
    pred = history.shift(period).loc[y_test.index]
    return pred.fillna(y_train.iloc[-period:].mean())


def plot_forecast(y_test: pd.Series, predictions: dict[str, pd.Series]) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    y_test.plot(ax=ax, label="Actual", linewidth=1.0, color="black")
    for name, pred in predictions.items():
        pred.plot(ax=ax, label=name, linewidth=0.8, alpha=0.85)
    ax.set_title("Hourly Load Forecast Comparison, 2018 Test Period")
    ax.set_ylabel("MW")
    ax.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_load_forecast_test_full.png", dpi=220)
    plt.close(fig)

    detail_end = y_test.index.min() + pd.Timedelta(days=30)
    fig, ax = plt.subplots(figsize=(13, 5))
    y_test.loc[:detail_end].plot(ax=ax, label="Actual", linewidth=1.1, color="black")
    for name, pred in predictions.items():
        pred.loc[:detail_end].plot(ax=ax, label=name, linewidth=0.9, alpha=0.85)
    ax.set_title("Hourly Load Forecast Detail, First 30 Test Days")
    ax.set_ylabel("MW")
    ax.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_load_forecast_first30days.png", dpi=220)
    plt.close(fig)


def write_result_notes(
    elapsed_sec: float,
    metric_table: pd.DataFrame,
    dm_table: pd.DataFrame,
    diag_table: pd.DataFrame,
    selected_order: tuple,
    selected_seasonal: tuple,
) -> None:
    lines = [
        f"SARIMA/SARIMAX run summary ({elapsed_sec / 60:.1f} min)",
        f"Selected order: SARIMA{selected_order}{selected_seasonal}",
        "",
        "Forecast metrics on 2018 test set:",
        metric_table.to_string(),
        "",
        "Diebold-Mariano (squared loss):",
        dm_table[["comparison", "dm_stat", "p_value"]].to_string(index=False),
        "",
        "Residual Ljung-Box at lag 24:",
    ]
    for model in diag_table["model"].unique():
        row = diag_table[(diag_table["model"] == model) & (diag_table["lag"] == 24)]
        if not row.empty:
            lines.append(f"  {model}: p = {row['lb_p'].iloc[0]:.3f}")
    (TABLE_DIR / "sarima_result_notes.txt").write_text("\n".join(lines) + "\n",
                                                       encoding="utf-8")


def main() -> None:
    ensure_dirs()
    t_start = time.perf_counter()
    df = load_hourly_panel()
    exog = make_exog(df)
    enhanced_exog = make_enhanced_exog(df)
    y_train, y_test, x_train, x_test = split_data(df, exog)
    _, _, x_enhanced_train, x_enhanced_test = split_data(df, enhanced_exog)

    print(f"train={len(y_train)}, test={len(y_test)}")
    print("picking SARIMA order on 2015-2016...")
    candidate_table = fit_candidate_table(y_train)
    best = candidate_table.dropna(subset=["aic"]).iloc[0]
    selected_order = tuple(int(v) for v in re.findall(r"\d+", best["order"]))
    selected_seasonal = tuple(int(v) for v in re.findall(r"\d+", best["seasonal_order"]))
    print(f"selected: order={selected_order}, seasonal={selected_seasonal}")

    print("fitting SARIMA...")
    sarima_res = fit_model(y_train, selected_order, selected_seasonal, maxiter=100)
    pred_sarima = pd.Series(sarima_res.forecast(steps=len(y_test)),
                            index=y_test.index, name="SARIMA")

    print("fitting SARIMAX...")
    sarimax_res = fit_model(y_train, selected_order, selected_seasonal,
                            exog=x_train, maxiter=100)
    pred_sarimax = pd.Series(sarimax_res.forecast(steps=len(y_test), exog=x_test),
                             index=y_test.index, name="SARIMAX")

    print("fitting enhanced SARIMAX...")
    enhanced_res = fit_model(y_train, selected_order, selected_seasonal,
                             exog=x_enhanced_train, maxiter=120)
    pred_enhanced = pd.Series(
        enhanced_res.forecast(steps=len(y_test), exog=x_enhanced_test),
        index=y_test.index, name="SARIMAX_Enhanced",
    )

    print("fitting ETS baseline...")
    ets_error = ""
    try:
        ets_res = fit_ets(y_train)
        pred_ets = pd.Series(ets_res.forecast(len(y_test)),
                             index=y_test.index, name="ETS")
    except Exception as exc:
        ets_error = str(exc)
        pred_ets = seasonal_naive(y_train, y_test).rename("SeasonalNaive")

    pred_naive = seasonal_naive(y_train, y_test).rename("SeasonalNaive")
    predictions = {
        "SARIMA": pred_sarima,
        "SARIMAX": pred_sarimax,
        "SARIMAX_Enhanced": pred_enhanced,
        "ETS" if not ets_error else "SeasonalNaive": pred_ets,
        "SeasonalNaive": pred_naive,
    }
    pd.DataFrame(predictions).to_csv(
        TABLE_DIR / "table_load_forecasts_test.csv", encoding="utf-8-sig"
    )

    metric_rows = []
    aic_lookup = {
        "SARIMA": sarima_res, "SARIMAX": sarimax_res, "SARIMAX_Enhanced": enhanced_res,
    }
    for name, pred in predictions.items():
        row = {"Model": name, **metrics(y_test.values, pred.values)}
        res = aic_lookup.get(name)
        row.update({
            "AIC": res.aic if res is not None else np.nan,
            "BIC": res.bic if res is not None else np.nan,
            "HQIC": res.hqic if res is not None else np.nan,
        })
        metric_rows.append(row)
    metric_table = pd.DataFrame(metric_rows).set_index("Model")
    metric_table.to_csv(TABLE_DIR / "table_load_forecast_metrics.csv", encoding="utf-8-sig")

    dm, pval = diebold_mariano(y_test.values - pred_sarima.values,
                               y_test.values - pred_sarimax.values)
    dm_enh, pval_enh = diebold_mariano(y_test.values - pred_sarimax.values,
                                       y_test.values - pred_enhanced.values)
    dm_table = pd.DataFrame([
        {"comparison": "SARIMAX vs SARIMA", "loss": "squared error",
         "dm_stat": dm, "p_value": pval,
         "interpretation": "positive means SARIMA loss > SARIMAX loss"},
        {"comparison": "SARIMAX_Enhanced vs SARIMAX", "loss": "squared error",
         "dm_stat": dm_enh, "p_value": pval_enh,
         "interpretation": "positive means SARIMAX loss > SARIMAX_Enhanced loss"},
    ])
    dm_table.to_csv(TABLE_DIR / "table_dm_sarimax_vs_sarima.csv",
                    index=False, encoding="utf-8-sig")

    diag_table = pd.concat([
        residual_diagnostics(sarima_res, "SARIMA"),
        residual_diagnostics(sarimax_res, "SARIMAX"),
        residual_diagnostics(enhanced_res, "SARIMAX_Enhanced"),
    ], ignore_index=True)
    diag_table.to_csv(TABLE_DIR / "table_sarima_residual_diagnostics.csv",
                      index=False, encoding="utf-8-sig")

    params = pd.concat([
        sarima_res.params.rename("SARIMA"),
        sarimax_res.params.rename("SARIMAX"),
        enhanced_res.params.rename("SARIMAX_Enhanced"),
    ], axis=1)
    params.to_csv(TABLE_DIR / "table_sarima_parameters.csv", encoding="utf-8-sig")
    (TABLE_DIR / "sarima_summary.txt").write_text(str(sarima_res.summary()), encoding="utf-8")
    (TABLE_DIR / "sarimax_summary.txt").write_text(str(sarimax_res.summary()), encoding="utf-8")
    (TABLE_DIR / "sarimax_enhanced_summary.txt").write_text(str(enhanced_res.summary()), encoding="utf-8")
    if ets_error:
        (TABLE_DIR / "ets_error.txt").write_text(ets_error, encoding="utf-8")

    plot_forecast(y_test, predictions)

    elapsed = time.perf_counter() - t_start
    write_result_notes(elapsed, metric_table, dm_table, diag_table,
                       selected_order, selected_seasonal)
    print(f"Done in {elapsed / 60:.1f} min. Outputs in {ROOT}")


if __name__ == "__main__":
    main()
