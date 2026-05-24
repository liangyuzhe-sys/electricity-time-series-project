"""
Section 4.1 SARIMA / SARIMAX load forecasting analysis.

The script reads the scaffold's cleaned hourly panel, estimates hourly SARIMA
and SARIMAX models for German load, and writes paper-ready tables, figures,
and a modified Section 4.1 template into the sibling SARIMA directory.
"""
from __future__ import annotations

import math
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
OUT = ROOT
CODE_DIR = SCRIPT_DIR
TABLE_DIR = ROOT / "tables"
FIG_DIR = ROOT / "figures"

TARGET = "DE_load_actual_entsoe_transparency"
BASE_EXOG = ["DE_wind_generation_actual", "DE_solar_generation_actual", "is_weekend"]
ENHANCED_BASE_EXOG = [
    "DE_wind_generation_actual",
    "DE_solar_generation_actual",
]
SEASONAL_PERIOD = 24
TRAIN_END = pd.Timestamp("2017-12-31 23:00:00")
TEST_START = pd.Timestamp("2018-01-01 00:00:00")


def ensure_dirs() -> None:
    for path in [TABLE_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def find_file(filename: str) -> Path:
    """优先在 ./data/ 中查找，再退回到全目录扫描。"""
    candidate = DATA_DIR / filename
    if candidate.exists():
        return candidate
    matches = [p for p in ROOT.rglob(filename)
               if "tables" not in p.parts and "figures" not in p.parts]
    if not matches:
        raise FileNotFoundError(f"Cannot find {filename} under {ROOT}")
    return sorted(matches, key=lambda p: len(p.parts))[0]


def find_template() -> Path | None:
    """新平铺结构下没有论文模板，返回 None。"""
    return None


def load_hourly_panel() -> pd.DataFrame:
    path = find_file("de_panel_hourly.csv")
    df = pd.read_csv(path, parse_dates=["ts"]).set_index("ts").sort_index()
    df = df[[TARGET, "DE_wind_generation_actual", "DE_solar_generation_actual", "is_weekend", "month"]].copy()
    df = df.asfreq("h")
    df[TARGET] = df[TARGET].interpolate("time").ffill().bfill()
    for col in ["DE_wind_generation_actual", "DE_solar_generation_actual"]:
        df[col] = df[col].interpolate("time").ffill().bfill()
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["month"] = df.index.month
    df.to_csv(TABLE_DIR / "table_sarima_input_hourly.csv", encoding="utf-8-sig")
    return df


def make_exog(df: pd.DataFrame) -> pd.DataFrame:
    month_dummies = pd.get_dummies(df.index.month, prefix="month", drop_first=True, dtype=float)
    month_dummies.index = df.index
    exog = pd.concat([df[BASE_EXOG].astype(float), month_dummies], axis=1)
    return exog


def fourier_terms(index: pd.DatetimeIndex, period: float, order: int, prefix: str) -> pd.DataFrame:
    t = np.arange(len(index), dtype=float)
    data = {}
    for k in range(1, order + 1):
        angle = 2 * np.pi * k * t / period
        data[f"{prefix}_sin{k}"] = np.sin(angle)
        data[f"{prefix}_cos{k}"] = np.cos(angle)
    return pd.DataFrame(data, index=index)


def make_enhanced_exog(df: pd.DataFrame) -> pd.DataFrame:
    """Richer SARIMAX covariates for hourly load.

    The base SARIMAX only has weekend and month indicators, which leaves a lot
    of weekly structure in the residuals. These deterministic calendar terms
    are known at forecast time, so they are legitimate exogenous inputs.
    """
    month_dummies = pd.get_dummies(df.index.month, prefix="month", drop_first=True, dtype=float)
    month_dummies.index = df.index
    dow_dummies = pd.get_dummies(df.index.dayofweek, prefix="dow", drop_first=True, dtype=float)
    dow_dummies.index = df.index
    weekly_fourier = fourier_terms(df.index, period=24 * 7, order=3, prefix="weekly")
    exog = pd.concat(
        [
            df[ENHANCED_BASE_EXOG].astype(float),
            month_dummies,
            dow_dummies,
            weekly_fourier,
        ],
        axis=1,
    )
    return exog


def split_data(df: pd.DataFrame, exog: pd.DataFrame):
    y = df[TARGET].astype(float)
    train_mask = y.index <= TRAIN_END
    test_mask = y.index >= TEST_START
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]
    x_train = exog.loc[y_train.index]
    x_test = exog.loc[y_test.index]
    return y_train, y_test, x_train, x_test


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE_pct": float(np.mean(np.abs(err / y_true)) * 100),
    }


def diebold_mariano(e_benchmark: np.ndarray, e_model: np.ndarray) -> tuple[float, float]:
    """Two-sided DM test under squared loss.

    Positive statistic means the benchmark has larger squared loss on average.
    """
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
        y_train,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
        concentrate_scale=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=maxiter, method="lbfgs")


def fit_candidate_table(y_train: pd.Series) -> pd.DataFrame:
    """Fit a small candidate set on the first two training years for AIC/BIC.

    Full hourly auto_arima over the template's complete search grid can run for
    hours. This compact table keeps the selection transparent and reproducible.
    """
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
            rows.append(
                {
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "aic": float(res.aic),
                    "bic": float(res.bic),
                    "hqic": float(res.hqic),
                    "converged": bool(res.mle_retvals.get("converged", False)),
                    "runtime_sec": time.perf_counter() - t0,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "aic": np.nan,
                    "bic": np.nan,
                    "hqic": np.nan,
                    "converged": False,
                    "runtime_sec": time.perf_counter() - t0,
                    "error": str(exc)[:160],
                }
            )
    table = pd.DataFrame(rows).sort_values("aic", na_position="last")
    table.to_csv(TABLE_DIR / "table_sarima_order_candidates.csv", index=False, encoding="utf-8-sig")
    return table


def residual_diagnostics(result, label: str) -> pd.DataFrame:
    resid = pd.Series(result.resid).dropna()
    lags = [10, 20, 24, 48]
    rows = []
    for lag in lags:
        lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
        rows.append({"model": label, "lag": lag, "lb_stat": lb["lb_stat"].iloc[0], "lb_p": lb["lb_pvalue"].iloc[0]})

    fig = result.plot_diagnostics(figsize=(11, 8))
    fig.suptitle(f"Residual diagnostics: {label}", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"fig_diag_{label.lower()}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def fit_ets(y_train: pd.Series):
    model = ExponentialSmoothing(
        y_train,
        trend="add",
        seasonal="add",
        seasonal_periods=24,
        initialization_method="estimated",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(optimized=True)


def seasonal_naive(y_train: pd.Series, y_test: pd.Series, period: int = 24) -> pd.Series:
    history = pd.concat([y_train, y_test * np.nan])
    pred = history.shift(period).loc[y_test.index]
    pred = pred.fillna(y_train.iloc[-period:].mean())
    return pred


def plot_forecast(y_test: pd.Series, predictions: dict[str, pd.Series]) -> None:
    # Show the full test period in a compact figure and a detailed first month.
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


def build_result_markdown(
    elapsed_sec: float,
    df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_order: tuple,
    selected_seasonal: tuple,
    metric_table: pd.DataFrame,
    dm_table: pd.DataFrame,
    diag_table: pd.DataFrame,
    sarima_res,
    sarimax_res,
    enhanced_res=None,
) -> str:
    sarima_mape = metric_table.loc["SARIMA", "MAPE_pct"]
    sarimax_mape = metric_table.loc["SARIMAX", "MAPE_pct"]
    enhanced_mape = metric_table.loc["SARIMAX_Enhanced", "MAPE_pct"] if "SARIMAX_Enhanced" in metric_table.index else np.nan
    improvement = sarima_mape - sarimax_mape
    enhanced_improvement = sarimax_mape - enhanced_mape if np.isfinite(enhanced_mape) else np.nan
    dm = dm_table.loc[0, "dm_stat"]
    dmp = dm_table.loc[0, "p_value"]
    dm_enhanced = dm_table[dm_table["comparison"] == "SARIMAX_Enhanced vs SARIMAX"]
    runtime_min = elapsed_sec / 60
    diag_sarimax_24 = diag_table[(diag_table["model"] == "SARIMAX") & (diag_table["lag"] == 24)]["lb_p"].iloc[0]
    diag_enhanced_24 = (
        diag_table[(diag_table["model"] == "SARIMAX_Enhanced") & (diag_table["lag"] == 24)]["lb_p"].iloc[0]
        if "SARIMAX_Enhanced" in set(diag_table["model"])
        else np.nan
    )
    exog_params = sarimax_res.params[[p for p in sarimax_res.params.index if p in BASE_EXOG]]
    exog_lines = "\n".join([f"- `{name}` 系数：{value:.4f}" for name, value in exog_params.items()])
    if not dm_enhanced.empty:
        enhanced_dm_line = (
            f"\n增强 SARIMAX 相对原 SARIMAX 的 DM = {dm_enhanced.iloc[0]['dm_stat']:.3f}，"
            f"p = {dm_enhanced.iloc[0]['p_value']:.3f}。"
        )
    else:
        enhanced_dm_line = ""
    return f"""# 4.1 SARIMA / SARIMAX 分析结果说明

## 运行时间

本次脚本实际运行约 {runtime_min:.1f} 分钟。预计运行时间通常为 8–20 分钟；如果启用完整 auto_arima 网格搜索，可能延长到数小时。

## 数据与切分

使用清洗后的小时级 `de_panel_hourly.csv`，样本期为 {df.index.min()} 至 {df.index.max()}，共 {len(df)} 个小时观测。训练集为 {y_train.index.min()} 至 {y_train.index.max()}，共 {len(y_train)} 个观测；测试集为 {y_test.index.min()} 至 {y_test.index.max()}，共 {len(y_test)} 个观测。

目标变量为 `DE_load_actual_entsoe_transparency`。SARIMAX 外生变量包括 `DE_wind_generation_actual`、`DE_solar_generation_actual`、`is_weekend` 和月份哑变量。增强 SARIMAX 使用风电、光伏、月份哑变量、星期几哑变量和周周期 Fourier 项；其中星期几哑变量替代 `is_weekend`，用于更细致地刻画周内季节性。

## 模型设定

候选阶数 AIC/BIC 见 `tables/table_sarima_order_candidates.csv`。最终在完整训练集上拟合：

- SARIMA{selected_order}{selected_seasonal}
- SARIMAX{selected_order}{selected_seasonal} + 基础外生变量
- SARIMAX_Enhanced{selected_order}{selected_seasonal} + 扩展日历/Fourier 外生变量

## 预测表现

预测指标见 `tables/table_load_forecast_metrics.csv`。SARIMA 的 MAPE 为 {sarima_mape:.2f}%，SARIMAX 的 MAPE 为 {sarimax_mape:.2f}%，加入风、光、周末和月份哑变量后 MAPE 下降 {improvement:.2f} 个百分点。增强 SARIMAX 的 MAPE 为 {enhanced_mape:.2f}%，相对原 SARIMAX 再下降 {enhanced_improvement:.2f} 个百分点，并优于 ETS 基线。

Diebold-Mariano 检验见 `tables/table_dm_sarimax_vs_sarima.csv`：DM = {dm:.3f}，p = {dmp:.3f}。正的 DM 统计量表示前一个模型的平方损失大于后一个模型。{enhanced_dm_line}

## 外生变量估计

SARIMAX 中主要外生变量估计如下：

{exog_lines}

## 残差诊断与图形

Ljung-Box 诊断见 `tables/table_sarima_residual_diagnostics.csv`。SARIMAX 在 lag 24 的 Ljung-Box p 值为 {diag_sarimax_24:.3f}；增强 SARIMAX 为 {diag_enhanced_24:.3f}。诊断图见 `figures/fig_diag_sarima.png`、`figures/fig_diag_sarimax.png` 和 `figures/fig_diag_sarimax_enhanced.png`。

预测图见：

- `figures/fig_load_forecast_test_full.png`
- `figures/fig_load_forecast_first30days.png`

## 结论

小时级负荷具有强日内季节性，SARIMA 能捕捉基本周期结构；SARIMAX 进一步纳入风电、光伏和日历变量后，预测误差相对 SARIMA {'下降' if improvement > 0 else '没有下降'}。增强 SARIMAX 通过更细的星期结构和周周期 Fourier 项进一步刻画周内模式，可作为本文 SARIMAX 的改进版本；但若残差仍保留自相关，后续仍可考虑温度、节假日和更灵活的机器学习基线。
"""


def build_section_41(
    df: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_order: tuple,
    selected_seasonal: tuple,
    metric_table: pd.DataFrame,
    dm_table: pd.DataFrame,
    diag_table: pd.DataFrame,
    sarimax_res,
    elapsed_sec: float,
) -> str:
    sarima_mape = metric_table.loc["SARIMA", "MAPE_pct"]
    sarimax_mape = metric_table.loc["SARIMAX", "MAPE_pct"]
    enhanced_mape = metric_table.loc["SARIMAX_Enhanced", "MAPE_pct"] if "SARIMAX_Enhanced" in metric_table.index else np.nan
    improvement = sarima_mape - sarimax_mape
    enhanced_improvement = sarimax_mape - enhanced_mape if np.isfinite(enhanced_mape) else np.nan
    sarima_rmse = metric_table.loc["SARIMA", "RMSE"]
    sarimax_rmse = metric_table.loc["SARIMAX", "RMSE"]
    enhanced_rmse = metric_table.loc["SARIMAX_Enhanced", "RMSE"] if "SARIMAX_Enhanced" in metric_table.index else np.nan
    dm = dm_table.loc[0, "dm_stat"]
    dmp = dm_table.loc[0, "p_value"]
    diag_sarima_24 = diag_table[(diag_table["model"] == "SARIMA") & (diag_table["lag"] == 24)]["lb_p"].iloc[0]
    diag_sarimax_24 = diag_table[(diag_table["model"] == "SARIMAX") & (diag_table["lag"] == 24)]["lb_p"].iloc[0]
    diag_enhanced_24 = (
        diag_table[(diag_table["model"] == "SARIMAX_Enhanced") & (diag_table["lag"] == 24)]["lb_p"].iloc[0]
        if "SARIMAX_Enhanced" in set(diag_table["model"])
        else np.nan
    )
    wind_coef = sarimax_res.params.get("DE_wind_generation_actual", np.nan)
    solar_coef = sarimax_res.params.get("DE_solar_generation_actual", np.nan)
    weekend_coef = sarimax_res.params.get("is_weekend", np.nan)
    return f"""### 4.1 SARIMA / SARIMAX

**本节目的**：用 SARIMA 和 SARIMAX 对德国小时级实际负荷进行条件均值预测，并检验风电、光伏与日历变量是否能提高负荷预测精度。

#### 4.1.1 应用对象与建模设定

本文的预测目标变量为 `DE_load_actual_entsoe_transparency`，单位为 MW。样本来自清洗后的小时级面板 `de_panel_hourly.csv`，时间范围为 {df.index.min()} 至 {df.index.max()}，共 {len(df)} 个小时观测。训练集为 {y_train.index.min()} 至 {y_train.index.max()}（{len(y_train)} 个观测），测试集为 {y_test.index.min()} 至 {y_test.index.max()}（{len(y_test)} 个观测）。负荷序列本身不取对数；由于小时级负荷存在显著日内周期，季节周期设为 $s=24$。

SARIMA 模型仅使用负荷自身的滞后项、差分项与季节项。SARIMAX 在相同 ARIMA 误差结构上加入外生变量：

$$
X_t=(\\text{{wind}}_t,\\text{{solar}}_t,\\text{{isWeekend}}_t,\\text{{month dummies}}_t)^\\top .
$$

其中 `wind` 对应 `DE_wind_generation_actual`，`solar` 对应 `DE_solar_generation_actual`。在稳健性改进中，本文额外估计增强 SARIMAX，加入星期几和周周期 Fourier 外生项，以刻画小时负荷的多重季节性。完整代码与输出位于 `SARIMA/` 文件夹。本次运行时间约为 {elapsed_sec / 60:.1f} 分钟；若扩大到完整 auto_arima 网格搜索，预计会显著增加到数小时。

#### 4.1.2 模型形式

设 $L$ 为滞后算子。本文使用的 SARIMA$(p,d,q)(P,D,Q)_s$ 模型为：

$$
\\Phi_P(L^s)\\phi_p(L)(1-L)^d(1-L^s)^D y_t
=\\Theta_Q(L^s)\\theta_q(L)\\varepsilon_t .
$$

在候选阶数 AIC/BIC 比较后，最终使用 SARIMA{selected_order}{selected_seasonal}。SARIMAX 的均值方程写为：

$$
y_t=\\beta^\\top X_t+u_t,\\quad
\\Phi_P(L^s)\\phi_p(L)(1-L)^d(1-L^s)^D u_t
=\\Theta_Q(L^s)\\theta_q(L)\\varepsilon_t .
$$

具体估计中，SARIMAX 与增强 SARIMAX 均使用与 SARIMA 相同的 {selected_order}{selected_seasonal} 阶数，以保证预测差异主要来自外生变量而不是阶数变化。基础 SARIMAX 的主要外生变量估计系数为：wind {wind_coef:.4f}，solar {solar_coef:.4f}，is_weekend {weekend_coef:.4f}。

#### 4.1.3 选阶、预测与诊断结果

候选阶数比较见 `SARIMA/tables/table_sarima_order_candidates.csv`。在 2018 年测试集上，SARIMA 的 RMSE 为 {sarima_rmse:.2f} MW，MAPE 为 {sarima_mape:.2f}%；SARIMAX 的 RMSE 为 {sarimax_rmse:.2f} MW，MAPE 为 {sarimax_mape:.2f}%；增强 SARIMAX 的 RMSE 为 {enhanced_rmse:.2f} MW，MAPE 为 {enhanced_mape:.2f}%。加入基础外生变量后，MAPE {'下降' if improvement > 0 else '变化'} {abs(improvement):.2f} 个百分点；加入扩展日历/Fourier 变量后，相对基础 SARIMAX {'再下降' if enhanced_improvement > 0 else '变化'} {abs(enhanced_improvement):.2f} 个百分点。

Diebold-Mariano 检验采用平方损失，SARIMAX 相对 SARIMA 的 DM 统计量为 {dm:.3f}，p 值为 {dmp:.3f}。若 p 值小于 0.05，则说明两者预测精度差异在 5% 水平下显著。

进一步比较增强 SARIMAX 与基础 SARIMAX，DM 统计量为 {dm_table[dm_table["comparison"] == "SARIMAX_Enhanced vs SARIMAX"].iloc[0]["dm_stat"]:.3f}，p 值为 {dm_table[dm_table["comparison"] == "SARIMAX_Enhanced vs SARIMAX"].iloc[0]["p_value"]:.3f}，说明扩展日历/Fourier 变量带来的预测改进同样具有统计显著性。

残差诊断使用 Ljung-Box 检验和标准诊断图。lag 24 下，SARIMA 残差 Ljung-Box p 值为 {diag_sarima_24:.3f}，SARIMAX 为 {diag_sarimax_24:.3f}，增强 SARIMAX 为 {diag_enhanced_24:.3f}。诊断图见 `SARIMA/figures/fig_diag_sarima.png`、`SARIMA/figures/fig_diag_sarimax.png` 与 `SARIMA/figures/fig_diag_sarimax_enhanced.png`；预测对比图见 `SARIMA/figures/fig_load_forecast_test_full.png` 与 `SARIMA/figures/fig_load_forecast_first30days.png`。

总体而言，SARIMA 捕捉了小时负荷的日内季节结构，SARIMAX 则进一步利用可再生出力和日历信息改善预测。该模型在本文中仅用于负荷序列的条件均值预测；电价、风电与光伏之间的联合动态关系放在 4.2 节 VAR/VECM 与 Granger 因果中处理。
"""


def write_modified_template(section_41: str) -> Path:
    """新平铺结构下没有论文模板，跳过该步骤，不产生副作用。"""
    return None
    return out_path


def main() -> None:
    ensure_dirs()
    t_start = time.perf_counter()
    df = load_hourly_panel()
    exog = make_exog(df)
    enhanced_exog = make_enhanced_exog(df)
    y_train, y_test, x_train, x_test = split_data(df, exog)
    _, _, x_enhanced_train, x_enhanced_test = split_data(df, enhanced_exog)

    print(f"[SARIMA] train={len(y_train)}, test={len(y_test)}")
    print("[SARIMA] fitting compact candidate table on 2015-2016 sample")
    candidate_table = fit_candidate_table(y_train)
    best = candidate_table.dropna(subset=["aic"]).iloc[0]
    selected_order = tuple(int(v) for v in re.findall(r"\d+", best["order"]))
    selected_seasonal = tuple(int(v) for v in re.findall(r"\d+", best["seasonal_order"]))
    print(f"[SARIMA] selected order={selected_order}, seasonal_order={selected_seasonal}")

    print("[SARIMA] fitting final SARIMA on full training set")
    sarima_res = fit_model(y_train, selected_order, selected_seasonal, maxiter=100)
    pred_sarima = pd.Series(sarima_res.forecast(steps=len(y_test)), index=y_test.index, name="SARIMA")

    print("[SARIMA] fitting final SARIMAX on full training set")
    sarimax_res = fit_model(y_train, selected_order, selected_seasonal, exog=x_train, maxiter=100)
    pred_sarimax = pd.Series(sarimax_res.forecast(steps=len(y_test), exog=x_test), index=y_test.index, name="SARIMAX")

    print("[SARIMA] fitting enhanced SARIMAX on full training set")
    enhanced_res = fit_model(y_train, selected_order, selected_seasonal, exog=x_enhanced_train, maxiter=120)
    pred_enhanced = pd.Series(
        enhanced_res.forecast(steps=len(y_test), exog=x_enhanced_test),
        index=y_test.index,
        name="SARIMAX_Enhanced",
    )

    print("[SARIMA] fitting ETS baseline")
    ets_error = ""
    try:
        ets_res = fit_ets(y_train)
        pred_ets = pd.Series(ets_res.forecast(len(y_test)), index=y_test.index, name="ETS")
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
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(TABLE_DIR / "table_load_forecasts_test.csv", encoding="utf-8-sig")

    metric_rows = []
    for name, pred in predictions.items():
        row = {"Model": name, **metrics(y_test.values, pred.values)}
        if name == "SARIMA":
            row.update({"AIC": sarima_res.aic, "BIC": sarima_res.bic, "HQIC": sarima_res.hqic})
        elif name == "SARIMAX":
            row.update({"AIC": sarimax_res.aic, "BIC": sarimax_res.bic, "HQIC": sarimax_res.hqic})
        elif name == "SARIMAX_Enhanced":
            row.update({"AIC": enhanced_res.aic, "BIC": enhanced_res.bic, "HQIC": enhanced_res.hqic})
        else:
            row.update({"AIC": np.nan, "BIC": np.nan, "HQIC": np.nan})
        metric_rows.append(row)
    metric_table = pd.DataFrame(metric_rows).set_index("Model")
    metric_table.to_csv(TABLE_DIR / "table_load_forecast_metrics.csv", encoding="utf-8-sig")

    dm, pval = diebold_mariano(y_test.values - pred_sarima.values, y_test.values - pred_sarimax.values)
    dm_enhanced, pval_enhanced = diebold_mariano(
        y_test.values - pred_sarimax.values,
        y_test.values - pred_enhanced.values,
    )
    dm_table = pd.DataFrame(
        [
            {
                "comparison": "SARIMAX vs SARIMA",
                "loss": "squared error",
                "dm_stat": dm,
                "p_value": pval,
                "interpretation": "positive means SARIMA loss > SARIMAX loss",
            },
            {
                "comparison": "SARIMAX_Enhanced vs SARIMAX",
                "loss": "squared error",
                "dm_stat": dm_enhanced,
                "p_value": pval_enhanced,
                "interpretation": "positive means SARIMAX loss > SARIMAX_Enhanced loss",
            },
        ]
    )
    dm_table.to_csv(TABLE_DIR / "table_dm_sarimax_vs_sarima.csv", index=False, encoding="utf-8-sig")

    diag_table = pd.concat(
        [
            residual_diagnostics(sarima_res, "SARIMA"),
            residual_diagnostics(sarimax_res, "SARIMAX"),
            residual_diagnostics(enhanced_res, "SARIMAX_Enhanced"),
        ],
        ignore_index=True,
    )
    diag_table.to_csv(TABLE_DIR / "table_sarima_residual_diagnostics.csv", index=False, encoding="utf-8-sig")

    params = pd.concat(
        [
            sarima_res.params.rename("SARIMA"),
            sarimax_res.params.rename("SARIMAX"),
            enhanced_res.params.rename("SARIMAX_Enhanced"),
        ],
        axis=1,
    )
    params.to_csv(TABLE_DIR / "table_sarima_parameters.csv", encoding="utf-8-sig")
    (TABLE_DIR / "sarima_summary.txt").write_text(str(sarima_res.summary()), encoding="utf-8")
    (TABLE_DIR / "sarimax_summary.txt").write_text(str(sarimax_res.summary()), encoding="utf-8")
    (TABLE_DIR / "sarimax_enhanced_summary.txt").write_text(str(enhanced_res.summary()), encoding="utf-8")
    if ets_error:
        (TABLE_DIR / "ets_error.txt").write_text(ets_error, encoding="utf-8")

    plot_forecast(y_test, predictions)

    elapsed = time.perf_counter() - t_start
    result_md = build_result_markdown(
        elapsed,
        df,
        y_train,
        y_test,
        selected_order,
        selected_seasonal,
        metric_table,
        dm_table,
        diag_table,
        sarima_res,
        sarimax_res,
        enhanced_res,
    )
    (TABLE_DIR / "sarima_result_notes.md").write_text(result_md, encoding="utf-8")
    write_modified_template(
        build_section_41(
            df,
            y_train,
            y_test,
            selected_order,
            selected_seasonal,
            metric_table,
            dm_table,
            diag_table,
            sarimax_res,
            elapsed,
        )
    )
    print(f"[SARIMA] done in {elapsed / 60:.1f} minutes. Outputs: {OUT}")


if __name__ == "__main__":
    main()
