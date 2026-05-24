# -*- coding: utf-8 -*-
"""
4.3 GARCH 族模型（修订版）
================================================================
本脚本严格按照 `GARCH模型结果总结_完整逻辑版.md` 的建模思路实现：

阶段一 —— 基础 GARCH 族探索
    1. 构造电价一阶差分 r_t = p_t - p_{t-1}
    2. 对 r_t 做 ARCH-LM 检验
    3. 估计五个基础模型（均值方程为 AR(1)）：
        - GARCH(1,1)-normal
        - GARCH(1,1)-t
        - EGARCH(1,1)-t
        - GJR-GARCH(1,1,1)-t
        - GARCH-X(1,1)-t（方差方程加入滞后 DE_renew_share）
    4. 对正态 vs Student-t 做似然比检验
    5. 对 GARCH-X 中 δ=0 做似然比检验
    6. 输出参数、AIC/BIC、残差诊断、标准化残差图

阶段二 —— 主模型：ARX(7)-GARCH(1,1)-skew-t
    1. 在均值方程中引入 7 阶滞后 r_t 与外生变量 X_t
        X_t = (renew_lag1, renew_diff_lag1, |renew_diff_lag1|,
               dow_Sat, dow_Sun)
    2. 方差方程 GARCH(1,1)
    3. 残差分布 Hansen (1994) skewed-t (eta, lambda)
    4. 残差诊断（Ljung-Box on z 与 z^2 at lag 10/20/30, ARCH-LM lag10）

阶段三 —— 子样本对比
    1. 2015-2016 与 2017-2018 两个子样本
    2. 使用同一 ARX(7)-GARCH(1,1)-skew-t 设定
    3. 比较 r_t 描述性差异、方差参数、厚尾/偏态参数与残差诊断

输出：
    OUT_DIR 下保存 csv 表格、png 图与 txt 报告
"""

import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import chi2

from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")


# ============================================================
# 0. 参数设置
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "tables" / "garch_results"
FIG_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DAILY_PARQUET = DATA_DIR / "de_panel_daily.parquet"
DAILY_CSV = DATA_DIR / "de_panel_daily.csv"

PRICE_COL = "DE_price_day_ahead"
RENEW_SHARE_COL = "DE_renew_share"

START_DATE = "2015-01-01"
END_DATE = "2018-09-30"

ARCH_LM_LAGS = 10
DIAG_LAGS = (10, 20, 30)

# 基础 GARCH 族探索使用标准化后的 r_t 以与历史结果保持一致；
# 主模型（ARX(7)-GARCH-skew-t）同样在标准化序列上估计，
# 这样所有阶段的参数都可以在同一尺度下比较。
STANDARDIZE_RETURN = True

# 主模型 / 子样本残差诊断的“最小通过 p 值”阈值
DIAG_ALPHA = 0.05

# 子样本切分
SUBSAMPLES = {
    "2015_2016": ("2015-01-01", "2016-12-31"),
    "2017_2018": ("2017-01-01", "2018-09-30"),
}


# ============================================================
# 1. 工具函数
# ============================================================

def read_daily_data():
    """读取日频数据。优先 parquet，否则 csv。"""
    if DAILY_PARQUET.exists():
        df = pd.read_parquet(DAILY_PARQUET)
    elif DAILY_CSV.exists():
        df = pd.read_csv(DAILY_CSV)
    else:
        raise FileNotFoundError(
            "没有找到 de_panel_daily.parquet 或 de_panel_daily.csv，"
            "请把数据文件放到当前目录。"
        )

    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
    else:
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df = df.loc[START_DATE:END_DATE].copy()

    for col in [PRICE_COL, RENEW_SHARE_COL]:
        if col not in df.columns:
            raise ValueError(f"数据中缺少变量：{col}")

    return df


def prepare_garch_data(df):
    """
    构造 GARCH 建模数据。

    1) 一阶差分 r_t = p_t - p_{t-1}
    2) 标准化（保持均值约 0、标准差约 1，便于优化）
    3) 构造 ARX(7) 用到的外生变量
    """
    data = pd.DataFrame(index=df.index)
    data["price"] = df[PRICE_COL]
    data["r_raw"] = df[PRICE_COL].diff()

    # 滞后变量
    data["renew_share"] = df[RENEW_SHARE_COL]
    data["renew_share_lag1"] = data["renew_share"].shift(1)
    data["renew_diff_lag1"] = data["renew_share"].diff().shift(1)
    data["abs_renew_diff_lag1"] = data["renew_diff_lag1"].abs()

    # 星期虚拟变量 (Mon=0 ... Sun=6)
    dow = data.index.dayofweek
    for d in range(7):
        data[f"dow_{d}"] = (dow == d).astype(float)

    data = data.dropna().copy()

    # 标准化 r_t
    if STANDARDIZE_RETURN:
        mu = data["r_raw"].mean()
        sd = data["r_raw"].std()
        data["r"] = (data["r_raw"] - mu) / sd
        scale_info = {"mean_raw": float(mu), "std_raw": float(sd)}
    else:
        data["r"] = data["r_raw"]
        scale_info = {"mean_raw": 0.0, "std_raw": 1.0}

    # 外生变量（用于 GARCH-X 标准化版本，便于 SLSQP 数值优化）
    for col in ["renew_share_lag1", "renew_diff_lag1", "abs_renew_diff_lag1"]:
        m = data[col].mean()
        s = data[col].std()
        if s > 0:
            data[col + "_std"] = (data[col] - m) / s
        else:
            data[col + "_std"] = 0.0
        scale_info[col + "_mean"] = float(m)
        scale_info[col + "_std"] = float(s)

    return data, scale_info


def arch_lm_test(series, lags=10):
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(series, nlags=lags)
    return {
        "lm_stat": float(lm_stat),
        "lm_pvalue": float(lm_pvalue),
        "f_stat": float(f_stat),
        "f_pvalue": float(f_pvalue),
    }


def ljung_box_test(series, lags):
    out = acorr_ljungbox(series, lags=list(lags), return_df=True)
    out.index.name = "lag"
    return out


def save_series_plot(data, out_dir):
    plt.figure(figsize=(12, 4))
    plt.plot(data.index, data["price"], linewidth=1, color="#1B4D79")
    plt.title("Daily day-ahead electricity price")
    plt.xlabel("Date")
    plt.ylabel("EUR/MWh")
    plt.tight_layout()
    plt.savefig(out_dir / "01_price_level.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(data.index, data["r_raw"], linewidth=1, color="#1B4D79")
    plt.axhline(0, linestyle="--", linewidth=1, color="grey")
    plt.title(r"Electricity price first difference: $r_t = p_t - p_{t-1}$")
    plt.xlabel("Date")
    plt.ylabel("EUR/MWh")
    plt.tight_layout()
    plt.savefig(out_dir / "02_price_difference.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(data.index, data["renew_share"], linewidth=1, color="#1B4D79")
    plt.title("Renewable energy share")
    plt.xlabel("Date")
    plt.ylabel("Share")
    plt.tight_layout()
    plt.savefig(out_dir / "03_renew_share.png", dpi=300)
    plt.close()


def save_acf_pacf(series, out_dir, name):
    plt.figure(figsize=(8, 4))
    plot_acf(series, lags=40, ax=plt.gca())
    plt.title(f"ACF of {name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"acf_{name}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4))
    plot_pacf(series, lags=40, ax=plt.gca(), method="ywm")
    plt.title(f"PACF of {name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"pacf_{name}.png", dpi=300)
    plt.close()


# ============================================================
# 2. 基础 GARCH 族模型（arch 包）
# ============================================================

def fit_basic_models(y):
    """估计 GARCH / EGARCH / GJR-GARCH，均值方程为 AR(1)。"""
    from arch import arch_model

    specs = [
        dict(name="GARCH11_normal", vol="GARCH", p=1, o=0, q=1, dist="normal"),
        dict(name="GARCH11_t",      vol="GARCH", p=1, o=0, q=1, dist="t"),
        dict(name="EGARCH11_t",     vol="EGARCH", p=1, o=1, q=1, dist="t"),
        dict(name="GJR_GARCH11_t",  vol="GARCH", p=1, o=1, q=1, dist="t"),
    ]
    models = {}
    for spec in specs:
        am = arch_model(
            y, mean="AR", lags=1,
            vol=spec["vol"], p=spec["p"], o=spec["o"], q=spec["q"],
            dist=spec["dist"], rescale=False,
        )
        res = am.fit(disp="off", show_warning=False)
        models[spec["name"]] = res
    return models


def extract_result_table(models):
    rows = []
    for name, res in models.items():
        row = {
            "model": name,
            "nobs": int(res.nobs),
            "loglik": float(res.loglikelihood),
            "aic": float(res.aic),
            "bic": float(res.bic),
        }
        for p in res.params.index:
            row[p] = float(res.params[p])
            row[p + "_pvalue"] = float(res.pvalues[p])
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostics_from_arch_result(res, name, lags=DIAG_LAGS):
    z = pd.Series(res.std_resid).dropna()
    z2 = z ** 2
    lb_z = ljung_box_test(z, lags=lags)
    lb_z2 = ljung_box_test(z2, lags=lags)
    arch_lm = arch_lm_test(z, lags=10)
    row = {"model": name}
    for L in lags:
        row[f"LB_z_lag{L}_pvalue"] = float(lb_z.loc[L, "lb_pvalue"])
        row[f"LB_z2_lag{L}_pvalue"] = float(lb_z2.loc[L, "lb_pvalue"])
    row["ARCH_LM_lag10_pvalue"] = float(arch_lm["lm_pvalue"])
    row["min_diag_pvalue"] = float(min(row[k] for k in row if k.endswith("_pvalue")))
    return row


def save_basic_residual_plots(models, out_dir):
    for name, res in models.items():
        idx = res.resid.index
        z = pd.Series(res.std_resid, index=idx).dropna()
        z2 = z ** 2
        cond_vol = pd.Series(res.conditional_volatility, index=idx).dropna()

        plt.figure(figsize=(12, 4))
        plt.plot(z.index, z.values, linewidth=0.7, color="#1B4D79")
        plt.axhline(0, linestyle="--", linewidth=1, color="grey")
        plt.title(f"Standardized residuals: {name}")
        plt.xlabel("Date"); plt.ylabel(r"$z_t$")
        plt.tight_layout()
        plt.savefig(out_dir / f"std_resid_{name}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(z2.index, z2.values, linewidth=0.7, color="#1B4D79")
        plt.title(f"Squared standardized residuals: {name}")
        plt.xlabel("Date"); plt.ylabel(r"$z_t^2$")
        plt.tight_layout()
        plt.savefig(out_dir / f"std_resid_squared_{name}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(cond_vol.index, cond_vol.values, linewidth=1, color="#1B4D79")
        plt.title(f"Conditional volatility: {name}")
        plt.xlabel("Date"); plt.ylabel(r"$\sigma_t$")
        plt.tight_layout()
        plt.savefig(out_dir / f"conditional_volatility_{name}.png", dpi=300)
        plt.close()


def likelihood_ratio_test(res_restricted, res_unrestricted, df_diff):
    lr = 2 * (res_unrestricted.loglikelihood - res_restricted.loglikelihood)
    pvalue = 1 - chi2.cdf(lr, df=df_diff)
    return float(lr), float(pvalue)


# ============================================================
# 3. GARCH-X(1,1)-t —— 自定义似然 + 似然比检验
# ============================================================

def student_t_logpdf_standardized(z, nu):
    if nu <= 2:
        return np.full_like(z, -1e10)
    c = (gammaln((nu + 1) / 2) - gammaln(nu / 2)
         - 0.5 * np.log(np.pi * (nu - 2)))
    return c - ((nu + 1) / 2) * np.log(1 + (z ** 2) / (nu - 2))


def garchx_negloglik(theta, y, x):
    mu, phi, omega, alpha, beta, delta, nu = theta
    n = len(y)
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    if omega <= 1e-8 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999 or nu <= 2.05:
        return 1e12
    eps[0] = y[0] - mu
    sigma2[0] = max(np.var(y), 1e-6)
    for t in range(1, n):
        eps[t] = y[t] - mu - phi * y[t - 1]
        raw_s2 = (omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
                  + delta * x[t - 1])
        sigma2[t] = max(raw_s2, 1e-8)
    z = eps / np.sqrt(sigma2)
    logpdf = student_t_logpdf_standardized(z, nu) - 0.5 * np.log(sigma2)
    if not np.all(np.isfinite(logpdf)):
        return 1e12
    return -np.sum(logpdf[1:])


def fit_garchx_t(y, x):
    y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
    theta0 = np.array([np.mean(y), 0.1, 0.05, 0.08, 0.88, 0.01, 8.0])
    bounds = [(None, None), (-0.99, 0.99), (1e-8, None),
              (0.0, 0.999), (0.0, 0.999), (None, None), (2.05, 100.0)]
    constraints = [{"type": "ineq", "fun": lambda th: 0.9999 - th[3] - th[4]}]
    opt = minimize(garchx_negloglik, theta0, args=(y, x),
                   method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 5000, "ftol": 1e-10, "disp": False})

    theta = opt.x
    loglik = -opt.fun
    k = len(theta); n = len(y)
    aic = 2 * k - 2 * loglik
    bic = np.log(n) * k - 2 * loglik

    mu, phi, omega, alpha, beta, delta, nu = theta
    eps = np.zeros(n); sigma2 = np.zeros(n)
    eps[0] = y[0] - mu; sigma2[0] = max(np.var(y), 1e-6)
    for t in range(1, n):
        eps[t] = y[t] - mu - phi * y[t - 1]
        raw_s2 = (omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
                  + delta * x[t - 1])
        sigma2[t] = max(raw_s2, 1e-8)
    z = eps / np.sqrt(sigma2)

    return {
        "model": "GARCHX11_t", "success": bool(opt.success),
        "params": dict(mu=theta[0], phi=theta[1], omega=theta[2],
                       alpha=theta[3], beta=theta[4], delta=theta[5], nu=theta[6]),
        "loglik": float(loglik), "aic": float(aic), "bic": float(bic),
        "eps": eps, "sigma2": sigma2, "std_resid": z,
    }


def garchx_restricted_negloglik(theta, y):
    """δ=0 的受限模型 = AR(1)-GARCH(1,1)-t。"""
    mu, phi, omega, alpha, beta, nu = theta
    n = len(y); eps = np.zeros(n); sigma2 = np.zeros(n)
    if omega <= 1e-8 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999 or nu <= 2.05:
        return 1e12
    eps[0] = y[0] - mu; sigma2[0] = max(np.var(y), 1e-6)
    for t in range(1, n):
        eps[t] = y[t] - mu - phi * y[t - 1]
        s2 = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        sigma2[t] = max(s2, 1e-8)
    z = eps / np.sqrt(sigma2)
    logpdf = student_t_logpdf_standardized(z, nu) - 0.5 * np.log(sigma2)
    if not np.all(np.isfinite(logpdf)):
        return 1e12
    return -np.sum(logpdf[1:])


def fit_garchx_restricted_t(y):
    y = np.asarray(y, dtype=float)
    theta0 = np.array([np.mean(y), 0.1, 0.05, 0.08, 0.88, 8.0])
    bounds = [(None, None), (-0.99, 0.99), (1e-8, None),
              (0.0, 0.999), (0.0, 0.999), (2.05, 100.0)]
    constraints = [{"type": "ineq", "fun": lambda th: 0.9999 - th[3] - th[4]}]
    opt = minimize(garchx_restricted_negloglik, theta0, args=(y,),
                   method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 5000, "ftol": 1e-10, "disp": False})
    return {"params": opt.x, "loglik": float(-opt.fun)}


# ============================================================
# 4. 主模型：ARX(7)-GARCH(1,1)-skew-t
# ============================================================

ARX_EXOG_BASE = [
    "renew_share_lag1",
    "renew_diff_lag1",
    "abs_renew_diff_lag1",
    "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
]


def build_arx_exog(data):
    """组装 ARX(7) 主模型的外生变量矩阵。
    使用原始（未标准化）的可再生能源变量；
    星期虚拟变量取 dow_1..dow_6（以周一为参照组）。
    """
    X = data[ARX_EXOG_BASE].copy()
    X.columns = ["renew_lag1", "renew_diff_lag1", "abs_renew_diff_lag1",
                 "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6"]
    return X


def fit_arx_garch_skewt(y, X, lags=7):
    """ARX(p)-GARCH(1,1)-skew-t 估计。"""
    from arch import arch_model
    am = arch_model(
        y, x=X, mean="ARX", lags=lags,
        vol="GARCH", p=1, o=0, q=1,
        dist="skewstudent", rescale=False,
    )
    res = am.fit(disp="off", show_warning=False)
    return res


def diag_pvalues(res, lags=DIAG_LAGS):
    z = pd.Series(res.std_resid).dropna()
    z2 = z ** 2
    lb_z = ljung_box_test(z, lags=lags)
    lb_z2 = ljung_box_test(z2, lags=lags)
    arch_lm = arch_lm_test(z, lags=10)
    out = {"ARCH_LM_lag10": float(arch_lm["lm_pvalue"])}
    for L in lags:
        out[f"LB_z_lag{L}"] = float(lb_z.loc[L, "lb_pvalue"])
        out[f"LB_z2_lag{L}"] = float(lb_z2.loc[L, "lb_pvalue"])
    out["min_p"] = float(min(out.values()))
    return out


def summarize_arx_result(res, sample_name):
    row = {
        "sample": sample_name,
        "nobs": int(res.nobs),
        "loglik": float(res.loglikelihood),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }
    for p in res.params.index:
        row[p] = float(res.params[p])
        row[p + "_pvalue"] = float(res.pvalues[p])
    diag = diag_pvalues(res)
    for k, v in diag.items():
        row[f"diag_{k}"] = v
    row["passes_5pct_diag"] = bool(diag["min_p"] >= DIAG_ALPHA)
    return row


def save_arx_plots(res, sample_name, out_dir):
    idx = res.resid.index
    z = pd.Series(res.std_resid, index=idx).dropna()
    z2 = z ** 2
    cond_vol = pd.Series(res.conditional_volatility, index=idx).dropna()

    plt.figure(figsize=(12, 4))
    plt.plot(z.index, z.values, linewidth=0.7, color="#1B4D79")
    plt.axhline(0, linestyle="--", linewidth=1, color="grey")
    plt.title(f"Standardized residuals: ARX(7)-GARCH(1,1)-skew-t ({sample_name})")
    plt.xlabel("Date"); plt.ylabel(r"$z_t$")
    plt.tight_layout()
    plt.savefig(out_dir / f"arx_std_resid_{sample_name}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(z2.index, z2.values, linewidth=0.7, color="#1B4D79")
    plt.title(f"Squared standardized residuals: ARX(7)-GARCH(1,1)-skew-t ({sample_name})")
    plt.xlabel("Date"); plt.ylabel(r"$z_t^2$")
    plt.tight_layout()
    plt.savefig(out_dir / f"arx_std_resid_squared_{sample_name}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(cond_vol.index, cond_vol.values, linewidth=1, color="#1B4D79")
    plt.title(f"Conditional volatility: ARX(7)-GARCH(1,1)-skew-t ({sample_name})")
    plt.xlabel("Date"); plt.ylabel(r"$\sigma_t$")
    plt.tight_layout()
    plt.savefig(out_dir / f"arx_conditional_volatility_{sample_name}.png", dpi=300)
    plt.close()


def mirror_key_figures_to_top_level() -> None:
    """Expose the main GARCH figures in figures/ for report and presentation use."""
    mapping = {
        "arx_conditional_volatility_full.png": "fig_arx_cond_vol_full.png",
        "arx_conditional_volatility_2015_2016.png": "fig_arx_cond_vol_2015_2016.png",
        "arx_conditional_volatility_2017_2018.png": "fig_arx_cond_vol_2017_2018.png",
    }
    for src_name, dst_name in mapping.items():
        src = OUT_DIR / src_name
        if src.exists():
            shutil.copy2(src, FIG_DIR / dst_name)


# ============================================================
# 5. 描述性统计
# ============================================================

def subsample_descriptive(data):
    rows = []
    rows.append({
        "sample": "full",
        "n": int(len(data)),
        "r_mean": float(data["r_raw"].mean()),
        "r_std": float(data["r_raw"].std()),
        "renew_lag1_mean": float(data["renew_share_lag1"].mean()),
        "renew_lag1_std": float(data["renew_share_lag1"].std()),
        "abs_renew_diff_lag1_mean": float(data["abs_renew_diff_lag1"].mean()),
    })
    for name, (s, e) in SUBSAMPLES.items():
        sub = data.loc[s:e]
        rows.append({
            "sample": name,
            "n": int(len(sub)),
            "r_mean": float(sub["r_raw"].mean()),
            "r_std": float(sub["r_raw"].std()),
            "renew_lag1_mean": float(sub["renew_share_lag1"].mean()),
            "renew_lag1_std": float(sub["renew_share_lag1"].std()),
            "abs_renew_diff_lag1_mean": float(sub["abs_renew_diff_lag1"].mean()),
        })
    return pd.DataFrame(rows)


# ============================================================
# 6. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("4.3 GARCH 族模型（修订版）")
    print("=" * 80)

    df = read_daily_data()
    data, scale_info = prepare_garch_data(df)

    print(f"[INFO] 数据范围：{data.index.min().date()} -- {data.index.max().date()}")
    print(f"[INFO] 样本数：{len(data)}")

    pd.Series(scale_info).to_csv(
        OUT_DIR / "scale_info.csv",
        header=["value"], encoding="utf-8-sig"
    )
    data.to_csv(OUT_DIR / "garch_model_input_data.csv", encoding="utf-8-sig")

    # ------------------------------------------------------------
    # (A) 描述性统计 + 基础图
    # ------------------------------------------------------------
    desc_table = subsample_descriptive(data)
    desc_table.to_csv(OUT_DIR / "descriptive_subsamples.csv",
                      index=False, encoding="utf-8-sig")
    print("\n[INFO] 子样本描述性统计：")
    print(desc_table)

    save_series_plot(data, OUT_DIR)
    save_acf_pacf(data["r"], OUT_DIR, "price_difference")

    # ------------------------------------------------------------
    # (B) ARCH-LM 预检验
    # ------------------------------------------------------------
    print("\n[INFO] ARCH-LM 检验（电价差分序列）：")
    arch_test = arch_lm_test(data["r"], lags=ARCH_LM_LAGS)
    pd.DataFrame([arch_test]).to_csv(
        OUT_DIR / "arch_lm_before_garch.csv",
        index=False, encoding="utf-8-sig"
    )
    print(arch_test)

    # ------------------------------------------------------------
    # (C) 基础 GARCH 族模型探索
    # ------------------------------------------------------------
    print("\n[INFO] 估计基础 GARCH 族模型...")
    basic_models = fit_basic_models(data["r"])
    basic_table = extract_result_table(basic_models)

    diag_rows = [diagnostics_from_arch_result(res, name) for name, res in basic_models.items()]
    basic_diag = pd.DataFrame(diag_rows)

    # GARCH-X 自定义估计
    print("[INFO] 估计 GARCH-X(1,1)-t（自定义似然）...")
    garchx_res = fit_garchx_t(
        y=data["r"].values,
        x=data["renew_share_lag1_std"].values,
    )
    # 受限模型用于 LR 检验
    restr = fit_garchx_restricted_t(data["r"].values)
    lr_x = 2 * (garchx_res["loglik"] - restr["loglik"])
    p_x = float(1 - chi2.cdf(lr_x, df=1))
    print(f"[INFO] GARCH-X δ=0 似然比检验：LR={lr_x:.4f}, p={p_x:.4f}")

    # GARCH-X 行加入比较表
    gx_row = {
        "model": "GARCHX11_t", "nobs": len(data["r"]),
        "loglik": garchx_res["loglik"], "aic": garchx_res["aic"], "bic": garchx_res["bic"],
        **{f"x_{k}": v for k, v in garchx_res["params"].items()},
    }
    basic_table_extra = pd.concat([basic_table, pd.DataFrame([gx_row])], ignore_index=True)
    basic_table_extra.to_csv(OUT_DIR / "basic_model_result_table.csv",
                             index=False, encoding="utf-8-sig")

    # GARCH-X 残差诊断
    z_gx = pd.Series(garchx_res["std_resid"]).dropna()
    lb_z = ljung_box_test(z_gx, lags=DIAG_LAGS)
    lb_z2 = ljung_box_test(z_gx ** 2, lags=DIAG_LAGS)
    arch_gx = arch_lm_test(z_gx, lags=10)
    gx_diag = {"model": "GARCHX11_t"}
    for L in DIAG_LAGS:
        gx_diag[f"LB_z_lag{L}_pvalue"] = float(lb_z.loc[L, "lb_pvalue"])
        gx_diag[f"LB_z2_lag{L}_pvalue"] = float(lb_z2.loc[L, "lb_pvalue"])
    gx_diag["ARCH_LM_lag10_pvalue"] = float(arch_gx["lm_pvalue"])
    gx_diag["min_diag_pvalue"] = float(min(v for k, v in gx_diag.items()
                                           if k.endswith("_pvalue")))
    basic_diag = pd.concat([basic_diag, pd.DataFrame([gx_diag])], ignore_index=True)
    basic_diag.to_csv(OUT_DIR / "basic_model_diagnostics.csv",
                      index=False, encoding="utf-8-sig")

    print("\n[INFO] 基础模型 AIC/BIC：")
    print(basic_table_extra[["model", "loglik", "aic", "bic"]])
    print("\n[INFO] 基础模型诊断：")
    print(basic_diag)

    # 正态 vs t LR 检验
    lr, pvalue = likelihood_ratio_test(
        basic_models["GARCH11_normal"], basic_models["GARCH11_t"], df_diff=1)
    pd.DataFrame([{
        "restricted_model": "GARCH11_normal",
        "unrestricted_model": "GARCH11_t",
        "LR_stat": lr, "df_diff": 1, "pvalue": pvalue,
    }]).to_csv(OUT_DIR / "lrt_normal_vs_t.csv",
               index=False, encoding="utf-8-sig")
    print(f"\n[INFO] LR(normal vs t)={lr:.4f}, p={pvalue:.3e}")

    # GARCH-X LR 检验
    pd.DataFrame([{
        "restricted": "AR1_GARCH11_t (delta=0)",
        "unrestricted": "GARCHX11_t",
        "ll_restricted": restr["loglik"],
        "ll_unrestricted": garchx_res["loglik"],
        "LR_stat": lr_x, "df": 1, "pvalue": p_x,
    }]).to_csv(OUT_DIR / "lrt_garchx_delta.csv",
               index=False, encoding="utf-8-sig")

    save_basic_residual_plots(basic_models, OUT_DIR)

    # ------------------------------------------------------------
    # (D) 主模型：ARX(7)-GARCH(1,1)-skew-t（全样本）
    # ------------------------------------------------------------
    print("\n[INFO] 估计主模型：ARX(7)-GARCH(1,1)-skew-t（全样本）...")
    X_full = build_arx_exog(data)
    res_arx_full = fit_arx_garch_skewt(data["r"], X_full, lags=7)

    arx_full_summary = summarize_arx_result(res_arx_full, "full")
    pd.DataFrame([arx_full_summary]).to_csv(
        OUT_DIR / "arx_garch_skewt_full_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    # 主要参数表
    full_params = pd.DataFrame({
        "param": res_arx_full.params.index,
        "estimate": res_arx_full.params.values,
        "std_err": res_arx_full.std_err.values,
        "tvalue": res_arx_full.tvalues.values,
        "pvalue": res_arx_full.pvalues.values,
    })
    full_params.to_csv(OUT_DIR / "arx_full_param_table.csv",
                       index=False, encoding="utf-8-sig")
    print("\n[INFO] 主模型（全样本）参数：")
    print(full_params.to_string(index=False))

    print("\n[INFO] 主模型（全样本）残差诊断：")
    print(diag_pvalues(res_arx_full))

    save_arx_plots(res_arx_full, "full", OUT_DIR)

    # ------------------------------------------------------------
    # (E) 子样本：2015-2016 / 2017-2018
    # ------------------------------------------------------------
    print("\n[INFO] 子样本估计：")
    sub_summaries = [arx_full_summary]
    sub_param_frames = [full_params.assign(sample="full")]
    for name, (s, e) in SUBSAMPLES.items():
        sub_data = data.loc[s:e].copy()
        # 重要：子样本里 dow 哑变量需要重新构造
        X_sub = build_arx_exog(sub_data)
        y_sub = sub_data["r"]
        res_sub = fit_arx_garch_skewt(y_sub, X_sub, lags=7)
        summ = summarize_arx_result(res_sub, name)
        sub_summaries.append(summ)
        sub_param_frames.append(pd.DataFrame({
            "param": res_sub.params.index,
            "estimate": res_sub.params.values,
            "std_err": res_sub.std_err.values,
            "tvalue": res_sub.tvalues.values,
            "pvalue": res_sub.pvalues.values,
            "sample": name,
        }))
        save_arx_plots(res_sub, name, OUT_DIR)
        print(f"\n[INFO] 子样本 {name} 残差诊断：")
        print(diag_pvalues(res_sub))

    pd.DataFrame(sub_summaries).to_csv(
        OUT_DIR / "arx_garch_skewt_full_and_subsamples.csv",
        index=False, encoding="utf-8-sig"
    )
    pd.concat(sub_param_frames, ignore_index=True).to_csv(
        OUT_DIR / "arx_param_full_and_subsamples.csv",
        index=False, encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # (F) 报告文本片段
    # ------------------------------------------------------------
    notes = OUT_DIR / "garch_report_notes.txt"
    with open(notes, "w", encoding="utf-8") as f:
        f.write("4.3 GARCH 族模型（修订版）结果说明\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. 基础 GARCH 族模型（AR(1) 均值方程，标准化 r_t）\n")
        f.write("   ARCH-LM、LR(normal vs t)、GARCH-X(δ=0) LR 检验结果见 csv。\n\n")
        f.write("2. 主模型：ARX(7)-GARCH(1,1)-skew-t（全样本）\n")
        f.write(f"   n={arx_full_summary['nobs']}, log-lik={arx_full_summary['loglik']:.2f}, "
                f"AIC={arx_full_summary['aic']:.2f}, BIC={arx_full_summary['bic']:.2f}\n")
        f.write(f"   通过 5% 诊断: {arx_full_summary['passes_5pct_diag']}, "
                f"min p={arx_full_summary['diag_min_p']:.4f}\n\n")
        for s in sub_summaries[1:]:
            f.write(f"   子样本 {s['sample']}: log-lik={s['loglik']:.2f}, "
                    f"AIC={s['aic']:.2f}, BIC={s['bic']:.2f}, "
                    f"通过 5% 诊断: {s['passes_5pct_diag']}, "
                    f"min p={s['diag_min_p']:.4f}\n")

    mirror_key_figures_to_top_level()
    print(f"\n[DONE] 全部结果保存到 {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
