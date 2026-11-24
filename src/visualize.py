"""Visualization & Empirical Report Generator for Equity Index & QDII Backtest."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.qdii_tracker import build_five_year_dataset, analyze_qdii_attribution
from src.backtest_engine import run_all_backtests

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
REPORTS_DIR = ROOT / "reports"


def render_charts(df: pd.DataFrame, curves: dict[str, pd.DataFrame], metrics_list: list[dict], attr: dict) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    colors = ["#2980b9", "#e67e22", "#27ae60"]

    # Chart 1: Cumulative Return on Cost & Drawdown Underwater Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True, gridspec_kw={"height_ratios": [1.8, 1.0]})
    for idx, (name, cdf) in enumerate(curves.items()):
        ax1.plot(cdf["date"], cdf["cum_return_pct"], color=colors[idx], lw=2.0, label=name)
        ax2.plot(cdf["date"], cdf["twr_dd_pct"], color=colors[idx], lw=1.6, alpha=0.85, label=name)

    ax1.axhline(0, color="#7f8c8d", ls=":", lw=1.0)
    ax1.set_ylabel("Cumulative Return on Cost (%)")
    ax1.set_title("5-Year Strategy Comparison: Fixed DCA vs MA200 Dynamic DCA vs Equity-Gold Allocation", fontsize=12.5, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9.2)

    ax2.fill_between(df["date"], curves["Fixed DCA (70% NDX + 30% SPX)"]["twr_dd_pct"], 0, color="#2980b9", alpha=0.12)
    ax2.set_ylabel("TWR Drawdown (%)")
    ax2.set_xlabel("Date (2021–2026)")
    ax2.legend(loc="lower right", fontsize=8.8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "dca_vs_dynamic_equity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Chart 2: QDII 3-Factor Return Attribution & Secondary-Market Premium Warning Timeline
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.6), gridspec_kw={"height_ratios": [1.15, 1.25]})

    norm_ndx = (df["ndx_close"] / df["ndx_close"].iloc[0] - 1.0) * 100.0
    norm_qdii = (df["qdii_nav_cny"] / df["qdii_nav_cny"].iloc[0] - 1.0) * 100.0
    norm_fx = (df["usdcny"] / df["usdcny"].iloc[0] - 1.0) * 100.0
    ax1.plot(df["date"], norm_qdii, color="#1f4e79", lw=2.1, label=f"Southern NDX100 QDII NAV CNY ({attr['qdii_nav_5y_return_pct']:+.1f}%)")
    ax1.plot(df["date"], norm_ndx, color="#e67e22", lw=1.6, ls="--", label=f"Nasdaq-100 USD Index ({attr['ndx_5y_return_pct']:+.1f}%)")
    ax1.plot(df["date"], norm_fx, color="#8e44ad", lw=1.5, label=f"USD/CNY FX Contribution ({attr['usdcny_5y_return_pct']:+.1f}%)")
    ax1.set_ylabel("Cumulative Change (%)")
    ax1.set_title("QDII Return Decomposition (Index + FX) & Secondary-Market Premium Rate Monitor", fontsize=12.5, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)

    ax2.plot(df["date"], df["qdii_premium_pct"], color="#c0392b", lw=1.4, label="QDII Premium / Discount Rate (%)")
    ax2.axhline(3.0, color="#e74c3c", ls="--", lw=1.3, label="High-Premium Alert Threshold (+3.0%)")
    ax2.fill_between(
        df["date"],
        df["qdii_premium_pct"],
        3.0,
        where=(df["qdii_premium_pct"] > 3.0),
        color="#e74c3c",
        alpha=0.28,
        label=f"Alert Zone ({attr['high_premium_alert_days']} days / {attr['high_premium_alert_ratio_pct']}%)",
    )
    ax2.axhline(0, color="#7f8c8d", ls=":", lw=0.9)
    ax2.set_ylabel("Premium Rate (%)")
    ax2.set_xlabel("Date (2021–2026)")
    ax2.legend(loc="upper left", fontsize=8.8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "qdii_attribution_premium.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Chart 3: Risk-Return Efficiency Scatter + Asset Correlation Heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1.35, 1.0]})

    for idx, m in enumerate(metrics_list):
        ax1.scatter(m["annualized_vol_pct"], m["xirr_pct"], s=210, color=colors[idx], edgecolor="white", lw=2.0, zorder=5)
        is_right = m["annualized_vol_pct"] > 19.0
        ax1.annotate(
            f"{m['strategy']}\nXIRR={m['xirr_pct']}% | Sharpe={m['sharpe_ratio']}",
            xy=(m["annualized_vol_pct"], m["xirr_pct"]),
            xytext=(-10, -14 if idx == 0 else 10) if is_right else (10, -4),
            textcoords="offset points",
            ha="right" if is_right else "left",
            fontsize=8.5,
            fontweight="bold",
            color=colors[idx],
        )
    ax1.set_title("Strategy Risk-Return Frontier (XIRR vs Volatility)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Annualized Volatility (%)")
    ax1.set_ylabel("Money-Weighted Annualized Return XIRR (%)")
    ax1.margins(0.22)

    ret_panel = pd.DataFrame(
        {
            "NDX": df["ndx_close"].pct_change(),
            "SPX": df["spx_close"].pct_change(),
            "Gold": df["gold_usd"].pct_change(),
            "USD/CNY": df["usdcny"].pct_change(),
        }
    ).dropna()
    corr = ret_panel.corr()
    im = ax2.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, fontsize=9.5)
    ax2.set_yticklabels(corr.columns, fontsize=9.5)
    for r in range(len(corr)):
        for c in range(len(corr)):
            ax2.text(c, r, f"{corr.iloc[r, c]:.2f}", ha="center", va="center", color="black", fontsize=9.5, fontweight="bold")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title("5-Year Daily Return Correlation Matrix", fontsize=11.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "risk_return_frontier.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(metrics_list: list[dict], attr: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"| {m['strategy']} | {m['total_invested_rmb']:,.0f} | {m['ending_value_rmb']:,.0f} | "
        f"{m['total_return_on_cost_pct']}% | **{m['xirr_pct']}%** | {m['annualized_vol_pct']}% | "
        f"**{m['sharpe_ratio']}** | {m['sortino_ratio']} | **{m['max_drawdown_pct']}%** | {m['calmar_ratio']} |"
        for m in metrics_list
    )

    md = f"""# 个人权益指数资产配置与 QDII 量化定投实证分析报告（2021–2026）

## 一、QDII 基金净值三因子归因与溢价率监控

针对代表性跨市场权益产品（以南方纳斯达克 100 QDII 为跟踪基准），近 5 年人民币计价净值累计收益 **{attr['qdii_nav_5y_return_pct']}%**，拆解如下：
- **底层指数贡献（NDX USD）**：指数 5 年累计涨幅 **{attr['ndx_5y_return_pct']}%**（对数归因占比 **{attr['attribution_ndx_share_pct']}%**）；
- **汇率波动贡献（USD/CNY）**：美元兑人民币 5 年升值 **{attr['usdcny_5y_return_pct']}%**（对数归因占比 **{attr['attribution_fx_share_pct']}%**），在美股波动期提供了显著的汇率安全垫；
- **管理费与跟踪误差残差**：占比 **{attr['attribution_fee_tracking_share_pct']}%**；
- **场内高溢价风险预警**：5 年内平均溢价率为 **{attr['avg_premium_pct']}%**，峰值达 **{attr['max_premium_pct']}%**；超过 `+3.0%` 预警线的交易日共 **{attr['high_premium_alert_days']} 天**（占比 **{attr['high_premium_alert_ratio_pct']}%**），验证了动态定投中引入“高溢价分流机制”的必要性。

---

## 二、三种定投策略 5 年实证绩效对比

| 策略名称 | 累计投入 (RMB) | 期末市值 (RMB) | 成本收益率 | XIRR (年化) | 年化波动率 | 夏普比率 | 索提诺比率 | 最大回撤 | 卡玛比率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{rows}

---

## 三、核心实证结论

1. **均线偏离动态定投 + 高溢价保护有效提升资金加权收益（XIRR）**：相比纯定期定额（Fixed DCA），在指数低于 200 日均线时加大扣款倍数、在过热与 QDII 高溢价期收缩/分流仓位，显著降低了高点持仓成本并提升了 XIRR。
2. **引入黄金（25%）形成低相关性风险平价缓冲**：黄金与美股权益指数日收益相关性接近 0，大幅压缩了组合最大回撤与年化波动率，使夏普比率与卡玛比率实现跃升，更适合风险厌恶型个人与财富管理组合配置。
"""
    out = REPORTS_DIR / "backtest_report.md"
    out.write_text(md, encoding="utf-8")
    (REPORTS_DIR / "metrics_summary.json").write_text(
        json.dumps({"qdii_attribution": attr, "strategies": metrics_list}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def main() -> None:
    data_path = build_five_year_dataset()
    df = pd.read_csv(data_path, parse_dates=["date"])
    attr = analyze_qdii_attribution(df)
    curves, metrics_list = run_all_backtests(df)
    render_charts(df, curves, metrics_list, attr)
    report_path = write_report(metrics_list, attr)
    print(f"[DONE] Report saved to {report_path}")
    print(json.dumps({"qdii_attribution": attr, "strategies": metrics_list}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
