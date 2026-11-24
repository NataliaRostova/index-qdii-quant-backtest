"""5-Year Quant Backtest Engine: Fixed DCA vs. MA-Deviation Dynamic DCA vs. Equity-Gold Allocation.

Evaluates:
1. Fixed DCA (70% NDX QDII + 30% SPX CNY-equivalent)
2. MA200-Deviation Dynamic DCA (scales contribution 0.5x~2.0x inversely with MA200 deviation,
   plus QDII premium guard diverting excess premium purchases to lower-premium SPX)
3. Multi-Asset Equity + Gold Dynamic Allocation (50% NDX + 25% SPX + 25% Gold CNY)

Computes XIRR (money-weighted annualized return), Time-Weighted CAGR, Volatility,
Sharpe Ratio, Sortino Ratio, Max Drawdown, and Calmar Ratio.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def compute_xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float:
    """Compute exact XIRR for irregular cashflows (negative = contribution, positive = ending value)."""
    t0 = cashflows[0][0]
    years = np.array([(dt - t0).days / 365.25 for dt, _ in cashflows], dtype=float)
    amounts = np.array([amt for _, amt in cashflows], dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(amounts / np.power(1.0 + rate, years)))

    try:
        return float(brentq(npv, -0.95, 5.0))
    except ValueError:
        return 0.0


def _simulate_dca_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    weights: dict[str, float],
    dynamic_ma: bool = False,
    premium_guard: bool = False,
    base_invest_rmb: float = 2000.0,
    rf_annual: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    """Run bi-weekly (every 10 trading days) investment simulation in CNY terms."""
    work = df.copy()
    # CNY-equivalent price series for SPX & Gold, plus actual QDII market execution price for NDX
    work["px_ndx_qdii"] = work["qdii_mkt_px_cny"]
    work["px_spx_cny"] = (work["spx_close"] * work["usdcny"]) / (work["spx_close"].iloc[0] * work["usdcny"].iloc[0])
    work["px_gold_cny"] = (work["gold_usd"] * work["usdcny"]) / (work["gold_usd"].iloc[0] * work["usdcny"].iloc[0])

    work["ndx_ma200"] = work["ndx_close"].rolling(200, min_periods=40).mean()
    work["ndx_dev"] = (work["ndx_close"] - work["ndx_ma200"]) / work["ndx_ma200"]

    shares = {"ndx": 0.0, "spx": 0.0, "gold": 0.0}
    cum_invested = 0.0
    cashflows: list[tuple[pd.Timestamp, float]] = []

    nav_series = []
    invested_series = []
    twr_unit_nav = [1.0]

    for i, row in work.iterrows():
        px_map = {"ndx": float(row["px_ndx_qdii"]), "spx": float(row["px_spx_cny"]), "gold": float(row["px_gold_cny"])}

        # Pre-cashflow portfolio valuation
        port_before = sum(shares[k] * px_map[k] for k in shares)
        if i > 0:
            prev_val = nav_series[-1]
            daily_twr = (port_before / prev_val - 1.0) if prev_val > 0 else 0.0
            twr_unit_nav.append(twr_unit_nav[-1] * (1.0 + daily_twr))

        # Contribute every 10 trading days (approx. bi-weekly)
        if i % 10 == 0:
            mult = 1.0
            if dynamic_ma:
                dev = float(row["ndx_dev"]) if pd.notna(row["ndx_dev"]) else 0.0
                # Invest up to 2.0x when NDX is deeply below MA200, scale down to 0.5x when overheated
                mult = float(np.clip(1.0 - 2.8 * dev, 0.5, 2.0))

            contrib = base_invest_rmb * mult
            eff_weights = dict(weights)

            # Premium guard: if QDII premium > 3.5%, shift half of NDX allocation into SPX to avoid overpaying
            if premium_guard and float(row["qdii_premium_pct"]) > 3.5 and eff_weights.get("ndx", 0) > 0:
                shift = eff_weights["ndx"] * 0.5
                eff_weights["ndx"] -= shift
                eff_weights["spx"] = eff_weights.get("spx", 0.0) + shift

            for asset, w in eff_weights.items():
                if w > 0:
                    shares[asset] += (contrib * w) / px_map[asset]

            cum_invested += contrib
            cashflows.append((row["date"], -contrib))

        port_after = sum(shares[k] * px_map[k] for k in shares)
        nav_series.append(port_after)
        invested_series.append(cum_invested)

    cashflows.append((work["date"].iloc[-1], nav_series[-1]))
    xirr = compute_xirr(cashflows)

    res_df = pd.DataFrame(
        {
            "date": work["date"],
            "portfolio_value": nav_series,
            "cum_invested": invested_series,
            "twr_nav": twr_unit_nav,
        }
    )
    res_df["cum_return_pct"] = (res_df["portfolio_value"] / res_df["cum_invested"] - 1.0) * 100.0
    res_df["twr_ret"] = res_df["twr_nav"].pct_change().fillna(0.0)
    res_df["twr_dd_pct"] = (res_df["twr_nav"] / res_df["twr_nav"].cummax() - 1.0) * 100.0

    years = (work["date"].iloc[-1] - work["date"].iloc[0]).days / 365.25
    twr_cagr = (res_df["twr_nav"].iloc[-1] ** (1.0 / years)) - 1.0
    ann_vol = float(res_df["twr_ret"].std() * np.sqrt(252.0))
    downside_vol = float(res_df.loc[res_df["twr_ret"] < 0, "twr_ret"].std() * np.sqrt(252.0))
    sharpe = (twr_cagr - rf_annual) / ann_vol if ann_vol > 1e-9 else 0.0
    sortino = (twr_cagr - rf_annual) / downside_vol if downside_vol > 1e-9 else 0.0
    max_dd = float(res_df["twr_dd_pct"].min())
    calmar = (twr_cagr * 100.0) / abs(max_dd) if abs(max_dd) > 1e-6 else 0.0

    metrics = {
        "strategy": strategy_name,
        "total_invested_rmb": round(cum_invested, 2),
        "ending_value_rmb": round(nav_series[-1], 2),
        "net_profit_rmb": round(nav_series[-1] - cum_invested, 2),
        "total_return_on_cost_pct": round((nav_series[-1] / cum_invested - 1.0) * 100.0, 2),
        "xirr_pct": round(xirr * 100.0, 2),
        "twr_cagr_pct": round(twr_cagr * 100.0, 2),
        "annualized_vol_pct": round(ann_vol * 100.0, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar_ratio": round(calmar, 2),
    }
    return res_df, metrics


def run_all_backtests(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    curves = {}
    metrics_list = []

    configs = [
        ("Fixed DCA (70% NDX + 30% SPX)", {"ndx": 0.70, "spx": 0.30, "gold": 0.0}, False, False),
        ("MA200 Dynamic DCA + Premium Guard", {"ndx": 0.70, "spx": 0.30, "gold": 0.0}, True, True),
        ("Equity + Gold Dynamic (50/25/25)", {"ndx": 0.50, "spx": 0.25, "gold": 0.25}, True, True),
    ]

    for name, w, dyn, guard in configs:
        curve_df, m = _simulate_dca_strategy(df, name, w, dynamic_ma=dyn, premium_guard=guard)
        curves[name] = curve_df
        metrics_list.append(m)

    return curves, metrics_list
