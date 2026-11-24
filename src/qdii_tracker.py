"""5-Year Market Data & QDII Fund Tracker (NDX, SPX, Gold, USD/CNY, Southern NDX100 QDII).

Generates a calibrated 5-year daily dataset (2021-11 to 2026-11) capturing realistic macro
regimes (2022 rate-hike drawdown, 2023-2024 AI rally, 2025-2026 gold strength & QDII quota
premium spikes), and provides an optional live yfinance fetcher (`--live`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def build_five_year_dataset(seed: int = 202605) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range("2021-11-22", "2026-11-20")
    n = len(dates)

    # Correlated daily shocks: [NDX, SPX, GOLD, USDCNY]
    corr = np.array(
        [
            [1.00, 0.91, 0.06, -0.10],
            [0.91, 1.00, 0.08, -0.12],
            [0.06, 0.08, 1.00, -0.18],
            [-0.10, -0.12, -0.18, 1.00],
        ]
    )
    vols = np.array([0.0140, 0.0100, 0.0085, 0.0025])
    cov = np.outer(vols, vols) * corr
    shocks = rng.multivariate_normal(np.zeros(4), cov, size=n)
    # Center realized shocks to match historical 2021-2026 regime moments cleanly
    shocks = shocks - shocks.mean(axis=0, keepdims=True)

    ndx = np.empty(n)
    spx = np.empty(n)
    gold = np.empty(n)
    fx = np.empty(n)
    ndx[0], spx[0], gold[0], fx[0] = 16200.0, 4680.0, 1810.0, 6.39

    for t in range(1, n):
        dt = dates[t]
        # Macro regime drifts calibrated to 2021-2026 cycles (2022 bear, 2023-24 AI rally, 2025-26 gold & equity cycle)
        if dt < pd.Timestamp("2022-10-15"):
            d_ndx, d_spx, d_gold, d_fx = -0.00145, -0.00095, -0.00010, 0.00050
        elif dt < pd.Timestamp("2024-12-31"):
            d_ndx, d_spx, d_gold, d_fx = 0.00118, 0.00086, 0.00068, 0.00004
        elif dt < pd.Timestamp("2025-05-15"):
            d_ndx, d_spx, d_gold, d_fx = -0.00030, -0.00015, 0.00080, 0.00001
        else:
            d_ndx, d_spx, d_gold, d_fx = 0.00055, 0.00042, 0.00048, -0.00004

        ndx[t] = ndx[t - 1] * np.exp(d_ndx + shocks[t, 0])
        spx[t] = spx[t - 1] * np.exp(d_spx + shocks[t, 1])
        gold[t] = gold[t - 1] * np.exp(d_gold + shocks[t, 2])
        fx[t] = np.clip(fx[t - 1] * np.exp(d_fx + shocks[t, 3]), 6.30, 7.36)

    # Southern Nasdaq-100 QDII (CNY-denominated NAV): tracks NDX * FX minus ~0.85% annual fee + tiny tracking noise
    ndx_ret = pd.Series(ndx).pct_change().fillna(0.0).to_numpy()
    fx_ret = pd.Series(fx).pct_change().fillna(0.0).to_numpy()
    daily_fee = 0.0085 / 252.0
    tracking_err = rng.normal(0.0, 0.0003, size=n)
    tracking_err = tracking_err - tracking_err.mean()

    qdii_nav = np.empty(n)
    qdii_nav[0] = 1.0000
    for t in range(1, n):
        qdii_nav[t] = qdii_nav[t - 1] * (1.0 + ndx_ret[t]) * (1.0 + fx_ret[t]) * (1.0 - daily_fee + tracking_err[t])

    # QDII secondary-market premium/discount rate (spikes during strong NDX momentum & quota limits)
    ndx_mom_20 = pd.Series(ndx).pct_change(20).fillna(0.0).to_numpy()
    prem = np.empty(n)
    prem[0] = 0.004
    for t in range(1, n):
        quota_squeeze = 0.042 if (pd.Timestamp("2024-01-15") <= dates[t] <= pd.Timestamp("2024-07-15")) or (
            pd.Timestamp("2025-11-01") <= dates[t] <= pd.Timestamp("2026-04-30")
        ) else 0.008
        target = quota_squeeze + 0.18 * max(ndx_mom_20[t], -0.05)
        prem[t] = np.clip(0.86 * prem[t - 1] + 0.14 * target + rng.normal(0, 0.0055), -0.025, 0.095)

    qdii_mkt_px = qdii_nav * (1.0 + prem)

    df = pd.DataFrame(
        {
            "date": dates,
            "ndx_close": np.round(ndx, 2),
            "spx_close": np.round(spx, 2),
            "gold_usd": np.round(gold, 2),
            "usdcny": np.round(fx, 4),
            "qdii_nav_cny": np.round(qdii_nav, 4),
            "qdii_mkt_px_cny": np.round(qdii_mkt_px, 4),
            "qdii_premium_pct": np.round(prem * 100.0, 2),
        }
    )

    out_path = DATA_DIR / "five_year_market_qdii.csv"
    df.to_csv(out_path, index=False)
    return out_path


def analyze_qdii_attribution(df: pd.DataFrame) -> dict:
    """Decompose 5-year Southern Nasdaq-100 QDII NAV return into NDX + FX + Residual."""
    ndx_total = float(df["ndx_close"].iloc[-1] / df["ndx_close"].iloc[0] - 1.0)
    fx_total = float(df["usdcny"].iloc[-1] / df["usdcny"].iloc[0] - 1.0)
    qdii_total = float(df["qdii_nav_cny"].iloc[-1] / df["qdii_nav_cny"].iloc[0] - 1.0)
    # Multiplicative attribution in log-return space mapped cleanly to percentage points
    log_ndx = np.log(1.0 + ndx_total)
    log_fx = np.log(1.0 + fx_total)
    log_tot = np.log(1.0 + qdii_total)
    log_res = log_tot - log_ndx - log_fx

    high_prem_days = int((df["qdii_premium_pct"] > 3.0).sum())
    return {
        "ndx_5y_return_pct": round(ndx_total * 100.0, 2),
        "usdcny_5y_return_pct": round(fx_total * 100.0, 2),
        "qdii_nav_5y_return_pct": round(qdii_total * 100.0, 2),
        "attribution_ndx_share_pct": round(float(log_ndx / log_tot * 100.0), 1),
        "attribution_fx_share_pct": round(float(log_fx / log_tot * 100.0), 1),
        "attribution_fee_tracking_share_pct": round(float(log_res / log_tot * 100.0), 1),
        "avg_premium_pct": round(float(df["qdii_premium_pct"].mean()), 2),
        "max_premium_pct": round(float(df["qdii_premium_pct"].max()), 2),
        "high_premium_alert_days": high_prem_days,
        "high_premium_alert_ratio_pct": round(high_prem_days / len(df) * 100.0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="5-Year Index & QDII Dataset Builder")
    parser.add_argument("--live", action="store_true", help="Try downloading live NDX/SPX/Gold via yfinance")
    args = parser.parse_args()

    if args.live:
        try:
            import yfinance as yf  # type: ignore

            raw = yf.download(["^NDX", "^GSPC", "GC=F", "CNY=X"], period="5y")["Close"].dropna()
            out = DATA_DIR / "live_yfinance_5y.csv"
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            raw.to_csv(out)
            print(f"[LIVE] Saved live 5Y market panel to {out}")
            return
        except Exception as exc:
            print(f"[WARN] Live download skipped ({exc}); using calibrated 5Y dataset.")

    path = build_five_year_dataset()
    df = pd.read_csv(path, parse_dates=["date"])
    print("[QDII Attribution Summary]", analyze_qdii_attribution(df))


if __name__ == "__main__":
    main()
