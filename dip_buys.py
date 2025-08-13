#!/usr/bin/env python3
# Dip-buy alerts for SPMO & QQQ (UK execution next LSE session).
# - Uses curl_cffi Chrome session ONLY for yfinance.
# - Normalizes yfinance MultiIndex -> flat columns.
# - Fixes timezone + pandas scalar warnings.
# - Prints detailed console output and ALWAYS posts to Telegram (buy or no-buy).

import os
import math
import time
import random
import re
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from curl_cffi import requests as cf_requests   # Chrome-impersonated session for Yahoo
import requests                                  # standard requests for Telegram ONLY

# ---------- Config ----------
TICKERS = ["SPMO", "QQQ"]
ANNUAL_BUDGET_GBP = 30000.0
BASE_DAILY_GBP = ANNUAL_BUDGET_GBP / 365.0
RSI_PERIOD = 14
DIP_MULTIPLIERS = [(0.00,0.05,0.0),(0.05,0.07,1.0),(0.07,0.12,2.0),(0.12,9.99,4.0)]
SKIP_IF_DEEP_BEAR = True
MAX_RETRIES = 4
YF_PERIOD = "400d"
YF_INTERVAL = "1d"

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# One Chrome-impersonated session for Yahoo only
yf_session = cf_requests.Session(impersonate="chrome")

# ---------- Helpers ----------
def to_float(x) -> float:
    if isinstance(x, (float, int, np.floating, np.integer)):
        return float(x)
    try:
        return float(np.asarray(x).item())
    except Exception:
        return float(x)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def dip_multiplier(dip_abs: float) -> float:
    for lo, hi, m in DIP_MULTIPLIERS:
        if lo <= dip_abs < hi:
            return m
    return DIP_MULTIPLIERS[-1][2]

def next_lse_trading_day(from_dt_utc: datetime) -> str:
    d = from_dt_utc.date() + timedelta(days=1)
    while d.weekday() >= 5:  # Sat/Sun
        d += timedelta(days=1)
    return d.strftime("%a %d %b %Y")

def normalize_single_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    # yfinance sometimes returns MultiIndex even for a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        elif ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0)
        else:
            df.columns = [c[0] for c in df.columns]
    return df

def rel_str_score(close: pd.Series, window: int = 63) -> float:
    if len(close) <= window:
        return 0.0
    return to_float(close.iloc[-1] / close.iloc[-window-1] - 1.0)

def fetch_df(ticker: str) -> pd.DataFrame:
    last_err = None
    for i in range(MAX_RETRIES):
        try:
            time.sleep(random.uniform(0.6, 2.2))  # jitter
            df = yf.download(
                ticker,
                period=YF_PERIOD,
                interval=YF_INTERVAL,
                auto_adjust=True,
                progress=False,
                threads=False,
                session=yf_session,  # Chrome session only for Yahoo
            )
            if df is not None and not df.empty:
                df = df.dropna()
                df = normalize_single_ticker(df, ticker)
                if "Close" in df.columns:
                    return df
        except Exception as e:
            last_err = e
        time.sleep((2 ** i) + random.uniform(0, 0.5))  # backoff
    if last_err:
        print(f"[WARN] yfinance failed for {ticker}: {last_err}")
    return pd.DataFrame()

# ---------- Telegram ----------
_MD_CHARS = r'_*[]()~`>#+-=|{}.!'
_MD_RE = re.compile(r'([_\*\[\]\(\)~`>#+\-=\|{}\.\!])')

def _escape_md_v2(text: str) -> str:
    # Escape special chars for MarkdownV2
    return _MD_RE.sub(r'\\\1', text)

def send_telegram(text: str):
    """
    Always try to send. Prefer MarkdownV2; if that fails, fall back to plain text.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] Telegram env vars not set; printing message instead:\n")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    md_text = _escape_md_v2(text)

    # Primary: MarkdownV2
    resp = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": md_text,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": True
    }, timeout=30)

    if resp.status_code == 200:
        return

    print(f"[INFO] MarkdownV2 send failed {resp.status_code}: {resp.text}")

    # Fallback: plain text
    resp2 = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True
    }, timeout=30)
    resp2.raise_for_status()

# ---------- Main ----------
def main():
    now_utc = datetime.now(timezone.utc)
    exec_date_uk = next_lse_trading_day(now_utc)
    run_id = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Load & enrich
    data = {}
    for t in TICKERS:
        df = fetch_df(t)
        if df.empty:
            print(f"[ERROR] No data for {t}")
            continue
        c = df["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.squeeze()
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
        c = c.astype(float)

        df["rsi"] = rsi(c, RSI_PERIOD)
        df["ma200"] = c.rolling(200).mean()
        df["ma200_slope_up"] = df["ma200"].diff(5) > 0
        df["30d_high"] = c.rolling(30).max()
        df["ath"] = c.cummax()
        data[t] = df

    # If we failed entirely, still send a heartbeat
    if not data:
        msg = (
            f"🧭 Dip-Buy Analysis (US close)\n"
            f"Run ID: {run_id}\n"
            f"Date (UTC): {now_utc.strftime('%a %d %b %Y')}\n"
            f"Next LSE execution: {exec_date_uk}\n"
            f"Data source: Yahoo Finance (session=Chrome)\n\n"
            f"‼️ Couldn’t fetch price data for SPMO/QQQ today. Check yfinance/connectivity."
        )
        print(msg)
        try:
            send_telegram(msg)
            print("\n[OK] Telegram sent (no data).")
        except Exception as e:
            print(f"\n[WARN] Telegram send failed: {e}")
        return 0

    # Relative strength weights (>=0)
    rs_scores = {t: max(0.0, rel_str_score(df["Close"].astype(float))) for t, df in data.items()}
    denom = sum(rs_scores.values()) or 1.0
    rs_weights = {t: (rs_scores[t] / denom) for t in data.keys()}

    print("\n=== Dip-Buy Analysis (based on US close) ===")
    print(f"Run ID: {run_id}")
    print(f"Date (UTC): {now_utc.date()} | Next LSE execution date: {exec_date_uk}")
    print(f"Base daily pace: £{BASE_DAILY_GBP:,.2f}\n")

    tg_lines = [
        "🧭 Dip-Buy Analysis (US close)",
        f"Run ID: {run_id}",
        f"Date (UTC): {now_utc.strftime('%a %d %b %Y')}",
        f"Next LSE execution: {exec_date_uk}",
        f"Base daily pace: £{BASE_DAILY_GBP:,.0f}",
        "Data source: Yahoo Finance (session=Chrome)",
        ""
    ]

    any_buy = False

    for t, df in data.items():
        last = df.iloc[-1]

        close = to_float(last["Close"])
        rsi_last = to_float(last["rsi"])
        ma200 = 0.0 if pd.isna(last["ma200"]) else to_float(last["ma200"])
        above_200 = (close > ma200 > 0)
        high30 = to_float(last["30d_high"])
        ath = to_float(last["ath"])
        dip_abs = max(0.0, (high30 - close) / high30) if high30 > 0 else 0.0
        draw_ath = (close / ath - 1.0) if ath > 0 else 0.0

        has_dip = (dip_abs >= 0.05) and (rsi_last < 40) and above_200
        deep_bear = (close < ma200) and (draw_ath <= -0.20) if SKIP_IF_DEEP_BEAR else False

        rs_w = rs_weights[t]

        # Console diagnostics
        print(
            f"[{t}] Close: {close:.2f} | 30D High: {high30:.2f} | Dip: {dip_abs*100:.2f}% | "
            f"RSI: {rsi_last:.1f} | 200DMA: {ma200:.2f} | Above 200DMA: {above_200} | "
            f"ATH DD: {draw_ath*100:.2f}% | RS weight: {rs_w:.2f}"
        )

        # Telegram block (buy or no-buy)
        if not has_dip or deep_bear:
            reasons = []
            if deep_bear:
                reasons.append("Deep-bear guardrail: below 200DMA and >20% off ATH.")
            if dip_abs < 0.05:
                reasons.append("Dip < 5% from 30D high.")
            if rsi_last >= 40:
                reasons.append("RSI ≥ 40 (not oversold).")
            if not above_200 and not deep_bear:
                reasons.append("Below 200DMA.")

            print(f" -> NO BUY: {', '.join(reasons)}\n")

            tg_lines += [
                f"🟨 {t}: No Buy",
                f"Dip: {dip_abs*100:.1f}% | RSI: {rsi_last:.1f} | Above 200DMA: {'Yes' if above_200 else 'No'}",
                f"Reason: {' '.join(reasons)}",
                ""
            ]
            continue

        mult = dip_multiplier(dip_abs)
        rec_total = BASE_DAILY_GBP * mult
        rec_gbp = max(0.0, round(rec_total * rs_w))

        if rec_gbp < 1:
            print(" -> Signal too small (<£1 after rounding); NO BUY.\n")
            tg_lines += [
                f"🟨 {t}: No Buy",
                f"Dip: {dip_abs*100:.1f}% | RSI: {rsi_last:.1f} | Above 200DMA: {'Yes' if above_200 else 'No'}",
                "Reason: Recommended amount < £1.",
                ""
            ]
            continue

        any_buy = True
        print(f" -> BUY: £{rec_gbp:,.0f} (base £{BASE_DAILY_GBP:,.0f} × mult {mult:g} × RS {rs_w:.2f})")
        print(f" -> Execute next LSE session: {exec_date_uk}\n")

        tg_lines += [
            f"📉 {t}: Buy £{rec_gbp:,.0f}",
            f"Dip: {dip_abs*100:.1f}% | RSI: {rsi_last:.1f} | Above 200DMA: {'Yes' if above_200 else 'No'}",
            f"Sizing: base £{BASE_DAILY_GBP:,.0f} × mult {mult:g} × RS {rs_w:.2f}",
            ""
        ]

    if not any_buy:
        tg_lines.append("No buys today based on rules. ✅ Patience preserved.")

    tg_text = "\n".join(tg_lines)

    print("\n--- Telegram message to be sent ---\n")
    print(tg_text)

    try:
        send_telegram(tg_text)
        print("\n[OK] Telegram sent.")
    except Exception as e:
        print(f"\n[WARN] Telegram send failed: {e}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
