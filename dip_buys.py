#!/usr/bin/env python3
# Sends Telegram alerts with recommended GBP buys for SPMO & QQQ after US close.
# Frequency: daily (21:05 Europe/London). One trade max per instrument per day.
# Execution guidance: next LSE session (single trade).
import os, math, datetime as dt
import pandas as pd, numpy as np, requests, yfinance as yf

TICKERS = ["SPMO", "QQQ"]
ANNUAL_BUDGET_GBP = 30000.0
BASE_DAILY_GBP = ANNUAL_BUDGET_GBP / 365.0
RSI_PERIOD = 14

# Dip buckets (from 30D high) -> multiplier
DIP_MULTIPLIERS = [(0.00,0.05,0.0),(0.05,0.07,1.0),(0.07,0.12,2.0),(0.12,9.99,4.0)]
SKIP_IF_DEEP_BEAR = True  # below 200DMA AND >20% off ATH

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}).raise_for_status()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def dip_multiplier(dip_abs: float) -> float:
    for lo, hi, m in DIP_MULTIPLIERS:
        if lo <= dip_abs < hi: return m
    return DIP_MULTIPLIERS[-1][2]

def next_lse_trading_day(from_date_utc: dt.date) -> dt.date:
    # Basic weekend skip; ignores bank holidays (acceptable for alerts).
    d = from_date_utc + dt.timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat,6=Sun
        d += dt.timedelta(days=1)
    return d

def rel_str_score(close: pd.Series, window: int = 63) -> float:
    if len(close) <= window: return 0.0
    return float(close.iloc[-1] / close.iloc[-window-1] - 1.0)

def fetch_df(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="400d", interval="1d", auto_adjust=True, progress=False)
    if df.empty: return df
    df = df.dropna()
    c = df["Close"].astype(float)
    df["rsi"] = rsi(c, RSI_PERIOD)
    df["ma200"] = c.rolling(200).mean()
    df["ma200_slope_up"] = df["ma200"].diff(5) > 0
    df["30d_high"] = c.rolling(30).max()
    df["ath"] = c.cummax()
    return df

def main():
    today_utc = dt.datetime.utcnow().date()
    exec_date_uk = next_lse_trading_day(today_utc)

    # Load data
    data = {t: fetch_df(t) for t in TICKERS}
    data = {t: df for t, df in data.items() if not df.empty}
    if not data:
        print("No data; abort.")
        return 0

    # Relative strength weights (softmax-ish >=0)
    rs = {t: max(0.0, rel_str_score(df["Close"])) for t, df in data.items()}
    denom = sum(rs.values()) or 1.0
    weights = {t: rs[t]/denom for t in data.keys()}

    msgs = []
    for t, df in data.items():
        last = df.iloc[-1]
        close = float(last["Close"])
        rsi_last = float(last["rsi"])
        ma200 = float(last["ma200"]) if not math.isnan(last["ma200"]) else 0.0
        above_200 = close > ma200 > 0
        high30 = float(last["30d_high"])
        ath = float(last["ath"])
        dip_abs = max(0.0, (high30 - close) / high30) if high30 > 0 else 0.0
        draw_ath = (close / ath - 1.0) if ath > 0 else 0.0

        has_dip = (dip_abs >= 0.05) and (rsi_last < 40) and above_200
        deep_bear = (close < ma200) and (draw_ath <= -0.20) if SKIP_IF_DEEP_BEAR else False
        if not has_dip or deep_bear:
            continue

        mult = dip_multiplier(dip_abs)
        base = BASE_DAILY_GBP
        rec_total = base * mult
        rec_gbp = rec_total * weights[t]
        rec_gbp = max(0.0, round(rec_gbp))  # whole pounds

        if rec_gbp < 1:
            continue

        msgs.append(
            f"📉 <b>{t} Dip Buy</b>\n"
            f"Dip vs 30D high: <b>{dip_abs*100:.1f}%</b> | RSI(14): <b>{rsi_last:.1f}</b>\n"
            f"Above 200DMA: <b>{'Yes' if above_200 else 'No'}</b> | 200DMA slope up: <b>{'Yes' if bool(last['ma200_slope_up']) else 'No'}</b>\n"
            f"💷 <b>Buy £{rec_gbp:,.0f}</b> "
            f"(base £{base:,.0f} × mult {mult:g} × RS {weights[t]:.2f})\n"
            f"🕘 Execute next LSE session: <b>{exec_date_uk.strftime('%a %d %b %Y')}</b> (single trade)\n"
        )

    if msgs:
        send_telegram("\n".join(msgs))
    else:
        # Quiet day – no Telegram to avoid spam. Uncomment to get a ping every day:
        # send_telegram("No qualifying dip-buy signals today.")
        print("No qualifying signals.")
    return 0

if __name__ == "__main__":
    main()
