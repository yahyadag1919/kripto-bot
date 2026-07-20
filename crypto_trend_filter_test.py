"""
Kripto botu - COK FAKTORLU FILTRE TESTI
Gemini ile konusulan, henuz canliya eklenmemis fikirlerin hepsini TEK SEFERDE
test eder. Ayni iki strateji (VWAP Sapmasi, Hacim Z-Skor), 3 farkli filtrenin
ACIK/KAPALI tum kombinasyonlariyla (2^3=8 kombinasyon) test edilir:

  T (Trend):    4sa 200 EMA'ya gore - sadece trend yonune uygun bounce'lar alinir
  F (Funding):  Funding rate'e gore - LONG icin negatif funding (asiri short
                pozisyon => yukari bounce daha olasi), SHORT icin pozitif funding
                (asiri long pozisyon => asagi duzeltme daha olasi) sart kosulur
  V (Volatilite): ATR14'un kendi trailing yuzdelik dilimine gore - ASIRI oynak
                donemlerde (ust %30) islem yapilmaz (kontrolsuz whipsaw'dan kacinma)

Sonuc: 8 kombinasyon x 2 strateji = 16 satirlik bir karsilastirma tablosu.
Hangi filtre(ler) gercekten isabet oranini/net getiriyi yukseltiyor gorulur.

Calistirmak icin: pip install ccxt pandas numpy requests --break-system-packages
                  python3 crypto_trend_filter_test.py
NOT: Sadece halka acik (public) veri kullanilir, API key GEREKMEZ.
"""

import os
import time
import itertools
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("(Telegram token/chat id yok, sadece konsola yaziliyor)")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    lines = text.split("\n")
    chunk = ""
    for line in lines:
        if len(chunk) + len(line) + 1 > 3500:
            try:
                requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=15)
            except Exception as e:
                print(f"Telegram gonderim hatasi: {e}")
            chunk = ""
        chunk += line + "\n"
    if chunk.strip():
        try:
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=15)
        except Exception as e:
            print(f"Telegram gonderim hatasi: {e}")


exchange = ccxt.binanceusdm({"enableRateLimit": True})

COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "SUI",
    "DOT", "TRX", "ATOM", "NEAR", "TON", "LTC", "BCH", "ETC", "FIL", "APT",
    "ARB", "OP", "INJ", "SEI", "ICP", "HBAR", "VET", "ALGO", "XLM", "EOS",
    "XTZ", "SAND", "MANA", "AAVE", "UNI", "CRV", "GRT", "THETA", "EGLD",
    "FLOW", "CHZ", "DYDX", "GALA", "IMX", "ONDO", "WLD",
    "PEPE", "SHIB", "TIA", "STRK", "JUP", "PYTH", "JTO", "ENA", "ETHFI", "ORDI",
]  # tam listenin bir alt kumesi - 3 ayri veri fetch'i oldugu icin sureyi makul tutmak icin

WATCHLIST = [f"{c}/USDT:USDT" for c in COINS]

TIMEFRAME = "15m"
TREND_TIMEFRAME = "4h"
TREND_EMA_PERIOD = 200
VWAP_WINDOW = 96
VWAP_DEV_LONG_MAX = -2.0
VWAP_DEV_SHORT_MIN = 2.0
VOLUME_ZSCORE_THRESHOLD = 2.0
COMMISSION_PCT = 0.1

VOLATILITY_PERCENTILE_WINDOW = 200   # ATR'nin yuzdelik dilimini hesaplarken bakilacak gecmis mum sayisi
VOLATILITY_MAX_PERCENTILE = 70       # ATR bu yuzdelik dilimin USTUNDEYSE (asiri oynak) islem yapilmaz

CHECKPOINTS = [(4, "1sa", 0.3), (16, "4sa", 0.6), (48, "12sa", 1.0), (96, "24sa", 1.5)]  # 15dk mum sayisi


def fetch_ohlcv_df(symbol, timeframe, limit):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def fetch_funding_df(symbol, limit=300):
    try:
        raw = exchange.fetch_funding_rate_history(symbol, limit=limit)
    except Exception as e:
        print(f"{symbol}: funding rate cekilemedi ({e}), funding filtresi bu coin icin devre disi kalacak")
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    if not raw:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.rename(columns={"fundingRate": "funding_rate"})
    return df[["timestamp", "funding_rate"]].sort_values("timestamp")


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_std20"] = df["volume"].rolling(20).std()
    df["vol_zscore"] = (df["volume"] - df["vol_sma20"]) / df["vol_std20"].replace(0, np.nan)

    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    df["vwap"] = pv.rolling(VWAP_WINDOW).sum() / df["volume"].rolling(VWAP_WINDOW).sum()
    df["vwap_dev_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_percentile"] = df["atr14"].rolling(VOLATILITY_PERCENTILE_WINDOW).rank(pct=True) * 100
    return df


def compute_trend_ema(df_4h: pd.DataFrame) -> pd.DataFrame:
    df_4h = df_4h.copy()
    df_4h["ema200_4h"] = df_4h["close"].ewm(span=TREND_EMA_PERIOD, adjust=False).mean()
    return df_4h[["timestamp", "ema200_4h"]]


def map_asof(df_15m: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Her 15dk mumuna, o ana kadarki en son kapanmis degeri esler (bakis-onu onlemek icin)."""
    if other.empty:
        return df_15m
    df = df_15m.sort_values("timestamp")
    other = other.sort_values("timestamp")
    return pd.merge_asof(df, other, on="timestamp", direction="backward")


def check_vwap_signal(row):
    if pd.isna(row.get("vwap_dev_pct")):
        return None
    if row["vwap_dev_pct"] <= VWAP_DEV_LONG_MAX:
        return "LONG"
    if row["vwap_dev_pct"] >= VWAP_DEV_SHORT_MIN:
        return "SHORT"
    return None


def check_zscore_signal(row):
    if pd.isna(row.get("vol_zscore")) or row["vol_zscore"] < VOLUME_ZSCORE_THRESHOLD:
        return None
    if row["close"] < row["open"]:
        return "LONG"
    elif row["close"] > row["open"]:
        return "SHORT"
    return None


def passes_trend_filter(row, direction):
    if pd.isna(row.get("ema200_4h")):
        return False
    above_trend = row["close"] > row["ema200_4h"]
    return above_trend if direction == "LONG" else not above_trend


def passes_funding_filter(row, direction):
    if pd.isna(row.get("funding_rate")):
        return False
    # LONG: piyasa asiri short'lanmis olmali (negatif funding) - yukari bounce daha olasi
    # SHORT: piyasa asiri long'lanmis olmali (pozitif funding) - asagi duzeltme daha olasi
    return row["funding_rate"] < 0 if direction == "LONG" else row["funding_rate"] > 0


def passes_volatility_filter(row):
    if pd.isna(row.get("atr_percentile")):
        return False
    return row["atr_percentile"] <= VOLATILITY_MAX_PERCENTILE


def run_backtest(df, signal_fn, use_trend, use_funding, use_volatility, min_gap_bars=3):
    outcomes = []
    max_cp = max(c[0] for c in CHECKPOINTS)
    n = len(df)
    last_i = -min_gap_bars - 1
    i = max(VWAP_WINDOW, VOLATILITY_PERCENTILE_WINDOW, 25)
    while i < n - max_cp - 1:
        if i - last_i < min_gap_bars:
            i += 1
            continue
        row = df.iloc[i]
        direction = signal_fn(row)
        if direction is None:
            i += 1
            continue
        if use_trend and not passes_trend_filter(row, direction):
            i += 1
            continue
        if use_funding and not passes_funding_filter(row, direction):
            i += 1
            continue
        if use_volatility and not passes_volatility_filter(row):
            i += 1
            continue

        entry_price = row["close"]
        hit = False
        raw_pct = None
        for bars_ahead, label, target_pct in CHECKPOINTS:
            future_price = df.iloc[i + bars_ahead]["close"]
            r = (future_price - entry_price) / entry_price * 100
            pct = r if direction == "LONG" else -r
            if pct >= target_pct:
                hit = True
                raw_pct = pct
                break
        if not hit:
            future_price = df.iloc[i + max_cp]["close"]
            r = (future_price - entry_price) / entry_price * 100
            raw_pct = r if direction == "LONG" else -r

        outcomes.append(raw_pct)
        last_i = i
        i += 1
    return outcomes


def summarize(label, outcomes):
    if not outcomes:
        return {"varyant": label, "sinyal": 0, "isabet_%": None, "ort_net_%": None, "toplam_net_%": None}
    arr = np.array(outcomes) - COMMISSION_PCT
    return {
        "varyant": label,
        "sinyal": len(arr),
        "isabet_%": round((arr > 0).mean() * 100, 1),
        "ort_net_%": round(arr.mean(), 3),
        "toplam_net_%": round(arr.sum(), 1),
    }


def combo_label(use_trend, use_funding, use_volatility):
    parts = []
    parts.append("T" if use_trend else "-")
    parts.append("F" if use_funding else "-")
    parts.append("V" if use_volatility else "-")
    return "".join(parts)


def main():
    send_telegram_message(
        "🏁 Cok faktorlu filtre testi basliyor.\n"
        "Filtreler: T=Trend(4sa 200EMA) F=Funding G=Volatilite\n"
        "8 kombinasyon x 2 strateji (VWAP, Hacim Z-Skor) test edilecek, biraz surebilir..."
    )

    combos = list(itertools.product([False, True], repeat=3))  # (trend, funding, volatility)
    results = {}
    for strat_name in ["VWAP", "ZSkor"]:
        for combo in combos:
            results[(strat_name, combo)] = []

    for symbol in WATCHLIST:
        try:
            df_15m = fetch_ohlcv_df(symbol, TIMEFRAME, limit=1200)
            if len(df_15m) < 250:
                print(f"{symbol}: yetersiz 15m veri, atlandi")
                continue
            df_4h = fetch_ohlcv_df(symbol, TREND_TIMEFRAME, limit=300)
            funding_df = fetch_funding_df(symbol, limit=200)

            df_15m = compute_indicators(df_15m)
            if len(df_4h) >= TREND_EMA_PERIOD // 4:
                trend_4h = compute_trend_ema(df_4h)
                df_15m = map_asof(df_15m, trend_4h)
            else:
                df_15m["ema200_4h"] = np.nan
            df_15m = map_asof(df_15m, funding_df)

            for strat_name, signal_fn in [("VWAP", check_vwap_signal), ("ZSkor", check_zscore_signal)]:
                for combo in combos:
                    use_trend, use_funding, use_volatility = combo
                    out = run_backtest(df_15m, signal_fn, use_trend, use_funding, use_volatility)
                    results[(strat_name, combo)].extend(out)

            print(f"{symbol}: tamamlandi ({len(df_15m)} mum)")
            time.sleep(0.25)
        except Exception as e:
            print(f"{symbol}: hata - {e}")

    rows = []
    for strat_name in ["VWAP", "ZSkor"]:
        for combo in combos:
            label = f"{strat_name}-{combo_label(*combo)}"
            rows.append(summarize(label, results[(strat_name, combo)]))

    table = pd.DataFrame(rows).sort_values("ort_net_%", ascending=False, na_position="last")
    print("\n--- COK FAKTORLU FILTRE TEST SONUCLARI ---")
    print("(T=Trend F=Funding V=Volatilite, harf varsa o filtre ACIK, '-' varsa KAPALI)")
    print(table.to_string(index=False))
    table.to_csv("cok_faktorlu_filtre_sonuclari.csv", index=False)

    send_telegram_message(
        "📊 COK FAKTORLU FILTRE SONUCLARI (buyukten kucuge ort_net_%)\n"
        "T=Trend F=Funding V=Volatilite, harf=ACIK '-'=KAPALI\n\n"
        + table.to_string(index=False)
    )
    finish_msg = f"✅ Cok faktorlu filtre testi tamamlandi - {datetime.now().isoformat()}"
    print(finish_msg)
    send_telegram_message(finish_msg)


if __name__ == "__main__":
    main()
