"""
Kripto botu - SONRAKI NESIL TEST (Gemini'nin 3 fikri, tek script'te)
Onceki testte VWAP-TF (Trend+Funding filtreli) en iyi kaliteyi verdi ama
sinyal sayisini cok dusurdu. Bu test, kaliteyi korurken sinyal sayisini
ARTIRMAYI hedefleyen 3 fikri birlikte dener:

1) COKLU ZAMAN DILIMI: Ayni filtreli VWAP mantigi 5m/15m/30m'de ayri ayri
   calistirilir - farkli zaman dilimleri farkli sinyaller yakalayabilir,
   havuzu buyutebilir.
2) HACIM Z-SKOR YONU TERSINE COVRILIYOR: Mevcut sistem "klimaks hacim ->
   tersine donus (reversal)" varsayimiyla calisiyordu ve zararli cikmisti.
   Bu testte "klimaks hacim -> kirilim yonunde devam (momentum)" varsayimi
   deneniyor - ayni yon (mumun kendi yonu) ile islem aciliyor, tersi degil.
3) DINAMIK VWAP ESIGI: Sabit %2 yerine, esik her coin/zaman icin kendi
   ATR'sine gore (ATR/fiyat * carpan) daralip genisliyor.

Her iki strateji de (VWAP, ZSkor) onceki turda en iyi cikan Trend+Funding
filtresiyle (T+F) test ediliyor - kazanan filtre sabit tutuldu, degisken
olan zaman dilimi + esik/yon mantigi.

NOT: Farkli zaman dilimleri farkli veri hacmi gerektirir. Tek API cagrisi
limiti (1000 mum) nedeniyle her zaman diliminin kapsadigi takvim suresi
farkli olur (5m ~3.5 gun, 15m ~10 gun, 30m ~20 gun) - sonuclari
yorumlarken bunu goz onunde bulundur.

Calistirmak icin: pip install ccxt pandas numpy requests --break-system-packages
                  python3 crypto_next_gen_test.py
NOT: Sadece halka acik (public) veri kullanilir, API key GEREKMEZ.
"""

import os
import time
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
    "PEPE", "SHIB", "TIA", "STRK", "JUP", "PYTH",
]  # onceki testten biraz daha az coin - 3 zaman dilimi x 2 strateji oldugu icin

WATCHLIST = [f"{c}/USDT:USDT" for c in COINS]

TIMEFRAMES = [("5m", 5), ("15m", 15), ("30m", 30)]  # (etiket, dakika)
FETCH_LIMIT = 1000  # tek API cagrisi siniri

TREND_TIMEFRAME = "4h"
TREND_EMA_PERIOD = 200
VWAP_DEV_LONG_MAX = -2.0     # sabit esik varyanti icin
VWAP_DEV_SHORT_MIN = 2.0
DYNAMIC_ATR_MULT = 2.5       # dinamik esik varyanti icin: esik = (ATR/fiyat)*100*bu_carpan
VOLUME_ZSCORE_THRESHOLD = 2.5  # Gemini'nin onerisiyle 2.0'dan 2.5'e cikarildi (momentum icin daha net sinyal)
COMMISSION_PCT = 0.1

# Checkpoint'ler DAKIKA cinsinden sabit - her zaman dilimi kendi mum sayisina cevirir
CHECKPOINTS_MIN = [(60, "1sa", 0.3), (240, "4sa", 0.6), (720, "12sa", 1.0), (1440, "24sa", 1.5)]


def fetch_ohlcv_df(symbol, timeframe, limit):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def fetch_funding_df(symbol, limit=200):
    try:
        raw = exchange.fetch_funding_rate_history(symbol, limit=limit)
    except Exception as e:
        print(f"{symbol}: funding rate cekilemedi ({e})")
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    if not raw:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.rename(columns={"fundingRate": "funding_rate"})
    return df[["timestamp", "funding_rate"]].sort_values("timestamp")


def compute_indicators(df: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    df = df.copy()
    vwap_window = max(20, (24 * 60) // tf_minutes)  # ~24 saatlik kayan pencere

    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_std20"] = df["volume"].rolling(20).std()
    df["vol_zscore"] = (df["volume"] - df["vol_sma20"]) / df["vol_std20"].replace(0, np.nan)

    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    df["vwap"] = pv.rolling(vwap_window).sum() / df["volume"].rolling(vwap_window).sum()
    df["vwap_dev_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["dynamic_threshold_pct"] = (df["atr14"] / df["close"]) * 100 * DYNAMIC_ATR_MULT
    return df


def compute_trend_ema(df_4h: pd.DataFrame) -> pd.DataFrame:
    df_4h = df_4h.copy()
    df_4h["ema200_4h"] = df_4h["close"].ewm(span=TREND_EMA_PERIOD, adjust=False).mean()
    return df_4h[["timestamp", "ema200_4h"]]


def map_asof(df_main: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    if other.empty:
        return df_main
    df = df_main.sort_values("timestamp")
    other = other.sort_values("timestamp")
    return pd.merge_asof(df, other, on="timestamp", direction="backward")


# --- Sinyal fonksiyonlari ---

def signal_vwap_fixed(row):
    if pd.isna(row.get("vwap_dev_pct")):
        return None
    if row["vwap_dev_pct"] <= VWAP_DEV_LONG_MAX:
        return "LONG"
    if row["vwap_dev_pct"] >= VWAP_DEV_SHORT_MIN:
        return "SHORT"
    return None


def signal_vwap_dynamic(row):
    if pd.isna(row.get("vwap_dev_pct")) or pd.isna(row.get("dynamic_threshold_pct")):
        return None
    th = row["dynamic_threshold_pct"]
    if row["vwap_dev_pct"] <= -th:
        return "LONG"
    if row["vwap_dev_pct"] >= th:
        return "SHORT"
    return None


def signal_zscore_reversal(row):
    """Eski (mevcut canli bot) mantigi: klimaks hacim -> TERSINE donus."""
    if pd.isna(row.get("vol_zscore")) or row["vol_zscore"] < VOLUME_ZSCORE_THRESHOLD:
        return None
    if row["close"] < row["open"]:
        return "LONG"
    elif row["close"] > row["open"]:
        return "SHORT"
    return None


def signal_zscore_momentum(row):
    """Gemini'nin onerisi: klimaks hacim -> KIRILIM YONUNDE DEVAM (momentum)."""
    if pd.isna(row.get("vol_zscore")) or row["vol_zscore"] < VOLUME_ZSCORE_THRESHOLD:
        return None
    if row["close"] > row["open"]:
        return "LONG"
    elif row["close"] < row["open"]:
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
    return row["funding_rate"] < 0 if direction == "LONG" else row["funding_rate"] > 0


def run_backtest(df, signal_fn, tf_minutes, apply_filters=True, min_gap_bars=3):
    outcomes = []
    checkpoints_bars = [(minutes // tf_minutes, label, target) for minutes, label, target in CHECKPOINTS_MIN]
    max_cp = max(c[0] for c in checkpoints_bars)
    n = len(df)
    last_i = -min_gap_bars - 1
    i = 30
    while i < n - max_cp - 1:
        if i - last_i < min_gap_bars:
            i += 1
            continue
        row = df.iloc[i]
        direction = signal_fn(row)
        if direction is None:
            i += 1
            continue
        if apply_filters:
            if not passes_trend_filter(row, direction):
                i += 1
                continue
            if not passes_funding_filter(row, direction):
                i += 1
                continue

        entry_price = row["close"]
        hit = False
        raw_pct = None
        for bars_ahead, label, target_pct in checkpoints_bars:
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


def main():
    send_telegram_message(
        "🏁 Sonraki nesil test basliyor (Gemini'nin 3 fikri): "
        "coklu zaman dilimi (5m/15m/30m) + Z-Skor momentum donusu + dinamik VWAP esigi. "
        "Hepsi Trend+Funding filtresiyle. Biraz surebilir..."
    )

    variant_defs = []
    for tf_label, tf_minutes in TIMEFRAMES:
        variant_defs.append((f"VWAP-sabit-{tf_label}-TF", signal_vwap_fixed, tf_minutes))
        variant_defs.append((f"VWAP-dinamik-{tf_label}-TF", signal_vwap_dynamic, tf_minutes))
        variant_defs.append((f"ZSkor-reversal-{tf_label}-TF", signal_zscore_reversal, tf_minutes))
        variant_defs.append((f"ZSkor-momentum-{tf_label}-TF", signal_zscore_momentum, tf_minutes))

    results = {name: [] for name, _, _ in variant_defs}

    # veri onbellegi: ayni sembol icin farkli zaman dilimlerini ayri ayri cekiyoruz
    for symbol in WATCHLIST:
        try:
            df_4h = fetch_ohlcv_df(symbol, TREND_TIMEFRAME, limit=300)
            funding_df = fetch_funding_df(symbol, limit=200)
            trend_4h = compute_trend_ema(df_4h) if len(df_4h) >= TREND_EMA_PERIOD // 4 else None

            for tf_label, tf_minutes in TIMEFRAMES:
                df_tf = fetch_ohlcv_df(symbol, tf_label, limit=FETCH_LIMIT)
                if len(df_tf) < 250:
                    print(f"{symbol} {tf_label}: yetersiz veri, atlandi")
                    continue
                df_tf = compute_indicators(df_tf, tf_minutes)
                if trend_4h is not None:
                    df_tf = map_asof(df_tf, trend_4h)
                else:
                    df_tf["ema200_4h"] = np.nan
                df_tf = map_asof(df_tf, funding_df)

                for name, signal_fn, _ in [v for v in variant_defs if v[0].endswith(f"-{tf_label}-TF")]:
                    out = run_backtest(df_tf, signal_fn, tf_minutes, apply_filters=True)
                    results[name].extend(out)

                time.sleep(0.2)

            print(f"{symbol}: tamamlandi")
        except Exception as e:
            print(f"{symbol}: hata - {e}")

    rows = [summarize(name, results[name]) for name, _, _ in variant_defs]
    table = pd.DataFrame(rows).sort_values("ort_net_%", ascending=False, na_position="last")
    print("\n--- SONRAKI NESIL TEST SONUCLARI ---")
    print(table.to_string(index=False))
    table.to_csv("sonraki_nesil_sonuclari.csv", index=False)

    send_telegram_message("📊 SONRAKI NESIL TEST SONUCLARI (buyukten kucuge ort_net_%)\n\n" + table.to_string(index=False))
    finish_msg = f"✅ Sonraki nesil test tamamlandi - {datetime.now().isoformat()}"
    print(finish_msg)
    send_telegram_message(finish_msg)


if __name__ == "__main__":
    main()
