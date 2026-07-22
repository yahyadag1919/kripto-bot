"""
tradfi_vs_crypto_walk_forward.py

Kriptoda 4 farkli yaklasim (VWAP, Hacim Z-Skor, Donchian trend-takip, mum
momentum, ve tersi) walk-forward testte gercek bir edge gostermedi. Bu
script AYNI basit "momentum" mantigini (ardisik ayni yonlu mumlar VE tek
guclu mum - simple_momentum_walk_forward.py'daki gibi), Binance'in TradFi
(gelenksel varlik) vadeli islem urunlerinde de test ediyor:

  - Emtialar: Altin (XAU), Gumus (XAG), Platin (XPT), Paladyum (XPD),
    WTI Petrol (CL), Brent Petrol (BZ), Dogalgaz (NATGAS)
  - Hisseler: Tesla, Amazon, Intel, Robinhood, Strategy (MSTR), SoFi,
    Palo Alto Networks

Ayni testi kripto majors ile YAN YANA calistirip, TRAIN/TEST sonuclarini
KATEGORI BAZINDA (kripto / emtia / hisse) ayri ayri raporluyor - boylece
"hangi pazar turu bu basit stratejiye daha uygun" sorusuna veri ile cevap
bulabiliyoruz.

NOT: TradFi sembollerinin TAM DOGRU YAZIMI/kullanilabilirligi zamanla
degisebilir (Binance surekli yeni urun ekliyor/kaldiriyor). Sembol
listesini calistirmadan once Binance uygulamasindaki "TradFi" sekmesinden
teyit et - bir sembol calismazsa script onu atlayip devam eder, hata
vermez.

ONEMLI KISIT: Bu script'i Claude'un sandbox'i CALISTIRAMIYOR (internet
erisimi kapali). Railway'de digerleri gibi calistir:
    pip install ccxt pandas numpy requests
    python tradfi_vs_crypto_walk_forward.py
"""

import os
import time

import ccxt
import numpy as np
import pandas as pd
import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print(f"Telegram gonderim hatasi: {e}")


# ---------------------------------------------------------------------------
# Ayarlar - canli bottaki GUNCEL degerlerle ayni (basit sabit TP/stop)
# ---------------------------------------------------------------------------

CATEGORIES = {
    "KRIPTO": [
        "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "SUI",
    ],
    "EMTIA": [
        "XAU", "XAG", "XPT", "XPD", "CL", "BZ", "NATGAS",
    ],
    "HISSE": [
        "TSLA", "AMZN", "INTC", "HOOD", "MSTR", "SOFI", "PANW",
    ],
}

TIMEFRAME = "15m"
DAYS_OF_HISTORY = 45
TRAIN_FRACTION = 0.6

CONSECUTIVE_CANDLES = 3
STRONG_BODY_MULT = 1.8
BODY_AVG_WINDOW = 20

# Canli bottaki guncel (basitlestirilmis) TP/STOP degerleriyle ayni
TP_PCT = 0.4        # ROUNDTRIP_COMMISSION_PCT(0.1) + SIMPLE_PROFIT_TARGET_PCT(0.3)
STOP_PCT = 0.6       # SIMPLE_STOP_PCT
ROUNDTRIP_COMMISSION_PCT = 0.1
MAX_HOLD_CANDLES = 96

exchange = ccxt.binanceusdm({"enableRateLimit": True})


def fetch_full_history(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    ms_per_candle = 15 * 60 * 1000
    since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    all_rows = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1500)
        if not batch:
            break
        all_rows += batch
        last_ts = batch[-1][0]
        since = last_ts + ms_per_candle
        if len(batch) < 1500:
            break
        time.sleep(exchange.rateLimit / 1000)
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["body"] = (df["close"] - df["open"]).abs()
    df["is_green"] = df["close"] > df["open"]
    df["avg_body20"] = df["body"].rolling(BODY_AVG_WINDOW).mean()
    return df


def simulate_momentum(df: pd.DataFrame) -> list:
    trades = []
    n = len(df)
    i = CONSECUTIVE_CANDLES + 1
    while i < n - 1:
        window = df.iloc[i - CONSECUTIVE_CANDLES:i]
        if window["is_green"].all():
            direction = "LONG"
        elif (~window["is_green"]).all():
            direction = "SHORT"
        else:
            i += 1
            continue
        trade = run_fixed_tp_sl(df, i, direction)
        if trade is not None:
            trades.append(trade)
        i += 1
    return trades


def simulate_strong_candle(df: pd.DataFrame) -> list:
    trades = []
    n = len(df)
    i = BODY_AVG_WINDOW + 1
    while i < n - 1:
        row = df.iloc[i]
        if pd.isna(row["avg_body20"]) or row["avg_body20"] == 0:
            i += 1
            continue
        if row["body"] >= row["avg_body20"] * STRONG_BODY_MULT:
            direction = "LONG" if row["is_green"] else "SHORT"
            trade = run_fixed_tp_sl(df, i, direction)
            if trade is not None:
                trades.append(trade)
        i += 1
    return trades


def run_fixed_tp_sl(df: pd.DataFrame, i: int, direction: str) -> dict:
    entry_price = df.iloc[i]["close"]
    n = len(df)

    if direction == "LONG":
        tp_price = entry_price * (1 + TP_PCT / 100)
        sl_price = entry_price * (1 - STOP_PCT / 100)
    else:
        tp_price = entry_price * (1 - TP_PCT / 100)
        sl_price = entry_price * (1 + STOP_PCT / 100)

    outcome, exit_price = None, None
    for j in range(i + 1, min(i + 1 + MAX_HOLD_CANDLES, n)):
        candle = df.iloc[j]
        if direction == "LONG":
            hit_sl = candle["low"] <= sl_price
            hit_tp = candle["high"] >= tp_price
        else:
            hit_sl = candle["high"] >= sl_price
            hit_tp = candle["low"] <= tp_price
        if hit_sl:
            exit_price, outcome = sl_price, "SL"
            break
        elif hit_tp:
            exit_price, outcome = tp_price, "TP"
            break

    if outcome is None:
        last_j = min(i + MAX_HOLD_CANDLES, n - 1)
        exit_price = df.iloc[last_j]["close"]
        outcome = "SURE_DOLDU"

    raw = (exit_price - entry_price) / entry_price * 100
    pct_change = (raw if direction == "LONG" else -raw) - ROUNDTRIP_COMMISSION_PCT
    return {"pct_change": pct_change, "outcome": outcome}


def summarize(trades: list) -> dict:
    if not trades:
        return {"n": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t["pct_change"] > 0)
    total_net = sum(t["pct_change"] for t in trades)
    return {"n": n, "hit_rate": wins / n * 100, "avg_net": total_net / n, "total_net": total_net}


def fmt(s: dict, label: str) -> str:
    if s["n"] == 0:
        return f"  {label}: sinyal yok"
    return (f"  {label}: {s['n']} işlem | isabet %{s['hit_rate']:.1f} | "
            f"ort. net %{s['avg_net']:+.3f} | toplam %{s['total_net']:+.1f}")


def main():
    print("TradFi vs Kripto walk-forward test basliyor\n")
    all_symbols = sum(CATEGORIES.values(), [])
    send_telegram_message(
        f"🌐 TradFi vs Kripto walk-forward test başladı ({len(all_symbols)} sembol: "
        f"kripto/emtia/hisse). Sonuç gelince buraya düşecek, biraz sürebilir..."
    )

    # kategori -> strateji -> train/test -> islem listesi
    results = {cat: {"momentum": {"train": [], "test": []}, "strong": {"train": [], "test": []}} for cat in CATEGORIES}
    skipped = []

    for category, coins in CATEGORIES.items():
        for coin in coins:
            symbol = f"{coin}/USDT:USDT"
            print(f"--- [{category}] {symbol} ---")
            try:
                df = fetch_full_history(symbol, TIMEFRAME, DAYS_OF_HISTORY)
            except Exception as e:
                print(f"  veri cekilemedi, atlaniyor: {e}")
                skipped.append(symbol)
                continue

            if len(df) < BODY_AVG_WINDOW * 3:
                print("  yeterli veri yok, atlaniyor")
                skipped.append(symbol)
                continue

            df = compute_indicators(df)
            split_idx = int(len(df) * TRAIN_FRACTION)
            train_df = df.iloc[:split_idx].reset_index(drop=True)
            test_df = df.iloc[split_idx:].reset_index(drop=True)

            results[category]["momentum"]["train"] += simulate_momentum(train_df)
            results[category]["momentum"]["test"] += simulate_momentum(test_df)
            results[category]["strong"]["train"] += simulate_strong_candle(train_df)
            results[category]["strong"]["test"] += simulate_strong_candle(test_df)

    print("\n================= SONUC (kategori bazli) =================")
    lines = [f"🌐 TradFi vs Kripto walk-forward SONUÇ (TP%{TP_PCT}/STOP%{STOP_PCT}):\n"]

    for category in CATEGORIES:
        lines.append(f"── {category} ──")
        print(f"\n{category}:")
        for strat_key, strat_label in [("momentum", "Ardışık mum"), ("strong", "Güçlü mum")]:
            train_s = summarize(results[category][strat_key]["train"])
            test_s = summarize(results[category][strat_key]["test"])
            print(f" {strat_label}:")
            print(fmt(train_s, "  TRAIN"))
            print(fmt(test_s, "  TEST "))
            lines.append(f"{strat_label}:")
            lines.append(fmt(train_s, "TRAIN"))
            lines.append(fmt(test_s, "TEST"))
        lines.append("")

    if skipped:
        lines.append(f"Atlanan semboller (veri çekilemedi): {', '.join(skipped)}")

    lines.append(
        "Yorum: Hangi kategoride TRAIN'de de TEST'te de pozitifse, o pazar "
        "türü bu basit stratejiye kriptodan daha uygun olabilir."
    )
    send_telegram_message("\n".join(lines))


if __name__ == "__main__":
    main()
