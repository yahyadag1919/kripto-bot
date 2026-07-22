"""
simple_momentum_walk_forward.py

Onceki tum denemeler (VWAP sapmasi, hacim z-skor, Donchian trend-takip)
gercek edge tasimadi. Kullanicinin BOTTAN ONCE, kendi elleriyle yaptigi ve
gercekten kar eden yontem cok daha basitti: mumlara bakip bir yone dogru
"guc" var mi diye bakip aciyordu, komisyonun biraz ustunde kucuk bir kar
hedefi + stop koyuyordu. Hicbir gosterge, filtre, trailing stop yoktu.

Bu script O BASIT MANTIGI iki olasi yorumla test ediyor (hangisi onun
yaptigina daha yakinsa ilerde onu esas alacagiz):

  A) MOMENTUM: son N mum ayni yonde kapanmis (ust uste yesil/kirmizi)
  B) GUCLU MUM: tek bir mumun govdesi, son 20 mumun ortalama govdesinden
     belirgin sekilde buyuk (ani/guclu hareket)

Her ikisi de AYRI AYRI test ediliyor (birbirinden bagimsiz iki basit
strateji), ayni sabit kucuk TP + dar stop ile. Amac: hangisi (varsa)
gercekten TRAIN'de de TEST'te de (hic gorulmemis veride) pozitif cikiyor,
onu bulmak.

TP/STOP: TP_PCT = komisyon + kucuk bir hedef (varsayilan %0.4 -> yani
gercek kar hedefi ~%0.3). STOP_PCT = biraz daha genis (varsayilan %0.6) -
"kucuk kar hedefi + stop" tanimina uyacak sekilde. Bu sayilari degistirip
tekrar calistirmak cok kolay, en alttaki ayarlar bolumunden.

CALISTIRMA (once walk_forward_test.py'yi calistirdigin gibi, Railway'de
Start Command'i gecici degistirerek):
    pip install ccxt pandas numpy requests
    python simple_momentum_walk_forward.py
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
# Ayarlar
# ---------------------------------------------------------------------------

COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "SUI",
    "DOT", "TRX", "ATOM", "NEAR", "TON", "LTC", "BCH", "ETC", "FIL", "APT",
    "ARB", "OP", "INJ", "SEI", "ICP",
]
WATCHLIST = [f"{c}/USDT:USDT" for c in COINS]

TIMEFRAME = "15m"               # kullanici bunu gun ici, kisa vadeli yapiyordu
DAYS_OF_HISTORY = 45
TRAIN_FRACTION = 0.6

CONSECUTIVE_CANDLES = 3         # Strateji A: ust uste kac mum ayni yonde olsun
STRONG_BODY_MULT = 1.8          # Strateji B: govde, ort. govdenin kac kati olsun
BODY_AVG_WINDOW = 20

TP_PCT = 0.4                    # brut kar hedefi (komisyon dahil, net ~%0.3 kar)
STOP_PCT = 0.6                  # stop mesafesi
ROUNDTRIP_COMMISSION_PCT = 0.1
MAX_HOLD_CANDLES = 96           # 15m * 96 = 24 saat, cok uzun beklemesin

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
    """Strateji A: ust uste CONSECUTIVE_CANDLES mum ayni yonde -> o yonde gir."""
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
        if trade:
            trades.append(trade)
        i += 1

    return trades


def simulate_strong_candle(df: pd.DataFrame) -> list:
    """Strateji B: tek mumun govdesi, ortalamanin STRONG_BODY_MULT kati -> o yonde gir."""
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
            if trade:
                trades.append(trade)
        i += 1

    return trades


def run_fixed_tp_sl(df: pd.DataFrame, i: int, direction: str) -> dict:
    """i. mumun kapanisindan giris yapip, sabit TP/STOP ile ilerideki mumlari tarar."""
    entry_price = df.iloc[i]["close"]
    n = len(df)

    if direction == "LONG":
        tp_price = entry_price * (1 + TP_PCT / 100)
        sl_price = entry_price * (1 - STOP_PCT / 100)
    else:
        tp_price = entry_price * (1 - TP_PCT / 100)
        sl_price = entry_price * (1 + STOP_PCT / 100)

    outcome = None
    exit_price = None

    for j in range(i + 1, min(i + 1 + MAX_HOLD_CANDLES, n)):
        candle = df.iloc[j]
        if direction == "LONG":
            hit_sl = candle["low"] <= sl_price
            hit_tp = candle["high"] >= tp_price
        else:
            hit_sl = candle["high"] >= sl_price
            hit_tp = candle["low"] <= tp_price

        # ayni mum icinde ikisi de tetiklenebilir - kotumser varsayim: stop once vurdu say
        if hit_sl and hit_tp:
            exit_price, outcome = sl_price, "SL"
            break
        elif hit_sl:
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

    return {
        "timestamp": df.iloc[i]["timestamp"], "direction": direction,
        "entry": entry_price, "exit": exit_price,
        "outcome": outcome, "pct_change": pct_change,
    }


def summarize(trades: list, label: str) -> dict:
    if not trades:
        return {"label": label, "n": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t["pct_change"] > 0)
    total_net = sum(t["pct_change"] for t in trades)
    return {
        "label": label, "n": n, "hit_rate": wins / n * 100,
        "avg_net": total_net / n, "total_net": total_net,
    }


def fmt(s: dict) -> str:
    if s["n"] == 0:
        return f"  {s['label']}: sinyal yok"
    return (
        f"  {s['label']}: {s['n']} işlem | isabet %{s['hit_rate']:.1f} | "
        f"ort. net %{s['avg_net']:+.3f} | toplam net %{s['total_net']:+.2f}"
    )


def main():
    print(f"Basit momentum walk-forward test basliyor - {len(COINS)} coin, {DAYS_OF_HISTORY} gun\n")
    send_telegram_message(
        f"🕯️ Basit momentum (ardışık mum / güçlü mum) walk-forward test başladı "
        f"({len(COINS)} coin). Sonuç gelince buraya düşecek..."
    )

    results = {"momentum": {"train": [], "test": []}, "strong_candle": {"train": [], "test": []}}

    for symbol in WATCHLIST:
        print(f"--- {symbol} ---")
        try:
            df = fetch_full_history(symbol, TIMEFRAME, DAYS_OF_HISTORY)
        except Exception as e:
            print(f"  veri cekilemedi: {e}")
            continue

        if len(df) < BODY_AVG_WINDOW * 3:
            print("  yeterli veri yok, atlaniyor")
            continue

        df = compute_indicators(df)
        split_idx = int(len(df) * TRAIN_FRACTION)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        results["momentum"]["train"] += simulate_momentum(train_df)
        results["momentum"]["test"] += simulate_momentum(test_df)
        results["strong_candle"]["train"] += simulate_strong_candle(train_df)
        results["strong_candle"]["test"] += simulate_strong_candle(test_df)

    print("\n================= SONUC =================")
    lines = ["📊 Basit momentum walk-forward SONUÇ:\n"]
    for key, label in [("momentum", f"A) Ardışık {CONSECUTIVE_CANDLES} mum"), ("strong_candle", "B) Tek güçlü mum")]:
        train_s = summarize(results[key]["train"], "TRAIN (ilk %60)")
        test_s = summarize(results[key]["test"], "TEST  (son %40, hiç görülmemiş)")
        print(f"{label}:")
        print(fmt(train_s))
        print(fmt(test_s))
        lines.append(f"{label}:")
        lines.append(fmt(train_s))
        lines.append(fmt(test_s))
        lines.append("")

    lines.append(
        f"(TP %{TP_PCT} / STOP %{STOP_PCT}, komisyon %{ROUNDTRIP_COMMISSION_PCT} düşülmüş halde)\n"
        "Hangisi TRAIN'de de TEST'te de pozitifse, o senin eski manuel "
        "yöntemine en yakın adaydır."
    )
    send_telegram_message("\n".join(lines))


if __name__ == "__main__":
    main()
