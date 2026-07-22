"""
trend_follow_walk_forward.py

VWAP Sapmasi ve Hacim Z-Skor stratejileri walk-forward testte hem TRAIN hem
TEST'te zararli cikti (gercek edge yok). Bu script FARKLI bir yaklasimi test
ediyor: kisa vadeli scalping yerine 4 saatlik mumlarla KLASIK TREND-TAKIP
(Donchian breakout) + kazananin buyumesine izin veren TRAILING STOP
(chandelier exit). Mantik: "kazananı büyüt, kaybedeni hızlı kes" - sabit kucuk
TP yerine, trend devam ettikce pozisyon acik kaliyor, sadece stop yukari
cekiliyor.

STRATEJI
--------
Giris:
  - Fiyat, son DONCHIAN_PERIOD mumun en yuksegini kirarsa -> LONG
  - Fiyat, son DONCHIAN_PERIOD mumun en dusugunu kirarsa  -> SHORT
Trend filtresi:
  - LONG sadece fiyat TREND_EMA_PERIOD'luk EMA'nin USTUNDEYSE aciliyor
  - SHORT sadece fiyat EMA'nin ALTINDAYSA aciliyor
Cikis (chandelier trailing stop):
  - LONG: stop = (pozisyon acildiginda beri gorulen EN YUKSEK kapanis) - ATR*CHANDELIER_MULT
    Bu seviye SADECE yukari cekilir, asla geri gevsetilmez.
  - SHORT: aynisinin ters yonu
  - Fiyat trailing stop'u kirinca pozisyon kapanir (kar ne kadar buyumusse o kadar alinir)
  - MAX_HOLD_CANDLES asilirsa (cok uzun surerse) zorla kapatilir

Ayni onceki script gibi veriyi TRAIN (ilk %60) / TEST (son %40) diye
ZAMAN SIRASINA GORE ikiye boluyor, TREN filtresini ve giris/cikis mantigini
TEK SEFER tanimlayip degistirmeden ikisine de uyguluyor - boylece "TRAIN'de
iyi gorunup TEST'te cokuyor mu" sorusuna yine dogru cevap alacagiz.

ONEMLI KISIT: Bu script'i Claude'un sandbox'i CALISTIRAMIYOR (internet
erisimi kapali). Railway'de walk_forward_test.py'yi calistirdigin gibi
calistir - Start Command'i gecici olarak
    python trend_follow_walk_forward.py
yap, sonuc Telegram'a gelince eski komuta geri don.

    pip install ccxt pandas numpy requests
    python trend_follow_walk_forward.py
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

TIMEFRAME = "4h"
DAYS_OF_HISTORY = 400          # ~2400 adet 4h mum - Donchian(55)+EMA(200) icin yeterli isinma + bol sinyal
TRAIN_FRACTION = 0.6

DONCHIAN_PERIOD = 55            # giris esigi - klasik "turtle" sistemine yakin
TREND_EMA_PERIOD = 200
ATR_PERIOD = 14
CHANDELIER_MULT = 3.0           # trailing stop mesafesi = ATR * bu katsayi
MAX_HOLD_CANDLES = 180          # 4h * 180 = 30 gun - cok uzayan pozisyonu zorla kapat
ROUNDTRIP_COMMISSION_PCT = 0.1

exchange = ccxt.binanceusdm({"enableRateLimit": True})


def fetch_full_history(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    ms_per_candle = {"4h": 4 * 60 * 60 * 1000, "1d": 24 * 60 * 60 * 1000}[timeframe]
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
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["donchian_high"] = df["high"].shift(1).rolling(DONCHIAN_PERIOD).max()
    df["donchian_low"] = df["low"].shift(1).rolling(DONCHIAN_PERIOD).min()
    df["ema_trend"] = df["close"].ewm(span=TREND_EMA_PERIOD, adjust=False).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD).mean()
    return df


def simulate(df: pd.DataFrame) -> list:
    trades = []
    n = len(df)
    warmup = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD, ATR_PERIOD) + 5
    i = warmup

    while i < n - 1:
        row = df.iloc[i]
        if pd.isna(row["donchian_high"]) or pd.isna(row["ema_trend"]) or pd.isna(row["atr"]):
            i += 1
            continue

        direction = None
        if row["close"] > row["donchian_high"] and row["close"] > row["ema_trend"]:
            direction = "LONG"
        elif row["close"] < row["donchian_low"] and row["close"] < row["ema_trend"]:
            direction = "SHORT"

        if direction is None:
            i += 1
            continue

        entry_price = row["close"]
        atr = row["atr"]
        extreme = entry_price  # LONG icin en yuksek kapanis, SHORT icin en dusuk kapanis
        if direction == "LONG":
            stop = entry_price - atr * CHANDELIER_MULT
        else:
            stop = entry_price + atr * CHANDELIER_MULT

        outcome = None
        pct_change = None
        exit_price = None

        for j in range(i + 1, min(i + 1 + MAX_HOLD_CANDLES, n)):
            candle = df.iloc[j]

            if direction == "LONG":
                if candle["low"] <= stop:
                    exit_price = stop
                    outcome = "TRAILING_STOP"
                    break
                extreme = max(extreme, candle["close"])
                new_stop = extreme - candle["atr"] * CHANDELIER_MULT if pd.notna(candle["atr"]) else stop
                stop = max(stop, new_stop)  # sadece yukari cek
            else:
                if candle["high"] >= stop:
                    exit_price = stop
                    outcome = "TRAILING_STOP"
                    break
                extreme = min(extreme, candle["close"])
                new_stop = extreme + candle["atr"] * CHANDELIER_MULT if pd.notna(candle["atr"]) else stop
                stop = min(stop, new_stop)  # sadece asagi cek

        if outcome is None:
            last_j = min(i + MAX_HOLD_CANDLES, n - 1)
            exit_price = df.iloc[last_j]["close"]
            outcome = "MAX_HOLD"

        raw = (exit_price - entry_price) / entry_price * 100
        pct_change = (raw if direction == "LONG" else -raw) - ROUNDTRIP_COMMISSION_PCT

        trades.append({
            "timestamp": row["timestamp"], "direction": direction,
            "entry": entry_price, "exit": exit_price,
            "outcome": outcome, "pct_change": pct_change,
        })

        i += 1

    return trades


def summarize(trades: list, label: str) -> dict:
    if not trades:
        return {"label": label, "n": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t["pct_change"] > 0)
    total_net = sum(t["pct_change"] for t in trades)
    avg_net = total_net / n
    # kazanan/kaybeden ortalamasi ayri - "kazananı büyüt, kaybedeni hızlı kes" gercekten oluyor mu gorelim
    win_pcts = [t["pct_change"] for t in trades if t["pct_change"] > 0]
    loss_pcts = [t["pct_change"] for t in trades if t["pct_change"] <= 0]
    avg_win = sum(win_pcts) / len(win_pcts) if win_pcts else 0
    avg_loss = sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0
    return {
        "label": label, "n": n, "hit_rate": wins / n * 100,
        "avg_net": avg_net, "total_net": total_net,
        "avg_win": avg_win, "avg_loss": avg_loss,
    }


def print_summary(s: dict) -> str:
    if s["n"] == 0:
        return f"  {s['label']}: sinyal yok"
    return (
        f"  {s['label']}: {s['n']} işlem | isabet %{s['hit_rate']:.1f} | "
        f"ort. net %{s['avg_net']:+.3f} | toplam net %{s['total_net']:+.2f} | "
        f"ort. kazanan %{s['avg_win']:+.2f} | ort. kaybeden %{s['avg_loss']:+.2f}"
    )


def main():
    print(f"Trend-takip walk-forward test basliyor - {len(COINS)} coin, {DAYS_OF_HISTORY} gun (4h mum)\n")
    send_telegram_message(
        f"🐢 Trend-takip (Donchian+Trailing Stop) walk-forward test başladı "
        f"({len(COINS)} coin, {DAYS_OF_HISTORY} gün, 4h mum). Sonuç gelince buraya düşecek..."
    )

    all_train, all_test = [], []

    for symbol in WATCHLIST:
        print(f"--- {symbol} ---")
        try:
            df = fetch_full_history(symbol, TIMEFRAME, DAYS_OF_HISTORY)
        except Exception as e:
            print(f"  veri cekilemedi: {e}")
            continue

        if len(df) < TREND_EMA_PERIOD + DONCHIAN_PERIOD + 20:
            print("  yeterli veri yok, atlaniyor")
            continue

        df = compute_indicators(df)
        split_idx = int(len(df) * TRAIN_FRACTION)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        all_train += simulate(train_df)
        all_test += simulate(test_df)

    train_s = summarize(all_train, "TRAIN (ilk %60)")
    test_s = summarize(all_test, "TEST  (son %40, hiç görülmemiş)")

    print("\n================= SONUC =================")
    print(print_summary(train_s))
    print(print_summary(test_s))

    lines = [
        "📊 Trend-takip (Donchian + Trailing Stop) walk-forward SONUÇ:",
        "",
        print_summary(train_s),
        print_summary(test_s),
        "",
        "Yorum: TEST, TRAIN'e yakınsa (ikisi de pozitifse) bu strateji önceki "
        "VWAP/Hacim Z-Skor'un aksine gerçek bir edge taşıyor olabilir. "
        "'Ort. kazanan' 'ort. kaybeden'den kat kat büyükse trailing-stop "
        "mantığı (kazananı büyüt, kaybedeni hızlı kes) gerçekten işliyor demektir.",
    ]
    send_telegram_message("\n".join(lines))


if __name__ == "__main__":
    main()
