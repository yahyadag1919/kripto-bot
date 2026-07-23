"""
validate_emtia_donchian_reversed.py

AMAC: 18 kombinasyonluk (3 strateji x 3 kategori x orijinal/ters) testte
TEK bir kombinasyon hem TRAIN hem TEST'te pozitif cikti:

    EMTIA + Donchian Kirilim + TERS yon
    TRAIN: 183 islem, isabet %48.1, toplam $+15.48
    TEST : 119 islem, isabet %52.9, toplam $+8.37

Ama 18 kombinasyon denenince birinin sans eseri pozitif cikmasi istatistiksel
olarak BEKLENEN bir sey - tipki 60+ strateji denendiginde bazilarinin "iyi"
gorunmesi gibi. Bu script SADECE bu tek kombinasyonu, DAHA UZUN bir veri
penceresinde (120 gun, oncekinin ~3 kati) ve 3 farkli train/test bolme
noktasinda (%50, %60, %70) tekrar test ediyor. Eger sonuc:
  - Farkli pencerelerde/bolmelerde TUTARLI sekilde pozitif kaliyorsa ->
    gercek bir kenar olma ihtimali yuksek
  - Pencereye/bolmeye gore pozitif-negatif arasinda ZIPLIYORSA -> onceki
    bulgu byuk ihtimalle rastlantisaldi (18 kombinasyondan biri sansla
    cikti), gercek degil

Strateji: Donchian(20) kirilim + EMA200 trend filtresi, YON TERSINE CEVRILMIS
(yani normalde LONG sinyali SHORT olarak, SHORT sinyali LONG olarak aciliyor).
Emtia sembolleri: XAU, XAG, XPT, XPD, CL, BZ, NATGAS (onceki testle ayni).
TP%5/STOP%5, sabit marjin %2 x 20x kaldirac, 100$ referans bakiye - hepsi
onceki testle BIREBIR AYNI, sadece veri suresi ve bolme noktalari degisti.

ONEMLI KISIT: Bu script'i Claude'un sandbox'i CALISTIRAMIYOR. Railway'de
digerleri gibi calistir:
    pip install ccxt pandas numpy requests
    python validate_emtia_donchian_reversed.py
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

COMMODITIES = ["XAU", "XAG", "XPT", "XPD", "CL", "BZ", "NATGAS"]

TIMEFRAME = "15m"
DAYS_OF_HISTORY = 120     # oncekinin ~3 kati - daha genis, daha guvenilir veri
SPLIT_POINTS = [0.5, 0.6, 0.7]  # farkli train/test bolme noktalarinda tutarlilik testi

DONCHIAN_PERIOD = 20
TREND_EMA_PERIOD = 200

TP_PRICE_PCT = 5.0
STOP_PRICE_PCT = 5.0
ROUNDTRIP_COMMISSION_PCT = 0.1
MAX_HOLD_CANDLES = 480

STARTING_BALANCE = 100.0
POSITION_PCT_OF_BALANCE = 2.0
LEVERAGE = 20

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
    df["donchian_high"] = df["high"].shift(1).rolling(DONCHIAN_PERIOD).max()
    df["donchian_low"] = df["low"].shift(1).rolling(DONCHIAN_PERIOD).min()
    df["ema_trend"] = df["close"].ewm(span=TREND_EMA_PERIOD, adjust=False).mean()
    return df


def find_entries_donchian_reversed(df: pd.DataFrame) -> list:
    """Normal Donchian kirilim + trend filtresi, ama yon TERSINE cevrilmis."""
    entries = []
    n = len(df)
    i = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD) + 5
    while i < n - 1:
        row = df.iloc[i]
        if pd.isna(row["donchian_high"]) or pd.isna(row["ema_trend"]):
            i += 1
            continue
        raw_direction = None
        if row["close"] > row["donchian_high"] and row["close"] > row["ema_trend"]:
            raw_direction = "LONG"
        elif row["close"] < row["donchian_low"] and row["close"] < row["ema_trend"]:
            raw_direction = "SHORT"
        if raw_direction:
            reversed_direction = "SHORT" if raw_direction == "LONG" else "LONG"
            entries.append((i, reversed_direction))
        i += 1
    return entries


def simulate_dollar_pnl(df: pd.DataFrame, entries: list) -> list:
    trades = []
    n = len(df)
    fixed_margin = STARTING_BALANCE * (POSITION_PCT_OF_BALANCE / 100)
    fixed_notional = fixed_margin * LEVERAGE

    for i, direction in entries:
        entry_price = df.iloc[i]["close"]
        if direction == "LONG":
            tp_price = entry_price * (1 + TP_PRICE_PCT / 100)
            sl_price = entry_price * (1 - STOP_PRICE_PCT / 100)
        else:
            tp_price = entry_price * (1 - TP_PRICE_PCT / 100)
            sl_price = entry_price * (1 + STOP_PRICE_PCT / 100)

        position_units = fixed_notional / entry_price
        commission_dollar = fixed_notional * (ROUNDTRIP_COMMISSION_PCT / 100)

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

        price_move = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
        net_pnl_dollar = position_units * price_move - commission_dollar
        trades.append({"pnl_dollar": net_pnl_dollar})

    return trades


def summarize(trades: list) -> dict:
    if not trades:
        return {"n": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t["pnl_dollar"] > 0)
    total_dollar = sum(t["pnl_dollar"] for t in trades)
    return {"n": n, "hit_rate": wins / n * 100, "total_dollar": total_dollar, "avg_dollar": total_dollar / n}


def fmt(s: dict, label: str) -> str:
    if s["n"] == 0:
        return f"  {label}: sinyal yok"
    return f"  {label}: {s['n']} işlem | isabet %{s['hit_rate']:.1f} | toplam ${s['total_dollar']:+.2f} (ort ${s['avg_dollar']:+.3f})"


def main():
    print(f"Emtia+Donchian+Ters DOGRULAMA testi basliyor - {len(COMMODITIES)} sembol, {DAYS_OF_HISTORY} gun\n")
    send_telegram_message(
        f"🔬 Doğrulama testi başladı: Emtia + Donchian + TERS ({len(COMMODITIES)} sembol, "
        f"{DAYS_OF_HISTORY} gün, 3 farklı train/test bölme noktası). Sonuç gelince buraya düşecek..."
    )

    dfs = {}
    skipped = []
    for coin in COMMODITIES:
        symbol = f"{coin}/USDT:USDT"
        print(f"--- {symbol} ---")
        try:
            df = fetch_full_history(symbol, TIMEFRAME, DAYS_OF_HISTORY)
        except Exception as e:
            print(f"  veri cekilemedi: {e}")
            skipped.append(symbol)
            continue
        if len(df) < TREND_EMA_PERIOD + 20:
            skipped.append(symbol)
            continue
        dfs[symbol] = compute_indicators(df)

    lines = [f"🔬 Doğrulama SONUÇ (Emtia + Donchian + TERS, TP%{TP_PRICE_PCT}/STOP%{STOP_PRICE_PCT}, {DAYS_OF_HISTORY} gün):\n"]

    all_consistent = True
    for split in SPLIT_POINTS:
        train_trades, test_trades = [], []
        for symbol, df in dfs.items():
            split_idx = int(len(df) * split)
            train_df = df.iloc[:split_idx].reset_index(drop=True)
            test_df = df.iloc[split_idx:].reset_index(drop=True)
            train_trades += simulate_dollar_pnl(train_df, find_entries_donchian_reversed(train_df))
            test_trades += simulate_dollar_pnl(test_df, find_entries_donchian_reversed(test_df))

        train_s = summarize(train_trades)
        test_s = summarize(test_trades)
        print(f"\nBölme %{int(split*100)}/{int((1-split)*100)}:")
        print(fmt(train_s, "TRAIN"))
        print(fmt(test_s, "TEST"))

        lines.append(f"Bölme %{int(split*100)}/{int((1-split)*100)}:")
        lines.append(fmt(train_s, "TRAIN"))
        lines.append(fmt(test_s, "TEST"))
        lines.append("")

        if test_s.get("n", 0) == 0 or test_s.get("total_dollar", -1) <= 0:
            all_consistent = False

    if skipped:
        lines.append(f"Atlanan semboller: {', '.join(skipped)}")

    if all_consistent:
        lines.append(
            "✅ SONUÇ: Her 3 bölme noktasında da TEST pozitif çıktı - bu tutarlılık "
            "gerçek bir kenar olduğuna işaret ediyor, rastlantı ihtimali düşük."
        )
    else:
        lines.append(
            "❌ SONUÇ: Bölme noktasına göre TEST sonucu değişken/negatife dönüyor - "
            "önceki bulgu büyük ihtimalle rastlantısaldı (18 kombinasyondan biri "
            "şansla pozitif çıkmıştı), gerçek bir kenar değil."
        )
    send_telegram_message("\n".join(lines))


if __name__ == "__main__":
    main()
