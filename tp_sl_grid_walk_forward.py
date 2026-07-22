"""
tp_sl_grid_walk_forward.py

simple_momentum_walk_forward.py sonucu umut verici cikti: isabet orani
TRAIN'de %57.2/%56.6, TEST'te %58.8/%55.4 - birbirine COK yakin, yani bu
"guc var mi" mantigi gercek/tekrarlanabilir bir sey yakaliyor olabilir.

Ama TP %0.4 / STOP %0.6 kombinasyonuyla, bu isabet oraniyla bile hala hafif
zararli cikti - cunku kazandiginda az kazaniyor, kaybettiginde daha cok
kaybediyor. Bu script AYNI iki giris mantigini (A: ardisik mum, B: guclu
mum) DEGISTIRMEDEN, farkli TP/STOP kombinasyonlariyla tarayip hangisi bu
isabet oranini GERCEK pozitif beklenti getirisine cevirebiliyor onu buluyor.

Performans icin: her coin/donem icin giris sinyalleri (hangi mumda, hangi
yonde) SADECE BIR KERE tespit ediliyor. Sonra TP/STOP izgarasindaki her
kombinasyon icin, ayni giris noktalari uzerinden farkli hedef/stop
seviyeleriyle sonuc hesaplaniyor - boylece giris tespiti tekrar tekrar
yapilmiyor, sadece cikis hesaplamasi tekrarlaniyor.

CALISTIRMA (aynen oncekiler gibi, Railway Start Command'i degistirerek):
    pip install ccxt pandas numpy requests
    python tp_sl_grid_walk_forward.py
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


COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "SUI",
    "DOT", "TRX", "ATOM", "NEAR", "TON", "LTC", "BCH", "ETC", "FIL", "APT",
    "ARB", "OP", "INJ", "SEI", "ICP",
]
WATCHLIST = [f"{c}/USDT:USDT" for c in COINS]

TIMEFRAME = "15m"
DAYS_OF_HISTORY = 45
TRAIN_FRACTION = 0.6

CONSECUTIVE_CANDLES = 3
STRONG_BODY_MULT = 1.8
BODY_AVG_WINDOW = 20
ROUNDTRIP_COMMISSION_PCT = 0.1
MAX_HOLD_CANDLES = 96

# Taranacak TP/STOP kombinasyonlari (brut yuzde, komisyon ayrica dusuluyor)
TP_GRID = [0.4, 0.6, 0.8, 1.0]
STOP_GRID = [0.4, 0.6, 0.8]

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


def find_entries_momentum(df: pd.DataFrame) -> list:
    entries = []
    n = len(df)
    i = CONSECUTIVE_CANDLES + 1
    while i < n - 1:
        window = df.iloc[i - CONSECUTIVE_CANDLES:i]
        if window["is_green"].all():
            entries.append((i, "LONG"))
        elif (~window["is_green"]).all():
            entries.append((i, "SHORT"))
        i += 1
    return entries


def find_entries_strong_candle(df: pd.DataFrame) -> list:
    entries = []
    n = len(df)
    i = BODY_AVG_WINDOW + 1
    while i < n - 1:
        row = df.iloc[i]
        if pd.isna(row["avg_body20"]) or row["avg_body20"] == 0:
            i += 1
            continue
        if row["body"] >= row["avg_body20"] * STRONG_BODY_MULT:
            direction = "LONG" if row["is_green"] else "SHORT"
            entries.append((i, direction))
        i += 1
    return entries


def evaluate_entries(df: pd.DataFrame, entries: list, tp_pct: float, stop_pct: float) -> list:
    trades = []
    n = len(df)
    for i, direction in entries:
        entry_price = df.iloc[i]["close"]
        if direction == "LONG":
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - stop_pct / 100)
        else:
            tp_price = entry_price * (1 - tp_pct / 100)
            sl_price = entry_price * (1 + stop_pct / 100)

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

            if hit_sl:  # ayni mumda ikisi de olursa kotumser varsayim: stop once vurdu
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
        trades.append(pct_change)

    return trades


def summarize(pct_list: list) -> dict:
    if not pct_list:
        return {"n": 0}
    n = len(pct_list)
    wins = sum(1 for p in pct_list if p > 0)
    total_net = sum(pct_list)
    return {"n": n, "hit_rate": wins / n * 100, "avg_net": total_net / n, "total_net": total_net}


def fmt(s: dict, label: str) -> str:
    if s["n"] == 0:
        return f"  {label}: sinyal yok"
    return f"  {label}: {s['n']} işlem | isabet %{s['hit_rate']:.1f} | ort. net %{s['avg_net']:+.3f} | toplam %{s['total_net']:+.1f}"


def main():
    print(f"TP/STOP izgara testi basliyor - {len(COINS)} coin\n")
    send_telegram_message(
        f"🎯 TP/STOP izgara testi başladı ({len(TP_GRID)}x{len(STOP_GRID)} kombinasyon, "
        f"{len(COINS)} coin). Sonuç gelince buraya düşecek..."
    )

    # her coin/donem icin: (train_df, test_df, train_entries_A, test_entries_A, train_entries_B, test_entries_B)
    per_coin_data = []

    for symbol in WATCHLIST:
        print(f"--- {symbol} ---")
        try:
            df = fetch_full_history(symbol, TIMEFRAME, DAYS_OF_HISTORY)
        except Exception as e:
            print(f"  veri cekilemedi: {e}")
            continue
        if len(df) < BODY_AVG_WINDOW * 3:
            continue

        df = compute_indicators(df)
        split_idx = int(len(df) * TRAIN_FRACTION)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)

        per_coin_data.append({
            "train_df": train_df, "test_df": test_df,
            "train_A": find_entries_momentum(train_df), "test_A": find_entries_momentum(test_df),
            "train_B": find_entries_strong_candle(train_df), "test_B": find_entries_strong_candle(test_df),
        })

    print("\n================= IZGARA SONUCLARI =================")
    lines = ["🎯 TP/STOP izgara test SONUÇ:\n"]

    for strategy_key, strategy_label in [("A", "Ardışık 3 mum"), ("B", "Tek güçlü mum")]:
        lines.append(f"{strategy_label}:")
        print(f"\n{strategy_label}:")
        best_test_avg = None
        best_combo = None
        for tp in TP_GRID:
            for stop in STOP_GRID:
                train_pcts, test_pcts = [], []
                for coin_data in per_coin_data:
                    train_pcts += evaluate_entries(coin_data["train_df"], coin_data[f"train_{strategy_key}"], tp, stop)
                    test_pcts += evaluate_entries(coin_data["test_df"], coin_data[f"test_{strategy_key}"], tp, stop)
                train_s = summarize(train_pcts)
                test_s = summarize(test_pcts)
                line = f"  TP%{tp}/STOP%{stop} -> TRAIN net%{train_s.get('avg_net', 0):+.3f} ({train_s.get('n',0)} işlem) | TEST net%{test_s.get('avg_net', 0):+.3f} ({test_s.get('n',0)} işlem)"
                print(line)
                lines.append(line)
                if test_s.get("n", 0) >= 30 and (best_test_avg is None or test_s["avg_net"] > best_test_avg):
                    best_test_avg = test_s["avg_net"]
                    best_combo = (tp, stop, train_s, test_s)
        if best_combo:
            tp, stop, train_s, test_s = best_combo
            lines.append(f"  → EN İYİ (TEST bazlı): TP%{tp}/STOP%{stop}")
            lines.append(f"    {fmt(train_s, 'TRAIN')}")
            lines.append(f"    {fmt(test_s, 'TEST')}")
        lines.append("")

    lines.append(
        "Yorum: TRAIN'de de TEST'te de aynı kombinasyon pozitif ve yakınsa, "
        "gerçek/tekrarlanabilir bir kenar bulmuş olabiliriz."
    )
    send_telegram_message("\n".join(lines))


if __name__ == "__main__":
    main()
