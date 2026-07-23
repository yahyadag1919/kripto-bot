"""
worst_strategies_reversed_walk_forward.py

AMAC: Su ana kadarki testlerde en kotu (en zararli) cikan strateji/kategori
kombinasyonlarini TEK dosyada, hem ORIJINAL hem TERSINE CEVRILMIS yonleriyle
test ediyoruz. Mantik: eger bir strateji GERCEKTEN sistematik olarak yanlis
yone bahis oynuyorsa (rastgele degil), tersini almak onu kara cevirebilir.
Daha once VWAP+Hacim komboyu tersine cevirmistik ama isabet orani zaten
%45-50 (yari yariya rastgele) oldugu icin tersi de ise yaramamisti. Bu
sefer en dusuk isabet oranli / en tutarli zararli olanlari deniyoruz -
bunlarin sistematik olma ihtimali daha yuksek.

TEST EDILEN STRATEJILER (hepsi ayni 3 kategoride: KRIPTO / EMTIA / HISSE):
  A) Ardisik mum (momentum) - en cok toplam dolar kaybi veren
  B) Guclu mum - kripto'da buyuk kayip verdi
  C) Donchian breakout + trend filtresi - EN DUSUK ISABET ORANI (%28 TRAIN)
     gorulen strateji, "sistematik yanlis yon" adayi olarak eklendi

HER STRATEJI + HER KATEGORI icin HEM ORIJINAL HEM TERSINE CEVRILMIS yon
test ediliyor - 3 strateji x 3 kategori x 2 yon x 2 donem (train/test) =
36 sonuc blogu. Sonuc DOLAR cinsinden (100$ referans, sabit marjin %2 x
20x kaldirac, TP%5/STOP%5 - kullanicinin belirledigi son ayarlarla ayni).

ONEMLI KISIT: Bu script'i Claude'un sandbox'i CALISTIRAMIYOR. Railway'de
digerleri gibi calistir:
    pip install ccxt pandas numpy requests
    python worst_strategies_reversed_walk_forward.py
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
    """4096 karakter Telegram siniri var - uzun mesajlari parcalayip gonderir."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > 3500:
            chunks.append(current)
            current = ""
        current += line + "\n"
    if current:
        chunks.append(current)
    for chunk in chunks:
        try:
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=10)
            time.sleep(1)
        except Exception as e:
            print(f"Telegram gonderim hatasi: {e}")


# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------

CATEGORIES = {
    "KRIPTO": ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "SUI"],
    "EMTIA": ["XAU", "XAG", "XPT", "XPD", "CL", "BZ", "NATGAS"],
    "HISSE": ["TSLA", "AMZN", "INTC", "HOOD", "MSTR", "SOFI", "PANW"],
}

TIMEFRAME = "15m"
DAYS_OF_HISTORY = 45
TRAIN_FRACTION = 0.6

CONSECUTIVE_CANDLES = 3
STRONG_BODY_MULT = 1.8
BODY_AVG_WINDOW = 20
DONCHIAN_PERIOD = 20     # 15m*20 = 5 saat - kisa vadeli kirilim
TREND_EMA_PERIOD = 200   # 15m*200 = ~50 saat, ~2 gun trend filtresi

TP_PRICE_PCT = 5.0
STOP_PRICE_PCT = 5.0
ROUNDTRIP_COMMISSION_PCT = 0.1
MAX_HOLD_CANDLES = 480   # 5 gun

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
    df["body"] = (df["close"] - df["open"]).abs()
    df["is_green"] = df["close"] > df["open"]
    df["avg_body20"] = df["body"].rolling(BODY_AVG_WINDOW).mean()
    df["donchian_high"] = df["high"].shift(1).rolling(DONCHIAN_PERIOD).max()
    df["donchian_low"] = df["low"].shift(1).rolling(DONCHIAN_PERIOD).min()
    df["ema_trend"] = df["close"].ewm(span=TREND_EMA_PERIOD, adjust=False).mean()
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


def find_entries_donchian(df: pd.DataFrame) -> list:
    entries = []
    n = len(df)
    i = max(DONCHIAN_PERIOD, TREND_EMA_PERIOD) + 5
    while i < n - 1:
        row = df.iloc[i]
        if pd.isna(row["donchian_high"]) or pd.isna(row["ema_trend"]):
            i += 1
            continue
        if row["close"] > row["donchian_high"] and row["close"] > row["ema_trend"]:
            entries.append((i, "LONG"))
        elif row["close"] < row["donchian_low"] and row["close"] < row["ema_trend"]:
            entries.append((i, "SHORT"))
        i += 1
    return entries


def flip(direction: str) -> str:
    return "SHORT" if direction == "LONG" else "LONG"


def simulate_dollar_pnl(df: pd.DataFrame, entries: list, reverse: bool) -> list:
    trades = []
    n = len(df)
    fixed_margin = STARTING_BALANCE * (POSITION_PCT_OF_BALANCE / 100)
    fixed_notional = fixed_margin * LEVERAGE

    for i, raw_direction in entries:
        direction = flip(raw_direction) if reverse else raw_direction
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


STRATEGIES = {
    "A) Ardışık mum": find_entries_momentum,
    "B) Güçlü mum": find_entries_strong_candle,
    "C) Donchian kırılım": find_entries_donchian,
}


def main():
    all_symbols = sum(CATEGORIES.values(), [])
    print(f"En-kotu-stratejiler + tersi testi basliyor - {len(all_symbols)} sembol, {len(STRATEGIES)} strateji x 2 yon\n")
    send_telegram_message(
        f"🔻🔺 En kötü stratejiler + tersleri testi başladı ({len(all_symbols)} sembol, "
        f"{len(STRATEGIES)} strateji × orijinal/ters yön). TP%{TP_PRICE_PCT}/STOP%{STOP_PRICE_PCT}, "
        f"sabit marjin %{POSITION_PCT_OF_BALANCE}x{LEVERAGE}x, {STARTING_BALANCE:.0f}$ ref. Sonuç uzun sürebilir..."
    )

    data = {cat: {} for cat in CATEGORIES}
    skipped = []
    for category, coins in CATEGORIES.items():
        for coin in coins:
            symbol = f"{coin}/USDT:USDT"
            print(f"--- [{category}] {symbol} ---")
            try:
                df = fetch_full_history(symbol, TIMEFRAME, DAYS_OF_HISTORY)
            except Exception as e:
                print(f"  veri cekilemedi: {e}")
                skipped.append(symbol)
                continue
            if len(df) < TREND_EMA_PERIOD + 20:
                skipped.append(symbol)
                continue
            df = compute_indicators(df)
            split_idx = int(len(df) * TRAIN_FRACTION)
            data[category][symbol] = (df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True))

    lines = [f"🔻🔺 SONUÇ (TP%{TP_PRICE_PCT}/STOP%{STOP_PRICE_PCT}, marjin%{POSITION_PCT_OF_BALANCE}x{LEVERAGE}x):\n"]

    for strat_label, entry_func in STRATEGIES.items():
        print(f"\n========== {strat_label} ==========")
        lines.append(f"════ {strat_label} ════")
        for category in CATEGORIES:
            print(f"-- {category} --")
            lines.append(f"── {category} ──")
            for reverse, mode_label in [(False, "Orijinal"), (True, "TERS")]:
                train_trades, test_trades = [], []
                for symbol, (train_df, test_df) in data[category].items():
                    train_entries = entry_func(train_df)
                    test_entries = entry_func(test_df)
                    train_trades += simulate_dollar_pnl(train_df, train_entries, reverse)
                    test_trades += simulate_dollar_pnl(test_df, test_entries, reverse)
                train_s = summarize(train_trades)
                test_s = summarize(test_trades)
                print(f" {mode_label}: {fmt(train_s, 'TRAIN')} | {fmt(test_s, 'TEST')}")
                lines.append(f"{mode_label}:")
                lines.append(fmt(train_s, "TRAIN"))
                lines.append(fmt(test_s, "TEST"))
        lines.append("")

    if skipped:
        lines.append(f"Atlanan semboller: {', '.join(skipped)}")
    lines.append(
        "Yorum: TERS modunda TRAIN'de de TEST'te de pozitife dönenler varsa, "
        "o strateji gerçekten sistematik yanlış yöne bahis oynuyormuş demektir - "
        "gerçek/tekrarlanabilir bir kenar bulmuş olabiliriz."
    )
    send_telegram_message("\n".join(lines))


if __name__ == "__main__":
    main()
