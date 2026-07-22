"""
dollar_based_walk_forward.py

ONCEKI TESTLERIN SORUNU: Hepsi "fiyat yuzde kac hareket etti" uzerinden
sonuc verdi (orn. "ort. net %-0.11"). Bu, gercek bir hesapta ne kadar
dolar kazanip kaybedecegini DOGRUDAN gostermiyor - cunku gercek kazanc/
kayip pozisyon BUYUKLUGUNE (bakiyenin yuzde kaci risk edildigine,
kaldiraca, marjin tavanina) bagli, sadece fiyat yuzdesine degil.

Bu script, HER ISLEMI SABIT pozisyon buyuklugu ile simule ediyor - kullanicinin
istedigi gibi: bakiyenin POSITION_PCT_OF_BALANCE'i marjin olarak kullanilir,
LEVERAGE ile carpilir, boylece notional (pozisyon buyuklugu) HER ZAMAN AYNI
kalir - stop mesafesinden bagimsiz. Bu, canli bottaki "TAM OTOMATIK" modun
temel mantigiyla ayni (bakiyenin %2'si | 20x kaldirac).
  1) Sabit bir baslangic bakiyesi (STARTING_BALANCE = 100 USD, referans icin)
  2) Pozisyon notional'i = bakiye x POSITION_PCT_OF_BALANCE x LEVERAGE (SABIT)
  3) TP/STOP fiyat seviyeleri (TP_PRICE_PCT / STOP_PRICE_PCT) sana ait, asagidan
     degistirebilirsin - test SENIN belirledigin seviyelerle calisir
  4) Sonuc DOLAR olarak raporlanir (100$'lik bir hesapta, islemler birbirinden
     BAGIMSIZ/compounding YOK)

SONUC: her islem icin gercek DOLAR kazanc/kayip (100$'lik bir hesapta,
islemler birbirinden BAGIMSIZ/compounding YOK - yani her islem sanki
hep 100$'la basliyormus gibi hesaplaniyor, boylece hangi coin/kategori
hangi ayarla ne kadar dolar kazandirir/kaybettirir NET gorulur).

AYARLAR (canli bottaki degerlerle AYNI baslangic, degistirebilirsin):
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
# AYARLAR - bunlari degistirerek kendi TP/Stop/risk tercihini test edebilirsin
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
MAX_HOLD_CANDLES = 480       # 15m*480 = 5 gun - %5 gibi genis hedefler icin 24sa yetersiz kalabilirdi

# --- Fiyat seviyeleri (SADECE pozisyon boyutu ve TP/SL fiyat noktasini
#     belirlemek icin kullanilir - nihai sonuc DOLAR olarak raporlanir) ---
TP_PRICE_PCT = 5.0           # kullanici tercihi - eskiden 0.4 idi
STOP_PRICE_PCT = 5.0         # kullanici tercihi - eskiden 0.6 idi
ROUNDTRIP_COMMISSION_PCT = 0.1

# --- Bakiye / marjin / kaldirac - canli bottaki GERCEK degerlerle AYNI ---
STARTING_BALANCE = 100.0    # referans hesap - her islem bagimsiz, compounding YOK
POSITION_PCT_OF_BALANCE = 2.0  # marjin - bakiyenin bu yuzdesi her islemde SABIT marjin olarak kullanilir
LEVERAGE = 20                # canli bottaki kaldirac - notional = marjin x LEVERAGE, SABIT

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


def simulate_dollar_pnl(df: pd.DataFrame, entries: list) -> list:
    """Her giris icin: SABIT pozisyon buyuklugu - bakiyenin POSITION_PCT_OF_BALANCE'i
    marjin olarak kullanilir, LEVERAGE ile carpilir. Bu, stop mesafesinden BAGIMSIZ,
    her zaman AYNI notional buyuklugu verir (kullanicinin istedigi "20x kaldirac
    seklinde" sabit pozisyon mantigi). Her islem STARTING_BALANCE'tan bagimsiz
    baslar (compounding yok)."""
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
        notional = fixed_notional
        commission_dollar = notional * (ROUNDTRIP_COMMISSION_PCT / 100)

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
        gross_pnl_dollar = position_units * price_move
        net_pnl_dollar = gross_pnl_dollar - commission_dollar

        trades.append({
            "pnl_dollar": net_pnl_dollar,
            "pnl_pct_of_balance": net_pnl_dollar / STARTING_BALANCE * 100,
            "outcome": outcome,
        })

    return trades


def summarize(trades: list) -> dict:
    if not trades:
        return {"n": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t["pnl_dollar"] > 0)
    total_dollar = sum(t["pnl_dollar"] for t in trades)
    total_pct_of_balance = sum(t["pnl_pct_of_balance"] for t in trades)
    return {
        "n": n, "hit_rate": wins / n * 100,
        "avg_dollar": total_dollar / n, "total_dollar": total_dollar,
        "total_pct_of_balance": total_pct_of_balance,
    }


def fmt(s: dict, label: str) -> str:
    if s["n"] == 0:
        return f"  {label}: sinyal yok"
    return (
        f"  {label}: {s['n']} işlem | isabet %{s['hit_rate']:.1f} | "
        f"ort. ${s['avg_dollar']:+.3f}/işlem | toplam ${s['total_dollar']:+.2f} "
        f"(100$'lık hesabın %{s['total_pct_of_balance']:+.1f}'i, işlemler bağımsız/compounding yok)"
    )


def main():
    print(
        f"Dolar-bazlı walk-forward test başlıyor - "
        f"TP%{TP_PRICE_PCT}/STOP%{STOP_PRICE_PCT}, bakiyenin %{POSITION_PCT_OF_BALANCE}'i marjin x {LEVERAGE}x kaldıraç (sabit), "
        f"{STARTING_BALANCE}$ referans bakiye\n"
    )
    all_symbols = sum(CATEGORIES.values(), [])
    send_telegram_message(
        f"💵 Dolar-bazlı walk-forward test başladı ({len(all_symbols)} sembol). "
        f"TP%{TP_PRICE_PCT}/STOP%{STOP_PRICE_PCT}, sabit marjin %{POSITION_PCT_OF_BALANCE} x {LEVERAGE}x kaldıraç, "
        f"{STARTING_BALANCE:.0f}$ referans bakiye. Sonuç gelince buraya düşecek..."
    )

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
                skipped.append(symbol)
                continue

            df = compute_indicators(df)
            split_idx = int(len(df) * TRAIN_FRACTION)
            train_df = df.iloc[:split_idx].reset_index(drop=True)
            test_df = df.iloc[split_idx:].reset_index(drop=True)

            train_entries_m = find_entries_momentum(train_df)
            test_entries_m = find_entries_momentum(test_df)
            train_entries_s = find_entries_strong_candle(train_df)
            test_entries_s = find_entries_strong_candle(test_df)

            results[category]["momentum"]["train"] += simulate_dollar_pnl(train_df, train_entries_m)
            results[category]["momentum"]["test"] += simulate_dollar_pnl(test_df, test_entries_m)
            results[category]["strong"]["train"] += simulate_dollar_pnl(train_df, train_entries_s)
            results[category]["strong"]["test"] += simulate_dollar_pnl(test_df, test_entries_s)

    print("\n================= SONUC (dolar bazli) =================")
    lines = [
        f"💵 Dolar-bazlı SONUÇ (TP%{TP_PRICE_PCT}/STOP%{STOP_PRICE_PCT}, "
        f"sabit marjin %{POSITION_PCT_OF_BALANCE} x {LEVERAGE}x, {STARTING_BALANCE:.0f}$ ref. bakiye):\n"
    ]

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
        lines.append(f"Atlanan semboller: {', '.join(skipped)}")

    lines.append(
        "Not: Her işlem 100$'lık bağımsız bir hesap gibi hesaplandı (compounding "
        "yok), gerçek pozisyon boyutlandırma (sabit $ risk / stop mesafesi, "
        "marjin tavanı, kaldıraç) canlı bottakiyle birebir aynı mantıkla uygulandı."
    )
    send_telegram_message("\n".join(lines))


if __name__ == "__main__":
    main()
