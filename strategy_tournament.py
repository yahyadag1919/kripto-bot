import os
import csv
import time
from datetime import datetime, timedelta

import ccxt
import numpy as np
import pandas as pd
import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------

# Turnuva icin daha kucuk ama likit bir liste kullaniyoruz (172 coin cok uzun surer).
# Istersen bu listeyi crypto_breakout_bot.py'deki COINS ile degistirebilirsin.
TOURNAMENT_COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "SUI",
    "DOT", "TRX", "ATOM", "NEAR", "TON", "LTC", "BCH", "ETC", "FIL", "APT",
    "ARB", "OP", "INJ", "SEI", "ICP", "HBAR", "PEPE", "WIF", "ENA", "TIA",
]
TOURNAMENT_WATCHLIST = [f"{c}/USDT:USDT" for c in TOURNAMENT_COINS]

TIMEFRAME = "15m"
LOOKBACK_DAYS = 30

RSI6_PERIOD = 6
RSI14_PERIOD = 14
ATR_PERIOD = 14
BREAKOUT_LOOKBACK = 20
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Senin kurallarin: hedef/azami tutus ve basari esigi
CHECK_CANDLES = [1, 2, 3]   # 15/30/45 dk sonrasi (mum sayisi)
SUCCESS_THRESHOLD_PCT = 0.15
MIN_SIGNALS_TO_RANK = 15    # bu sayidan az sinyal ureten strateji siralamaya alinmaz (gurultu)

exchange = ccxt.okx({
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
})


def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("UYARI: Telegram bilgileri yok, sadece konsola yazdiriliyor.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=15)
    except Exception as e:
        print(f"Telegram gonderim hatasi: {e}")


# ---------------------------------------------------------------------------
# Gecmis veri cekme (sayfalama ile - OKX tek seferde sinirli mum veriyor)
# ---------------------------------------------------------------------------

def fetch_historical_ohlcv(symbol: str, days: int = LOOKBACK_DAYS, timeframe: str = TIMEFRAME) -> pd.DataFrame:
    ms_per_candle = 15 * 60 * 1000
    total_candles_needed = int(days * 24 * 60 / 15)
    since = exchange.milliseconds() - total_candles_needed * ms_per_candle

    all_candles = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=300)
        if not batch:
            break
        all_candles.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= since:
            break
        since = last_ts + ms_per_candle
        if len(batch) < 300:
            break
        time.sleep(0.15)  # rate limit'e takilmamak icin

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ---------------------------------------------------------------------------
# Ortak indikatorler (tum stratejiler ayni hazirlanmis dataframe'i kullanir)
# ---------------------------------------------------------------------------

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["vol_sma20"] = df["volume"].rolling(20).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(ATR_PERIOD).mean()

    df["body"] = (df["close"] - df["open"]).abs()
    df["is_bull"] = df["close"] > df["open"]

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_ratio"] = df["lower_wick_ratio"].fillna(0)
    df["upper_wick_ratio"] = df["upper_wick_ratio"].fillna(0)
    df["close_position"] = (df["close"] - df["low"]) / candle_range
    df["close_position"] = df["close_position"].fillna(0.5)

    def rsi(series, period):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    df["rsi6"] = rsi(df["close"], RSI6_PERIOD)
    df["rsi14"] = rsi(df["close"], RSI14_PERIOD)

    boll_mid = df["close"].rolling(BOLLINGER_PERIOD).mean()
    boll_std = df["close"].rolling(BOLLINGER_PERIOD).std()
    df["boll_upper"] = boll_mid + BOLLINGER_STD * boll_std
    df["boll_lower"] = boll_mid - BOLLINGER_STD * boll_std

    df["breakout_high"] = df["close"].shift(1).rolling(BREAKOUT_LOOKBACK).max()
    df["breakout_low"] = df["close"].shift(1).rolling(BREAKOUT_LOOKBACK).min()

    return df


# ---------------------------------------------------------------------------
# 8 farkli strateji - her biri (df, i) alir, None/"LONG"/"SHORT" doner
# i: degerlendirilen mumun index'i (bu mum KAPANMIS kabul edilir)
# ---------------------------------------------------------------------------

def strategy_1_mean_reversion(df, i):
    """Eski sistem: RSI asiri uc + fitil + hacim patlamasi -> tersine bahis."""
    row = df.iloc[i]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 2.5:
        return None
    if row["lower_wick_ratio"] >= 0.5 and row["rsi6"] <= 20:
        return "LONG"
    if row["upper_wick_ratio"] >= 0.5 and row["rsi6"] >= 80:
        return "SHORT"
    return None


def strategy_2_momentum_breakout(df, i):
    """Su anki canli sistem: kirilim + hacim + guclu govde + guclu kapanis + RSI momentum bolgesi."""
    row = df.iloc[i]
    if pd.isna(row["breakout_high"]) or pd.isna(row["atr14"]) or row["atr14"] == 0:
        return None
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 2.0:
        return None
    body_ratio = row["body"] / row["atr14"]
    if body_ratio < 0.6:
        return None
    if (row["close"] > row["breakout_high"] and row["close_position"] >= 0.65
            and 50 <= row["rsi14"] <= 78):
        return "LONG"
    if (row["close"] < row["breakout_low"] and row["close_position"] <= 0.35
            and 22 <= row["rsi14"] <= 50):
        return "SHORT"
    return None


def strategy_3_ema_cross(df, i):
    """Klasik trend takibi: EMA20, EMA50'yi yeni kesti + hacim destegi."""
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ok = row["volume"] >= row["vol_sma20"]
    if prev["ema20"] <= prev["ema50"] and row["ema20"] > row["ema50"] and vol_ok:
        return "LONG"
    if prev["ema20"] >= prev["ema50"] and row["ema20"] < row["ema50"] and vol_ok:
        return "SHORT"
    return None


def strategy_4_bollinger_breakout(df, i):
    """Bollinger disina hacimle tasma - momentum devam bahsi (tersine bahis degil)."""
    row = df.iloc[i]
    if pd.isna(row["boll_upper"]) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 1.8:
        return None
    if row["close"] >= row["boll_upper"] and row["is_bull"]:
        return "LONG"
    if row["close"] <= row["boll_lower"] and not row["is_bull"]:
        return "SHORT"
    return None


def strategy_5_volume_spike(df, i):
    """Sadece cok buyuk hacim patlamasi + yon - filtre az, ham momentum."""
    row = df.iloc[i]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 3.5:
        return None
    if row["is_bull"]:
        return "LONG"
    return "SHORT"


def strategy_6_rsi_midline(df, i):
    """RSI(14) 50 cizgisini yeni kesti - momentum yon degistiriyor."""
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if prev["rsi14"] < 50 and row["rsi14"] >= 53:
        return "LONG"
    if prev["rsi14"] > 50 and row["rsi14"] <= 47:
        return "SHORT"
    return None


def strategy_7_donchian_simple(df, i):
    """Saf kanal kirilimi - hicbir ek filtre yok (kiyas/baseline)."""
    row = df.iloc[i]
    if pd.isna(row["breakout_high"]) or pd.isna(row["breakout_low"]):
        return None
    if row["close"] > row["breakout_high"]:
        return "LONG"
    if row["close"] < row["breakout_low"]:
        return "SHORT"
    return None


def strategy_8_confluence(df, i):
    """Hem tukenme hem kirilim ayni anda ayni yonu gostermeli - nadir, yuksek guvenli."""
    s1 = strategy_1_mean_reversion(df, i)
    s2 = strategy_2_momentum_breakout(df, i)
    if s1 is not None and s2 is not None and s1 == s2:
        return s1
    return None


STRATEGIES = {
    "1-Tersine Bahis (eski)": strategy_1_mean_reversion,
    "2-Momentum Kirilim (canli)": strategy_2_momentum_breakout,
    "3-EMA Kesisimi": strategy_3_ema_cross,
    "4-Bollinger Kirilim": strategy_4_bollinger_breakout,
    "5-Hacim Patlamasi": strategy_5_volume_spike,
    "6-RSI Orta Cizgi": strategy_6_rsi_midline,
    "7-Saf Kanal Kirilimi": strategy_7_donchian_simple,
    "8-Kombinasyon": strategy_8_confluence,
}


# ---------------------------------------------------------------------------
# Degerlendirme: her sinyal 15/30/45 dk sonra basarili mi
# ---------------------------------------------------------------------------

def evaluate_signals(df, strategy_fn):
    total, correct = 0, 0
    max_check = max(CHECK_CANDLES)

    for i in range(BREAKOUT_LOOKBACK + 5, len(df) - max_check):
        direction = strategy_fn(df, i)
        if direction is None:
            continue

        entry_price = df.iloc[i]["close"]
        success = False
        for c in CHECK_CANDLES:
            future_price = df.iloc[i + c]["close"]
            pct_change = (future_price - entry_price) / entry_price * 100
            if direction == "LONG" and pct_change >= SUCCESS_THRESHOLD_PCT:
                success = True
                break
            if direction == "SHORT" and pct_change <= -SUCCESS_THRESHOLD_PCT:
                success = True
                break

        total += 1
        if success:
            correct += 1

    return total, correct


# ---------------------------------------------------------------------------
# Ana turnuva
# ---------------------------------------------------------------------------

def run_tournament():
    print(f"Turnuva basliyor: {len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} gunluk veri, {len(STRATEGIES)} strateji")
    send_telegram_message(
        f"🏆 Strateji turnuvası başladı.\n"
        f"{len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} günlük geçmiş veri, {len(STRATEGIES)} strateji test ediliyor.\n"
        f"Bu biraz zaman alabilir, bitince sonuçları göndereceğim."
    )

    results = {name: {"total": 0, "correct": 0} for name in STRATEGIES}
    unsupported = set()

    for idx, symbol in enumerate(TOURNAMENT_WATCHLIST):
        print(f"[{idx+1}/{len(TOURNAMENT_WATCHLIST)}] {symbol} verisi cekiliyor...")
        try:
            df = fetch_historical_ohlcv(symbol)
            if df.empty or len(df) < BREAKOUT_LOOKBACK + 50:
                print(f"  {symbol}: yetersiz veri, atlaniyor")
                continue
            df = compute_all_indicators(df)
        except Exception as e:
            if "does not have" in str(e).lower():
                unsupported.add(symbol)
            print(f"  {symbol} hata: {e}")
            continue

        for name, fn in STRATEGIES.items():
            total, correct = evaluate_signals(df, fn)
            results[name]["total"] += total
            results[name]["correct"] += correct

    # Sonuclari sirala
    leaderboard = []
    for name, r in results.items():
        total, correct = r["total"], r["correct"]
        win_rate = (correct / total * 100) if total > 0 else 0
        leaderboard.append((name, total, correct, win_rate))

    ranked = [x for x in leaderboard if x[1] >= MIN_SIGNALS_TO_RANK]
    unranked = [x for x in leaderboard if x[1] < MIN_SIGNALS_TO_RANK]
    ranked.sort(key=lambda x: x[3], reverse=True)

    lines = [f"🏆 TURNUVA SONUÇLARI ({LOOKBACK_DAYS} günlük veri, {len(TOURNAMENT_WATCHLIST)} coin)\n"]
    lines.append("Sıralama (en az {} sinyal üreten stratejiler):\n".format(MIN_SIGNALS_TO_RANK))
    for rank, (name, total, correct, win_rate) in enumerate(ranked, 1):
        madalya = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        lines.append(f"{madalya} {name}\n   Sinyal: {total} | Doğru: {correct} | İsabet: %{win_rate:.1f}")

    if unranked:
        lines.append("\nYetersiz örneklem (sıralamaya alınmadı):")
        for name, total, correct, win_rate in unranked:
            lines.append(f"- {name}: {total} sinyal (%{win_rate:.1f})")

    msg = "\n".join(lines)
    print("\n" + msg)
    send_telegram_message(msg)

    with open("tournament_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strateji", "toplam_sinyal", "dogru", "isabet_yuzde"])
        for name, total, correct, win_rate in leaderboard:
            writer.writerow([name, total, correct, f"{win_rate:.2f}"])

    print("\nTurnuva tamamlandi. Sonuclar tournament_results.csv dosyasina kaydedildi.")


if __name__ == "__main__":
    run_tournament()
