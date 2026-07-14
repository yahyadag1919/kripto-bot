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

TOURNAMENT_COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "SUI",
    "DOT", "TRX", "ATOM", "NEAR", "TON", "LTC", "BCH", "ETC", "FIL", "APT",
    "ARB", "OP", "INJ", "SEI", "ICP", "HBAR", "PEPE", "WIF", "ENA", "TIA",
]
TOURNAMENT_WATCHLIST = [f"{c}/USDT:USDT" for c in TOURNAMENT_COINS]
BTC_SYMBOL = "BTC/USDT:USDT"

TIMEFRAME = "15m"
LOOKBACK_DAYS = 30

RSI14_PERIOD = 14
ATR_PERIOD = 14
BREAKOUT_LOOKBACK = 20
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
SQUEEZE_LOOKBACK = 50       # bant genisliginin karsilastirildigi gecmis pencere
ZSCORE_PERIOD = 20

CHECK_CANDLES = [1, 2, 3]   # 15/30/45 dk sonrasi (mum sayisi)
SUCCESS_THRESHOLD_PCT = 0.15
MIN_SIGNALS_TO_RANK = 15

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
# Gecmis veri cekme (sayfalama ile)
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
        time.sleep(0.15)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ---------------------------------------------------------------------------
# Ortak indikatorler
# ---------------------------------------------------------------------------

def compute_all_indicators(df: pd.DataFrame, btc_df: pd.DataFrame = None) -> pd.DataFrame:
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
    df["close_position"] = (df["close"] - df["low"]) / candle_range
    df["close_position"] = df["close_position"].fillna(0.5)

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI14_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI14_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = (100 - (100 / (1 + rs))).fillna(50)

    boll_mid = df["close"].rolling(BOLLINGER_PERIOD).mean()
    boll_std = df["close"].rolling(BOLLINGER_PERIOD).std()
    df["boll_upper"] = boll_mid + BOLLINGER_STD * boll_std
    df["boll_lower"] = boll_mid - BOLLINGER_STD * boll_std
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / boll_mid * 100
    df["boll_width_percentile"] = df["boll_width"].rolling(SQUEEZE_LOOKBACK).rank(pct=True)

    df["breakout_high"] = df["close"].shift(1).rolling(BREAKOUT_LOOKBACK).max()
    df["breakout_low"] = df["close"].shift(1).rolling(BREAKOUT_LOOKBACK).min()

    # On-Balance-Volume
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_slope"] = df["obv"].diff(5)

    # Z-skor (fiyatin kendi ortalamasindan kac standart sapma uzakta oldugu)
    roll_mean = df["close"].rolling(ZSCORE_PERIOD).mean()
    roll_std = df["close"].rolling(ZSCORE_PERIOD).std()
    df["zscore"] = (df["close"] - roll_mean) / roll_std.replace(0, np.nan)

    # BTC'ye gore goreceli guc (sadece BTC verisi verildiyse)
    if btc_df is not None:
        btc_small = btc_df[["timestamp", "close"]].rename(columns={"close": "btc_close"})
        df = pd.merge_asof(df.sort_values("timestamp"), btc_small.sort_values("timestamp"),
                            on="timestamp", direction="backward")
        df["btc_change_20"] = df["btc_close"].pct_change(20) * 100
        df["coin_change_20"] = df["close"].pct_change(20) * 100
        df["relative_strength"] = df["coin_change_20"] - df["btc_change_20"]
    else:
        df["relative_strength"] = np.nan

    return df


# ---------------------------------------------------------------------------
# 8 YENI strateji - farkli mekanizmalar, oncekilerle ayni mantik degil
# ---------------------------------------------------------------------------

def strategy_a_breakout_retest(df, i):
    """Kirilim olur, fiyat geri donup seviyeyi test eder, tutarsa devam - sahte kirilimlari eler."""
    if i < 6:
        return None
    row = df.iloc[i]
    if pd.isna(row["breakout_high"]) or pd.isna(row["breakout_low"]):
        return None
    window = df.iloc[i - 5:i]

    broke_up = (window["close"] > window["breakout_high"]).any()
    if broke_up:
        level = window["breakout_high"].iloc[-1]
        retested = (window["low"] <= level * 1.003).any()
        if retested and row["close"] > level and row["is_bull"]:
            return "LONG"

    broke_down = (window["close"] < window["breakout_low"]).any()
    if broke_down:
        level = window["breakout_low"].iloc[-1]
        retested = (window["high"] >= level * 0.997).any()
        if retested and row["close"] < level and not row["is_bull"]:
            return "SHORT"

    return None


def strategy_b_volatility_squeeze(df, i):
    """Bollinger bandi uzun sure daralmis (sikisma), simdi hacimle patliyor."""
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(prev["boll_width_percentile"]) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    was_squeezed = prev["boll_width_percentile"] <= 0.2   # son 50 mumun en dar %20'si
    vol_ratio = row["volume"] / row["vol_sma20"]
    if not was_squeezed or vol_ratio < 2.0:
        return None
    if row["is_bull"] and row["close"] > row["boll_upper"]:
        return "LONG"
    if not row["is_bull"] and row["close"] < row["boll_lower"]:
        return "SHORT"
    return None


def strategy_c_triple_timeframe(df, i):
    """15m mumun kendisi + EMA20/50 (orta vade) + fiyatin EMA50'ye gore konumu (uzun vade) hizali mi."""
    row = df.iloc[i]
    if pd.isna(row["ema20"]) or pd.isna(row["ema50"]) or pd.isna(row["rsi14"]):
        return None
    short_term_bull = row["is_bull"] and row["close_position"] >= 0.6
    short_term_bear = (not row["is_bull"]) and row["close_position"] <= 0.4
    mid_term_bull = row["ema20"] > row["ema50"]
    mid_term_bear = row["ema20"] < row["ema50"]
    long_term_bull = row["close"] > row["ema50"] * 1.01
    long_term_bear = row["close"] < row["ema50"] * 0.99

    if short_term_bull and mid_term_bull and long_term_bull and row["rsi14"] < 75:
        return "LONG"
    if short_term_bear and mid_term_bear and long_term_bear and row["rsi14"] > 25:
        return "SHORT"
    return None


def strategy_d_relative_strength(df, i):
    """Coin, son 20 mumda BTC'den belirgin sekilde daha guclu/zayif - farkin devamina bahis."""
    row = df.iloc[i]
    if pd.isna(row.get("relative_strength")):
        return None
    if row["relative_strength"] >= 4.0 and row["is_bull"]:
        return "LONG"
    if row["relative_strength"] <= -4.0 and not row["is_bull"]:
        return "SHORT"
    return None


def strategy_e_pullback_in_trend(df, i):
    """Belirlenmis bir trend var, fiyat EMA20'ye geri cekiliyor, oradan sekiyor (dipte al, trend yonunde)."""
    row = df.iloc[i]
    if pd.isna(row["ema20"]) or pd.isna(row["ema50"]) or pd.isna(row["atr14"]) or row["atr14"] == 0:
        return None
    uptrend = row["ema20"] > row["ema50"] * 1.005
    downtrend = row["ema20"] < row["ema50"] * 0.995
    near_ema20 = abs(row["low"] - row["ema20"]) <= row["atr14"] * 0.5 or abs(row["high"] - row["ema20"]) <= row["atr14"] * 0.5

    if uptrend and near_ema20 and row["is_bull"] and row["close"] > row["ema20"]:
        return "LONG"
    if downtrend and near_ema20 and (not row["is_bull"]) and row["close"] < row["ema20"]:
        return "SHORT"
    return None


def strategy_f_obv_confirmation(df, i):
    """Fiyat ve kumulatif hacim akisi (OBV) ayni yonde - gercek alim/satim baskisi var mi."""
    row = df.iloc[i]
    if pd.isna(row["obv_slope"]) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 1.5:
        return None
    if row["is_bull"] and row["obv_slope"] > 0 and row["close_position"] >= 0.6:
        return "LONG"
    if (not row["is_bull"]) and row["obv_slope"] < 0 and row["close_position"] <= 0.4:
        return "SHORT"
    return None


def strategy_g_zscore_reversion(df, i):
    """Fiyat, kendi ortalamasindan istatistiksel olarak asiri sapmis (z-skor) - tersine bahis."""
    row = df.iloc[i]
    if pd.isna(row["zscore"]):
        return None
    if row["zscore"] <= -2.2:
        return "LONG"
    if row["zscore"] >= 2.2:
        return "SHORT"
    return None


def strategy_h_three_candle_momentum(df, i):
    """Art arda 3 ayni yonlu mum, artan hacim - 4. mumda devam bahsi."""
    if i < 3:
        return None
    c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    if pd.isna(c1["atr14"]) or c1["atr14"] == 0:
        return None

    all_bull = c1["is_bull"] and c2["is_bull"] and c3["is_bull"]
    all_bear = (not c1["is_bull"]) and (not c2["is_bull"]) and (not c3["is_bull"])
    rising_volume = c3["volume"] > c2["volume"] > c1["volume"]
    decent_bodies = all(c["body"] >= c["atr14"] * 0.3 for c in [c1, c2, c3])

    if all_bull and rising_volume and decent_bodies:
        return "LONG"
    if all_bear and rising_volume and decent_bodies:
        return "SHORT"
    return None


STRATEGIES = {
    "A-Kirilim+Yeniden Test": strategy_a_breakout_retest,
    "B-Volatilite Sikismasi": strategy_b_volatility_squeeze,
    "C-Uc Zaman Dilimi Uyumu": strategy_c_triple_timeframe,
    "D-BTC'ye Gore Goreceli Guc": strategy_d_relative_strength,
    "E-Trend Ici Geri Cekilme": strategy_e_pullback_in_trend,
    "F-OBV Hacim Teyidi": strategy_f_obv_confirmation,
    "G-Z-Skor Istatistiksel Sapma": strategy_g_zscore_reversion,
    "H-Uc Mum Momentum Kaliciligi": strategy_h_three_candle_momentum,
}


# ---------------------------------------------------------------------------
# Degerlendirme
# ---------------------------------------------------------------------------

def evaluate_signals(df, strategy_fn):
    total, correct = 0, 0
    max_check = max(CHECK_CANDLES)
    start_idx = max(BREAKOUT_LOOKBACK, SQUEEZE_LOOKBACK, ZSCORE_PERIOD) + 5

    for i in range(start_idx, len(df) - max_check):
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
    print(f"Turnuva basliyor: {len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} gunluk veri, {len(STRATEGIES)} YENI strateji")
    send_telegram_message(
        f"🏆 Strateji turnuvası (2. tur - yeni stratejiler) başladı.\n"
        f"{len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} günlük geçmiş veri, {len(STRATEGIES)} strateji test ediliyor.\n"
        f"Bitince sonuçları göndereceğim."
    )

    print("BTC verisi cekiliyor (goreceli guc karsilastirmasi icin)...")
    btc_df = fetch_historical_ohlcv(BTC_SYMBOL)

    results = {name: {"total": 0, "correct": 0} for name in STRATEGIES}

    for idx, symbol in enumerate(TOURNAMENT_WATCHLIST):
        print(f"[{idx+1}/{len(TOURNAMENT_WATCHLIST)}] {symbol} verisi cekiliyor...")
        try:
            df = fetch_historical_ohlcv(symbol)
            if df.empty or len(df) < SQUEEZE_LOOKBACK + 50:
                print(f"  {symbol}: yetersiz veri, atlaniyor")
                continue
            df = compute_all_indicators(df, btc_df=btc_df)
        except Exception as e:
            print(f"  {symbol} hata: {e}")
            continue

        for name, fn in STRATEGIES.items():
            total, correct = evaluate_signals(df, fn)
            results[name]["total"] += total
            results[name]["correct"] += correct

    leaderboard = []
    for name, r in results.items():
        total, correct = r["total"], r["correct"]
        win_rate = (correct / total * 100) if total > 0 else 0
        leaderboard.append((name, total, correct, win_rate))

    ranked = [x for x in leaderboard if x[1] >= MIN_SIGNALS_TO_RANK]
    unranked = [x for x in leaderboard if x[1] < MIN_SIGNALS_TO_RANK]
    ranked.sort(key=lambda x: x[3], reverse=True)

    lines = [f"🏆 TURNUVA SONUÇLARI - 2. TUR ({LOOKBACK_DAYS} günlük veri, {len(TOURNAMENT_WATCHLIST)} coin)\n"]
    lines.append(f"Sıralama (en az {MIN_SIGNALS_TO_RANK} sinyal üreten stratejiler):\n")
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

    with open("tournament_results_v2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strateji", "toplam_sinyal", "dogru", "isabet_yuzde"])
        for name, total, correct, win_rate in leaderboard:
            writer.writerow([name, total, correct, f"{win_rate:.2f}"])

    print("\nTurnuva tamamlandi. Sonuclar tournament_results_v2.csv dosyasina kaydedildi.")


if __name__ == "__main__":
    run_tournament()
