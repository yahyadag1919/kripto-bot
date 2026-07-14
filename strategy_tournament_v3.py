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

TIMEFRAME = "15m"
LOOKBACK_DAYS = 30

ATR_PERIOD = 14
RSI_PERIOD = 14
SWING_LOOKBACK = 5          # swing high/low tespiti icin her iki yandan kac mum
VWAP_WINDOW = 96             # ~24 saat (15m mumla) - kayan VWAP penceresi
FIB_SWING_LOOKBACK = 30

CHECK_CANDLES = [1, 2, 3]    # 15/30/45 dk
SUCCESS_THRESHOLD_PCT = 0.15
MIN_SIGNALS_TO_RANK = 15

# Tahmini yuvarlanmis (round-trip) komisyon + kayma maliyeti (%).
# OKX vadeli islem taker ucreti genelde ~%0.05, giris+cikis = ~%0.10,
# + kayma payi icin biraz daha ekliyoruz. Kendi gercek ucretine gore degistirebilirsin.
COMMISSION_ROUNDTRIP_PCT = 0.12

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
# Gecmis veri cekme
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

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["vol_sma20"] = df["volume"].rolling(20).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(ATR_PERIOD).mean()

    df["body"] = (df["close"] - df["open"]).abs()
    df["is_bull"] = df["close"] > df["open"]

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    # MACD
    df["macd_line"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # Stochastic osilator (14,3)
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df["stoch_k"] = ((df["close"] - low14) / (high14 - low14).replace(0, np.nan)) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Keltner Kanali (EMA20 +- ATR*1.5)
    df["keltner_upper"] = df["ema20"] + df["atr14"] * 1.5
    df["keltner_lower"] = df["ema20"] - df["atr14"] * 1.5

    # Kayan VWAP (yaklasik - gercek gunluk sifirlanan VWAP degil, kayan pencere)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]
    df["vwap"] = pv.rolling(VWAP_WINDOW).sum() / df["volume"].rolling(VWAP_WINDOW).sum()
    df["vwap_dev_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100

    # Swing high/low (basit fraktal: N mum sagi ve solu daha dusuk/yuksek)
    df["swing_high"] = df["high"][(df["high"] == df["high"].rolling(SWING_LOOKBACK * 2 + 1, center=True).max())]
    df["swing_low"] = df["low"][(df["low"] == df["low"].rolling(SWING_LOOKBACK * 2 + 1, center=True).min())]

    return df


# ---------------------------------------------------------------------------
# 8 YENI strateji - 1. ve 2. turdakilerden farkli mekanizmalar
# ---------------------------------------------------------------------------

def strategy_macd_cross(df, i):
    """Klasik MACD: MACD cizgisi sinyal cizgisini kesiyor + histogram donuyor."""
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["macd_line"]) or pd.isna(prev["macd_line"]):
        return None
    if prev["macd_line"] <= prev["macd_signal"] and row["macd_line"] > row["macd_signal"] and row["macd_hist"] > 0:
        return "LONG"
    if prev["macd_line"] >= prev["macd_signal"] and row["macd_line"] < row["macd_signal"] and row["macd_hist"] < 0:
        return "SHORT"
    return None


def strategy_stochastic(df, i):
    """Stochastic %K, %D'yi asiri bolgede kesiyor."""
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["stoch_k"]) or pd.isna(prev["stoch_k"]):
        return None
    if prev["stoch_k"] <= prev["stoch_d"] and row["stoch_k"] > row["stoch_d"] and row["stoch_k"] < 30:
        return "LONG"
    if prev["stoch_k"] >= prev["stoch_d"] and row["stoch_k"] < row["stoch_d"] and row["stoch_k"] > 70:
        return "SHORT"
    return None


def strategy_keltner_breakout(df, i):
    """Fiyat Keltner kanalinin (ATR tabanli) disina hacimle tasiyor."""
    row = df.iloc[i]
    if pd.isna(row["keltner_upper"]) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 1.8:
        return None
    if row["close"] > row["keltner_upper"] and row["is_bull"]:
        return "LONG"
    if row["close"] < row["keltner_lower"] and not row["is_bull"]:
        return "SHORT"
    return None


def strategy_vwap_deviation(df, i):
    """Fiyat kayan VWAP'tan asiri sapmis - tersine bahis."""
    row = df.iloc[i]
    if pd.isna(row["vwap_dev_pct"]):
        return None
    if row["vwap_dev_pct"] <= -3.0:
        return "LONG"
    if row["vwap_dev_pct"] >= 3.0:
        return "SHORT"
    return None


def strategy_fibonacci_bounce(df, i):
    """Son swing hareketin %50-%61.8 seviyesine geri cekilip trend yonunde sekiyor."""
    if i < FIB_SWING_LOOKBACK:
        return None
    window = df.iloc[i - FIB_SWING_LOOKBACK:i]
    swing_high = window["high"].max()
    swing_low = window["low"].min()
    swing_range = swing_high - swing_low
    if swing_range <= 0:
        return None

    row = df.iloc[i]
    fib_50 = swing_high - swing_range * 0.5
    fib_618 = swing_high - swing_range * 0.618

    high_idx = window["high"].idxmax()
    low_idx = window["low"].idxmin()
    uptrend_swing = high_idx > low_idx   # dip once, tepe sonra olustu -> yukselen swing

    if uptrend_swing and fib_618 <= row["low"] <= fib_50 and row["is_bull"] and row["close"] > fib_50:
        return "LONG"
    if (not uptrend_swing) and fib_50 <= row["high"] <= fib_618 and (not row["is_bull"]) and row["close"] < fib_50:
        return "SHORT"
    return None


def strategy_market_structure_break(df, i):
    """Son swing high/low kirilirsa yapisal kirilim - trend yonunde giris."""
    if i < 30:
        return None
    row = df.iloc[i]
    recent = df.iloc[i - 30:i]
    last_swing_high = recent["swing_high"].dropna()
    last_swing_low = recent["swing_low"].dropna()
    if last_swing_high.empty or last_swing_low.empty:
        return None

    if row["close"] > last_swing_high.iloc[-1] and row["is_bull"]:
        return "LONG"
    if row["close"] < last_swing_low.iloc[-1] and not row["is_bull"]:
        return "SHORT"
    return None


def strategy_rsi_divergence(df, i):
    """Fiyat yeni dip/tepe yapiyor ama RSI onaylamiyor - uyumsuzluk, tersine donus sinyali."""
    if i < 20:
        return None
    row = df.iloc[i]
    window = df.iloc[i - 20:i]
    if pd.isna(row["rsi"]):
        return None

    price_new_low = row["close"] <= window["close"].min()
    price_new_high = row["close"] >= window["close"].max()
    rsi_higher_than_window_min = row["rsi"] > window["rsi"].min() + 5
    rsi_lower_than_window_max = row["rsi"] < window["rsi"].max() - 5

    if price_new_low and rsi_higher_than_window_min and row["is_bull"]:
        return "LONG"
    if price_new_high and rsi_lower_than_window_max and not row["is_bull"]:
        return "SHORT"
    return None


def strategy_engulfing_pattern(df, i):
    """Yutan mum formasyonu (bullish/bearish engulfing) + hacim teyidi."""
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 1.3:
        return None

    bullish_engulf = (not prev["is_bull"]) and row["is_bull"] and row["open"] <= prev["close"] and row["close"] >= prev["open"]
    bearish_engulf = prev["is_bull"] and (not row["is_bull"]) and row["open"] >= prev["close"] and row["close"] <= prev["open"]

    if bullish_engulf:
        return "LONG"
    if bearish_engulf:
        return "SHORT"
    return None


def strategy_ema_ribbon_alignment(df, i):
    """EMA12/20/26/50 siralanmis (hepsi ayni yonde) + fiyat hepsinin ustunde/altinda - guclu trend teyidi."""
    row = df.iloc[i]
    vals = [row["ema12"], row["ema20"], row["ema26"], row["ema50"]]
    if any(pd.isna(v) for v in vals):
        return None
    bullish_aligned = row["ema12"] > row["ema20"] > row["ema26"] > row["ema50"] and row["close"] > row["ema12"]
    bearish_aligned = row["ema12"] < row["ema20"] < row["ema26"] < row["ema50"] and row["close"] < row["ema12"]
    if bullish_aligned and row["is_bull"]:
        return "LONG"
    if bearish_aligned and not row["is_bull"]:
        return "SHORT"
    return None


STRATEGIES = {
    "MACD Kesisimi": strategy_macd_cross,
    "Stochastic Osilator": strategy_stochastic,
    "Keltner Kanali Kirilimi": strategy_keltner_breakout,
    "VWAP Sapmasi": strategy_vwap_deviation,
    "Fibonacci Geri Cekilme": strategy_fibonacci_bounce,
    "Piyasa Yapisi Kirilimi": strategy_market_structure_break,
    "RSI Uyumsuzlugu": strategy_rsi_divergence,
    "Yutan Mum + Hacim": strategy_engulfing_pattern,
}


# ---------------------------------------------------------------------------
# Degerlendirme - artik gercek yuzde kar/zarar + komisyon
# ---------------------------------------------------------------------------

def evaluate_signals(df, strategy_fn):
    total = 0
    wins = 0
    net_pct_list = []
    max_check = max(CHECK_CANDLES)
    start_idx = 60

    for i in range(start_idx, len(df) - max_check):
        direction = strategy_fn(df, i)
        if direction is None:
            continue

        entry_price = df.iloc[i]["close"]
        realized_pct = None

        for c in CHECK_CANDLES:
            future_price = df.iloc[i + c]["close"]
            pct_change = (future_price - entry_price) / entry_price * 100
            favorable_pct = pct_change if direction == "LONG" else -pct_change

            if favorable_pct >= SUCCESS_THRESHOLD_PCT:
                realized_pct = favorable_pct
                break

        if realized_pct is None:
            final_price = df.iloc[i + max_check]["close"]
            pct_change = (final_price - entry_price) / entry_price * 100
            realized_pct = pct_change if direction == "LONG" else -pct_change

        net_pct = realized_pct - COMMISSION_ROUNDTRIP_PCT

        total += 1
        if realized_pct >= SUCCESS_THRESHOLD_PCT:
            wins += 1
        net_pct_list.append(net_pct)

    avg_net_pct = float(np.mean(net_pct_list)) if net_pct_list else 0.0
    total_net_pct = float(np.sum(net_pct_list)) if net_pct_list else 0.0

    return total, wins, avg_net_pct, total_net_pct


# ---------------------------------------------------------------------------
# Ana turnuva
# ---------------------------------------------------------------------------

def run_tournament():
    print(f"Turnuva basliyor (3. tur, komisyon dahil): {len(TOURNAMENT_WATCHLIST)} coin, "
          f"{LOOKBACK_DAYS} gunluk veri, {len(STRATEGIES)} strateji")
    send_telegram_message(
        f"🏆 Strateji turnuvası (3. tur - yeni stratejiler + gerçek kâr/zarar) başladı.\n"
        f"{len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} günlük veri, {len(STRATEGIES)} strateji.\n"
        f"Tahmini komisyon: %{COMMISSION_ROUNDTRIP_PCT} (gidiş-dönüş).\n"
        f"Bitince sonuçları göndereceğim."
    )

    results = {name: {"total": 0, "wins": 0, "net_pct_sum": 0.0, "signal_count_for_avg": 0} for name in STRATEGIES}

    for idx, symbol in enumerate(TOURNAMENT_WATCHLIST):
        print(f"[{idx+1}/{len(TOURNAMENT_WATCHLIST)}] {symbol} verisi cekiliyor...")
        try:
            df = fetch_historical_ohlcv(symbol)
            if df.empty or len(df) < 100:
                print(f"  {symbol}: yetersiz veri, atlaniyor")
                continue
            df = compute_all_indicators(df)
        except Exception as e:
            print(f"  {symbol} hata: {e}")
            continue

        for name, fn in STRATEGIES.items():
            total, wins, avg_net_pct, total_net_pct = evaluate_signals(df, fn)
            results[name]["total"] += total
            results[name]["wins"] += wins
            results[name]["net_pct_sum"] += total_net_pct

    leaderboard = []
    for name, r in results.items():
        total, wins = r["total"], r["wins"]
        win_rate = (wins / total * 100) if total > 0 else 0
        avg_net_pct = (r["net_pct_sum"] / total) if total > 0 else 0
        leaderboard.append((name, total, wins, win_rate, avg_net_pct, r["net_pct_sum"]))

    ranked = [x for x in leaderboard if x[1] >= MIN_SIGNALS_TO_RANK]
    unranked = [x for x in leaderboard if x[1] < MIN_SIGNALS_TO_RANK]
    # Komisyon sonrasi ortalama net kazanca gore sirala (gercek karlilik)
    ranked.sort(key=lambda x: x[4], reverse=True)

    lines = [f"🏆 TURNUVA SONUÇLARI - 3. TUR ({LOOKBACK_DAYS} günlük veri, {len(TOURNAMENT_WATCHLIST)} coin)"]
    lines.append(f"Komisyon (%{COMMISSION_ROUNDTRIP_PCT}) düşülmüş, işlem başına ORTALAMA NET kazanca göre sıralı.\n")
    for rank, (name, total, wins, win_rate, avg_net_pct, total_net_pct) in enumerate(ranked, 1):
        madalya = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        yon = "✅ KARLI" if avg_net_pct > 0 else "❌ ZARARLI"
        lines.append(
            f"{madalya} {name} {yon}\n"
            f"   Sinyal: {total} | İsabet: %{win_rate:.1f}\n"
            f"   İşlem başı ort. net: %{avg_net_pct:+.3f} | Toplam: %{total_net_pct:+.1f}"
        )

    if unranked:
        lines.append("\nYetersiz örneklem (sıralamaya alınmadı):")
        for name, total, wins, win_rate, avg_net_pct, total_net_pct in unranked:
            lines.append(f"- {name}: {total} sinyal (%{win_rate:.1f} isabet, ort net %{avg_net_pct:+.3f})")

    msg = "\n".join(lines)
    print("\n" + msg)
    send_telegram_message(msg)

    with open("tournament_results_v3.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strateji", "toplam_sinyal", "dogru", "isabet_yuzde", "ort_net_yuzde", "toplam_net_yuzde"])
        for name, total, wins, win_rate, avg_net_pct, total_net_pct in leaderboard:
            writer.writerow([name, total, wins, f"{win_rate:.2f}", f"{avg_net_pct:.4f}", f"{total_net_pct:.2f}"])

    print("\nTurnuva tamamlandi. Sonuclar tournament_results_v3.csv dosyasina kaydedildi.")


if __name__ == "__main__":
    run_tournament()
