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

CHECK_CANDLES = [1, 2, 3]
SUCCESS_THRESHOLD_PCT = 0.15
MIN_SIGNALS_TO_RANK = 15
COMMISSION_ROUNDTRIP_PCT = 0.12

PIVOT_WINDOW = 96          # ~24 saat (15m mumla) - pivot hesaplama penceresi
VOLUME_PROFILE_WINDOW = 96
VOLUME_PROFILE_BUCKETS = 24

exchange = ccxt.okx({
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
})


def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("UYARI: Telegram bilgileri yok.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=15)
    except Exception as e:
        print(f"Telegram gonderim hatasi: {e}")


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

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    # --- Ichimoku ---
    tenkan_high = df["high"].rolling(9).max()
    tenkan_low = df["low"].rolling(9).min()
    df["tenkan"] = (tenkan_high + tenkan_low) / 2
    kijun_high = df["high"].rolling(26).max()
    kijun_low = df["low"].rolling(26).min()
    df["kijun"] = (kijun_high + kijun_low) / 2
    df["senkou_a"] = ((df["tenkan"] + df["kijun"]) / 2).shift(26)
    senkou_b_high = df["high"].rolling(52).max()
    senkou_b_low = df["low"].rolling(52).min()
    df["senkou_b"] = ((senkou_b_high + senkou_b_low) / 2).shift(26)
    df["cloud_top"] = df[["senkou_a", "senkou_b"]].max(axis=1)
    df["cloud_bottom"] = df[["senkou_a", "senkou_b"]].min(axis=1)

    # --- Parabolic SAR (basitlestirilmis iteratif hesap) ---
    df["sar"] = compute_parabolic_sar(df)

    # --- Heikin-Ashi ---
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = np.zeros(len(df))
    ha_open[0] = df["open"].iloc[0]
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close.iloc[i - 1]) / 2
    df["ha_open"] = ha_open
    df["ha_close"] = ha_close
    df["ha_is_bull"] = df["ha_close"] > df["ha_open"]
    df["ha_lower_wick"] = df[["ha_open", "ha_close"]].min(axis=1) - df["low"]
    df["ha_upper_wick"] = df["high"] - df[["ha_open", "ha_close"]].max(axis=1)

    # --- Klasik Pivot Noktalari (kayan 24 saatlik pencere) ---
    period_high = df["high"].rolling(PIVOT_WINDOW).max().shift(1)
    period_low = df["low"].rolling(PIVOT_WINDOW).min().shift(1)
    period_close = df["close"].shift(PIVOT_WINDOW)
    pivot = (period_high + period_low + period_close) / 3
    df["pivot_r1"] = 2 * pivot - period_low
    df["pivot_s1"] = 2 * pivot - period_high
    df["pivot_p"] = pivot

    # --- Chandelier Exit (ATR takip eden stop) ---
    highest_22 = df["high"].rolling(22).max()
    lowest_22 = df["low"].rolling(22).min()
    df["chandelier_long"] = highest_22 - df["atr14"] * 3
    df["chandelier_short"] = lowest_22 + df["atr14"] * 3

    # --- Elder Triple Screen icin uzun vade trend ---
    df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()

    # Seans saati (UTC saat, basit oturum ayrimi icin)
    df["hour"] = df["timestamp"].dt.hour

    return df


def compute_parabolic_sar(df, af_step=0.02, af_max=0.2):
    n = len(df)
    sar = np.zeros(n)
    if n < 2:
        return sar
    trend_up = True
    af = af_step
    ep = df["high"].iloc[0]
    sar[0] = df["low"].iloc[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if trend_up:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = min(sar[i], df["low"].iloc[i - 1], df["low"].iloc[max(0, i - 2)])
            if df["high"].iloc[i] > ep:
                ep = df["high"].iloc[i]
                af = min(af + af_step, af_max)
            if df["low"].iloc[i] < sar[i]:
                trend_up = False
                sar[i] = ep
                ep = df["low"].iloc[i]
                af = af_step
        else:
            sar[i] = prev_sar + af * (ep - prev_sar)
            sar[i] = max(sar[i], df["high"].iloc[i - 1], df["high"].iloc[max(0, i - 2)])
            if df["low"].iloc[i] < ep:
                ep = df["low"].iloc[i]
                af = min(af + af_step, af_max)
            if df["high"].iloc[i] > sar[i]:
                trend_up = True
                sar[i] = ep
                ep = df["high"].iloc[i]
                af = af_step
    return sar


def get_volume_profile_poc(df, i, window=VOLUME_PROFILE_WINDOW, buckets=VOLUME_PROFILE_BUCKETS):
    """Son 'window' mumun hacim dagilimina bakip en yuksek hacimli fiyat bolgesini (POC) bulur."""
    if i < window:
        return None
    segment = df.iloc[i - window:i]
    low, high = segment["low"].min(), segment["high"].max()
    if high <= low:
        return None
    bucket_edges = np.linspace(low, high, buckets + 1)
    bucket_volumes = np.zeros(buckets)
    typical = (segment["high"] + segment["low"] + segment["close"]) / 3
    bucket_idx = np.clip(np.digitize(typical, bucket_edges) - 1, 0, buckets - 1)
    for idx, vol in zip(bucket_idx, segment["volume"]):
        bucket_volumes[idx] += vol
    poc_bucket = np.argmax(bucket_volumes)
    poc_price = (bucket_edges[poc_bucket] + bucket_edges[poc_bucket + 1]) / 2
    return poc_price


# ---------------------------------------------------------------------------
# 8 YENI strateji (Round 5)
# ---------------------------------------------------------------------------

def strategy_ichimoku_cloud(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["cloud_top"]) or pd.isna(row["cloud_bottom"]):
        return None
    if prev["close"] <= prev["cloud_top"] and row["close"] > row["cloud_top"] and row["is_bull"]:
        return "LONG"
    if prev["close"] >= prev["cloud_bottom"] and row["close"] < row["cloud_bottom"] and not row["is_bull"]:
        return "SHORT"
    return None


def strategy_parabolic_sar_flip(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["sar"]) or pd.isna(prev["sar"]):
        return None
    prev_below = prev["close"] > prev["sar"]
    now_below = row["close"] > row["sar"]
    if not prev_below and now_below:
        return "LONG"
    if prev_below and not now_below:
        return "SHORT"
    return None


def strategy_pivot_bounce(df, i):
    row = df.iloc[i]
    if pd.isna(row["pivot_s1"]) or pd.isna(row["atr14"]) or row["atr14"] == 0:
        return None
    near_s1 = abs(row["low"] - row["pivot_s1"]) <= row["atr14"] * 0.3
    near_r1 = abs(row["high"] - row["pivot_r1"]) <= row["atr14"] * 0.3
    if near_s1 and row["is_bull"] and row["close"] > row["pivot_s1"]:
        return "LONG"
    if near_r1 and not row["is_bull"] and row["close"] < row["pivot_r1"]:
        return "SHORT"
    return None


def strategy_heikin_ashi_trend(df, i):
    if i < 2:
        return None
    c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    if pd.isna(c3["ha_lower_wick"]):
        return None
    all_bull_no_lower_wick = (c1["ha_is_bull"] and c2["ha_is_bull"] and c3["ha_is_bull"]
                               and c3["ha_lower_wick"] <= 0)
    all_bear_no_upper_wick = ((not c1["ha_is_bull"]) and (not c2["ha_is_bull"]) and (not c3["ha_is_bull"])
                               and c3["ha_upper_wick"] <= 0)
    if all_bull_no_lower_wick:
        return "LONG"
    if all_bear_no_upper_wick:
        return "SHORT"
    return None


def strategy_volume_profile_rejection(df, i):
    poc = get_volume_profile_poc(df, i)
    if poc is None:
        return None
    row = df.iloc[i]
    if pd.isna(row["atr14"]) or row["atr14"] == 0:
        return None
    near_poc = abs(row["close"] - poc) <= row["atr14"] * 0.5
    if not near_poc:
        return None
    if row["low"] <= poc <= row["high"] and row["is_bull"] and row["close"] > poc:
        return "LONG"
    if row["low"] <= poc <= row["high"] and not row["is_bull"] and row["close"] < poc:
        return "SHORT"
    return None


def strategy_chandelier_exit(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["chandelier_long"]) or pd.isna(prev["chandelier_long"]):
        return None
    if prev["close"] <= prev["chandelier_long"] and row["close"] > row["chandelier_long"]:
        return "LONG"
    if prev["close"] >= prev["chandelier_short"] and row["close"] < row["chandelier_short"]:
        return "SHORT"
    return None


def strategy_session_filtered_breakout(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    # Sadece ABD/Avrupa ortak seans saatlerinde (13:00-21:00 UTC) islem ara - daha yuksek likidite
    if not (13 <= row["hour"] <= 21):
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 2.2:
        return None
    if row["is_bull"] and row["body"] / row["atr14"] >= 0.6:
        return "LONG"
    if not row["is_bull"] and row["body"] / row["atr14"] >= 0.6:
        return "SHORT"
    return None


def strategy_elder_triple_screen(df, i):
    row = df.iloc[i]
    if pd.isna(row["ema100"]) or pd.isna(row["rsi"]):
        return None
    long_term_up = row["close"] > row["ema100"]
    long_term_down = row["close"] < row["ema100"]
    oscillator_pullback_up = 40 <= row["rsi"] <= 55
    oscillator_pullback_down = 45 <= row["rsi"] <= 60
    trigger_bull = row["is_bull"] and row["close"] > row["ema20"]
    trigger_bear = (not row["is_bull"]) and row["close"] < row["ema20"]

    if long_term_up and oscillator_pullback_up and trigger_bull:
        return "LONG"
    if long_term_down and oscillator_pullback_down and trigger_bear:
        return "SHORT"
    return None


STRATEGIES = {
    "Ichimoku Bulutu": strategy_ichimoku_cloud,
    "Parabolic SAR Donusu": strategy_parabolic_sar_flip,
    "Klasik Pivot Sekmesi": strategy_pivot_bounce,
    "Heikin-Ashi Trend": strategy_heikin_ashi_trend,
    "Hacim Profili POC Reddi": strategy_volume_profile_rejection,
    "Chandelier Exit Donusu": strategy_chandelier_exit,
    "Seans Saatine Gore Kirilim": strategy_session_filtered_breakout,
    "Elder Uclu Ekran": strategy_elder_triple_screen,
}


def evaluate_signals(df, strategy_fn):
    total, wins = 0, 0
    net_pct_list = []
    max_check = max(CHECK_CANDLES)
    start_idx = 110  # Ichimoku/pivot gibi uzun pencereli indikatorler icin guvenli baslangic

    for i in range(start_idx, len(df) - max_check):
        direction = strategy_fn(df, i)
        if direction is None:
            continue
        entry_price = df.iloc[i]["close"]
        realized_pct = None
        for c in CHECK_CANDLES:
            future_price = df.iloc[i + c]["close"]
            pct_change = (future_price - entry_price) / entry_price * 100
            favorable = pct_change if direction == "LONG" else -pct_change
            if favorable >= SUCCESS_THRESHOLD_PCT:
                realized_pct = favorable
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

    avg_net = float(np.mean(net_pct_list)) if net_pct_list else 0.0
    total_net = float(np.sum(net_pct_list)) if net_pct_list else 0.0
    return total, wins, avg_net, total_net


def run_tournament():
    print(f"Turnuva basliyor (5. tur): {len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} gunluk veri, {len(STRATEGIES)} strateji")
    send_telegram_message(
        f"🏆 Strateji turnuvası (5. tur - Ichimoku, SAR, Pivot, Heikin-Ashi vb.) başladı.\n"
        f"{len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} günlük veri, {len(STRATEGIES)} strateji.\n"
        f"Bitince göndereceğim."
    )

    results = {name: {"total": 0, "wins": 0, "net_pct_sum": 0.0} for name in STRATEGIES}

    for idx, symbol in enumerate(TOURNAMENT_WATCHLIST):
        print(f"[{idx+1}/{len(TOURNAMENT_WATCHLIST)}] {symbol} verisi cekiliyor...")
        try:
            df = fetch_historical_ohlcv(symbol)
            if df.empty or len(df) < 150:
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
        avg_net = (r["net_pct_sum"] / total) if total > 0 else 0
        leaderboard.append((name, total, wins, win_rate, avg_net, r["net_pct_sum"]))

    ranked = [x for x in leaderboard if x[1] >= MIN_SIGNALS_TO_RANK]
    unranked = [x for x in leaderboard if x[1] < MIN_SIGNALS_TO_RANK]
    ranked.sort(key=lambda x: x[4], reverse=True)

    lines = [f"🏆 TURNUVA SONUÇLARI - 5. TUR ({LOOKBACK_DAYS} günlük veri, {len(TOURNAMENT_WATCHLIST)} coin)"]
    lines.append(f"Komisyon (%{COMMISSION_ROUNDTRIP_PCT}) düşülmüş, ort. net kazanca göre sıralı.\n")
    for rank, (name, total, wins, win_rate, avg_net, total_net) in enumerate(ranked, 1):
        madalya = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        yon = "✅ KARLI" if avg_net > 0 else "❌ ZARARLI"
        lines.append(
            f"{madalya} {name} {yon}\n"
            f"   Sinyal: {total} | İsabet: %{win_rate:.1f}\n"
            f"   İşlem başı ort. net: %{avg_net:+.3f} | Toplam: %{total_net:+.1f}"
        )

    if unranked:
        lines.append("\nYetersiz örneklem:")
        for name, total, wins, win_rate, avg_net, total_net in unranked:
            lines.append(f"- {name}: {total} sinyal (%{win_rate:.1f}, ort net %{avg_net:+.3f})")

    msg = "\n".join(lines)
    print("\n" + msg)
    send_telegram_message(msg)

    with open("tournament_results_v5.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strateji", "toplam_sinyal", "dogru", "isabet_yuzde", "ort_net_yuzde", "toplam_net_yuzde"])
        for name, total, wins, win_rate, avg_net, total_net in leaderboard:
            writer.writerow([name, total, wins, f"{win_rate:.2f}", f"{avg_net:.4f}", f"{total_net:.2f}"])

    print("\nTurnuva tamamlandi.")


if __name__ == "__main__":
    run_tournament()
