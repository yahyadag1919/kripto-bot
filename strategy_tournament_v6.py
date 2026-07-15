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
BTC_SYMBOL = "BTC/USDT:USDT"

TIMEFRAME = "15m"
LOOKBACK_DAYS = 30
ATR_PERIOD = 14

CHECK_CANDLES = [1, 2, 3]
SUCCESS_THRESHOLD_PCT = 0.15
MIN_SIGNALS_TO_RANK = 15
COMMISSION_ROUNDTRIP_PCT = 0.12

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


def fetch_funding_history(symbol: str, days: int = LOOKBACK_DAYS):
    try:
        since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        all_rates = []
        cursor = since
        while True:
            batch = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=100)
            if not batch:
                break
            all_rates.extend(batch)
            last_ts = batch[-1]["timestamp"]
            if last_ts <= cursor or len(batch) < 100:
                break
            cursor = last_ts + 1
            time.sleep(0.15)
        if not all_rates:
            return None
        fr = pd.DataFrame(all_rates)
        fr["timestamp"] = pd.to_datetime(fr["timestamp"], unit="ms")
        fr = fr[["timestamp", "fundingRate"]].dropna().sort_values("timestamp")
        return fr
    except Exception as e:
        print(f"  Funding gecmisi alinamadi ({symbol}): {e}")
        return None


def compute_all_indicators(df: pd.DataFrame, btc_df: pd.DataFrame = None, funding_df: pd.DataFrame = None) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["vol_sma20"] = df["volume"].rolling(20).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(ATR_PERIOD).mean()

    df["body"] = (df["close"] - df["open"]).abs()
    df["is_bull"] = df["close"] > df["open"]

    df["jaw"] = df["close"].rolling(13).mean().shift(8)
    df["teeth"] = df["close"].rolling(8).mean().shift(5)
    df["lips"] = df["close"].rolling(5).mean().shift(3)

    median_price = (df["high"] + df["low"]) / 2
    df["ao"] = median_price.rolling(5).mean() - median_price.rolling(34).mean()

    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical.rolling(20).mean()
    mean_dev = typical.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = (typical - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))

    high14 = df["high"].rolling(14).max()
    low14 = df["low"].rolling(14).min()
    df["williams_r"] = (high14 - df["close"]) / (high14 - low14).replace(0, np.nan) * -100

    df["date"] = df["timestamp"].dt.date
    daily_high_by_date = df.groupby("date")["high"].max()
    daily_low_by_date = df.groupby("date")["low"].min()
    df["prev_day_high"] = df["date"].map(daily_high_by_date.shift(1))
    df["prev_day_low"] = df["date"].map(daily_low_by_date.shift(1))

    df["renko_direction"] = compute_renko_direction(df)

    if btc_df is not None:
        btc_small = btc_df[["timestamp", "close"]].rename(columns={"close": "btc_close"})
        btc_small = btc_small.copy()
        btc_small["timestamp"] = btc_small["timestamp"].astype("datetime64[ns]")
        df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
        df = pd.merge_asof(df.sort_values("timestamp"), btc_small.sort_values("timestamp"),
                            on="timestamp", direction="backward")
        df["ratio"] = df["close"] / df["btc_close"]
        ratio_mean = df["ratio"].rolling(96).mean()
        ratio_std = df["ratio"].rolling(96).std()
        df["ratio_zscore"] = (df["ratio"] - ratio_mean) / ratio_std.replace(0, np.nan)
    else:
        df["ratio_zscore"] = np.nan

    if funding_df is not None and not funding_df.empty:
        funding_df = funding_df.copy()
        funding_df["timestamp"] = funding_df["timestamp"].astype("datetime64[ns]")
        df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
        df = pd.merge_asof(df.sort_values("timestamp"), funding_df.sort_values("timestamp"),
                            on="timestamp", direction="backward")
    else:
        df["fundingRate"] = np.nan

    return df


def compute_renko_direction(df, brick_multiplier=1.0):
    n = len(df)
    direction = np.zeros(n)
    atr = df["atr14"].values
    closes = df["close"].values

    last_brick_price = closes[0] if n > 0 else 0
    trend = 0
    consecutive = 0

    for i in range(n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            direction[i] = 0
            continue
        brick_size = atr[i] * brick_multiplier
        diff = closes[i] - last_brick_price

        if diff >= brick_size:
            bricks = int(diff // brick_size)
            last_brick_price += bricks * brick_size
            if trend >= 0:
                consecutive += bricks
            else:
                consecutive = bricks
            trend = 1
        elif diff <= -brick_size:
            bricks = int(-diff // brick_size)
            last_brick_price -= bricks * brick_size
            if trend <= 0:
                consecutive += bricks
            else:
                consecutive = bricks
            trend = -1

        direction[i] = trend * consecutive

    return direction


def strategy_alligator(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["jaw"]) or pd.isna(row["teeth"]) or pd.isna(row["lips"]):
        return None
    bullish_order = row["lips"] > row["teeth"] > row["jaw"]
    bearish_order = row["lips"] < row["teeth"] < row["jaw"]
    prev_not_bullish = not (prev["lips"] > prev["teeth"] > prev["jaw"])
    prev_not_bearish = not (prev["lips"] < prev["teeth"] < prev["jaw"])

    if bullish_order and prev_not_bullish and row["is_bull"]:
        return "LONG"
    if bearish_order and prev_not_bearish and not row["is_bull"]:
        return "SHORT"
    return None


def strategy_awesome_oscillator(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["ao"]) or pd.isna(prev["ao"]):
        return None
    if prev["ao"] <= 0 and row["ao"] > 0:
        return "LONG"
    if prev["ao"] >= 0 and row["ao"] < 0:
        return "SHORT"
    return None


def strategy_cci_extreme(df, i):
    row = df.iloc[i]
    if pd.isna(row["cci"]):
        return None
    if row["cci"] <= -150:
        return "LONG"
    if row["cci"] >= 150:
        return "SHORT"
    return None


def strategy_williams_r_extreme(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["williams_r"]) or pd.isna(prev["williams_r"]):
        return None
    if prev["williams_r"] <= -90 and row["williams_r"] > -90:
        return "LONG"
    if prev["williams_r"] >= -10 and row["williams_r"] < -10:
        return "SHORT"
    return None


def strategy_real_funding_extreme(df, i):
    row = df.iloc[i]
    if "fundingRate" not in df.columns or pd.isna(row.get("fundingRate")):
        return None
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 1.5:
        return None
    if row["fundingRate"] <= -0.001 and row["is_bull"]:
        return "LONG"
    if row["fundingRate"] >= 0.001 and not row["is_bull"]:
        return "SHORT"
    return None


def strategy_daily_breakout(df, i):
    row = df.iloc[i]
    if pd.isna(row["prev_day_high"]) or pd.isna(row["prev_day_low"]):
        return None
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 1.8:
        return None
    if row["close"] > row["prev_day_high"] and row["is_bull"]:
        return "LONG"
    if row["close"] < row["prev_day_low"] and not row["is_bull"]:
        return "SHORT"
    return None


def strategy_renko_trend(df, i):
    row = df.iloc[i]
    if pd.isna(row["renko_direction"]):
        return None
    if row["renko_direction"] >= 3:
        return "LONG"
    if row["renko_direction"] <= -3:
        return "SHORT"
    return None


def strategy_ratio_mean_reversion(df, i):
    row = df.iloc[i]
    if pd.isna(row.get("ratio_zscore")):
        return None
    if row["ratio_zscore"] <= -2.0 and row["is_bull"]:
        return "LONG"
    if row["ratio_zscore"] >= 2.0 and not row["is_bull"]:
        return "SHORT"
    return None


STRATEGIES = {
    "Bill Williams Alligator": strategy_alligator,
    "Awesome Oscillator": strategy_awesome_oscillator,
    "CCI Asiri Uc": strategy_cci_extreme,
    "Williams %R Asiri Uc": strategy_williams_r_extreme,
    "Gercek Funding Asiri Uc": strategy_real_funding_extreme,
    "Gunluk Kirilim": strategy_daily_breakout,
    "Renko Trend": strategy_renko_trend,
    "Coin/BTC Orani Sapmasi": strategy_ratio_mean_reversion,
}


def evaluate_signals(df, strategy_fn):
    total, wins = 0, 0
    net_pct_list = []
    max_check = max(CHECK_CANDLES)
    start_idx = 100

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
    print(f"Turnuva basliyor (6. tur): {len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} gunluk veri, {len(STRATEGIES)} strateji")
    send_telegram_message(
        f"🏆 Strateji turnuvası (6. tur - Alligator, AO, CCI, Williams %R, gerçek funding, Renko vb.) başladı.\n"
        f"{len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} günlük veri, {len(STRATEGIES)} strateji.\n"
        f"Bitince göndereceğim."
    )

    print("BTC verisi cekiliyor (oran karsilastirmasi icin)...")
    btc_df = fetch_historical_ohlcv(BTC_SYMBOL)

    results = {name: {"total": 0, "wins": 0, "net_pct_sum": 0.0} for name in STRATEGIES}

    for idx, symbol in enumerate(TOURNAMENT_WATCHLIST):
        print(f"[{idx+1}/{len(TOURNAMENT_WATCHLIST)}] {symbol} verisi cekiliyor...")
        try:
            df = fetch_historical_ohlcv(symbol)
            if df.empty or len(df) < 150:
                print(f"  {symbol}: yetersiz veri, atlaniyor")
                continue
            funding_df = fetch_funding_history(symbol)
            df = compute_all_indicators(df, btc_df=btc_df, funding_df=funding_df)
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

    lines = [f"🏆 TURNUVA SONUÇLARI - 6. TUR ({LOOKBACK_DAYS} günlük veri, {len(TOURNAMENT_WATCHLIST)} coin)"]
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

    with open("tournament_results_v6.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strateji", "toplam_sinyal", "dogru", "isabet_yuzde", "ort_net_yuzde", "toplam_net_yuzde"])
        for name, total, wins, win_rate, avg_net, total_net in leaderboard:
            writer.writerow([name, total, wins, f"{win_rate:.2f}", f"{avg_net:.4f}", f"{total_net:.2f}"])

    print("\nTurnuva tamamlandi.")


if __name__ == "__main__":
    run_tournament()
