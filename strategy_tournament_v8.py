import os
import csv
import time
import random
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
RSI_PERIOD = 14

CHECK_CANDLES = [1, 2, 3]
SUCCESS_THRESHOLD_PCT = 0.15
MIN_SIGNALS_TO_RANK = 15
COMMISSION_ROUNDTRIP_PCT = 0.12

exchange = ccxt.okx({
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
})

random.seed(42)


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
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").astype("datetime64[ns]")
    return df


def compute_all_indicators(df: pd.DataFrame, btc_df: pd.DataFrame = None) -> pd.DataFrame:
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
    df["realized_vol"] = df["close"].pct_change().rolling(20).std() * 100
    df["vol_percentile"] = df["realized_vol"].rolling(200).rank(pct=True)

    df["body"] = (df["close"] - df["open"]).abs()
    df["is_bull"] = df["close"] > df["open"]

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_ratio"] = df["lower_wick_ratio"].fillna(0)
    df["upper_wick_ratio"] = df["upper_wick_ratio"].fillna(0)

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    df["macd_line"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()

    boll_mid = df["close"].rolling(20).mean()
    boll_std = df["close"].rolling(20).std()
    df["boll_upper"] = boll_mid + 2 * boll_std
    df["boll_lower"] = boll_mid - 2 * boll_std

    z_mean = df["close"].rolling(20).mean()
    z_std = df["close"].rolling(20).std()
    df["zscore"] = (df["close"] - z_mean) / z_std.replace(0, np.nan)

    vwap_typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = vwap_typical * df["volume"]
    df["vwap"] = pv.rolling(96).sum() / df["volume"].rolling(96).sum()
    df["vwap_dev_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100

    df["date"] = df["timestamp"].dt.date
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["time_of_day"] = df["timestamp"].dt.time

    # Acilis araligi (UTC gunun ilk 1 saati = ilk 4 mum)
    df["_minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    opening_mask = df["_minute_of_day"] < 60
    opening_high = df[opening_mask].groupby("date")["high"].max()
    opening_low = df[opening_mask].groupby("date")["low"].min()
    df["opening_range_high"] = df["date"].map(opening_high)
    df["opening_range_low"] = df["date"].map(opening_low)

    if btc_df is not None:
        btc_small = btc_df[["timestamp", "close"]].rename(columns={"close": "btc_close"})
        btc_small = btc_small.copy()
        btc_small["timestamp"] = btc_small["timestamp"].astype("datetime64[ns]")
        df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
        df = pd.merge_asof(df.sort_values("timestamp"), btc_small.sort_values("timestamp"),
                            on="timestamp", direction="backward")
        df["coin_ret"] = df["close"].pct_change()
        df["btc_ret"] = df["btc_close"].pct_change()
        df["rolling_corr_short"] = df["coin_ret"].rolling(20).corr(df["btc_ret"])
        df["rolling_corr_long"] = df["coin_ret"].rolling(100).corr(df["btc_ret"])
    else:
        df["rolling_corr_short"] = np.nan
        df["rolling_corr_long"] = np.nan

    return df


# ---------------------------------------------------------------------------
# Alt sinyaller (topluluk/ensemble icin yapi taslari)
# ---------------------------------------------------------------------------

def _sub_rsi_extreme(row):
    if pd.isna(row["rsi"]):
        return None
    if row["rsi"] <= 25:
        return "LONG"
    if row["rsi"] >= 75:
        return "SHORT"
    return None


def _sub_macd(row):
    if pd.isna(row["macd_line"]):
        return None
    if row["macd_line"] > row["macd_signal"]:
        return "LONG"
    if row["macd_line"] < row["macd_signal"]:
        return "SHORT"
    return None


def _sub_bollinger(row):
    if pd.isna(row["boll_upper"]):
        return None
    if row["close"] <= row["boll_lower"]:
        return "LONG"
    if row["close"] >= row["boll_upper"]:
        return "SHORT"
    return None


def _sub_ema_trend(row):
    if pd.isna(row["ema20"]) or pd.isna(row["ema50"]):
        return None
    if row["ema20"] > row["ema50"]:
        return "LONG"
    if row["ema20"] < row["ema50"]:
        return "SHORT"
    return None


def _sub_zscore(row):
    if pd.isna(row["zscore"]):
        return None
    if row["zscore"] <= -2.0:
        return "LONG"
    if row["zscore"] >= 2.0:
        return "SHORT"
    return None


def _sub_wick(row):
    if row["lower_wick_ratio"] >= 0.5:
        return "LONG"
    if row["upper_wick_ratio"] >= 0.5:
        return "SHORT"
    return None


SUB_SIGNALS = [_sub_rsi_extreme, _sub_macd, _sub_bollinger, _sub_ema_trend, _sub_zscore, _sub_wick]


# ---------------------------------------------------------------------------
# 8 YENI strateji (Round 8) - coklu sinyal / topluluk temelli
# ---------------------------------------------------------------------------

def strategy_majority_vote(df, i):
    row = df.iloc[i]
    votes = [fn(row) for fn in SUB_SIGNALS]
    votes = [v for v in votes if v is not None]
    if len(votes) < 4:
        return None
    long_votes = votes.count("LONG")
    short_votes = votes.count("SHORT")
    if long_votes >= 4:
        return "LONG"
    if short_votes >= 4:
        return "SHORT"
    return None


def strategy_day_of_week_breakout(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    # Hafta ortasi (Sal-Per) genelde daha yuksek likidite - sadece o gunlerde ara
    if row["day_of_week"] not in [1, 2, 3]:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 2.0:
        return None
    if row["is_bull"] and row["body"] / row["atr14"] >= 0.5:
        return "LONG"
    if not row["is_bull"] and row["body"] / row["atr14"] >= 0.5:
        return "SHORT"
    return None


def _make_random_rule(seed):
    rng = random.Random(seed)
    feature = rng.choice(["rsi", "zscore", "vwap_dev_pct", "vol_percentile"])
    long_threshold_pct = rng.uniform(0.05, 0.25)
    short_threshold_pct = rng.uniform(0.75, 0.95)

    def rule(df_window, row):
        val = row.get(feature)
        if pd.isna(val):
            return None
        series = df_window[feature].dropna()
        if len(series) < 20:
            return None
        low_cut = series.quantile(long_threshold_pct)
        high_cut = series.quantile(short_threshold_pct)
        if val <= low_cut:
            return "LONG"
        if val >= high_cut:
            return "SHORT"
        return None

    return rule


_BAGGING_RULES = [_make_random_rule(s) for s in range(5)]


def strategy_bagging_ensemble(df, i):
    if i < 60:
        return None
    window = df.iloc[max(0, i - 200):i]
    row = df.iloc[i]
    votes = [rule(window, row) for rule in _BAGGING_RULES]
    votes = [v for v in votes if v is not None]
    if len(votes) < 3:
        return None
    long_votes = votes.count("LONG")
    short_votes = votes.count("SHORT")
    if long_votes >= 3:
        return "LONG"
    if short_votes >= 3:
        return "SHORT"
    return None


def strategy_opening_range_breakout(df, i):
    row = df.iloc[i]
    if pd.isna(row["opening_range_high"]) or pd.isna(row["opening_range_low"]):
        return None
    if row["_minute_of_day"] < 75:  # acilis penceresinin hemen disinda islem yapma
        return None
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 1.8:
        return None
    if row["close"] > row["opening_range_high"] and row["is_bull"]:
        return "LONG"
    if row["close"] < row["opening_range_low"] and not row["is_bull"]:
        return "SHORT"
    return None


def strategy_correlation_breakdown(df, i):
    row = df.iloc[i]
    if pd.isna(row["rolling_corr_short"]) or pd.isna(row["rolling_corr_long"]):
        return None
    breakdown = (row["rolling_corr_long"] - row["rolling_corr_short"]) >= 0.4
    if not breakdown:
        return None
    if row["is_bull"]:
        return "LONG"
    return "SHORT"


def strategy_rsi_vwap_confluence(df, i):
    row = df.iloc[i]
    if pd.isna(row["rsi"]) or pd.isna(row["vwap_dev_pct"]):
        return None
    if row["rsi"] <= 30 and row["vwap_dev_pct"] <= -2.0:
        return "LONG"
    if row["rsi"] >= 70 and row["vwap_dev_pct"] >= 2.0:
        return "SHORT"
    return None


def strategy_adaptive_volatility_threshold(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_percentile"]) or pd.isna(row["rsi"]):
        return None
    # Volatilite dusukken (sakin piyasa) daha hassas esik, yuksekken daha sıkı esik kullan
    if row["vol_percentile"] <= 0.3:
        rsi_low, rsi_high = 35, 65
    else:
        rsi_low, rsi_high = 20, 80

    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 1.8:
        return None

    if row["rsi"] <= rsi_low and row["lower_wick_ratio"] >= 0.4:
        return "LONG"
    if row["rsi"] >= rsi_high and row["upper_wick_ratio"] >= 0.4:
        return "SHORT"
    return None


def strategy_two_of_three_reversion(df, i):
    row = df.iloc[i]
    checks_long = 0
    checks_short = 0
    if not pd.isna(row["rsi"]):
        if row["rsi"] <= 25:
            checks_long += 1
        if row["rsi"] >= 75:
            checks_short += 1
    if not pd.isna(row["zscore"]):
        if row["zscore"] <= -2.0:
            checks_long += 1
        if row["zscore"] >= 2.0:
            checks_short += 1
    if row["lower_wick_ratio"] >= 0.5:
        checks_long += 1
    if row["upper_wick_ratio"] >= 0.5:
        checks_short += 1

    if checks_long >= 2:
        return "LONG"
    if checks_short >= 2:
        return "SHORT"
    return None


STRATEGIES = {
    "Cogunluk Oyu Toplulugu": strategy_majority_vote,
    "Haftanin Gunune Gore Kirilim": strategy_day_of_week_breakout,
    "Rastgele Kural Harmanlamasi": strategy_bagging_ensemble,
    "Acilis Araligi Kirilimi": strategy_opening_range_breakout,
    "Korelasyon Kopmasi": strategy_correlation_breakdown,
    "RSI+VWAP Birlesik Teyit": strategy_rsi_vwap_confluence,
    "Uyarlanabilir Volatilite Esigi": strategy_adaptive_volatility_threshold,
    "Iki-Ucte-Iki Tersine Donus": strategy_two_of_three_reversion,
}


def evaluate_signals(df, strategy_fn):
    total, wins = 0, 0
    net_pct_list = []
    max_check = max(CHECK_CANDLES)
    start_idx = 210

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
    print(f"Turnuva basliyor (8. tur): {len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} gunluk veri")
    send_telegram_message(
        f"🏆 Strateji turnuvası (8. tur - Topluluk/Ensemble sistemleri) başladı.\n"
        f"{len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} günlük veri.\n"
        f"Bitince göndereceğim."
    )

    print("BTC verisi cekiliyor (korelasyon icin)...")
    btc_df = fetch_historical_ohlcv(BTC_SYMBOL)

    results = {name: {"total": 0, "wins": 0, "net_pct_sum": 0.0} for name in STRATEGIES}

    for idx, symbol in enumerate(TOURNAMENT_WATCHLIST):
        print(f"[{idx+1}/{len(TOURNAMENT_WATCHLIST)}] {symbol} verisi cekiliyor...")
        try:
            df = fetch_historical_ohlcv(symbol)
            if df.empty or len(df) < 250:
                print(f"  {symbol}: yetersiz veri, atlaniyor")
                continue
            df = compute_all_indicators(df, btc_df=btc_df)
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

    lines = [f"🏆 TURNUVA SONUÇLARI - 8. TUR ({LOOKBACK_DAYS} günlük veri, {len(TOURNAMENT_WATCHLIST)} coin)"]
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

    with open("tournament_results_v8.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strateji", "toplam_sinyal", "dogru", "isabet_yuzde", "ort_net_yuzde", "toplam_net_yuzde"])
        for name, total, wins, win_rate, avg_net, total_net in leaderboard:
            writer.writerow([name, total, wins, f"{win_rate:.2f}", f"{avg_net:.4f}", f"{total_net:.2f}"])

    print("\nTurnuva tamamlandi.")


if __name__ == "__main__":
    run_tournament()
