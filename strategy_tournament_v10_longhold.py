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

# ---------------------------------------------------------------------------
# 10. TUR: 9 turda test edilen 66 stratejiden 60'i (ML ve cross-sectional
# haric - farkli bir test yapisi gerektiriyorlardi), bu sefer UZUN tutus
# sureleriyle: 1sa / 4sa / 12sa / 24sa, buyuyen hedeflerle (%0.3/%0.6/%1.0/%1.5)
# ---------------------------------------------------------------------------

TOURNAMENT_COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "SUI",
    "DOT", "TRX", "ATOM", "NEAR", "TON", "LTC", "BCH", "ETC", "FIL", "APT",
    "ARB", "OP", "INJ", "SEI", "ICP", "HBAR", "PEPE", "WIF", "ENA", "TIA",
]
TOURNAMENT_WATCHLIST = [f"{c}/USDT:USDT" for c in TOURNAMENT_COINS]
BTC_SYMBOL = "BTC/USDT:USDT"
ETH_SYMBOL = "ETH/USDT:USDT"

TIMEFRAME = "15m"
LOOKBACK_DAYS = 30
ATR_PERIOD = 14
RSI_PERIOD = 14

CHECK_POINTS = [(4, 0.3), (16, 0.6), (48, 1.0), (96, 1.5)]
MIN_SIGNALS_TO_RANK = 15
COMMISSION_ROUNDTRIP_PCT = 0.12

HURST_WINDOW = 100
SR_LOOKBACK = 150
SR_TOUCH_TOLERANCE_ATR = 0.3
SR_MIN_TOUCHES = 3
VOLUME_PROFILE_WINDOW = 96
VOLUME_PROFILE_BUCKETS = 24
FIB_SWING_LOOKBACK = 30
PIVOT_WINDOW = 96
VWAP_WINDOW = 96
ZSCORE_PERIOD = 20
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
SQUEEZE_LOOKBACK = 50

random.seed(42)

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
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").astype("datetime64[ns]")
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
        fr["timestamp"] = pd.to_datetime(fr["timestamp"], unit="ms").astype("datetime64[ns]")
        fr = fr[["timestamp", "fundingRate"]].dropna().sort_values("timestamp")
        return fr
    except Exception:
        return None


def fetch_oi_history(symbol: str, days: int = LOOKBACK_DAYS):
    try:
        since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        all_oi = []
        cursor = since
        while True:
            batch = exchange.fetch_open_interest_history(symbol, timeframe="1h", since=cursor, limit=100)
            if not batch:
                break
            all_oi.extend(batch)
            last_ts = batch[-1]["timestamp"]
            if last_ts <= cursor or len(batch) < 100:
                break
            cursor = last_ts + 1
            time.sleep(0.15)
        if not all_oi:
            return None
        oi = pd.DataFrame(all_oi)
        oi["timestamp"] = pd.to_datetime(oi["timestamp"], unit="ms").astype("datetime64[ns]")
        value_col = "openInterestAmount" if "openInterestAmount" in oi.columns else "openInterestValue"
        if value_col not in oi.columns:
            return None
        oi = oi[["timestamp", value_col]].rename(columns={value_col: "oi_value"}).dropna().sort_values("timestamp")
        return oi
    except Exception:
        return None


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
            consecutive = consecutive + bricks if trend >= 0 else bricks
            trend = 1
        elif diff <= -brick_size:
            bricks = int(-diff // brick_size)
            last_brick_price -= bricks * brick_size
            consecutive = consecutive + bricks if trend <= 0 else bricks
            trend = -1
        direction[i] = trend * consecutive
    return direction


def rolling_hurst(prices: np.ndarray, window: int) -> np.ndarray:
    n = len(prices)
    hurst = np.full(n, np.nan)
    for i in range(window, n):
        segment = prices[i - window:i]
        returns = np.diff(np.log(segment + 1e-9))
        if len(returns) < 10 or np.std(returns) == 0:
            continue
        mean_r = np.mean(returns)
        deviations = np.cumsum(returns - mean_r)
        R = np.max(deviations) - np.min(deviations)
        S = np.std(returns)
        if S == 0:
            continue
        rs = R / S
        if rs > 0:
            hurst[i] = np.log(rs) / np.log(len(returns))
    return hurst


def kalman_filter_trend(prices: np.ndarray, process_var=1e-4, measurement_var=1e-2) -> np.ndarray:
    n = len(prices)
    estimate = np.zeros(n)
    if n == 0:
        return estimate
    estimate[0] = prices[0]
    error = 1.0
    for i in range(1, n):
        pred = estimate[i - 1]
        pred_error = error + process_var
        kalman_gain = pred_error / (pred_error + measurement_var)
        estimate[i] = pred + kalman_gain * (prices[i] - pred)
        error = (1 - kalman_gain) * pred_error
    return estimate


def rolling_autocorr(returns: pd.Series, window: int, lag: int = 1) -> pd.Series:
    def _ac(x):
        if len(x) < lag + 5 or np.std(x) == 0:
            return np.nan
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]
    return returns.rolling(window).apply(_ac, raw=True)


def compute_ou_halflife(prices: pd.Series, window: int = 50) -> pd.Series:
    log_prices = np.log(prices.replace(0, np.nan))
    halflife = pd.Series(np.nan, index=prices.index)
    for i in range(window, len(prices)):
        segment = log_prices.iloc[i - window:i].values
        y = np.diff(segment)
        x = segment[:-1] - np.mean(segment[:-1])
        if np.std(x) == 0 or len(x) < 10:
            continue
        beta = np.sum(x * y) / np.sum(x * x)
        if beta >= 0:
            continue
        hl = -np.log(2) / beta
        if 0 < hl < 500:
            halflife.iloc[i] = hl
    return halflife


def find_sr_levels(df, i, lookback=SR_LOOKBACK, atr_tolerance=SR_TOUCH_TOLERANCE_ATR, min_touches=SR_MIN_TOUCHES):
    if i < lookback:
        return None
    row = df.iloc[i]
    atr = row["atr14"]
    if pd.isna(atr) or atr == 0:
        return None
    segment = df.iloc[i - lookback:i]
    candidate_prices = np.concatenate([segment["high"].values, segment["low"].values])
    tolerance = atr * atr_tolerance
    candidate_prices = np.sort(candidate_prices)
    best_level, best_touches = None, 0
    step = max(1, len(candidate_prices) // 40)
    for p in candidate_prices[::step]:
        touches = np.sum(np.abs(candidate_prices - p) <= tolerance)
        if touches > best_touches:
            best_touches = touches
            best_level = p
    if best_touches >= min_touches:
        return best_level
    return None


def get_volume_profile_poc(df, i, window=VOLUME_PROFILE_WINDOW, buckets=VOLUME_PROFILE_BUCKETS):
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


def _make_random_rule(seed):
    rng = random.Random(seed)
    feature = rng.choice(["rsi", "zscore", "vwap_dev_pct", "vol_percentile"])
    long_q = rng.uniform(0.05, 0.25)
    short_q = rng.uniform(0.75, 0.95)

    def rule(df_window, row):
        val = row.get(feature)
        if pd.isna(val):
            return None
        series = df_window[feature].dropna()
        if len(series) < 20:
            return None
        low_cut = series.quantile(long_q)
        high_cut = series.quantile(short_q)
        if val <= low_cut:
            return "LONG"
        if val >= high_cut:
            return "SHORT"
        return None

    return rule


_BAGGING_RULES = [_make_random_rule(s) for s in range(5)]


def compute_all_indicators(df: pd.DataFrame, btc_df=None, eth_df=None, funding_df=None, oi_df=None) -> pd.DataFrame:
    df = df.copy()
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_std20"] = df["volume"].rolling(20).std()

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
    df["lower_wick_ratio"] = ((df[["open", "close"]].min(axis=1) - df["low"]) / candle_range).fillna(0)
    df["upper_wick_ratio"] = ((df["high"] - df[["open", "close"]].max(axis=1)) / candle_range).fillna(0)
    df["close_position"] = ((df["close"] - df["low"]) / candle_range).fillna(0.5)

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    df["macd_line"] = df["ema12"] - df["ema26"]
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df["stoch_k"] = ((df["close"] - low14) / (high14 - low14).replace(0, np.nan)) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    df["keltner_upper"] = df["ema20"] + df["atr14"] * 1.5
    df["keltner_lower"] = df["ema20"] - df["atr14"] * 1.5

    boll_mid = df["close"].rolling(BOLLINGER_PERIOD).mean()
    boll_std = df["close"].rolling(BOLLINGER_PERIOD).std()
    df["boll_upper"] = boll_mid + BOLLINGER_STD * boll_std
    df["boll_lower"] = boll_mid - BOLLINGER_STD * boll_std
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / boll_mid * 100
    df["boll_width_percentile"] = df["boll_width"].rolling(SQUEEZE_LOOKBACK).rank(pct=True)

    z_mean = df["close"].rolling(ZSCORE_PERIOD).mean()
    z_std = df["close"].rolling(ZSCORE_PERIOD).std()
    df["zscore"] = (df["close"] - z_mean) / z_std.replace(0, np.nan)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]
    df["vwap"] = pv.rolling(VWAP_WINDOW).sum() / df["volume"].rolling(VWAP_WINDOW).sum()
    df["vwap_dev_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100
    df["vwma20"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()

    df["breakout_high"] = df["close"].shift(1).rolling(20).max()
    df["breakout_low"] = df["close"].shift(1).rolling(20).min()

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

    df["sar"] = compute_parabolic_sar(df)

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

    period_high = df["high"].rolling(PIVOT_WINDOW).max().shift(1)
    period_low = df["low"].rolling(PIVOT_WINDOW).min().shift(1)
    period_close = df["close"].shift(PIVOT_WINDOW)
    pivot = (period_high + period_low + period_close) / 3
    df["pivot_r1"] = 2 * pivot - period_low
    df["pivot_s1"] = 2 * pivot - period_high

    highest_22 = df["high"].rolling(22).max()
    lowest_22 = df["low"].rolling(22).min()
    df["chandelier_long"] = highest_22 - df["atr14"] * 3
    df["chandelier_short"] = lowest_22 + df["atr14"] * 3

    df["jaw"] = df["close"].rolling(13).mean().shift(8)
    df["teeth"] = df["close"].rolling(8).mean().shift(5)
    df["lips"] = df["close"].rolling(5).mean().shift(3)

    median_price = (df["high"] + df["low"]) / 2
    df["ao"] = median_price.rolling(5).mean() - median_price.rolling(34).mean()

    typical2 = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical2.rolling(20).mean()
    mean_dev = typical2.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = (typical2 - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))

    df["williams_r"] = (high14 - df["close"]) / (high14 - low14).replace(0, np.nan) * -100

    df["date"] = df["timestamp"].dt.date
    daily_high_by_date = df.groupby("date")["high"].max()
    daily_low_by_date = df.groupby("date")["low"].min()
    df["prev_day_high"] = df["date"].map(daily_high_by_date.shift(1))
    df["prev_day_low"] = df["date"].map(daily_low_by_date.shift(1))
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["_minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    opening_mask = df["_minute_of_day"] < 60
    opening_high = df[opening_mask].groupby("date")["high"].max()
    opening_low = df[opening_mask].groupby("date")["low"].min()
    df["opening_range_high"] = df["date"].map(opening_high)
    df["opening_range_low"] = df["date"].map(opening_low)

    df["renko_direction"] = compute_renko_direction(df)
    df["hurst"] = rolling_hurst(df["close"].values, HURST_WINDOW)
    df["kalman_trend"] = kalman_filter_trend(df["close"].values)
    returns = df["close"].pct_change()
    df["autocorr"] = rolling_autocorr(returns, window=30, lag=1)
    roc = df["close"].pct_change(5) * 100
    df["momentum_accel"] = roc.diff(3)
    df["ou_halflife"] = compute_ou_halflife(df["close"])
    vol_above_avg = df["realized_vol"] > df["realized_vol"].rolling(50).mean()
    df["vol_clustering"] = vol_above_avg.rolling(3).sum()

    if btc_df is not None:
        btc_small = btc_df[["timestamp", "close"]].rename(columns={"close": "btc_close"}).copy()
        btc_small["timestamp"] = btc_small["timestamp"].astype("datetime64[ns]")
        df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
        df = pd.merge_asof(df.sort_values("timestamp"), btc_small.sort_values("timestamp"),
                            on="timestamp", direction="backward")
        df["ratio"] = df["close"] / df["btc_close"]
        ratio_mean = df["ratio"].rolling(96).mean()
        ratio_std = df["ratio"].rolling(96).std()
        df["ratio_zscore"] = (df["ratio"] - ratio_mean) / ratio_std.replace(0, np.nan)
        df["coin_ret"] = df["close"].pct_change()
        df["btc_ret"] = df["btc_close"].pct_change()
        df["rolling_corr_short"] = df["coin_ret"].rolling(20).corr(df["btc_ret"])
        df["rolling_corr_long"] = df["coin_ret"].rolling(100).corr(df["btc_ret"])
        df["btc_is_bull"] = df["btc_close"].diff() > 0
        df["return_8"] = df["close"].pct_change(8) * 100
    else:
        df["ratio_zscore"] = np.nan
        df["rolling_corr_short"] = np.nan
        df["rolling_corr_long"] = np.nan
        df["btc_is_bull"] = np.nan

    if eth_df is not None and "is_bull" in eth_df.columns:
        eth_small = eth_df[["timestamp", "is_bull"]].rename(columns={"is_bull": "eth_is_bull"}).copy()
        eth_small["timestamp"] = eth_small["timestamp"].astype("datetime64[ns]")
        df = pd.merge_asof(df.sort_values("timestamp"), eth_small.sort_values("timestamp"),
                            on="timestamp", direction="backward")
    else:
        df["eth_is_bull"] = np.nan

    if funding_df is not None and not funding_df.empty:
        funding_df = funding_df.copy()
        funding_df["timestamp"] = funding_df["timestamp"].astype("datetime64[ns]")
        df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
        df = pd.merge_asof(df.sort_values("timestamp"), funding_df.sort_values("timestamp"),
                            on="timestamp", direction="backward")
    else:
        df["fundingRate"] = np.nan

    if oi_df is not None and not oi_df.empty:
        oi_df = oi_df.copy()
        oi_df["timestamp"] = oi_df["timestamp"].astype("datetime64[ns]")
        df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
        df = pd.merge_asof(df.sort_values("timestamp"), oi_df.sort_values("timestamp"),
                            on="timestamp", direction="backward")
        df["oi_change_pct"] = df["oi_value"].pct_change(4) * 100
    else:
        df["oi_value"] = np.nan
        df["oi_change_pct"] = np.nan

    return df


def s01_mean_reversion(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 2.5:
        return None
    if row["lower_wick_ratio"] >= 0.5 and row["rsi"] <= 20:
        return "LONG"
    if row["upper_wick_ratio"] >= 0.5 and row["rsi"] >= 80:
        return "SHORT"
    return None


def s02_momentum_breakout(df, i):
    row = df.iloc[i]
    if pd.isna(row["breakout_high"]) or pd.isna(row["atr14"]) or row["atr14"] == 0:
        return None
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 2.0:
        return None
    if row["body"] / row["atr14"] < 0.6:
        return None
    if row["close"] > row["breakout_high"] and row["close_position"] >= 0.65 and 50 <= row["rsi"] <= 78:
        return "LONG"
    if row["close"] < row["breakout_low"] and row["close_position"] <= 0.35 and 22 <= row["rsi"] <= 50:
        return "SHORT"
    return None


def s03_ema_cross(df, i):
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


def s04_bollinger_breakout(df, i):
    row = df.iloc[i]
    if pd.isna(row["boll_upper"]) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 1.8:
        return None
    if row["close"] >= row["boll_upper"] and row["is_bull"]:
        return "LONG"
    if row["close"] <= row["boll_lower"] and not row["is_bull"]:
        return "SHORT"
    return None


def s05_volume_spike(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 3.5:
        return None
    return "LONG" if row["is_bull"] else "SHORT"


def s06_rsi_midline(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if prev["rsi"] < 50 and row["rsi"] >= 53:
        return "LONG"
    if prev["rsi"] > 50 and row["rsi"] <= 47:
        return "SHORT"
    return None


def s07_donchian_simple(df, i):
    row = df.iloc[i]
    if pd.isna(row["breakout_high"]) or pd.isna(row["breakout_low"]):
        return None
    if row["close"] > row["breakout_high"]:
        return "LONG"
    if row["close"] < row["breakout_low"]:
        return "SHORT"
    return None


def s08_confluence(df, i):
    a, b = s01_mean_reversion(df, i), s02_momentum_breakout(df, i)
    return a if (a is not None and a == b) else None


def s09_breakout_retest(df, i):
    if i < 6:
        return None
    row = df.iloc[i]
    if pd.isna(row["breakout_high"]) or pd.isna(row["breakout_low"]):
        return None
    window = df.iloc[i - 5:i]
    if (window["close"] > window["breakout_high"]).any():
        level = window["breakout_high"].iloc[-1]
        if (window["low"] <= level * 1.003).any() and row["close"] > level and row["is_bull"]:
            return "LONG"
    if (window["close"] < window["breakout_low"]).any():
        level = window["breakout_low"].iloc[-1]
        if (window["high"] >= level * 0.997).any() and row["close"] < level and not row["is_bull"]:
            return "SHORT"
    return None


def s10_volatility_squeeze(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(prev["boll_width_percentile"]) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if prev["boll_width_percentile"] > 0.2 or row["volume"] / row["vol_sma20"] < 2.0:
        return None
    if row["is_bull"] and row["close"] > row["boll_upper"]:
        return "LONG"
    if not row["is_bull"] and row["close"] < row["boll_lower"]:
        return "SHORT"
    return None


def s11_triple_timeframe(df, i):
    row = df.iloc[i]
    if pd.isna(row["ema20"]) or pd.isna(row["ema50"]) or pd.isna(row["rsi"]):
        return None
    st_bull = row["is_bull"] and row["close_position"] >= 0.6
    st_bear = (not row["is_bull"]) and row["close_position"] <= 0.4
    mt_bull, mt_bear = row["ema20"] > row["ema50"], row["ema20"] < row["ema50"]
    lt_bull, lt_bear = row["close"] > row["ema50"] * 1.01, row["close"] < row["ema50"] * 0.99
    if st_bull and mt_bull and lt_bull and row["rsi"] < 75:
        return "LONG"
    if st_bear and mt_bear and lt_bear and row["rsi"] > 25:
        return "SHORT"
    return None


def s12_relative_strength(df, i):
    row = df.iloc[i]
    if pd.isna(row.get("return_8")):
        return None
    if row["return_8"] >= 4.0 and row["is_bull"]:
        return "LONG"
    if row["return_8"] <= -4.0 and not row["is_bull"]:
        return "SHORT"
    return None


def s13_pullback_in_trend(df, i):
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


def s14_obv_confirmation(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 1.5:
        return None
    if row["is_bull"] and row["close_position"] >= 0.6:
        return "LONG"
    if not row["is_bull"] and row["close_position"] <= 0.4:
        return "SHORT"
    return None


def s15_zscore_reversion(df, i):
    row = df.iloc[i]
    if pd.isna(row["zscore"]):
        return None
    if row["zscore"] <= -2.2:
        return "LONG"
    if row["zscore"] >= 2.2:
        return "SHORT"
    return None


def s16_three_candle_momentum(df, i):
    if i < 3:
        return None
    c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    if pd.isna(c1["atr14"]) or c1["atr14"] == 0:
        return None
    all_bull = c1["is_bull"] and c2["is_bull"] and c3["is_bull"]
    all_bear = (not c1["is_bull"]) and (not c2["is_bull"]) and (not c3["is_bull"])
    rising_vol = c3["volume"] > c2["volume"] > c1["volume"]
    decent = all(c["body"] >= c["atr14"] * 0.3 for c in [c1, c2, c3])
    if all_bull and rising_vol and decent:
        return "LONG"
    if all_bear and rising_vol and decent:
        return "SHORT"
    return None


def s17_macd_cross(df, i):
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


def s18_stochastic(df, i):
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


def s19_keltner_breakout(df, i):
    row = df.iloc[i]
    if pd.isna(row["keltner_upper"]) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 1.8:
        return None
    if row["close"] > row["keltner_upper"] and row["is_bull"]:
        return "LONG"
    if row["close"] < row["keltner_lower"] and not row["is_bull"]:
        return "SHORT"
    return None


def s20_vwap_deviation(df, i):
    row = df.iloc[i]
    if pd.isna(row["vwap_dev_pct"]):
        return None
    if row["vwap_dev_pct"] <= -3.0:
        return "LONG"
    if row["vwap_dev_pct"] >= 3.0:
        return "SHORT"
    return None


def s21_fibonacci_bounce(df, i):
    if i < FIB_SWING_LOOKBACK:
        return None
    window = df.iloc[i - FIB_SWING_LOOKBACK:i]
    swing_high, swing_low = window["high"].max(), window["low"].min()
    swing_range = swing_high - swing_low
    if swing_range <= 0:
        return None
    row = df.iloc[i]
    fib_50 = swing_high - swing_range * 0.5
    fib_618 = swing_high - swing_range * 0.618
    uptrend_swing = window["high"].idxmax() > window["low"].idxmin()
    if uptrend_swing and fib_618 <= row["low"] <= fib_50 and row["is_bull"] and row["close"] > fib_50:
        return "LONG"
    if (not uptrend_swing) and fib_50 <= row["high"] <= fib_618 and (not row["is_bull"]) and row["close"] < fib_50:
        return "SHORT"
    return None


def s22_market_structure_break(df, i):
    if i < 30:
        return None
    row = df.iloc[i]
    window = df.iloc[i - 30:i]
    if row["close"] > window["high"].max() and row["is_bull"]:
        return "LONG"
    if row["close"] < window["low"].min() and not row["is_bull"]:
        return "SHORT"
    return None


def s23_rsi_divergence(df, i):
    if i < 20:
        return None
    row = df.iloc[i]
    window = df.iloc[i - 20:i]
    if pd.isna(row["rsi"]):
        return None
    price_new_low = row["close"] <= window["close"].min()
    price_new_high = row["close"] >= window["close"].max()
    rsi_higher = row["rsi"] > window["rsi"].min() + 5
    rsi_lower = row["rsi"] < window["rsi"].max() - 5
    if price_new_low and rsi_higher and row["is_bull"]:
        return "LONG"
    if price_new_high and rsi_lower and not row["is_bull"]:
        return "SHORT"
    return None


def s24_engulfing(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 1.3:
        return None
    bull_engulf = (not prev["is_bull"]) and row["is_bull"] and row["open"] <= prev["close"] and row["close"] >= prev["open"]
    bear_engulf = prev["is_bull"] and (not row["is_bull"]) and row["open"] >= prev["close"] and row["close"] <= prev["open"]
    if bull_engulf:
        return "LONG"
    if bear_engulf:
        return "SHORT"
    return None


def s27_regime_switching(df, i):
    row = df.iloc[i]
    if pd.isna(row["ema20"]) or pd.isna(row["ema50"]) or pd.isna(row["atr14"]) or row["atr14"] == 0:
        return None
    trend_strength = abs(row["ema20"] - row["ema50"]) / row["atr14"]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if trend_strength >= 1.2:
        if vol_ratio < 1.8 or row["body"] / row["atr14"] < 0.5:
            return None
        if row["ema20"] > row["ema50"] and row["is_bull"] and row["close_position"] >= 0.6:
            return "LONG"
        if row["ema20"] < row["ema50"] and not row["is_bull"] and row["close_position"] <= 0.4:
            return "SHORT"
    else:
        if vol_ratio < 2.2:
            return None
        if row["lower_wick_ratio"] >= 0.5 and row["rsi"] <= 25:
            return "LONG"
        if row["upper_wick_ratio"] >= 0.5 and row["rsi"] >= 75:
            return "SHORT"
    return None


def s28_liquidation_proxy(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 4.0:
        return None
    if row["lower_wick_ratio"] >= 0.6:
        return "LONG"
    if row["upper_wick_ratio"] >= 0.6:
        return "SHORT"
    return None


def s29_ichimoku(df, i):
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


def s30_parabolic_sar(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["sar"]) or pd.isna(prev["sar"]):
        return None
    prev_below, now_below = prev["close"] > prev["sar"], row["close"] > row["sar"]
    if not prev_below and now_below:
        return "LONG"
    if prev_below and not now_below:
        return "SHORT"
    return None


def s31_pivot_bounce(df, i):
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


def s32_heikin_ashi(df, i):
    if i < 2:
        return None
    c1, c2, c3 = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    if pd.isna(c3["ha_lower_wick"]):
        return None
    all_bull = c1["ha_is_bull"] and c2["ha_is_bull"] and c3["ha_is_bull"] and c3["ha_lower_wick"] <= 0
    all_bear = (not c1["ha_is_bull"]) and (not c2["ha_is_bull"]) and (not c3["ha_is_bull"]) and c3["ha_upper_wick"] <= 0
    if all_bull:
        return "LONG"
    if all_bear:
        return "SHORT"
    return None


def s33_volume_profile_poc(df, i):
    poc = get_volume_profile_poc(df, i)
    if poc is None:
        return None
    row = df.iloc[i]
    if pd.isna(row["atr14"]) or row["atr14"] == 0:
        return None
    if abs(row["close"] - poc) > row["atr14"] * 0.5:
        return None
    if row["low"] <= poc <= row["high"] and row["is_bull"] and row["close"] > poc:
        return "LONG"
    if row["low"] <= poc <= row["high"] and not row["is_bull"] and row["close"] < poc:
        return "SHORT"
    return None


def s34_chandelier_exit(df, i):
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


def s35_session_filtered_breakout(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    hour = row["timestamp"].hour
    if not (13 <= hour <= 21):
        return None
    if row["volume"] / row["vol_sma20"] < 2.2:
        return None
    if row["is_bull"] and row["body"] / row["atr14"] >= 0.6:
        return "LONG"
    if not row["is_bull"] and row["body"] / row["atr14"] >= 0.6:
        return "SHORT"
    return None


def s36_elder_triple_screen(df, i):
    row = df.iloc[i]
    if pd.isna(row["ema100"]) or pd.isna(row["rsi"]):
        return None
    lt_up, lt_down = row["close"] > row["ema100"], row["close"] < row["ema100"]
    osc_up, osc_down = 40 <= row["rsi"] <= 55, 45 <= row["rsi"] <= 60
    trig_bull = row["is_bull"] and row["close"] > row["ema20"]
    trig_bear = (not row["is_bull"]) and row["close"] < row["ema20"]
    if lt_up and osc_up and trig_bull:
        return "LONG"
    if lt_down and osc_down and trig_bear:
        return "SHORT"
    return None


def s37_alligator(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["jaw"]) or pd.isna(row["teeth"]) or pd.isna(row["lips"]):
        return None
    bull_order = row["lips"] > row["teeth"] > row["jaw"]
    bear_order = row["lips"] < row["teeth"] < row["jaw"]
    prev_not_bull = not (prev["lips"] > prev["teeth"] > prev["jaw"])
    prev_not_bear = not (prev["lips"] < prev["teeth"] < prev["jaw"])
    if bull_order and prev_not_bull and row["is_bull"]:
        return "LONG"
    if bear_order and prev_not_bear and not row["is_bull"]:
        return "SHORT"
    return None


def s38_awesome_oscillator(df, i):
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


def s39_cci_extreme(df, i):
    row = df.iloc[i]
    if pd.isna(row["cci"]):
        return None
    if row["cci"] <= -150:
        return "LONG"
    if row["cci"] >= 150:
        return "SHORT"
    return None


def s40_williams_r(df, i):
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


def s41_real_funding_extreme(df, i):
    row = df.iloc[i]
    if pd.isna(row.get("fundingRate")) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 1.5:
        return None
    if row["fundingRate"] <= -0.001 and row["is_bull"]:
        return "LONG"
    if row["fundingRate"] >= 0.001 and not row["is_bull"]:
        return "SHORT"
    return None


def s42_daily_breakout(df, i):
    row = df.iloc[i]
    if pd.isna(row["prev_day_high"]) or pd.isna(row["prev_day_low"]) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 1.8:
        return None
    if row["close"] > row["prev_day_high"] and row["is_bull"]:
        return "LONG"
    if row["close"] < row["prev_day_low"] and not row["is_bull"]:
        return "SHORT"
    return None


def s43_renko_trend(df, i):
    row = df.iloc[i]
    if pd.isna(row["renko_direction"]):
        return None
    if row["renko_direction"] >= 3:
        return "LONG"
    if row["renko_direction"] <= -3:
        return "SHORT"
    return None


def s44_ratio_mean_reversion(df, i):
    row = df.iloc[i]
    if pd.isna(row.get("ratio_zscore")):
        return None
    if row["ratio_zscore"] <= -2.0 and row["is_bull"]:
        return "LONG"
    if row["ratio_zscore"] >= 2.0 and not row["is_bull"]:
        return "SHORT"
    return None


def s45_hurst_regime(df, i):
    row = df.iloc[i]
    if pd.isna(row["hurst"]) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if row["hurst"] >= 0.55:
        if vol_ratio >= 1.8 and row["body"] / row["atr14"] >= 0.5:
            return "LONG" if row["is_bull"] else "SHORT"
    elif row["hurst"] <= 0.45:
        if vol_ratio >= 2.2 and row["lower_wick_ratio"] >= 0.5:
            return "LONG"
        if vol_ratio >= 2.2 and row["upper_wick_ratio"] >= 0.5:
            return "SHORT"
    return None


def s46_kalman_cross(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["kalman_trend"]) or pd.isna(prev["kalman_trend"]):
        return None
    if prev["close"] <= prev["kalman_trend"] and row["close"] > row["kalman_trend"] and row["is_bull"]:
        return "LONG"
    if prev["close"] >= prev["kalman_trend"] and row["close"] < row["kalman_trend"] and not row["is_bull"]:
        return "SHORT"
    return None


def s47_autocorr_reversion(df, i):
    row = df.iloc[i]
    if pd.isna(row["autocorr"]) or row["autocorr"] > -0.3:
        return None
    if row["lower_wick_ratio"] >= 0.4:
        return "LONG"
    if row["upper_wick_ratio"] >= 0.4:
        return "SHORT"
    return None


def s48_vwma_cross(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["vwma20"]) or pd.isna(prev["vwma20"]):
        return None
    if prev["close"] <= prev["vwma20"] and row["close"] > row["vwma20"] and row["is_bull"]:
        return "LONG"
    if prev["close"] >= prev["vwma20"] and row["close"] < row["vwma20"] and not row["is_bull"]:
        return "SHORT"
    return None


def s49_sr_multi_touch(df, i):
    level = find_sr_levels(df, i)
    if level is None:
        return None
    row = df.iloc[i]
    if pd.isna(row["atr14"]) or row["atr14"] == 0:
        return None
    if abs(row["close"] - level) > row["atr14"] * SR_TOUCH_TOLERANCE_ATR:
        return None
    if row["low"] <= level <= row["high"] and row["is_bull"] and row["close"] > level:
        return "LONG"
    if row["low"] <= level <= row["high"] and not row["is_bull"] and row["close"] < level:
        return "SHORT"
    return None


def s50_volume_zscore_outlier(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_std20"]) or row["vol_std20"] == 0 or pd.isna(row["vol_sma20"]):
        return None
    z = (row["volume"] - row["vol_sma20"]) / row["vol_std20"]
    if z < 3.0:
        return None
    if row["lower_wick_ratio"] >= 0.5:
        return "LONG"
    if row["upper_wick_ratio"] >= 0.5:
        return "SHORT"
    return None


def s51_momentum_acceleration(df, i):
    if i < 1:
        return None
    row, prev = df.iloc[i], df.iloc[i - 1]
    if pd.isna(row["momentum_accel"]) or pd.isna(prev["momentum_accel"]):
        return None
    if prev["momentum_accel"] <= 0 and row["momentum_accel"] > 0.05 and row["is_bull"]:
        return "LONG"
    if prev["momentum_accel"] >= 0 and row["momentum_accel"] < -0.05 and not row["is_bull"]:
        return "SHORT"
    return None


def s52_basket_confirmation(df, i):
    row = df.iloc[i]
    if pd.isna(row.get("btc_is_bull")) or pd.isna(row.get("eth_is_bull")) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 1.5:
        return None
    if bool(row["btc_is_bull"]) and bool(row["eth_is_bull"]) and row["is_bull"]:
        return "LONG"
    if (not bool(row["btc_is_bull"])) and (not bool(row["eth_is_bull"])) and (not row["is_bull"]):
        return "SHORT"
    return None


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
    return "LONG" if row["macd_line"] > row["macd_signal"] else "SHORT"


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
    return "LONG" if row["ema20"] > row["ema50"] else "SHORT"


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


_SUB_SIGNALS = [_sub_rsi_extreme, _sub_macd, _sub_bollinger, _sub_ema_trend, _sub_zscore, _sub_wick]


def s53_majority_vote(df, i):
    row = df.iloc[i]
    votes = [fn(row) for fn in _SUB_SIGNALS]
    votes = [v for v in votes if v is not None]
    if len(votes) < 4:
        return None
    if votes.count("LONG") >= 4:
        return "LONG"
    if votes.count("SHORT") >= 4:
        return "SHORT"
    return None


def s54_day_of_week_breakout(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["day_of_week"] not in [1, 2, 3]:
        return None
    if row["volume"] / row["vol_sma20"] < 2.0:
        return None
    if row["is_bull"] and row["body"] / row["atr14"] >= 0.5:
        return "LONG"
    if not row["is_bull"] and row["body"] / row["atr14"] >= 0.5:
        return "SHORT"
    return None


def s55_bagging_ensemble(df, i):
    if i < 60:
        return None
    window = df.iloc[max(0, i - 200):i]
    row = df.iloc[i]
    votes = [rule(window, row) for rule in _BAGGING_RULES]
    votes = [v for v in votes if v is not None]
    if len(votes) < 3:
        return None
    if votes.count("LONG") >= 3:
        return "LONG"
    if votes.count("SHORT") >= 3:
        return "SHORT"
    return None


def s56_opening_range_breakout(df, i):
    row = df.iloc[i]
    if pd.isna(row["opening_range_high"]) or pd.isna(row["opening_range_low"]):
        return None
    if row["_minute_of_day"] < 75 or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 1.8:
        return None
    if row["close"] > row["opening_range_high"] and row["is_bull"]:
        return "LONG"
    if row["close"] < row["opening_range_low"] and not row["is_bull"]:
        return "SHORT"
    return None


def s57_correlation_breakdown(df, i):
    row = df.iloc[i]
    if pd.isna(row["rolling_corr_short"]) or pd.isna(row["rolling_corr_long"]):
        return None
    if (row["rolling_corr_long"] - row["rolling_corr_short"]) < 0.4:
        return None
    return "LONG" if row["is_bull"] else "SHORT"


def s58_rsi_vwap_confluence(df, i):
    row = df.iloc[i]
    if pd.isna(row["rsi"]) or pd.isna(row["vwap_dev_pct"]):
        return None
    if row["rsi"] <= 30 and row["vwap_dev_pct"] <= -2.0:
        return "LONG"
    if row["rsi"] >= 70 and row["vwap_dev_pct"] >= 2.0:
        return "SHORT"
    return None


def s59_adaptive_volatility_threshold(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_percentile"]) or pd.isna(row["rsi"]) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 1.8:
        return None
    rsi_low, rsi_high = (35, 65) if row["vol_percentile"] <= 0.3 else (20, 80)
    if row["rsi"] <= rsi_low and row["lower_wick_ratio"] >= 0.4:
        return "LONG"
    if row["rsi"] >= rsi_high and row["upper_wick_ratio"] >= 0.4:
        return "SHORT"
    return None


def s60_two_of_three_reversion(df, i):
    row = df.iloc[i]
    checks_long = checks_short = 0
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


def s61_ou_halflife_reversion(df, i):
    row = df.iloc[i]
    if pd.isna(row["ou_halflife"]) or row["ou_halflife"] > 20:
        return None
    if row["lower_wick_ratio"] >= 0.45:
        return "LONG"
    if row["upper_wick_ratio"] >= 0.45:
        return "SHORT"
    return None


def s62_volatility_clustering(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_clustering"]) or row["vol_clustering"] < 3 or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["volume"] / row["vol_sma20"] < 1.5:
        return None
    return "LONG" if row["is_bull"] else "SHORT"


def s63_whale_volume_proxy(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_std20"]) or row["vol_std20"] == 0 or pd.isna(row["vol_sma20"]):
        return None
    z = (row["volume"] - row["vol_sma20"]) / row["vol_std20"]
    if z < 4.0 or row["body"] / row["atr14"] < 0.8:
        return None
    return "LONG" if row["is_bull"] else "SHORT"


def s64_positioning_score(df, i):
    row = df.iloc[i]
    score = 0
    if not pd.isna(row.get("fundingRate")):
        if row["fundingRate"] >= 0.0006:
            score -= 1
        elif row["fundingRate"] <= -0.0006:
            score += 1
    if not pd.isna(row.get("oi_change_pct")) and row["oi_change_pct"] >= 4:
        score += 1 if row["is_bull"] else -1
    if row["lower_wick_ratio"] >= 0.4:
        score += 1
    if row["upper_wick_ratio"] >= 0.4:
        score -= 1
    if score >= 2:
        return "LONG"
    if score <= -2:
        return "SHORT"
    return None


ALL_STRATEGIES = {
    "01-Tersine Bahis": s01_mean_reversion, "02-Momentum Kirilim": s02_momentum_breakout,
    "03-EMA Kesisimi": s03_ema_cross, "04-Bollinger Kirilim": s04_bollinger_breakout,
    "05-Hacim Patlamasi": s05_volume_spike, "06-RSI Orta Cizgi": s06_rsi_midline,
    "07-Saf Kanal Kirilimi": s07_donchian_simple, "08-Kombinasyon": s08_confluence,
    "09-Kirilim+Retest": s09_breakout_retest, "10-Volatilite Sikismasi": s10_volatility_squeeze,
    "11-Uc Zaman Dilimi": s11_triple_timeframe, "12-Goreceli Guc": s12_relative_strength,
    "13-Trend Ici Pullback": s13_pullback_in_trend, "14-Hacim/Kapanis Teyidi": s14_obv_confirmation,
    "15-Z-Skor": s15_zscore_reversion, "16-Uc Mum Momentum": s16_three_candle_momentum,
    "17-MACD": s17_macd_cross, "18-Stochastic": s18_stochastic,
    "19-Keltner": s19_keltner_breakout, "20-VWAP Sapmasi": s20_vwap_deviation,
    "21-Fibonacci": s21_fibonacci_bounce, "22-Piyasa Yapisi": s22_market_structure_break,
    "23-RSI Uyumsuzlugu": s23_rsi_divergence, "24-Yutan Mum": s24_engulfing,
    "27-Rejim Degistirme": s27_regime_switching, "28-Likidasyon Izi": s28_liquidation_proxy,
    "29-Ichimoku": s29_ichimoku, "30-Parabolic SAR": s30_parabolic_sar,
    "31-Pivot Sekmesi": s31_pivot_bounce, "32-Heikin-Ashi": s32_heikin_ashi,
    "33-Hacim Profili POC": s33_volume_profile_poc, "34-Chandelier Exit": s34_chandelier_exit,
    "35-Seans Filtreli": s35_session_filtered_breakout, "36-Elder Uclu Ekran": s36_elder_triple_screen,
    "37-Alligator": s37_alligator, "38-Awesome Oscillator": s38_awesome_oscillator,
    "39-CCI": s39_cci_extreme, "40-Williams %R": s40_williams_r,
    "41-Gercek Funding": s41_real_funding_extreme, "42-Gunluk Kirilim": s42_daily_breakout,
    "43-Renko": s43_renko_trend, "44-Coin/BTC Orani": s44_ratio_mean_reversion,
    "45-Hurst Rejim": s45_hurst_regime, "46-Kalman Kesisimi": s46_kalman_cross,
    "47-Otokorelasyon": s47_autocorr_reversion, "48-VWMA": s48_vwma_cross,
    "49-Coklu Temas S/R": s49_sr_multi_touch, "50-Hacim Z-Skor": s50_volume_zscore_outlier,
    "51-Momentum Ivmesi": s51_momentum_acceleration, "52-Sepet Teyidi": s52_basket_confirmation,
    "53-Cogunluk Oyu": s53_majority_vote, "54-Haftanin Gunu": s54_day_of_week_breakout,
    "55-Bagging": s55_bagging_ensemble, "56-Acilis Araligi": s56_opening_range_breakout,
    "57-Korelasyon Kopmasi": s57_correlation_breakdown, "58-RSI+VWAP": s58_rsi_vwap_confluence,
    "59-Uyarlanabilir Esik": s59_adaptive_volatility_threshold, "60-Iki-Ucte-Iki": s60_two_of_three_reversion,
    "61-OU Yari-Omru": s61_ou_halflife_reversion, "62-Volatilite Kumelenmesi": s62_volatility_clustering,
    "63-Buyuk Oyuncu Hacmi": s63_whale_volume_proxy, "64-Pozisyonlanma Skoru": s64_positioning_score,
}


def evaluate_signals_long(df, strategy_fn):
    total, wins = 0, 0
    net_pct_list = []
    max_check = max(c for c, _ in CHECK_POINTS)
    start_idx = 210

    for i in range(start_idx, len(df) - max_check):
        direction = strategy_fn(df, i)
        if direction is None:
            continue
        entry_price = df.iloc[i]["close"]
        realized_pct = None
        for candles, threshold in CHECK_POINTS:
            future_price = df.iloc[i + candles]["close"]
            pct_change = (future_price - entry_price) / entry_price * 100
            favorable = pct_change if direction == "LONG" else -pct_change
            if favorable >= threshold:
                realized_pct = favorable
                break
        if realized_pct is None:
            final_price = df.iloc[i + max_check]["close"]
            pct_change = (final_price - entry_price) / entry_price * 100
            realized_pct = pct_change if direction == "LONG" else -pct_change

        net_pct = realized_pct - COMMISSION_ROUNDTRIP_PCT
        total += 1
        if realized_pct >= CHECK_POINTS[0][1]:
            wins += 1
        net_pct_list.append(net_pct)

    avg_net = float(np.mean(net_pct_list)) if net_pct_list else 0.0
    total_net = float(np.sum(net_pct_list)) if net_pct_list else 0.0
    return total, wins, avg_net, total_net


def run_tournament():
    print(f"10. TUR (UZUN TUTUS) basliyor: {len(TOURNAMENT_WATCHLIST)} coin, {len(ALL_STRATEGIES)} strateji")
    send_telegram_message(
        f"🏆 Strateji turnuvası (10. TUR - UZUN TUTUŞ: 1sa/4sa/12sa/24sa) başladı.\n"
        f"{len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} günlük veri, {len(ALL_STRATEGIES)} strateji "
        f"(9 turdan 60'ı, ML ve cross-sectional hariç).\n"
        f"Bu çok uzun sürebilir (belki 20-30 dk). Sabırlı ol, ekranı kapatma."
    )

    print("BTC ve ETH verisi cekiliyor...")
    btc_df = fetch_historical_ohlcv(BTC_SYMBOL)
    btc_df_full = compute_all_indicators(btc_df) if not btc_df.empty else None
    eth_df = fetch_historical_ohlcv(ETH_SYMBOL)
    eth_df_full = compute_all_indicators(eth_df) if not eth_df.empty else None

    results = {name: {"total": 0, "wins": 0, "net_pct_sum": 0.0} for name in ALL_STRATEGIES}

    for idx, symbol in enumerate(TOURNAMENT_WATCHLIST):
        print(f"[{idx+1}/{len(TOURNAMENT_WATCHLIST)}] {symbol} verisi cekiliyor...")
        try:
            df = fetch_historical_ohlcv(symbol)
            if df.empty or len(df) < 250:
                print(f"  {symbol}: yetersiz veri, atlaniyor")
                continue
            funding_df = fetch_funding_history(symbol)
            oi_df = fetch_oi_history(symbol)
            df = compute_all_indicators(df, btc_df=btc_df_full, eth_df=eth_df_full,
                                         funding_df=funding_df, oi_df=oi_df)
        except Exception as e:
            print(f"  {symbol} hata: {e}")
            continue

        for name, fn in ALL_STRATEGIES.items():
            try:
                total, wins, avg_net_pct, total_net_pct = evaluate_signals_long(df, fn)
                results[name]["total"] += total
                results[name]["wins"] += wins
                results[name]["net_pct_sum"] += total_net_pct
            except Exception as e:
                print(f"  {name} hata ({symbol}): {e}")

    leaderboard = []
    for name, r in results.items():
        total, wins = r["total"], r["wins"]
        win_rate = (wins / total * 100) if total > 0 else 0
        avg_net = (r["net_pct_sum"] / total) if total > 0 else 0
        leaderboard.append((name, total, wins, win_rate, avg_net, r["net_pct_sum"]))

    ranked = [x for x in leaderboard if x[1] >= MIN_SIGNALS_TO_RANK]
    unranked = [x for x in leaderboard if x[1] < MIN_SIGNALS_TO_RANK]
    ranked.sort(key=lambda x: x[4], reverse=True)

    header = (
        f"🏆 TURNUVA SONUÇLARI - 10. TUR (UZUN TUTUŞ)\n"
        f"{LOOKBACK_DAYS} günlük veri, {len(TOURNAMENT_WATCHLIST)} coin, {len(ALL_STRATEGIES)} strateji.\n"
        f"Kontrol noktaları: 1sa(%0.3) / 4sa(%0.6) / 12sa(%1.0) / 24sa(%1.5)\n"
        f"Komisyon (%{COMMISSION_ROUNDTRIP_PCT}) düşülmüş, ort. net kazanca göre sıralı.\n"
    )
    send_telegram_message(header)

    chunk_lines = []
    for rank, (name, total, wins, win_rate, avg_net, total_net) in enumerate(ranked, 1):
        madalya = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
        yon = "✅ KARLI" if avg_net > 0 else "❌ ZARARLI"
        chunk_lines.append(
            f"{madalya} {name} {yon}\n"
            f"   Sinyal: {total} | İsabet: %{win_rate:.1f}\n"
            f"   Ort. net: %{avg_net:+.3f} | Toplam: %{total_net:+.1f}"
        )
        if len(chunk_lines) >= 10:
            send_telegram_message("\n".join(chunk_lines))
            chunk_lines = []
    if chunk_lines:
        send_telegram_message("\n".join(chunk_lines))

    if unranked:
        unranked_lines = ["Yetersiz örneklem:"]
        for name, total, wins, win_rate, avg_net, total_net in unranked:
            unranked_lines.append(f"- {name}: {total} sinyal (%{win_rate:.1f}, ort net %{avg_net:+.3f})")
        send_telegram_message("\n".join(unranked_lines))

    with open("tournament_results_v10_longhold.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strateji", "toplam_sinyal", "dogru", "isabet_yuzde", "ort_net_yuzde", "toplam_net_yuzde"])
        for name, total, wins, win_rate, avg_net, total_net in leaderboard:
            writer.writerow([name, total, wins, f"{win_rate:.2f}", f"{avg_net:.4f}", f"{total_net:.2f}"])

    print("\nTurnuva tamamlandi.")


if __name__ == "__main__":
    run_tournament()
