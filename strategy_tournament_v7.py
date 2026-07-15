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
ETH_SYMBOL = "ETH/USDT:USDT"

TIMEFRAME = "15m"
LOOKBACK_DAYS = 30
ATR_PERIOD = 14

CHECK_CANDLES = [1, 2, 3]
SUCCESS_THRESHOLD_PCT = 0.15
MIN_SIGNALS_TO_RANK = 15
COMMISSION_ROUNDTRIP_PCT = 0.12

HURST_WINDOW = 100
SR_LOOKBACK = 150
SR_TOUCH_TOLERANCE_ATR = 0.3
SR_MIN_TOUCHES = 3

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


# ---------------------------------------------------------------------------
# Ozel hesaplamalar
# ---------------------------------------------------------------------------

def rolling_hurst(prices: np.ndarray, window: int) -> np.ndarray:
    """Basitlestirilmis R/S yontemiyle kayan Hurst usteli. >0.5 trend, <0.5 yatay/ortalamaya donus egilimi."""
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
    """Basit 1 boyutlu Kalman filtresi ile duzlestirilmis trend tahmini."""
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


def find_sr_levels(df, i, lookback=SR_LOOKBACK, atr_tolerance=SR_TOUCH_TOLERANCE_ATR, min_touches=SR_MIN_TOUCHES):
    """Son 'lookback' mumda en cok test edilmis (dokunulmus) fiyat seviyesini bulur."""
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


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_std20"] = df["volume"].rolling(20).std()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(ATR_PERIOD).mean()

    df["body"] = (df["close"] - df["open"]).abs()
    df["is_bull"] = df["close"] > df["open"]

    df["hurst"] = rolling_hurst(df["close"].values, HURST_WINDOW)
    df["kalman_trend"] = kalman_filter_trend(df["close"].values)

    returns = df["close"].pct_change()
    df["autocorr"] = rolling_autocorr(returns, window=30, lag=1)

    df["vwma20"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()

    roc = df["close"].pct_change(5) * 100
    df["momentum_accel"] = roc.diff(3)

    return df


# ---------------------------------------------------------------------------
# 8 YENI strateji (Round 7)
# ---------------------------------------------------------------------------

def strategy_hurst_regime(df, i):
    row = df.iloc[i]
    if pd.isna(row["hurst"]) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]

    if row["hurst"] >= 0.55:
        # Trend rejimi: momentum
        if vol_ratio >= 1.8 and row["body"] / row["atr14"] >= 0.5:
            if row["is_bull"]:
                return "LONG"
            return "SHORT"
    elif row["hurst"] <= 0.45:
        # Ortalamaya donus rejimi
        upper_wick = (row["high"] - max(row["open"], row["close"])) / (row["high"] - row["low"] + 1e-9)
        lower_wick = (min(row["open"], row["close"]) - row["low"]) / (row["high"] - row["low"] + 1e-9)
        if vol_ratio >= 2.2 and lower_wick >= 0.5:
            return "LONG"
        if vol_ratio >= 2.2 and upper_wick >= 0.5:
            return "SHORT"
    return None


def strategy_kalman_cross(df, i):
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


def strategy_autocorr_reversion(df, i):
    row = df.iloc[i]
    if pd.isna(row["autocorr"]):
        return None
    if row["autocorr"] <= -0.3:
        upper_wick = (row["high"] - max(row["open"], row["close"])) / (row["high"] - row["low"] + 1e-9)
        lower_wick = (min(row["open"], row["close"]) - row["low"]) / (row["high"] - row["low"] + 1e-9)
        if lower_wick >= 0.4:
            return "LONG"
        if upper_wick >= 0.4:
            return "SHORT"
    return None


def strategy_vwma_cross(df, i):
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


def strategy_sr_multi_touch(df, i):
    level = find_sr_levels(df, i)
    if level is None:
        return None
    row = df.iloc[i]
    atr = row["atr14"]
    if pd.isna(atr) or atr == 0:
        return None
    near = abs(row["close"] - level) <= atr * SR_TOUCH_TOLERANCE_ATR
    if not near:
        return None
    if row["low"] <= level <= row["high"] and row["is_bull"] and row["close"] > level:
        return "LONG"
    if row["low"] <= level <= row["high"] and not row["is_bull"] and row["close"] < level:
        return "SHORT"
    return None


def strategy_volume_zscore_outlier(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_std20"]) or row["vol_std20"] == 0 or pd.isna(row["vol_sma20"]):
        return None
    z = (row["volume"] - row["vol_sma20"]) / row["vol_std20"]
    if z < 3.0:
        return None
    upper_wick = (row["high"] - max(row["open"], row["close"])) / (row["high"] - row["low"] + 1e-9)
    lower_wick = (min(row["open"], row["close"]) - row["low"]) / (row["high"] - row["low"] + 1e-9)
    if lower_wick >= 0.5:
        return "LONG"
    if upper_wick >= 0.5:
        return "SHORT"
    return None


def strategy_momentum_acceleration(df, i):
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


def strategy_basket_confirmation(df, i, btc_df, eth_df):
    row = df.iloc[i]
    ts = row["timestamp"]
    btc_match = btc_df[btc_df["timestamp"] == ts]
    eth_match = eth_df[eth_df["timestamp"] == ts]
    if btc_match.empty or eth_match.empty:
        return None
    btc_row = btc_match.iloc[0]
    eth_row = eth_match.iloc[0]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 1.5:
        return None

    if btc_row["is_bull"] and eth_row["is_bull"] and row["is_bull"]:
        return "LONG"
    if (not btc_row["is_bull"]) and (not eth_row["is_bull"]) and (not row["is_bull"]):
        return "SHORT"
    return None


def evaluate_signals(df, strategy_fn, extra_args=None):
    total, wins = 0, 0
    net_pct_list = []
    max_check = max(CHECK_CANDLES)
    start_idx = max(HURST_WINDOW, SR_LOOKBACK) + 5

    for i in range(start_idx, len(df) - max_check):
        if extra_args:
            direction = strategy_fn(df, i, *extra_args)
        else:
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
    print(f"Turnuva basliyor (7. tur): {len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} gunluk veri")
    send_telegram_message(
        f"🏆 Strateji turnuvası (7. tur - Hurst, Kalman, Otokorelasyon, VWMA, S/R, vb.) başladı.\n"
        f"{len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} günlük veri.\n"
        f"Bitince göndereceğim."
    )

    print("BTC ve ETH verisi cekiliyor (sepet teyidi icin)...")
    btc_df = fetch_historical_ohlcv(BTC_SYMBOL)
    btc_df = compute_all_indicators(btc_df)
    eth_df = fetch_historical_ohlcv(ETH_SYMBOL)
    eth_df = compute_all_indicators(eth_df)

    strategy_list = {
        "Hurst Rejim Tespiti": (strategy_hurst_regime, None),
        "Kalman Filtresi Kesisimi": (strategy_kalman_cross, None),
        "Otokorelasyon Ortalamaya Donus": (strategy_autocorr_reversion, None),
        "VWMA Kesisimi": (strategy_vwma_cross, None),
        "Coklu Temas Destek/Direnc": (strategy_sr_multi_touch, None),
        "Hacim Z-Skor Aykiri Deger": (strategy_volume_zscore_outlier, None),
        "Momentum Ivmesi": (strategy_momentum_acceleration, None),
        "Sepet Teyidi (BTC+ETH)": (strategy_basket_confirmation, (btc_df, eth_df)),
    }

    results = {name: {"total": 0, "wins": 0, "net_pct_sum": 0.0} for name in strategy_list}

    for idx, symbol in enumerate(TOURNAMENT_WATCHLIST):
        print(f"[{idx+1}/{len(TOURNAMENT_WATCHLIST)}] {symbol} verisi cekiliyor...")
        try:
            df = fetch_historical_ohlcv(symbol)
            if df.empty or len(df) < 200:
                print(f"  {symbol}: yetersiz veri, atlaniyor")
                continue
            df = compute_all_indicators(df)
        except Exception as e:
            print(f"  {symbol} hata: {e}")
            continue

        for name, (fn, extra_args) in strategy_list.items():
            total, wins, avg_net_pct, total_net_pct = evaluate_signals(df, fn, extra_args)
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

    lines = [f"🏆 TURNUVA SONUÇLARI - 7. TUR ({LOOKBACK_DAYS} günlük veri, {len(TOURNAMENT_WATCHLIST)} coin)"]
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

    with open("tournament_results_v7.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strateji", "toplam_sinyal", "dogru", "isabet_yuzde", "ort_net_yuzde", "toplam_net_yuzde"])
        for name, total, wins, win_rate, avg_net, total_net in leaderboard:
            writer.writerow([name, total, wins, f"{win_rate:.2f}", f"{avg_net:.4f}", f"{total_net:.2f}"])

    print("\nTurnuva tamamlandi.")


if __name__ == "__main__":
    run_tournament()
