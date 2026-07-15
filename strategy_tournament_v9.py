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


def fetch_oi_history(symbol: str, days: int = LOOKBACK_DAYS):
    """Gercek Open Interest gecmisi. OKX/ccxt destegi degiskenlik gosterebilir - basarisiz olursa None doner."""
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
    except Exception as e:
        print(f"  OI gecmisi alinamadi ({symbol}): {e}")
        return None


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
    except Exception as e:
        print(f"  Funding gecmisi alinamadi ({symbol}): {e}")
        return None


def compute_all_indicators(df: pd.DataFrame, oi_df: pd.DataFrame = None, funding_df: pd.DataFrame = None) -> pd.DataFrame:
    df = df.copy()
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_std20"] = df["volume"].rolling(20).std()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(ATR_PERIOD).mean()
    df["realized_vol"] = df["close"].pct_change().rolling(20).std() * 100

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

    # Ornstein-Uhlenbeck tarzi ortalamaya donus yari-omru (kayan regresyon)
    df["ou_halflife"] = compute_ou_halflife(df["close"])

    # Volatilite kumelenmesi: son 3 mumun hepsi ortalama-ustu volatilite mi
    vol_above_avg = df["realized_vol"] > df["realized_vol"].rolling(50).mean()
    df["vol_clustering"] = vol_above_avg.rolling(3).sum()

    if oi_df is not None and not oi_df.empty:
        oi_df = oi_df.copy()
        oi_df["timestamp"] = oi_df["timestamp"].astype("datetime64[ns]")
        df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
        df = pd.merge_asof(df.sort_values("timestamp"), oi_df.sort_values("timestamp"),
                            on="timestamp", direction="backward")
        df["oi_change_pct"] = df["oi_value"].pct_change(4) * 100  # ~1 saatlik OI degisimi
    else:
        df["oi_value"] = np.nan
        df["oi_change_pct"] = np.nan

    if funding_df is not None and not funding_df.empty:
        funding_df = funding_df.copy()
        funding_df["timestamp"] = funding_df["timestamp"].astype("datetime64[ns]")
        df["timestamp"] = df["timestamp"].astype("datetime64[ns]")
        df = pd.merge_asof(df.sort_values("timestamp"), funding_df.sort_values("timestamp"),
                            on="timestamp", direction="backward")
    else:
        df["fundingRate"] = np.nan

    return df


def compute_ou_halflife(prices: pd.Series, window: int = 50) -> pd.Series:
    """Kayan pencerede basit OU yari-omru tahmini (dusuk deger = hizli ortalamaya donus egilimi)."""
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
            continue  # trend rejiminde (ortalamaya donus yok), NaN birak
        hl = -np.log(2) / beta
        if 0 < hl < 500:
            halflife.iloc[i] = hl

    return halflife


# ---------------------------------------------------------------------------
# 6 YENI strateji (Round 9) - pozisyonlanma verisi + kalan istatistiksel fikirler
# ---------------------------------------------------------------------------

def strategy_real_oi_extreme(df, i):
    row = df.iloc[i]
    if pd.isna(row.get("oi_change_pct")):
        return None
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 1.5:
        return None
    # OI hizla artiyorken fiyat dususte -> yeni short pozisyon aciliyor, devam olasi
    if row["oi_change_pct"] >= 5 and not row["is_bull"]:
        return "SHORT"
    if row["oi_change_pct"] >= 5 and row["is_bull"]:
        return "LONG"
    return None


def strategy_funding_oi_combined(df, i):
    row = df.iloc[i]
    if pd.isna(row.get("fundingRate")) or pd.isna(row.get("oi_change_pct")):
        return None
    # Asiri pozitif funding + OI azaliyor -> asiri longlar kapaniyor -> asagi baski
    if row["fundingRate"] >= 0.0008 and row["oi_change_pct"] <= -3 and not row["is_bull"]:
        return "SHORT"
    # Asiri negatif funding + OI azaliyor -> asiri shortlar kapaniyor -> yukari baski
    if row["fundingRate"] <= -0.0008 and row["oi_change_pct"] <= -3 and row["is_bull"]:
        return "LONG"
    return None


def strategy_volatility_clustering(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_clustering"]) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    if row["vol_clustering"] < 3:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < 1.5:
        return None
    if row["is_bull"]:
        return "LONG"
    return "SHORT"


def strategy_ou_halflife_reversion(df, i):
    row = df.iloc[i]
    if pd.isna(row["ou_halflife"]):
        return None
    if row["ou_halflife"] > 20:
        return None  # yavas donus, guvenilir degil
    if row["lower_wick_ratio"] >= 0.45:
        return "LONG"
    if row["upper_wick_ratio"] >= 0.45:
        return "SHORT"
    return None


def strategy_whale_volume_proxy(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_std20"]) or row["vol_std20"] == 0 or pd.isna(row["vol_sma20"]):
        return None
    z = (row["volume"] - row["vol_sma20"]) / row["vol_std20"]
    if z < 4.0:
        return None
    body_ratio = row["body"] / row["atr14"] if row["atr14"] > 0 else 0
    if body_ratio < 0.8:
        return None
    if row["is_bull"]:
        return "LONG"
    return "SHORT"


def strategy_positioning_score(df, i):
    """Funding + OI + hacim - agirlikli pozisyonlanma skoru, esik gecince sinyal."""
    row = df.iloc[i]
    score = 0
    if not pd.isna(row.get("fundingRate")):
        if row["fundingRate"] >= 0.0006:
            score -= 1
        elif row["fundingRate"] <= -0.0006:
            score += 1
    if not pd.isna(row.get("oi_change_pct")):
        if row["oi_change_pct"] >= 4:
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


STRATEGIES = {
    "Gercek OI Degisimi Asiri Uc": strategy_real_oi_extreme,
    "Funding+OI Kombinasyonu": strategy_funding_oi_combined,
    "Volatilite Kumelenmesi": strategy_volatility_clustering,
    "OU Yari-Omru Ortalamaya Donus": strategy_ou_halflife_reversion,
    "Buyuk Oyuncu Hacim Izi": strategy_whale_volume_proxy,
    "Pozisyonlanma Skoru (Funding+OI+Fitil)": strategy_positioning_score,
}


def evaluate_signals(df, strategy_fn):
    total, wins = 0, 0
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
    print(f"Turnuva basliyor (9. tur): {len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} gunluk veri")
    send_telegram_message(
        f"🏆 Strateji turnuvası (9. tur - Pozisyonlanma verisi: gerçek OI+funding, OU yarı-ömrü vb.) başladı.\n"
        f"{len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} günlük veri.\n"
        f"Not: gerçek on-chain/duyarlılık verisine erişimim yok, bu en yakın proxy.\n"
        f"Bitince göndereceğim."
    )

    results = {name: {"total": 0, "wins": 0, "net_pct_sum": 0.0} for name in STRATEGIES}

    for idx, symbol in enumerate(TOURNAMENT_WATCHLIST):
        print(f"[{idx+1}/{len(TOURNAMENT_WATCHLIST)}] {symbol} verisi cekiliyor...")
        try:
            df = fetch_historical_ohlcv(symbol)
            if df.empty or len(df) < 100:
                print(f"  {symbol}: yetersiz veri, atlaniyor")
                continue
            oi_df = fetch_oi_history(symbol)
            funding_df = fetch_funding_history(symbol)
            df = compute_all_indicators(df, oi_df=oi_df, funding_df=funding_df)
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

    lines = [f"🏆 TURNUVA SONUÇLARI - 9. TUR ({LOOKBACK_DAYS} günlük veri, {len(TOURNAMENT_WATCHLIST)} coin)"]
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

    with open("tournament_results_v9.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strateji", "toplam_sinyal", "dogru", "isabet_yuzde", "ort_net_yuzde", "toplam_net_yuzde"])
        for name, total, wins, win_rate, avg_net, total_net in leaderboard:
            writer.writerow([name, total, wins, f"{win_rate:.2f}", f"{avg_net:.4f}", f"{total_net:.2f}"])

    print("\nTurnuva tamamlandi.")


if __name__ == "__main__":
    run_tournament()
