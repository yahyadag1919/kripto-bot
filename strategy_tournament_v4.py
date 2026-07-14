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

CHECK_CANDLES = [1, 2, 3]
SUCCESS_THRESHOLD_PCT = 0.15
MIN_SIGNALS_TO_RANK = 15
COMMISSION_ROUNDTRIP_PCT = 0.12

# ML icin: verinin ilk yuzde kaci egitim, geri kalani (gormedigi) test
TRAIN_SPLIT_RATIO = 0.7
ML_PROB_THRESHOLD = 0.55

# Rejim tespiti icin trend gucu esigi
REGIME_TREND_THRESHOLD = 1.2   # |EMA20-EMA50| / ATR bu esigi gecerse "trend", altindaysa "yatay"

# Likidasyon izi icin ekstrem esikler
LIQ_WICK_RATIO = 0.6
LIQ_VOLUME_MULTIPLIER = 4.0

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

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_ratio"] = df["lower_wick_ratio"].fillna(0)
    df["upper_wick_ratio"] = df["upper_wick_ratio"].fillna(0)
    df["close_position"] = (df["close"] - df["low"]) / candle_range
    df["close_position"] = df["close_position"].fillna(0.5)

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

    df["ema_gap_pct"] = (df["ema20"] - df["ema50"]) / df["ema50"] * 100
    df["trend_strength"] = (df["ema20"] - df["ema50"]).abs() / df["atr14"].replace(0, np.nan)

    df["return_8"] = df["close"].pct_change(8) * 100  # ~2 saatlik getiri (cross-sectional icin)

    return df


# ---------------------------------------------------------------------------
# YARDIMCI: basari etiketi (label) - ML egitimi ve genel degerlendirme icin ortak
# ---------------------------------------------------------------------------

def compute_forward_label(df):
    """Her mum icin: ileride LONG mu SHORT mu daha karli olurdu (ya da notr)."""
    n = len(df)
    label = np.zeros(n)  # 1 = LONG uygun, -1 = SHORT uygun, 0 = notr
    closes = df["close"].values
    max_check = max(CHECK_CANDLES)

    for i in range(n - max_check):
        entry = closes[i]
        long_hit, short_hit = False, False
        for c in CHECK_CANDLES:
            pct = (closes[i + c] - entry) / entry * 100
            if pct >= SUCCESS_THRESHOLD_PCT:
                long_hit = True
            if pct <= -SUCCESS_THRESHOLD_PCT:
                short_hit = True
        if long_hit and not short_hit:
            label[i] = 1
        elif short_hit and not long_hit:
            label[i] = -1
    return label


# ---------------------------------------------------------------------------
# SISTEM 1: Makine ogrenmesi (manuel lojistik regresyon, egitim/test ayrimiyla)
# ---------------------------------------------------------------------------

FEATURE_COLS = ["rsi", "macd_hist", "ema_gap_pct", "trend_strength", "close_position"]


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def train_logistic(X, y, epochs=400, lr=0.15, l2=0.02):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        pred = _sigmoid(z)
        grad_w = X.T @ (pred - y) / n + l2 * w
        grad_b = np.mean(pred - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def prepare_ml_features(df):
    feat = df[FEATURE_COLS].copy()
    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat


def build_ml_model(all_dfs_train):
    """Tum coinlerin egitim bolumunu birlestirip tek bir model egitir (havuzlanmis egitim)."""
    X_list, y_long_list, y_short_list = [], [], []
    for df in all_dfs_train:
        feat = prepare_ml_features(df)
        label = compute_forward_label(df)
        valid = feat.notna().all(axis=1)
        X_list.append(feat[valid].values)
        y_long_list.append((label[valid.values] == 1).astype(float))
        y_short_list.append((label[valid.values] == -1).astype(float))

    X = np.vstack(X_list)
    y_long = np.concatenate(y_long_list)
    y_short = np.concatenate(y_short_list)

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X_norm = (X - mean) / std

    w_long, b_long = train_logistic(X_norm, y_long)
    w_short, b_short = train_logistic(X_norm, y_short)

    return {"mean": mean, "std": std, "w_long": w_long, "b_long": b_long,
            "w_short": w_short, "b_short": b_short}


def ml_predict(model, df):
    """Egitilmis modelle, verilen (test) df icin her satira LONG/SHORT/None tahmini uretir."""
    feat = prepare_ml_features(df)
    valid = feat.notna().all(axis=1)
    X = feat.values
    X_norm = (X - model["mean"]) / model["std"]

    prob_long = _sigmoid(X_norm @ model["w_long"] + model["b_long"])
    prob_short = _sigmoid(X_norm @ model["w_short"] + model["b_short"])

    signals = [None] * len(df)
    for i in range(len(df)):
        if not valid.iloc[i]:
            continue
        if prob_long[i] >= ML_PROB_THRESHOLD and prob_long[i] > prob_short[i]:
            signals[i] = "LONG"
        elif prob_short[i] >= ML_PROB_THRESHOLD and prob_short[i] > prob_long[i]:
            signals[i] = "SHORT"
    return signals


# ---------------------------------------------------------------------------
# SISTEM 3: Rejime gore strateji degistiren meta-sistem
# ---------------------------------------------------------------------------

def strategy_regime_switching(df, i):
    row = df.iloc[i]
    if pd.isna(row["trend_strength"]) or pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None

    trending = row["trend_strength"] >= REGIME_TREND_THRESHOLD
    vol_ratio = row["volume"] / row["vol_sma20"]

    if trending:
        # Trend rejiminde: momentum/kirilim mantigi
        if vol_ratio < 1.8 or row["body"] / row["atr14"] < 0.5:
            return None
        if row["ema_gap_pct"] > 0 and row["is_bull"] and row["close_position"] >= 0.6:
            return "LONG"
        if row["ema_gap_pct"] < 0 and not row["is_bull"] and row["close_position"] <= 0.4:
            return "SHORT"
    else:
        # Yatay rejimde: tersine bahis mantigi
        if vol_ratio < 2.2:
            return None
        if row["lower_wick_ratio"] >= 0.5 and row["rsi"] <= 25:
            return "LONG"
        if row["upper_wick_ratio"] >= 0.5 and row["rsi"] >= 75:
            return "SHORT"
    return None


# ---------------------------------------------------------------------------
# SISTEM 4: Likidasyon dalgasi izi (proxy - gercek likidasyon verisi degil,
# asiri fitil + asiri hacimden cikarim yapiliyor)
# ---------------------------------------------------------------------------

def strategy_liquidation_proxy(df, i):
    row = df.iloc[i]
    if pd.isna(row["vol_sma20"]) or row["vol_sma20"] == 0:
        return None
    vol_ratio = row["volume"] / row["vol_sma20"]
    if vol_ratio < LIQ_VOLUME_MULTIPLIER:
        return None
    if row["lower_wick_ratio"] >= LIQ_WICK_RATIO:
        return "LONG"
    if row["upper_wick_ratio"] >= LIQ_WICK_RATIO:
        return "SHORT"
    return None


# ---------------------------------------------------------------------------
# Degerlendirme (komisyon dahil gercek net kar/zarar)
# ---------------------------------------------------------------------------

def evaluate_rule_strategy(df, strategy_fn, start_idx=0, end_idx=None):
    total, wins = 0, 0
    net_pct_list = []
    max_check = max(CHECK_CANDLES)
    end_idx = end_idx or (len(df) - max_check)

    for i in range(max(start_idx, 60), min(end_idx, len(df) - max_check)):
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


def evaluate_signal_list(df, signals, start_idx=0):
    """ML gibi onceden uretilmis sinyal listesini degerlendirir."""
    total, wins = 0, 0
    net_pct_list = []
    max_check = max(CHECK_CANDLES)

    for i in range(start_idx, len(df) - max_check):
        direction = signals[i]
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


# ---------------------------------------------------------------------------
# SISTEM 2: Goreceli siralama (cross-sectional) - ayri, coklu-sembol pipeline gerektirir
# ---------------------------------------------------------------------------

def run_cross_sectional_strategy(symbol_dfs: dict):
    """
    Tum coinleri ayni zaman damgasinda hizalayip, her anda en guclu/en zayif
    performans gosteren coinleri LONG/SHORT adayi olarak isaretler.
    """
    aligned = {}
    for symbol, df in symbol_dfs.items():
        s = df.set_index("timestamp")["return_8"]
        aligned[symbol] = s

    wide = pd.DataFrame(aligned)
    wide = wide.dropna(how="all")

    cross_mean = wide.mean(axis=1)
    cross_std = wide.std(axis=1)

    total, wins = 0, 0
    net_pct_list = []
    max_check = max(CHECK_CANDLES)

    for symbol, df in symbol_dfs.items():
        df_indexed = df.set_index("timestamp")
        closes = df["close"].values
        timestamps = df["timestamp"].values
        n = len(df)

        for i in range(60, n - max_check):
            ts = df["timestamp"].iloc[i]
            if ts not in wide.index:
                continue
            ret = wide.loc[ts, symbol] if symbol in wide.columns else np.nan
            mean_ts = cross_mean.loc[ts] if ts in cross_mean.index else np.nan
            std_ts = cross_std.loc[ts] if ts in cross_std.index else np.nan
            if pd.isna(ret) or pd.isna(std_ts) or std_ts == 0:
                continue

            z = (ret - mean_ts) / std_ts
            row = df.iloc[i]
            direction = None
            if z >= 1.5 and row["is_bull"] and row["rsi"] < 75:
                direction = "LONG"
            elif z <= -1.5 and not row["is_bull"] and row["rsi"] > 25:
                direction = "SHORT"

            if direction is None:
                continue

            entry_price = closes[i]
            realized_pct = None
            for c in CHECK_CANDLES:
                pct_change = (closes[i + c] - entry_price) / entry_price * 100
                favorable = pct_change if direction == "LONG" else -pct_change
                if favorable >= SUCCESS_THRESHOLD_PCT:
                    realized_pct = favorable
                    break
            if realized_pct is None:
                pct_change = (closes[i + max_check] - entry_price) / entry_price * 100
                realized_pct = pct_change if direction == "LONG" else -pct_change

            net_pct = realized_pct - COMMISSION_ROUNDTRIP_PCT
            total += 1
            if realized_pct >= SUCCESS_THRESHOLD_PCT:
                wins += 1
            net_pct_list.append(net_pct)

    avg_net = float(np.mean(net_pct_list)) if net_pct_list else 0.0
    total_net = float(np.sum(net_pct_list)) if net_pct_list else 0.0
    return total, wins, avg_net, total_net


# ---------------------------------------------------------------------------
# Ana turnuva
# ---------------------------------------------------------------------------

def run_tournament():
    print(f"Turnuva basliyor (4. tur - farkli sistemler): {len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} gunluk veri")
    send_telegram_message(
        f"🏆 Strateji turnuvası (4. tur - ML, göreceli sıralama, rejim, likidasyon izi) başladı.\n"
        f"{len(TOURNAMENT_WATCHLIST)} coin, {LOOKBACK_DAYS} günlük veri.\n"
        f"Bu tur daha karmaşık, biraz daha uzun sürebilir. Bitince göndereceğim."
    )

    print("Tum coinlerin verisi cekiliyor...")
    symbol_dfs = {}
    for idx, symbol in enumerate(TOURNAMENT_WATCHLIST):
        print(f"[{idx+1}/{len(TOURNAMENT_WATCHLIST)}] {symbol} verisi cekiliyor...")
        try:
            df = fetch_historical_ohlcv(symbol)
            if df.empty or len(df) < 200:
                print(f"  {symbol}: yetersiz veri, atlaniyor")
                continue
            df = compute_all_indicators(df)
            symbol_dfs[symbol] = df
        except Exception as e:
            print(f"  {symbol} hata: {e}")

    print(f"\n{len(symbol_dfs)} coin icin veri hazir.\n")

    results = {}

    # --- SISTEM 3: Rejim degistirme ---
    print("Sistem 3 (Rejim degistirme) degerlendiriliyor...")
    total, wins, avg_net, total_net = 0, 0, 0.0, 0.0
    all_net = []
    for symbol, df in symbol_dfs.items():
        t, w, a, tn = evaluate_rule_strategy(df, strategy_regime_switching)
        total += t
        wins += w
        total_net += tn
    avg_net = (total_net / total) if total > 0 else 0
    results["3-Rejime Gore Strateji Degistirme"] = (total, wins, avg_net, total_net)

    # --- SISTEM 4: Likidasyon izi ---
    print("Sistem 4 (Likidasyon izi) degerlendiriliyor...")
    total, wins, total_net = 0, 0, 0.0
    for symbol, df in symbol_dfs.items():
        t, w, a, tn = evaluate_rule_strategy(df, strategy_liquidation_proxy)
        total += t
        wins += w
        total_net += tn
    avg_net = (total_net / total) if total > 0 else 0
    results["4-Likidasyon Dalgasi Izi"] = (total, wins, avg_net, total_net)

    # --- SISTEM 2: Cross-sectional (goreceli siralama) ---
    print("Sistem 2 (Goreceli siralama) degerlendiriliyor...")
    total, wins, avg_net, total_net = run_cross_sectional_strategy(symbol_dfs)
    results["2-Goreceli Siralama (Cross-Sectional)"] = (total, wins, avg_net, total_net)

    # --- SISTEM 1: ML (egitim/test ayrimiyla) ---
    print("Sistem 1 (Makine ogrenmesi) egitiliyor...")
    train_dfs, test_dfs = {}, {}
    for symbol, df in symbol_dfs.items():
        split_idx = int(len(df) * TRAIN_SPLIT_RATIO)
        train_dfs[symbol] = df.iloc[:split_idx].reset_index(drop=True)
        test_dfs[symbol] = df.iloc[split_idx:].reset_index(drop=True)

    model = build_ml_model(list(train_dfs.values()))
    print("Model egitildi, gormedigi (test) veride degerlendiriliyor...")

    total, wins, total_net = 0, 0, 0.0
    for symbol, df in test_dfs.items():
        if len(df) < 70:
            continue
        signals = ml_predict(model, df)
        t, w, a, tn = evaluate_signal_list(df, signals, start_idx=60)
        total += t
        wins += w
        total_net += tn
    avg_net = (total_net / total) if total > 0 else 0
    results["1-Makine Ogrenmesi (gormedigi veride)"] = (total, wins, avg_net, total_net)

    # --- Sonuclari sirala ve gonder ---
    leaderboard = []
    for name, (total, wins, avg_net, total_net) in results.items():
        win_rate = (wins / total * 100) if total > 0 else 0
        leaderboard.append((name, total, wins, win_rate, avg_net, total_net))

    ranked = [x for x in leaderboard if x[1] >= MIN_SIGNALS_TO_RANK]
    unranked = [x for x in leaderboard if x[1] < MIN_SIGNALS_TO_RANK]
    ranked.sort(key=lambda x: x[4], reverse=True)

    lines = [f"🏆 TURNUVA SONUÇLARI - 4. TUR ({LOOKBACK_DAYS} günlük veri, {len(symbol_dfs)} coin)"]
    lines.append(f"Komisyon (%{COMMISSION_ROUNDTRIP_PCT}) düşülmüş. ML sistemi SADECE görmediği veride test edildi.\n")
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

    with open("tournament_results_v4.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strateji", "toplam_sinyal", "dogru", "isabet_yuzde", "ort_net_yuzde", "toplam_net_yuzde"])
        for name, total, wins, win_rate, avg_net, total_net in leaderboard:
            writer.writerow([name, total, wins, f"{win_rate:.2f}", f"{avg_net:.4f}", f"{total_net:.2f}"])

    print("\nTurnuva tamamlandi.")


if __name__ == "__main__":
    run_tournament()
