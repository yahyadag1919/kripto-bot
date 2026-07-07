import os
import csv
import ccxt
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError(
        "TELEGRAM_TOKEN ve TELEGRAM_CHAT_ID ortam degiskenleri tanimli degil. "
        "Railway'de Variables kismindan ekle."
    )

WATCHLIST = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "ADA/USDT", "SUI/USDT",
    "DOT/USDT", "TRX/USDT", "ATOM/USDT", "NEAR/USDT", "TON/USDT", "LTC/USDT",
]

TIMEFRAME = "15m"
CHECK_INTERVAL_MINUTES = 15

VOLUME_MULTIPLIER = 1.2
MIN_CONSECUTIVE_CANDLES = 2
MIN_BODY_ATR_RATIO = 0.5

# Tukenme (exhaustion) filtresi esikleri
EXHAUSTION_WICK_RATIO = 0.5      # ters yondeki fitil, mumun range'inin en az bu orani kadar olmali
EXHAUSTION_VOLUME_RATIO = 2.5    # hacim, SMA15'in en az bu kati olmali
EXHAUSTION_RSI_PERIOD = 6
EXHAUSTION_RSI_LOW = 20          # SHORT sinyalinde RSI bu altindaysa asiri satim (tukenme)
EXHAUSTION_RSI_HIGH = 80         # LONG sinyalinde RSI bu ustundeyse asiri alim (tukenme)

exchange = ccxt.okx()


def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Telegram gonderim hatasi: {e}")


def fetch_data(symbol: str, limit: int = 100) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["vol_sma15"] = df["volume"].rolling(15).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    df["body"] = (df["close"] - df["open"]).abs()
    df["is_bull"] = df["close"] > df["open"]

    # RSI (exhaustion filtresi icin)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / EXHAUSTION_RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / EXHAUSTION_RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    # Fitil oranlari (exhaustion filtresi icin)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_ratio"] = df["lower_wick_ratio"].fillna(0)
    df["upper_wick_ratio"] = df["upper_wick_ratio"].fillna(0)

    return df


def check_exhaustion(direction: str, row) -> bool:
    """
    Sinyal yonunun tersine bir tukenme (exhaustion) belirtisi var mi kontrol eder.
    SHORT sinyalinde: asiri satim + alt fitil buyukse -> yukari tepki (bounce) ihtimali
    LONG sinyalinde: asiri alim + ust fitil buyukse -> asagi tepki ihtimali
    """
    volume_ratio = row["volume"] / row["vol_sma15"] if row["vol_sma15"] else 0
    if volume_ratio < EXHAUSTION_VOLUME_RATIO:
        return False

    if direction == "SHORT":
        return row["lower_wick_ratio"] >= EXHAUSTION_WICK_RATIO and row["rsi"] <= EXHAUSTION_RSI_LOW
    if direction == "LONG":
        return row["upper_wick_ratio"] >= EXHAUSTION_WICK_RATIO and row["rsi"] >= EXHAUSTION_RSI_HIGH

    return False


def check_breakout(df: pd.DataFrame):
    if len(df) < 55:
        return None

    row = df.iloc[-2]
    prev_rows = df.iloc[-2 - MIN_CONSECUTIVE_CANDLES: -2]

    volume_ok = row["volume"] > row["vol_sma15"] * VOLUME_MULTIPLIER
    body_ok = row["body"] > row["atr14"] * MIN_BODY_ATR_RATIO

    same_color_streak = prev_rows["is_bull"].tolist() + [row["is_bull"]]
    consecutive_bull = all(same_color_streak)
    consecutive_bear = all(not x for x in same_color_streak)

    trend_up = row["ema20"] > row["ema50"]
    trend_down = row["ema20"] < row["ema50"]

    price_above_ema20 = row["close"] > row["ema20"]
    price_below_ema20 = row["close"] < row["ema20"]

    if volume_ok and body_ok and consecutive_bull and trend_up and price_above_ema20:
        return "LONG", row
    if volume_ok and body_ok and consecutive_bear and trend_down and price_below_ema20:
        return "SHORT", row

    return None


SIGNAL_LOG_FILE = "signal_history.csv"


def log_signal(symbol: str, direction: str, row, exhausted: bool):
    file_exists = os.path.isfile(SIGNAL_LOG_FILE)
    with open(SIGNAL_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "symbol", "direction", "price", "volume",
                "vol_sma15", "atr14", "ema20", "ema50", "rsi", "exhausted"
            ])
        writer.writerow([
            datetime.now().isoformat(), symbol, direction, row["close"],
            row["volume"], row["vol_sma15"], row["atr14"], row["ema20"],
            row["ema50"], row["rsi"], exhausted
        ])


def scan_once():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Tarama basliyor...")
    for symbol in WATCHLIST:
        try:
            df = fetch_data(symbol)
            df = compute_indicators(df)
            result = check_breakout(df)

            if result:
                direction, row = result
                exhausted = check_exhaustion(direction, row)
                log_signal(symbol, direction, row, exhausted)

                msg = (
                    f"{symbol} - {direction} sinyali\n"
                    f"Fiyat: {row['close']:.4f}\n"
                    f"Hacim: {row['volume']:.0f} (SMA15: {row['vol_sma15']:.0f})\n"
                    f"ATR14: {row['atr14']:.4f}\n"
                    f"EMA20/50: {row['ema20']:.4f} / {row['ema50']:.4f}\n"
                    f"RSI({EXHAUSTION_RSI_PERIOD}): {row['rsi']:.1f}\n"
                    f"Zaman dilimi: {TIMEFRAME}"
                )
                if exhausted:
                    ters_yon = "LONG" if direction == "SHORT" else "SHORT"
                    msg += (
                        f"\n\n⚠️ Olası tükenme belirtisi.\n"
                        f"Hareket zaten ilerlemiş olabilir, {ters_yon} yönünde "
                        f"tepki (bounce) ihtimali göz önünde bulundurulabilir."
                    )
                print(msg)
                send_telegram_message(msg)
            else:
                print(f"{symbol}: kriter yok")

        except Exception as e:
            print(f"{symbol} hata: {e}")


def run_forever():
    send_telegram_message("Kripto kirilim botu baslatildi.")
    while True:
        scan_once()
        time.sleep(CHECK_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    run_forever()