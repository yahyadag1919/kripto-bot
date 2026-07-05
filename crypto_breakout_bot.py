import ccxt
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime

TELEGRAM_TOKEN = "8642212045:AAF-7RSZ9-3IWL-1Dn6SnApPWFKTdo9vqiI"
TELEGRAM_CHAT_ID = "1959583102"

WATCHLIST = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "ADA/USDT", "SUI/USDT",
]

TIMEFRAME = "15m"
CHECK_INTERVAL_MINUTES = 15

VOLUME_MULTIPLIER = 1.2
MIN_CONSECUTIVE_CANDLES = 2
MIN_BODY_ATR_RATIO = 0.5

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
    return df


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


def scan_once():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Tarama basliyor...")
    for symbol in WATCHLIST:
        try:
            df = fetch_data(symbol)
            df = compute_indicators(df)
            result = check_breakout(df)

            if result:
                direction, row = result
                emoji = "green" if direction == "LONG" else "red"
                msg = (
                    f"{symbol} - {direction} sinyali\n"
                    f"Fiyat: {row['close']:.4f}\n"
                    f"Hacim: {row['volume']:.0f} (SMA15: {row['vol_sma15']:.0f})\n"
                    f"ATR14: {row['atr14']:.4f}\n"
                    f"EMA20/50: {row['ema20']:.4f} / {row['ema50']:.4f}\n"
                    f"Zaman dilimi: {TIMEFRAME}"
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