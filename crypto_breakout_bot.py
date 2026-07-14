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

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError(
        "TELEGRAM_TOKEN ve TELEGRAM_CHAT_ID ortam degiskenleri tanimli degil. "
        "Railway'de Variables kismindan ekle."
    )

# ---------------------------------------------------------------------------
# STRATEJI: Trend takibi / momentum kirilimi (tersine bahis DEGIL)
#
# Eski sistem "fiyat asiri uca gitti, tersine doner" diye bahis oynuyordu.
# Bu sistem tam tersi: "fiyat guclu ve teyitli bir kirilim yapti, kisa sure
# bu yonde devam eder" diye giriyor. Trendin tersine degil, yaninda durur.
# ---------------------------------------------------------------------------

COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "SUI",
    "DOT", "TRX", "ATOM", "NEAR", "TON", "LTC", "BCH", "ETC", "FIL", "APT",
    "ARB", "OP", "INJ", "SEI", "ICP", "HBAR", "VET", "ALGO", "XLM", "EOS",
    "XTZ", "SAND", "MANA", "AAVE", "UNI", "CRV", "GRT", "THETA", "EGLD",
    "FLOW", "CHZ", "DYDX", "GALA", "IMX", "ONDO", "WLD",
    "PEPE", "SHIB", "TIA", "STRK", "JUP", "PYTH", "JTO", "ENA", "ETHFI", "ORDI",
    "BLUR", "LDO", "RPL", "FXS", "SSV", "CFX", "WOO", "GMX", "ZRX", "BAT",
    "ENJ", "ZIL", "KDA", "ROSE", "ANKR", "CELO", "IOTA", "IOTX", "QTUM", "1INCH",
    "COMP", "SNX", "YFI", "BAL", "STORJ", "OCEAN", "MASK", "LRC", "GMT", "APE",
    "RSR", "SKL", "CTSI", "MTL", "DENT", "HOT", "RVN", "ICX", "ONT", "WAVES",
    "KSM", "ZEC", "DASH", "MINA",
    "ARKM", "AR", "RENDER", "AKT", "FET", "AGIX", "TAO", "NOT", "DOGS",
    "FLOKI", "BONK", "WIF", "BOME", "MEME", "TURBO", "1000SATS", "PENDLE",
    "ENS", "API3", "BAND", "UMA", "REN", "KNC", "SUSHI", "CAKE", "JOE", "RAY",
    "SRM", "ALPHA", "BADGER", "ALCX", "TRB", "OXT", "NKN", "CTK", "COTI",
    "ARPA", "LIT", "DUSK", "PERP", "MDT", "POLYX", "POWR", "REQ", "STMX",
    "STPT", "TLM", "ALICE", "AXS", "SLP", "ILV", "YGG", "MAGIC", "PRIME",
    "SUPER", "GHST", "AUDIO", "RLC", "NMR", "ORCA", "RAD", "GLMR", "MOVR",
    "ASTR", "ACA", "PHA", "KLAY", "ONE", "FTM", "METIS", "BOBA", "CELR",
]

WATCHLIST = [f"{c}/USDT:USDT" for c in COINS]

TIMEFRAME = "15m"
CHECK_INTERVAL_MINUTES = 15

RSI_PERIOD = 14
ATR_PERIOD = 14

# Kirilim (breakout) tanimi
BREAKOUT_LOOKBACK = 20          # son N mumun en yuksek/dusuk kapanisini kirmali
BREAKOUT_VOLUME_MULTIPLIER = 2.0
BREAKOUT_BODY_ATR_MULTIPLIER = 0.6   # mum govdesi ATR'nin bu katindan buyuk olmali (kararli hareket)
BREAKOUT_CLOSE_POSITION = 0.65        # mum kendi araliginin ust/alt %65'inde kapanmali (fitilli degil, guclu kapanis)

# RSI "momentum insa oluyor ama henuz tukenmedi" bolgesi
LONG_RSI_MIN, LONG_RSI_MAX = 50, 78
SHORT_RSI_MIN, SHORT_RSI_MAX = 22, 50

# Coin'in kendi 1h trendi kirilim yonunu desteklemeli (ya da en azindan karsi olmamali)
TREND_FILTER_TIMEFRAME = "1h"
TREND_EMA_GAP_THRESHOLD = 1.5   # bu esigin tersine guclu trend varsa kirilim reddedilir

INVALIDATION_ATR_BUFFER = 1.0   # kirilim seviyesinin gerisine ATR'nin bu kati kadar tampon

ORDERBOOK_IMBALANCE_RATIO = 1.2

TARGET_HOLD_MINUTES = 20
MAX_HOLD_MINUTES = 45

CHECK_MINUTES = [15, 30, 45]
SUCCESS_THRESHOLD_PCT = 0.15

exchange = ccxt.okx({
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
})

_unsupported_symbols = set()
_reminders = {}


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Telegram gonderim hatasi: {e}")


# ---------------------------------------------------------------------------
# Veri ve indikatorler
# ---------------------------------------------------------------------------

def fetch_ohlcv_df(symbol: str, timeframe: str, limit: int = 100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
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

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_position"] = (df["close"] - df["low"]) / candle_range
    df["close_position"] = df["close_position"].fillna(0.5)

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    # Kirilim seviyeleri: SON mum haric onceki N mumun en yuksek/dusuk kapanisi
    df["breakout_high"] = df["close"].shift(1).rolling(BREAKOUT_LOOKBACK).max()
    df["breakout_low"] = df["close"].shift(1).rolling(BREAKOUT_LOOKBACK).min()

    return df


# ---------------------------------------------------------------------------
# Kirilim kapisi (mandatory gate) - trend YONUNDE giris
# ---------------------------------------------------------------------------

def check_breakout_gate(df: pd.DataFrame):
    """
    Son KAPANMIS muma bakar (df.iloc[-2]). Tum sartlar birden tutmali:
      1. Kapanis, son N mumun en yuksek/dusuk kapanisini kirmis olmali
      2. Hacim ortalamanin en az BREAKOUT_VOLUME_MULTIPLIER kati olmali
      3. Mum govdesi ATR'ye gore guclu olmali (kararsiz/dojivari degil)
      4. Mum kendi araliginin guclu tarafinda kapanmis olmali (fitil degil, gercek kirilim)
      5. RSI "momentum insa oluyor" bolgesinde olmali (henuz asiri tukenmis degil)
    """
    if len(df) < BREAKOUT_LOOKBACK + 5:
        return None

    row = df.iloc[-2]
    if pd.isna(row["breakout_high"]) or pd.isna(row["breakout_low"]) or pd.isna(row["atr14"]):
        return None

    volume_ratio = row["volume"] / row["vol_sma20"] if row["vol_sma20"] else 0
    if volume_ratio < BREAKOUT_VOLUME_MULTIPLIER:
        return None

    body_ratio = row["body"] / row["atr14"] if row["atr14"] > 0 else 0
    if body_ratio < BREAKOUT_BODY_ATR_MULTIPLIER:
        return None

    # LONG: yukari kirilim
    if (row["close"] > row["breakout_high"]
            and row["close_position"] >= BREAKOUT_CLOSE_POSITION
            and LONG_RSI_MIN <= row["rsi"] <= LONG_RSI_MAX):
        return "LONG", row

    # SHORT: asagi kirilim
    if (row["close"] < row["breakout_low"]
            and row["close_position"] <= (1 - BREAKOUT_CLOSE_POSITION)
            and SHORT_RSI_MIN <= row["rsi"] <= SHORT_RSI_MAX):
        return "SHORT", row

    return None


def get_symbol_trend(symbol: str):
    """Coin'in kendi 1h trendine bakar. Kirilim bu trendin tersine olmamali."""
    try:
        df1h = fetch_ohlcv_df(symbol, TREND_FILTER_TIMEFRAME, limit=60)
        df1h = compute_indicators(df1h)
        row = df1h.iloc[-2]
        if pd.isna(row["ema50"]) or row["ema50"] == 0:
            return "BILINMIYOR", 0.0
        gap_pct = (row["ema20"] - row["ema50"]) / row["ema50"] * 100
        if gap_pct <= -TREND_EMA_GAP_THRESHOLD:
            return "DUSUS", gap_pct
        if gap_pct >= TREND_EMA_GAP_THRESHOLD:
            return "YUKSELIS", gap_pct
        return "YATAY", gap_pct
    except Exception as e:
        print(f"{symbol} icin 1h trend alinamadi: {e}")
        return "BILINMIYOR", 0.0


def score_orderbook(symbol: str, direction: str) -> tuple:
    try:
        ob = exchange.fetch_order_book(symbol, limit=20)
        bid_vol = sum(b[1] for b in ob["bids"])
        ask_vol = sum(a[1] for a in ob["asks"])
        if ask_vol == 0 or bid_vol == 0:
            return False, "veri yetersiz"
        ratio = bid_vol / ask_vol
        if direction == "LONG" and ratio >= ORDERBOOK_IMBALANCE_RATIO:
            return True, f"bid/ask {ratio:.2f} (alici agirlikli, destekliyor)"
        if direction == "SHORT" and ratio <= 1 / ORDERBOOK_IMBALANCE_RATIO:
            return True, f"bid/ask {ratio:.2f} (satici agirlikli, destekliyor)"
        return False, f"bid/ask {ratio:.2f} (notr)"
    except Exception as e:
        return False, f"order book alinamadi ({e})"


def compute_invalidation(direction: str, row) -> float:
    atr = row["atr14"] if pd.notna(row["atr14"]) else 0
    buffer = atr * INVALIDATION_ATR_BUFFER
    if direction == "LONG":
        return row["breakout_high"] - buffer
    return row["breakout_low"] + buffer


# ---------------------------------------------------------------------------
# Loglama
# ---------------------------------------------------------------------------

SIGNAL_LOG_FILE = "signal_history.csv"
PENDING_FILE = "pending_signals.csv"
OUTCOME_FILE = "signal_outcomes.csv"


def log_signal(symbol: str, direction: str, row, breakdown: list):
    file_exists = os.path.isfile(SIGNAL_LOG_FILE)
    with open(SIGNAL_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "direction", "price", "rsi", "breakdown"])
        writer.writerow([
            datetime.now().isoformat(), symbol, direction, row["close"], row["rsi"], " | ".join(breakdown)
        ])


def log_pending(symbol: str, direction: str, entry_price: float, entry_time: datetime, invalidation: float):
    file_exists = os.path.isfile(PENDING_FILE)
    with open(PENDING_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "symbol", "direction", "entry_price", "entry_time", "invalidation",
                "checked_15m", "checked_30m", "checked_45m"
            ])
        writer.writerow([symbol, direction, entry_price, entry_time.isoformat(), invalidation, "0", "0", "0"])


def _read_pending():
    if not os.path.isfile(PENDING_FILE):
        return []
    with open(PENDING_FILE, newline="") as f:
        return list(csv.DictReader(f))


def _write_pending(rows):
    fieldnames = ["symbol", "direction", "entry_price", "entry_time", "invalidation",
                  "checked_15m", "checked_30m", "checked_45m"]
    with open(PENDING_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def log_outcome(symbol, direction, entry_price, entry_time, minutes, current_price, pct_change, success):
    file_exists = os.path.isfile(OUTCOME_FILE)
    with open(OUTCOME_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "symbol", "direction", "entry_price", "entry_time", "minutes_after",
                "price_now", "pct_change", "success"
            ])
        writer.writerow([
            symbol, direction, entry_price, entry_time, minutes, current_price, f"{pct_change:.3f}", success
        ])


def check_pending_outcomes():
    rows = _read_pending()
    if not rows:
        pass
    else:
        now = datetime.now()
        still_pending = []
        for r in rows:
            entry_time = datetime.fromisoformat(r["entry_time"])
            entry_price = float(r["entry_price"])
            symbol = r["symbol"]
            direction = r["direction"]
            all_checked = True
            for m in CHECK_MINUTES:
                flag_key = f"checked_{m}m"
                if r.get(flag_key, "0") == "1":
                    continue
                all_checked = False
                if now >= entry_time + timedelta(minutes=m):
                    try:
                        ticker = exchange.fetch_ticker(symbol)
                        current_price = ticker["last"]
                        pct_change = (current_price - entry_price) / entry_price * 100
                        if direction == "LONG":
                            success = pct_change >= SUCCESS_THRESHOLD_PCT
                        else:
                            success = pct_change <= -SUCCESS_THRESHOLD_PCT
                        log_outcome(symbol, direction, entry_price, r["entry_time"], m,
                                    current_price, pct_change, success)
                        r[flag_key] = "1"
                    except Exception as e:
                        print(f"{symbol} sonuc kontrolu hatasi: {e}")
            if not all_checked:
                still_pending.append(r)
        _write_pending(still_pending)

    now = datetime.now()
    still_reminding = {}
    for symbol, info in _reminders.items():
        if now >= info["remind_at"]:
            try:
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker["last"]
                pct_change = (current_price - info["entry_price"]) / info["entry_price"] * 100
                if info["direction"] == "SHORT":
                    pct_change = -pct_change
                msg = (
                    f"⏰ {symbol} - hedef süre doldu (~{TARGET_HOLD_MINUTES} dk)\n"
                    f"Giriş: {info['entry_price']:.4f} | Şimdi: {current_price:.4f}\n"
                    f"Yöndeki değişim: {pct_change:+.2f}%\n\n"
                    f"Pozisyonu gözden geçirme zamanı (en fazla {MAX_HOLD_MINUTES} dk kuralın)."
                )
                send_telegram_message(msg)
            except Exception as e:
                print(f"{symbol} hatirlatma hatasi: {e}")
        else:
            still_reminding[symbol] = info
    _reminders.clear()
    _reminders.update(still_reminding)


# ---------------------------------------------------------------------------
# Ana tarama
# ---------------------------------------------------------------------------

def scan_once():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Tarama basliyor...")

    check_pending_outcomes()

    for symbol in WATCHLIST:
        if symbol in _unsupported_symbols:
            continue
        try:
            df = fetch_ohlcv_df(symbol, TIMEFRAME, limit=100)
            df = compute_indicators(df)

            gate_result = check_breakout_gate(df)
            if not gate_result:
                print(f"{symbol}: kirilim yok")
                continue

            direction, row = gate_result

            symbol_trend, trend_gap = get_symbol_trend(symbol)
            if direction == "LONG" and symbol_trend == "DUSUS":
                print(f"{symbol}: LONG kirilim var ama 1h trend dususte ({trend_gap:+.1f}%), reddedildi")
                continue
            if direction == "SHORT" and symbol_trend == "YUKSELIS":
                print(f"{symbol}: SHORT kirilim var ama 1h trend yukseliste ({trend_gap:+.1f}%), reddedildi")
                continue

            breakdown = [
                f"✅ Kırılım seviyesi: {row['breakout_high' if direction == 'LONG' else 'breakout_low']:.4f}",
                f"✅ Hacim {row['volume']/row['vol_sma20']:.2f}x ortalama",
                f"✅ Mum gövdesi {row['body']/row['atr14']:.2f}x ATR",
                f"✅ Kapanış konumu: {row['close_position']:.2f}",
                f"✅ RSI: {row['rsi']:.1f} (momentum bölgesinde)",
                f"{'✅' if symbol_trend != 'BILINMIYOR' else '➖'} 1h trend: {symbol_trend} ({trend_gap:+.1f}%)",
            ]

            ob_support, ob_note = score_orderbook(symbol, direction)
            breakdown.append(f"{'✅' if ob_support else '➖'} Order book: {ob_note}")

            log_signal(symbol, direction, row, breakdown)

            invalidation = compute_invalidation(direction, row)
            yon_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
            breakdown_text = "\n".join(f"- {b}" for b in breakdown)

            msg = (
                f"{yon_emoji} {symbol} - MOMENTUM KIRILIMI\n"
                f"(trend yönünde giriş, tersine bahis değil)\n\n"
                f"Giriş fiyatı: {row['close']:.4f}\n"
                f"Geçersizlik seviyesi: {invalidation:.4f}\n\n"
                f"⏱ Hedef tutuş: ~{TARGET_HOLD_MINUTES} dk | En fazla: {MAX_HOLD_MINUTES} dk\n\n"
                f"Teyit detayları:\n{breakdown_text}"
            )
            print(msg)
            send_telegram_message(msg)
            log_pending(symbol, direction, row["close"], datetime.now(), invalidation)
            _reminders[symbol] = {
                "entry_price": row["close"],
                "direction": direction,
                "remind_at": datetime.now() + timedelta(minutes=TARGET_HOLD_MINUTES),
            }

        except Exception as e:
            if "does not have" in str(e).lower():
                _unsupported_symbols.add(symbol)
                print(f"{symbol}: bu borsada islem gormuyor, listeden cikarildi")
            else:
                print(f"{symbol} hata: {e}")


def run_forever():
    send_telegram_message(
        "Kripto MOMENTUM botu (trend takibi) başlatıldı.\n"
        f"{len(WATCHLIST)} coin taranıyor.\n\n"
        "Strateji değişti: artık tersine bahis değil, trend yönünde giriş yapılıyor.\n"
        f"Kırılım şartları: hacim {BREAKOUT_VOLUME_MULTIPLIER}x+, güçlü gövde, güçlü kapanış, "
        "RSI momentum bölgesinde, coin'in kendi 1h trendi karşı yönde olmamalı.\n\n"
        f"Hedef tutuş: ~{TARGET_HOLD_MINUTES} dk, en fazla {MAX_HOLD_MINUTES} dk — "
        f"{TARGET_HOLD_MINUTES}. dakikada otomatik hatırlatma gelecek."
    )
    while True:
        scan_once()
        time.sleep(CHECK_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    run_forever()
