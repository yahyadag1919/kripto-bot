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
# STRATEJI: RSI + VWAP Birlesik Teyit (9 turluk, 66 stratejilik turnuvada
# en yuksek isabet oranini gosteren sistem - %65.4 isabet, en dusuk ort. zarar)
#
# Fiyat, kayan VWAP'tan (hacim agirlikli ortalama fiyat) belirgin sekilde
# sapmis VE RSI ayni yonde asiri iken, tersine (bounce) giris yapilir.
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
VWAP_WINDOW = 96          # ~24 saat (15m mumla) - kayan VWAP penceresi

# Turnuvada test edilen esikler
RSI_LONG_MAX = 30
RSI_SHORT_MIN = 70
VWAP_DEV_LONG_MAX = -2.0   # VWAP'in en az %2 altinda
VWAP_DEV_SHORT_MIN = 2.0   # VWAP'in en az %2 ustunde

INVALIDATION_ATR_BUFFER = 1.0

ORDERBOOK_IMBALANCE_RATIO = 1.2

# 10. tur (uzun tutus) sonuclarina gore checkpoint bazli cikis:
# (checkpoint_dakika, hedef_yuzde, etiket)
CHECKPOINTS = [
    (60, 0.3, "1sa"),
    (240, 0.6, "4sa"),
    (720, 1.0, "12sa"),
    (1440, 1.5, "24sa"),
]
MAX_HOLD_MINUTES = CHECKPOINTS[-1][0]

exchange = ccxt.okx({
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
})

_unsupported_symbols = set()


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
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]
    df["vwap"] = pv.rolling(VWAP_WINDOW).sum() / df["volume"].rolling(VWAP_WINDOW).sum()
    df["vwap_dev_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100

    return df


# ---------------------------------------------------------------------------
# RSI + VWAP birlesik teyit kapisi
# ---------------------------------------------------------------------------

def check_breakout_gate(df: pd.DataFrame):
    """
    Son KAPANMIS muma bakar (df.iloc[-2]).
    LONG: RSI asiri satimda VE fiyat VWAP'in belirgin altinda
    SHORT: RSI asiri alimda VE fiyat VWAP'in belirgin ustunde
    """
    if len(df) < VWAP_WINDOW + 5:
        return None

    row = df.iloc[-2]
    if pd.isna(row["vwap_dev_pct"]) or pd.isna(row["rsi"]) or pd.isna(row["atr14"]):
        return None

    if row["rsi"] <= RSI_LONG_MAX and row["vwap_dev_pct"] <= VWAP_DEV_LONG_MAX:
        return "LONG", row

    if row["rsi"] >= RSI_SHORT_MIN and row["vwap_dev_pct"] >= VWAP_DEV_SHORT_MIN:
        return "SHORT", row

    return None


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
        return row["close"] - buffer
    return row["close"] + buffer


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


PENDING_FIELDNAMES = ["symbol", "direction", "entry_price", "entry_time", "invalidation"] + [
    f"checked_{label}" for _, _, label in CHECKPOINTS
] + ["closed"]


def log_pending(symbol: str, direction: str, entry_price: float, entry_time: datetime, invalidation: float):
    file_exists = os.path.isfile(PENDING_FILE)
    with open(PENDING_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(PENDING_FIELDNAMES)
        row = [symbol, direction, entry_price, entry_time.isoformat(), invalidation]
        row += ["0" for _ in CHECKPOINTS]
        row += ["0"]
        writer.writerow(row)


def _read_pending():
    if not os.path.isfile(PENDING_FILE):
        return []
    with open(PENDING_FILE, newline="") as f:
        return list(csv.DictReader(f))


def _write_pending(rows):
    with open(PENDING_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PENDING_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def log_outcome(symbol, direction, entry_price, entry_time, minutes, label, target_pct,
                 current_price, pct_change, success):
    file_exists = os.path.isfile(OUTCOME_FILE)
    with open(OUTCOME_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "symbol", "direction", "entry_price", "entry_time", "minutes_after", "checkpoint",
                "target_pct", "price_now", "pct_change", "success"
            ])
        writer.writerow([
            symbol, direction, entry_price, entry_time, minutes, label,
            target_pct, current_price, f"{pct_change:.3f}", success
        ])


def check_pending_outcomes():
    rows = _read_pending()
    if not rows:
        return

    now = datetime.now()
    still_pending = []

    for r in rows:
        if r.get("closed", "0") == "1":
            continue

        entry_time = datetime.fromisoformat(r["entry_time"])
        entry_price = float(r["entry_price"])
        symbol = r["symbol"]
        direction = r["direction"]
        closed = False

        for minutes, target_pct, label in CHECKPOINTS:
            flag_key = f"checked_{label}"
            if r.get(flag_key, "0") == "1":
                continue
            if now < entry_time + timedelta(minutes=minutes):
                # bu checkpoint'e daha ulasilmadi, sonraki checkpoint'ler de beklemede
                break

            try:
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker["last"]
                raw_pct_change = (current_price - entry_price) / entry_price * 100
                pct_change = raw_pct_change if direction == "LONG" else -raw_pct_change
                success = pct_change >= target_pct

                log_outcome(symbol, direction, entry_price, r["entry_time"], minutes, label,
                            target_pct, current_price, pct_change, success)
                r[flag_key] = "1"

                if success:
                    msg = (
                        f"🎯 {symbol} {direction} - {label} checkpoint'te hedef tutturuldu\n"
                        f"Giriş: {entry_price:.4f} | Şimdi: {current_price:.4f}\n"
                        f"Değişim: {pct_change:+.2f}% (hedef: %{target_pct})\n\n"
                        f"Öneri: kârı realize etmeyi değerlendir."
                    )
                    send_telegram_message(msg)
                    r["closed"] = "1"
                    closed = True
                    break
                elif label == CHECKPOINTS[-1][2]:
                    msg = (
                        f"⏱ {symbol} {direction} - 24sa sonunda hiçbir checkpoint'te hedef tutmadı\n"
                        f"Giriş: {entry_price:.4f} | Şimdi: {current_price:.4f}\n"
                        f"Son değişim: {pct_change:+.2f}%\n\n"
                        f"Sinyal geçersiz sayılıyor, kapatılıyor."
                    )
                    send_telegram_message(msg)
                    r["closed"] = "1"
                    closed = True
            except Exception as e:
                print(f"{symbol} sonuc kontrolu hatasi: {e}")
                break

        if not closed:
            still_pending.append(r)

    _write_pending(still_pending)


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
                print(f"{symbol}: kriter yok")
                continue

            direction, row = gate_result

            breakdown = [
                f"✅ RSI: {row['rsi']:.1f} ({'asiri satim' if direction == 'LONG' else 'asiri alim'})",
                f"✅ VWAP sapması: %{row['vwap_dev_pct']:+.2f}",
                f"✅ Hacim {row['volume']/row['vol_sma20']:.2f}x ortalama" if pd.notna(row.get('vol_sma20')) and row.get('vol_sma20') else "➖ Hacim verisi yetersiz",
            ]

            ob_support, ob_note = score_orderbook(symbol, direction)
            breakdown.append(f"{'✅' if ob_support else '➖'} Order book: {ob_note}")

            log_signal(symbol, direction, row, breakdown)

            invalidation = compute_invalidation(direction, row)
            yon_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
            breakdown_text = "\n".join(f"- {b}" for b in breakdown)

            checkpoint_text = " / ".join(f"{label}(%{target})" for _, target, label in CHECKPOINTS)
            msg = (
                f"{yon_emoji} {symbol} - RSI+VWAP TÜKENME sinyali (bounce)\n"
                f"(10. tur / uzun tutuş sonuçlarına göre checkpoint bazlı çıkış)\n\n"
                f"Giriş fiyatı: {row['close']:.4f}\n"
                f"Geçersizlik seviyesi: {invalidation:.4f}\n\n"
                f"⏱ Checkpoint hedefleri: {checkpoint_text}\n"
                f"İlk tutan hedefte pozisyon kapanmış sayılır, en geç 24sa'da değerlendirme gelir.\n\n"
                f"Teyit detayları:\n{breakdown_text}"
            )
            print(msg)
            send_telegram_message(msg)
            log_pending(symbol, direction, row["close"], datetime.now(), invalidation)

        except Exception as e:
            if "does not have" in str(e).lower():
                _unsupported_symbols.add(symbol)
                print(f"{symbol}: bu borsada islem gormuyor, listeden cikarildi")
            else:
                print(f"{symbol} hata: {e}")


def run_forever():
    checkpoint_text = " / ".join(f"{label}(%{target})" for _, target, label in CHECKPOINTS)
    send_telegram_message(
        "Kripto botu (RSI+VWAP Birleşik Teyit) başlatıldı.\n"
        f"{len(WATCHLIST)} coin taranıyor.\n\n"
        "Strateji: 10. tur (uzun tutuş) sonuçlarına göre checkpoint bazlı çıkış.\n"
        f"Şart: RSI≤{RSI_LONG_MAX}/≥{RSI_SHORT_MIN} VE fiyat kayan VWAP'tan %2+ sapmış.\n\n"
        f"Checkpoint hedefleri: {checkpoint_text}\n"
        f"En fazla {MAX_HOLD_MINUTES // 60}sa tutuş, her checkpoint'te otomatik durum bildirimi gelecek."
    )
    while True:
        scan_once()
        time.sleep(CHECK_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    run_forever()
