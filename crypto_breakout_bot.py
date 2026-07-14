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
# Ayarlar
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

# Tukenme kapisi esikleri (tek seviye - yuksek kalite, gevsek kademe kaldirildi)
GATE_WICK_RATIO = 0.5
GATE_VOLUME_RATIO = 2.5
GATE_RSI_LOW = 20
GATE_RSI_HIGH = 80

RSI_PERIOD = 6
ATR_PERIOD = 14

# Onay mumu sonrasi ek "onay gucu" puanlamasi (0-4)
MIN_CONFIRMATION_STRENGTH = 2
MAX_CONFIRMATION_STRENGTH = 5
ORDERBOOK_IMBALANCE_RATIO = 1.3

# Gecersizlik seviyesi - 15m ATR'nin bu kati kadar tampon (hizli tutus icin makul genislikte)
INVALIDATION_ATR_BUFFER = 1.2

# Senin islem tarzin: anlik giris, kisa tutus
TARGET_HOLD_MINUTES = 20
MAX_HOLD_MINUTES = 45

# Performans takibi / hatirlatma zamanlari (dakika)
CHECK_MINUTES = [15, 30, 45]
SUCCESS_THRESHOLD_PCT = 0.15

exchange = ccxt.okx({
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
})

_unsupported_symbols = set()
_candidates = {}   # tukenme adaylari - onay mumu bekleniyor
_reminders = {}    # gonderilen sinyaller icin hatirlatma zamanlamasi


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
    df["vol_sma15"] = df["volume"].rolling(15).mean()

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

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_ratio"] = df["lower_wick_ratio"].fillna(0)
    df["upper_wick_ratio"] = df["upper_wick_ratio"].fillna(0)

    return df


# ---------------------------------------------------------------------------
# Tukenme kapisi (mandatory gate)
# ---------------------------------------------------------------------------

def check_exhaustion_gate(df: pd.DataFrame):
    """Uc sart birden tutmali: hacim patlamasi + fitil + RSI asiri uc."""
    if len(df) < 25:
        return None

    row = df.iloc[-2]
    volume_ratio = row["volume"] / row["vol_sma15"] if row["vol_sma15"] else 0
    if volume_ratio < GATE_VOLUME_RATIO:
        return None

    if row["lower_wick_ratio"] >= GATE_WICK_RATIO and row["rsi"] <= GATE_RSI_LOW:
        return "LONG", row
    if row["upper_wick_ratio"] >= GATE_WICK_RATIO and row["rsi"] >= GATE_RSI_HIGH:
        return "SHORT", row

    return None


# ---------------------------------------------------------------------------
# Onay mumu mekanizmasi
# ---------------------------------------------------------------------------

def check_candidate_confirmation(symbol: str, df: pd.DataFrame):
    """
    Bekleyen bir tukenme adayi varsa, en son kapanan mumun beklenen yonde
    kapanip kapanmadigina bakar.
    Donus: None ya da (status, direction, confirm_row, exhaustion_row)
    """
    candidate = _candidates.get(symbol)
    if not candidate:
        return None

    latest_row = df.iloc[-2]
    if latest_row["timestamp"] <= candidate["candle_time"]:
        return None

    direction = candidate["direction"]
    exhaustion_row = candidate["exhaustion_row"]
    del _candidates[symbol]

    confirmed = (
        (direction == "LONG" and bool(latest_row["is_bull"])) or
        (direction == "SHORT" and not bool(latest_row["is_bull"]))
    )
    status = "confirmed" if confirmed else "rejected"
    return (status, direction, latest_row, exhaustion_row)


# ---------------------------------------------------------------------------
# Onay gucu puanlamasi - sadece HIZLI, ilgili veriler (yavas trend filtreleri yok)
# ---------------------------------------------------------------------------

def score_confirmation_strength(direction: str, confirm_row, exhaustion_row) -> tuple:
    """
    Onay mumunun gercekten guclu bir donusu mu yoksa zayif bir tepkimi
    oldugunu olcer. Puanlar (0-4), her biri +1:
      1. Onay mumunun govdesi ATR'ye gore guclu mu (kararli hareket)
      2. Onay mumunun hacmi, tukenme mumunun hacminin en az %70'i mi (ilgi devam ediyor)
      3. RSI, tukenme anindan beri anlamli sekilde toparlandi mi (momentum gercekten donuyor)
      4. Fiyat, tukenme mumunun govde ortasini gecti mi (yarim kalan degil, tam donus)
    """
    score = 0
    breakdown = []

    atr = exhaustion_row["atr14"] if pd.notna(exhaustion_row["atr14"]) and exhaustion_row["atr14"] > 0 else None
    if atr:
        body_ratio = confirm_row["body"] / atr
        if body_ratio >= 0.5:
            score += 1
            breakdown.append(f"Onay mumu govdesi guclu ({body_ratio:.2f}x ATR) (+1)")
        else:
            breakdown.append(f"Onay mumu govdesi zayif ({body_ratio:.2f}x ATR) (+0)")
    else:
        breakdown.append("ATR verisi yetersiz (+0)")

    if exhaustion_row["volume"] > 0:
        vol_ratio = confirm_row["volume"] / exhaustion_row["volume"]
        if vol_ratio >= 0.7:
            score += 1
            breakdown.append(f"Hacim devam ediyor ({vol_ratio:.2f}x) (+1)")
        else:
            breakdown.append(f"Hacim zayifliyor ({vol_ratio:.2f}x) (+0)")
    else:
        breakdown.append("Hacim verisi yetersiz (+0)")

    rsi_shift = confirm_row["rsi"] - exhaustion_row["rsi"]
    if direction == "LONG" and rsi_shift >= 5:
        score += 1
        breakdown.append(f"RSI toparlaniyor ({rsi_shift:+.1f}) (+1)")
    elif direction == "SHORT" and rsi_shift <= -5:
        score += 1
        breakdown.append(f"RSI geriliyor ({rsi_shift:+.1f}) (+1)")
    else:
        breakdown.append(f"RSI degisimi zayif ({rsi_shift:+.1f}) (+0)")

    exhaustion_mid = (exhaustion_row["open"] + exhaustion_row["close"]) / 2
    if direction == "LONG" and confirm_row["close"] > exhaustion_mid:
        score += 1
        breakdown.append("Fiyat tukenme mumunun govde ortasini gecti (+1)")
    elif direction == "SHORT" and confirm_row["close"] < exhaustion_mid:
        score += 1
        breakdown.append("Fiyat tukenme mumunun govde ortasini gecti (+1)")
    else:
        breakdown.append("Fiyat henuz govde ortasini gecmedi (+0)")

    return score, breakdown


def score_orderbook(symbol: str, direction: str) -> tuple:
    """Emir defterinde anlik alici/satici baskisi yonu destekliyor mu (hizli giris icin onemli)."""
    try:
        ob = exchange.fetch_order_book(symbol, limit=20)
        bid_vol = sum(b[1] for b in ob["bids"])
        ask_vol = sum(a[1] for a in ob["asks"])
        if ask_vol == 0 or bid_vol == 0:
            return 0, "veri yetersiz"

        ratio = bid_vol / ask_vol
        if direction == "LONG" and ratio >= ORDERBOOK_IMBALANCE_RATIO:
            return 1, f"bid/ask {ratio:.2f} (alici agirlikli, destekliyor)"
        if direction == "SHORT" and ratio <= 1 / ORDERBOOK_IMBALANCE_RATIO:
            return 1, f"bid/ask {ratio:.2f} (satici agirlikli, destekliyor)"
        return 0, f"bid/ask {ratio:.2f} (destek yok)"
    except Exception as e:
        return 0, f"order book alinamadi ({e})"


def compute_invalidation(direction: str, row) -> float:
    atr = row["atr14"] if pd.notna(row["atr14"]) else 0
    buffer = atr * INVALIDATION_ATR_BUFFER
    if direction == "LONG":
        return row["low"] - buffer
    return row["high"] + buffer


# ---------------------------------------------------------------------------
# Loglama
# ---------------------------------------------------------------------------

SIGNAL_LOG_FILE = "signal_history.csv"
PENDING_FILE = "pending_signals.csv"
OUTCOME_FILE = "signal_outcomes.csv"


def log_signal(symbol: str, direction: str, row, score: int, breakdown: list):
    file_exists = os.path.isfile(SIGNAL_LOG_FILE)
    with open(SIGNAL_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "direction", "price", "rsi", "score", "breakdown"])
        writer.writerow([
            datetime.now().isoformat(), symbol, direction, row["close"],
            row["rsi"], score, " | ".join(breakdown)
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
            symbol, direction, entry_price, entry_time, minutes,
            current_price, f"{pct_change:.3f}", success
        ])


def check_pending_outcomes():
    """Bekleyen sinyalleri kontrol eder; 15/30/45 dk'da sonucu loglar, 20 dk'da hatirlatma gonderir."""
    rows = _read_pending()
    if not rows:
        return

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

    # Hedef tutus suresi (~20 dk) hatirlatmasi - senin islem tarzina ozel
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
                    f"⏰ {symbol} - hedef sure doldu (~{TARGET_HOLD_MINUTES} dk)\n"
                    f"Giris: {info['entry_price']:.4f} | Simdi: {current_price:.4f}\n"
                    f"Yondeki degisim: {pct_change:+.2f}%\n\n"
                    f"Pozisyonu gozden gecirme zamani (en fazla {MAX_HOLD_MINUTES} dk kuralin)."
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

            confirmation = check_candidate_confirmation(symbol, df)

            if confirmation is not None:
                status, direction, confirm_row, exhaustion_row = confirmation

                if status == "rejected":
                    print(f"{symbol}: aday onaylanmadi (beklenen yon {direction} degildi), iptal edildi")
                    continue

                score, breakdown = score_confirmation_strength(direction, confirm_row, exhaustion_row)
                pts, note = score_orderbook(symbol, direction)
                score += pts
                breakdown.append(f"Order book: {note} ({pts:+d})")

                log_signal(symbol, direction, confirm_row, score, breakdown)

                if score >= MIN_CONFIRMATION_STRENGTH:
                    invalidation = compute_invalidation(direction, exhaustion_row)
                    yon_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
                    breakdown_text = "\n".join(f"- {b}" for b in breakdown)

                    msg = (
                        f"{yon_emoji} {symbol} - TÜKENME sinyali (bounce)\n"
                        f"✅ Onay mumu ile teyit edildi\n"
                        f"Onay gücü: {score}/{MAX_CONFIRMATION_STRENGTH}\n\n"
                        f"Tükenme fiyatı: {exhaustion_row['close']:.4f} (RSI {exhaustion_row['rsi']:.1f})\n"
                        f"Onay/Giriş fiyatı: {confirm_row['close']:.4f}\n"
                        f"Geçersizlik seviyesi: {invalidation:.4f}\n\n"
                        f"⏱ Hedef tutuş: ~{TARGET_HOLD_MINUTES} dk | En fazla: {MAX_HOLD_MINUTES} dk\n\n"
                        f"Teyit detayları:\n{breakdown_text}"
                    )
                    print(msg)
                    send_telegram_message(msg)
                    log_pending(symbol, direction, confirm_row["close"], datetime.now(), invalidation)
                    _reminders[symbol] = {
                        "entry_price": confirm_row["close"],
                        "direction": direction,
                        "remind_at": datetime.now() + timedelta(minutes=TARGET_HOLD_MINUTES),
                    }
                else:
                    print(f"{symbol}: onaylandi ama onay gucu dusuk ({score}/{MAX_CONFIRMATION_STRENGTH})")

                continue

            gate_result = check_exhaustion_gate(df)
            if not gate_result:
                print(f"{symbol}: kapi gecilmedi")
                continue

            direction, exhaustion_row = gate_result
            _candidates[symbol] = {
                "direction": direction,
                "candle_time": exhaustion_row["timestamp"],
                "exhaustion_row": exhaustion_row,
            }
            print(f"{symbol}: tukenme adayi olustu ({direction}), onay mumu bekleniyor")

        except Exception as e:
            if "does not have" in str(e).lower():
                _unsupported_symbols.add(symbol)
                print(f"{symbol}: bu borsada islem gormuyor, listeden cikarildi")
            else:
                print(f"{symbol} hata: {e}")


def run_forever():
    send_telegram_message(
        "Kripto tükenme botu (hızlı işlem sürümü) başlatıldı.\n"
        f"{len(WATCHLIST)} coin taranıyor.\n"
        f"Minimum onay gücü: {MIN_CONFIRMATION_STRENGTH}/{MAX_CONFIRMATION_STRENGTH}\n"
        f"Tükenme tespit edilince hemen değil, bir sonraki mum onaylarsa sinyal gönderilir.\n"
        f"Hedef tutuş: ~{TARGET_HOLD_MINUTES} dk, en fazla {MAX_HOLD_MINUTES} dk — "
        f"{TARGET_HOLD_MINUTES}. dakikada otomatik hatırlatma gelecek."
    )
    while True:
        scan_once()
        time.sleep(CHECK_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    run_forever()
