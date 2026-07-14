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
    # Ek genisletme - L1/L2, AI, DeFi, oyun, meme
    "ARKM", "AR", "RENDER", "AKT", "FET", "AGIX", "TAO", "NOT", "DOGS",
    "FLOKI", "BONK", "WIF", "BOME", "MEME", "TURBO", "1000SATS", "PENDLE",
    "ENS", "API3", "BAND", "UMA", "REN", "KNC", "SUSHI", "CAKE", "JOE", "RAY",
    "SRM", "ALPHA", "BADGER", "ALCX", "TRB", "OXT", "NKN", "CTK", "COTI",
    "ARPA", "LIT", "DUSK", "PERP", "MDT", "POLYX", "POWR", "REQ", "STMX",
    "STPT", "TLM", "ALICE", "AXS", "SLP", "ILV", "YGG", "MAGIC", "PRIME",
    "SUPER", "GHST", "AUDIO", "RLC", "NMR", "ORCA", "RAD", "GLMR", "MOVR",
    "ASTR", "ACA", "PHA", "KLAY", "ONE", "FTM", "METIS", "BOBA", "CELR",
]

# OKX perpetual swap sembol formati
WATCHLIST = [f"{c}/USDT:USDT" for c in COINS]
BTC_SYMBOL = "BTC/USDT:USDT"

TIMEFRAME = "15m"
CHECK_INTERVAL_MINUTES = 15

# Grup A (fiyat bazli tukenme kapisi) esikleri - SIKI (altin) seviye
STRICT_WICK_RATIO = 0.5
STRICT_VOLUME_RATIO = 2.5
STRICT_RSI_LOW = 20
STRICT_RSI_HIGH = 80
STRICT_MIN_SCORE = 4

# Grup A esikleri - GEVSEK seviye
LOOSE_WICK_RATIO = 0.3
LOOSE_VOLUME_RATIO = 1.5
LOOSE_RSI_LOW = 30
LOOSE_RSI_HIGH = 70
LOOSE_MIN_SCORE = 2

RSI_PERIOD = 6
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Grup B / C ek puanlama esikleri
FUNDING_EXTREME = 0.0005       # %0.05 - tek periyotluk funding esigi (OKX 8 saatlik)
OI_CHANGE_MIN = 0.01           # OI %1'den fazla degisti mi
ORDERBOOK_IMBALANCE_RATIO = 1.5
BTC_DIVERGENCE_MULTIPLIER = 1.5  # coin hareketi BTC'nin en az bu kati kadar guclu olmali

MAX_CONFIRMATION_SCORE = 8

# 4 saatlik piyasa yonu / trend filtresi
TREND_FILTER_TIMEFRAME = "4h"
TREND_EMA_GAP_THRESHOLD = 3.0     # EMA20 ve EMA50 arasindaki % fark bu esigi gecerse "guclu trend"
TREND_CONSECUTIVE_CANDLES = 3     # son N 4h mumun cogunlugu ayni yonde olmali
INVALIDATION_ATR_BUFFER = 0.3     # gecersizlik seviyesi icin ATR'nin bu orani kadar tampon

# Performans takibi - sinyalden sonra kac saat sonra kontrol edilecek
CHECK_MINUTES = [15, 30, 60]
SUCCESS_THRESHOLD_PCT = 0.15   # yon dogru ise en az bu kadar % hareket etmis olmali

exchange = ccxt.okx({
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
})

# Bellek ici OI takibi (surekli calisan process icin, restart'ta sifirlanir)
_last_oi = {}
_unsupported_symbols = set()
_candidates = {}  # tukenme adaylari - onay mumu bekleniyor


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
# Veri cekme yardimcilari
# ---------------------------------------------------------------------------

def fetch_ohlcv_df(symbol: str, timeframe: str, limit: int = 100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
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

    boll_mid = df["close"].rolling(BOLLINGER_PERIOD).mean()
    boll_std = df["close"].rolling(BOLLINGER_PERIOD).std()
    df["boll_upper"] = boll_mid + BOLLINGER_STD * boll_std
    df["boll_lower"] = boll_mid - BOLLINGER_STD * boll_std

    return df


# ---------------------------------------------------------------------------
# Grup A: fiyat bazli tukenme kapisi (mandatory gate)
# ---------------------------------------------------------------------------

def check_exhaustion_gate(df: pd.DataFrame, tier: str = "strict"):
    """
    Uc sart birden tutmali: hacim patlamasi + fitil + RSI asiri uc.
    tier: "strict" (sıkı/altin) ya da "loose" (gevsek)
    Donus: (direction, row) ya da None
    """
    if len(df) < max(BOLLINGER_PERIOD, 20) + 2:
        return None

    if tier == "strict":
        wick_ratio, volume_ratio_min, rsi_low, rsi_high = (
            STRICT_WICK_RATIO, STRICT_VOLUME_RATIO, STRICT_RSI_LOW, STRICT_RSI_HIGH
        )
    else:
        wick_ratio, volume_ratio_min, rsi_low, rsi_high = (
            LOOSE_WICK_RATIO, LOOSE_VOLUME_RATIO, LOOSE_RSI_LOW, LOOSE_RSI_HIGH
        )

    row = df.iloc[-2]
    volume_ratio = row["volume"] / row["vol_sma15"] if row["vol_sma15"] else 0
    if volume_ratio < volume_ratio_min:
        return None

    if row["lower_wick_ratio"] >= wick_ratio and row["rsi"] <= rsi_low:
        return "LONG", row
    if row["upper_wick_ratio"] >= wick_ratio and row["rsi"] >= rsi_high:
        return "SHORT", row

    return None


# ---------------------------------------------------------------------------
# Piyasa yonu / 4 saatlik trend filtresi
# ---------------------------------------------------------------------------

def get_market_regime(symbol: str = None):
    """
    Verilen sembolun (varsayilan BTC) 4 saatlik grafigine bakarak trend yonunu belirler.
    Donus: (regime, gap_pct) - regime: 'GUCLU_DUSUS', 'GUCLU_YUKSELIS', 'YATAY'
    """
    target = symbol or BTC_SYMBOL
    try:
        df4h = fetch_ohlcv_df(target, TREND_FILTER_TIMEFRAME, limit=60)
        df4h = compute_indicators(df4h)
        row = df4h.iloc[-2]
        gap_pct = (row["ema20"] - row["ema50"]) / row["ema50"] * 100

        recent = df4h.iloc[-(TREND_CONSECUTIVE_CANDLES + 1):-1]
        bearish_count = (~recent["is_bull"]).sum()
        bullish_count = recent["is_bull"].sum()

        if gap_pct <= -TREND_EMA_GAP_THRESHOLD and bearish_count >= TREND_CONSECUTIVE_CANDLES - 1:
            return "GUCLU_DUSUS", gap_pct
        if gap_pct >= TREND_EMA_GAP_THRESHOLD and bullish_count >= TREND_CONSECUTIVE_CANDLES - 1:
            return "GUCLU_YUKSELIS", gap_pct
        return "YATAY", gap_pct
    except Exception as e:
        print(f"{target} icin trend alinamadi: {e}")
        return "BILINMIYOR", 0.0


def regime_label(regime: str) -> str:
    return {
        "GUCLU_DUSUS": "📉 Genel piyasa: GUCLU DUSUS",
        "GUCLU_YUKSELIS": "📈 Genel piyasa: GUCLU YUKSELIS",
        "YATAY": "➡️ Genel piyasa: YATAY/KARISIK",
        "BILINMIYOR": "❓ Genel piyasa: bilinmiyor",
    }.get(regime, "❓ Genel piyasa: bilinmiyor")


def compute_invalidation(direction: str, row) -> float:
    """Sinyali gecersiz sayacak / cikis referansi olacak fiyat seviyesi."""
    atr = row["atr14"] if pd.notna(row["atr14"]) else 0
    buffer = atr * INVALIDATION_ATR_BUFFER
    if direction == "LONG":
        return row["low"] - buffer
    return row["high"] + buffer


# ---------------------------------------------------------------------------
# Onay mumu mekanizmasi
# ---------------------------------------------------------------------------

def check_candidate_confirmation(symbol: str, df: pd.DataFrame):
    """
    Bu sembol icin bekleyen bir tukenme adayi varsa, en son kapanan mumun
    beklenen yonde kapanip kapanmadigina bakar.
    Donus: None (aday yok ya da henuz yeni mum kapanmadi)
           ("confirmed", direction, tier, confirm_row, exhaustion_row)
           ("rejected", direction, tier, confirm_row, exhaustion_row)
    """
    candidate = _candidates.get(symbol)
    if not candidate:
        return None

    latest_row = df.iloc[-2]
    if latest_row["timestamp"] <= candidate["candle_time"]:
        return None  # henuz yeni mum kapanmadi, beklemeye devam

    direction = candidate["direction"]
    tier = candidate["tier"]
    exhaustion_row = candidate["exhaustion_row"]
    del _candidates[symbol]  # tek seferlik: onaylansa da reddedilse de aday tuketildi

    confirmed = (
        (direction == "LONG" and bool(latest_row["is_bull"])) or
        (direction == "SHORT" and not bool(latest_row["is_bull"]))
    )

    status = "confirmed" if confirmed else "rejected"
    return (status, direction, tier, latest_row, exhaustion_row)


# ---------------------------------------------------------------------------
# Grup B: bagimsiz veri kaynaklari (funding, OI, order book)
# ---------------------------------------------------------------------------

def score_funding(symbol: str, direction: str) -> tuple:
    """Asiri funding rate, o yonde asiri pozisyonlanma anlamina gelir -> tersine donus ihtimali artar."""
    try:
        funding = exchange.fetch_funding_rate(symbol)
        rate = funding.get("fundingRate")
        if rate is None:
            return 0, "veri yok"
        # LONG sinyalinde (fiyat dipte) asiri negatif funding -> asiri short pozisyon -> bounce potansiyeli
        if direction == "LONG" and rate <= -FUNDING_EXTREME:
            return 2, f"funding {rate:.4%} (asiri short)"
        # SHORT sinyalinde (fiyat tepede) asiri pozitif funding -> asiri long pozisyon -> dusus potansiyeli
        if direction == "SHORT" and rate >= FUNDING_EXTREME:
            return 2, f"funding {rate:.4%} (asiri long)"
        return 0, f"funding {rate:.4%} (notr)"
    except Exception as e:
        return 0, f"funding alinamadi ({e})"


def score_open_interest(symbol: str, direction: str) -> tuple:
    """
    OI azaliyorsa (pozisyonlar kapaniyor) -> mevcut hareket tukeniyor olabilir, bounce'u destekler.
    OI artiyorsa (yeni para giriyor) -> trend taze, devam ihtimali yuksek, bounce'a karsi.
    """
    try:
        oi_data = exchange.fetch_open_interest(symbol)
        current_oi = oi_data.get("openInterestAmount") or oi_data.get("openInterestValue")
        if current_oi is None:
            return 0, "veri yok"

        prev_oi = _last_oi.get(symbol)
        _last_oi[symbol] = current_oi

        if prev_oi is None or prev_oi == 0:
            return 0, "ilk olcum"

        change = (current_oi - prev_oi) / prev_oi
        if change <= -OI_CHANGE_MIN:
            return 1, f"OI {change:+.1%} (pozisyonlar kapaniyor)"
        if change >= OI_CHANGE_MIN:
            return -1, f"OI {change:+.1%} (yeni pozisyon aciliyor)"
        return 0, f"OI {change:+.1%} (durgun)"
    except Exception as e:
        return 0, f"OI alinamadi ({e})"


def score_orderbook(symbol: str, direction: str) -> tuple:
    """Emir defterinde bounce yonunu destekleyen bir agirlik var mi."""
    try:
        ob = exchange.fetch_order_book(symbol, limit=20)
        bid_vol = sum(b[1] for b in ob["bids"])
        ask_vol = sum(a[1] for a in ob["asks"])
        if ask_vol == 0 or bid_vol == 0:
            return 0, "veri yetersiz"

        ratio = bid_vol / ask_vol
        if direction == "LONG" and ratio >= ORDERBOOK_IMBALANCE_RATIO:
            return 1, f"bid/ask {ratio:.2f} (alici agirlikli)"
        if direction == "SHORT" and ratio <= 1 / ORDERBOOK_IMBALANCE_RATIO:
            return 1, f"bid/ask {ratio:.2f} (satici agirlikli)"
        return 0, f"bid/ask {ratio:.2f} (dengeli)"
    except Exception as e:
        return 0, f"order book alinamadi ({e})"


# ---------------------------------------------------------------------------
# Grup C: yapisal teyit (coklu zaman dilimi, BTC korelasyonu)
# ---------------------------------------------------------------------------

def score_multi_timeframe(symbol: str, direction: str) -> tuple:
    """1 saatlik grafikte de RSI ayni yonde asiri mi (buyuk resim teyidi)."""
    try:
        df1h = fetch_ohlcv_df(symbol, "1h", limit=30)
        df1h = compute_indicators(df1h)
        row = df1h.iloc[-2]
        if direction == "LONG" and row["rsi"] <= STRICT_RSI_LOW + 10:
            return 1, f"1h RSI {row['rsi']:.1f} (destekliyor)"
        if direction == "SHORT" and row["rsi"] >= STRICT_RSI_HIGH - 10:
            return 1, f"1h RSI {row['rsi']:.1f} (destekliyor)"
        return 0, f"1h RSI {row['rsi']:.1f} (notr)"
    except Exception as e:
        return 0, f"1h veri alinamadi ({e})"


def score_bollinger(row) -> tuple:
    """Fiyat Bollinger bandinin disina tasmis mi (istatistiksel asiri hareket)."""
    try:
        if pd.isna(row["boll_upper"]) or pd.isna(row["boll_lower"]):
            return 0, "veri yetersiz"
        if row["close"] <= row["boll_lower"]:
            return 1, "alt bant disinda"
        if row["close"] >= row["boll_upper"]:
            return 1, "ust bant disinda"
        return 0, "bant icinde"
    except Exception:
        return 0, "hata"


def score_btc_divergence(symbol: str, df15: pd.DataFrame, btc_df15: pd.DataFrame, direction: str) -> tuple:
    """
    Coin'in son hareketi BTC'ninkinden belirgin sekilde guclu mu.
    Guclu ise coin'e ozel bir asiri tepki (overreaction) olma ihtimali daha yuksek,
    bu da bounce senaryosunu destekler.
    """
    try:
        coin_change = abs(df15["close"].iloc[-2] / df15["close"].iloc[-6] - 1)
        btc_change = abs(btc_df15["close"].iloc[-2] / btc_df15["close"].iloc[-6] - 1)
        if btc_change == 0:
            return 0, "BTC veri yetersiz"
        ratio = coin_change / btc_change
        if ratio >= BTC_DIVERGENCE_MULTIPLIER:
            return 1, f"coin/BTC hareket orani {ratio:.1f}x (coin'e ozel)"
        return 0, f"coin/BTC hareket orani {ratio:.1f}x (piyasa geneli)"
    except Exception as e:
        return 0, f"BTC karsilastirma hatasi ({e})"


# ---------------------------------------------------------------------------
# Sinyal loglama
# ---------------------------------------------------------------------------

SIGNAL_LOG_FILE = "signal_history.csv"
PENDING_FILE = "pending_signals.csv"
OUTCOME_FILE = "signal_outcomes.csv"


def log_signal(symbol: str, direction: str, row, score: int, breakdown: list, tier: str = "strict"):
    file_exists = os.path.isfile(SIGNAL_LOG_FILE)
    with open(SIGNAL_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "symbol", "direction", "tier", "price", "rsi", "score", "breakdown"
            ])
        writer.writerow([
            datetime.now().isoformat(), symbol, direction, tier, row["close"],
            row["rsi"], score, " | ".join(breakdown)
        ])


def log_pending(symbol: str, direction: str, entry_price: float, entry_time: datetime, invalidation_price: float):
    file_exists = os.path.isfile(PENDING_FILE)
    with open(PENDING_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "symbol", "direction", "entry_price", "entry_time", "invalidation_price",
                "checked_15m", "checked_30m", "checked_60m"
            ])
        writer.writerow([symbol, direction, entry_price, entry_time.isoformat(), invalidation_price, "0", "0", "0"])


def _read_pending():
    if not os.path.isfile(PENDING_FILE):
        return []
    with open(PENDING_FILE, newline="") as f:
        return list(csv.DictReader(f))


def _write_pending(rows):
    fieldnames = ["symbol", "direction", "entry_price", "entry_time", "invalidation_price",
                  "checked_15m", "checked_30m", "checked_60m"]
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
    """Bekleyen sinyalleri kontrol et, zamani gelenler icin sonucu logla."""
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


# ---------------------------------------------------------------------------
# Ana tarama
# ---------------------------------------------------------------------------

def scan_once():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Tarama basliyor...")

    check_pending_outcomes()

    try:
        btc_df15 = fetch_ohlcv_df(BTC_SYMBOL, TIMEFRAME, limit=30)
    except Exception as e:
        print(f"BTC verisi alinamadi: {e}")
        btc_df15 = None

    market_regime, regime_gap = get_market_regime()
    print(f"Piyasa yonu: {market_regime} (EMA gap: {regime_gap:+.2f}%)")

    for symbol in WATCHLIST:
        if symbol in _unsupported_symbols:
            continue
        try:
            df = fetch_ohlcv_df(symbol, TIMEFRAME, limit=100)
            df = compute_indicators(df)

            confirmation = check_candidate_confirmation(symbol, df)

            if confirmation is not None:
                status, direction, tier, confirm_row, exhaustion_row = confirmation

                if status == "rejected":
                    print(f"{symbol}: {tier} aday onaylanmadi (beklenen yon {direction} degildi), iptal edildi")
                    continue

                # status == "confirmed" -> asagida puanlamaya devam edilecek
                row = confirm_row

            else:
                gate_result = check_exhaustion_gate(df, tier="strict")
                tier = "strict"
                if not gate_result:
                    gate_result = check_exhaustion_gate(df, tier="loose")
                    tier = "loose"

                if not gate_result:
                    print(f"{symbol}: kapi gecilmedi")
                    continue

                direction, exhaustion_row = gate_result

                # Ters trend filtresi: hem BTC genel piyasasina hem coin'in KENDI trendine bakilir.
                # Coin, BTC yatay olsa bile kendi basina guclu bir trendde olabilir.
                symbol_regime, symbol_gap = get_market_regime(symbol)

                if direction == "LONG" and market_regime == "GUCLU_DUSUS":
                    print(f"{symbol}: {tier} kapi gecti ama genel piyasa (BTC) guclu dususte, LONG bounce engellendi")
                    continue
                if direction == "SHORT" and market_regime == "GUCLU_YUKSELIS":
                    print(f"{symbol}: {tier} kapi gecti ama genel piyasa (BTC) guclu yukseliste, SHORT bounce engellendi")
                    continue
                if direction == "LONG" and symbol_regime == "GUCLU_DUSUS":
                    print(f"{symbol}: {tier} kapi gecti ama coin'in kendi 4h trendi guclu dususte ({symbol_gap:+.1f}%), LONG bounce engellendi")
                    continue
                if direction == "SHORT" and symbol_regime == "GUCLU_YUKSELIS":
                    print(f"{symbol}: {tier} kapi gecti ama coin'in kendi 4h trendi guclu yukseliste ({symbol_gap:+.1f}%), SHORT bounce engellendi")
                    continue

                _candidates[symbol] = {
                    "direction": direction,
                    "tier": tier,
                    "candle_time": exhaustion_row["timestamp"],
                    "exhaustion_row": exhaustion_row,
                }
                print(f"{symbol}: {tier} tukenme adayi olustu ({direction}), onay mumu bekleniyor")
                continue

            min_score = STRICT_MIN_SCORE if tier == "strict" else LOOSE_MIN_SCORE

            breakdown = []
            score = 0

            pts, note = score_funding(symbol, direction)
            score += pts
            breakdown.append(f"Funding: {note} ({pts:+d})")

            pts, note = score_open_interest(symbol, direction)
            score += pts
            breakdown.append(f"OI: {note} ({pts:+d})")

            pts, note = score_orderbook(symbol, direction)
            score += pts
            breakdown.append(f"Order book: {note} ({pts:+d})")

            pts, note = score_multi_timeframe(symbol, direction)
            score += pts
            breakdown.append(f"1h teyit: {note} ({pts:+d})")

            pts, note = score_bollinger(row)
            score += pts
            breakdown.append(f"Bollinger: {note} ({pts:+d})")

            if btc_df15 is not None and symbol != BTC_SYMBOL:
                pts, note = score_btc_divergence(symbol, df, btc_df15, direction)
                score += pts
                breakdown.append(f"BTC karsilastirma: {note} ({pts:+d})")

            log_signal(symbol, direction, row, score, breakdown, tier)

            if score >= min_score:
                breakdown_text = "\n".join(f"- {b}" for b in breakdown)
                invalidation = compute_invalidation(direction, exhaustion_row)
                if tier == "strict":
                    baslik = f"🥇 {symbol} - SIKI TÜKENME sinyali ({direction} bounce)"
                else:
                    baslik = f"⚪ {symbol} - GEVŞEK tükenme sinyali ({direction} bounce)"

                msg = (
                    f"{baslik}\n"
                    f"✅ Onay mumu ile teyit edildi\n"
                    f"Guven skoru: {score}/{MAX_CONFIRMATION_SCORE}\n"
                    f"{regime_label(market_regime)}\n\n"
                    f"Tukenme fiyati: {exhaustion_row['close']:.4f} (RSI {exhaustion_row['rsi']:.1f})\n"
                    f"Onay/Giris fiyati: {row['close']:.4f}\n"
                    f"Gecersizlik seviyesi: {invalidation:.4f}\n"
                    f"Zaman dilimi: {TIMEFRAME}\n\n"
                    f"Teyit detaylari:\n{breakdown_text}"
                )
                print(msg)
                send_telegram_message(msg)
                log_pending(symbol, direction, row["close"], datetime.now(), invalidation)
            else:
                print(f"{symbol}: {tier} onaylandi ama skor dusuk ({score}/{MAX_CONFIRMATION_SCORE})")

        except Exception as e:
            if "does not have" in str(e).lower():
                _unsupported_symbols.add(symbol)
                print(f"{symbol}: bu borsada islem gormuyor, listeden cikarildi")
            else:
                print(f"{symbol} hata: {e}")


def run_forever():
    send_telegram_message(
        "Kripto tukenme botu (coklu analiz, 2 kademeli + trend filtresi + onay mumu) baslatildi.\n"
        f"{len(WATCHLIST)} coin taranıyor.\n"
        f"🥇 Siki: skor >= {STRICT_MIN_SCORE}/{MAX_CONFIRMATION_SCORE}\n"
        f"⚪ Gevsek: skor >= {LOOSE_MIN_SCORE}/{MAX_CONFIRMATION_SCORE}\n"
        f"Guclu trend varsa ters yon sinyali engellenir.\n"
        f"Tukenme tespit edilince hemen degil, bir sonraki mum onaylarsa sinyal gonderilir.\n"
        f"Her sinyalde gecersizlik seviyesi ve genel piyasa yonu gosterilir."
    )
    while True:
        scan_once()
        time.sleep(CHECK_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    run_forever()
