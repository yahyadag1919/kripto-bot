"""
trend_pullback_bot.py
======================
SIFIRDAN TEMIZ KURULUM v2 (2026-07-27, Gemini'nin 4 katmanli mimarisi)
+ TAMIRCI KATMANI (2026-07-28, Gemini ile birlikte, mevcut kod uzerine eklendi)

1. KATMAN: SISTEM DEDEKTIFI (Error Tracker & Profiler)
2. KATMAN: TAMIRCI (Auto-Healer / Self-Healing)
3. KATMAN: DEVRE KESICILER VE GUVENLIK
4. KATMAN: PIYASA BEYNI (Dynamic Market Allocator)
5. KATMAN: CEKIRDEK STRATEJI MOTORU (Trend-Pullback)
"""

import os
import csv
import json
import sys
import time
import traceback
import functools
from datetime import datetime, timedelta

import ccxt
import numpy as np
import pandas as pd
import requests

# ============================================================
# BORSA BAGLANTISI
# ============================================================
USE_TESTNET = os.environ.get("TESTNET", "true").lower() == "true"


def _redirect_all_urls_to_demo(urls_node):
    if isinstance(urls_node, dict):
        return {k: _redirect_all_urls_to_demo(v) for k, v in urls_node.items()}
    if isinstance(urls_node, list):
        return [_redirect_all_urls_to_demo(v) for v in urls_node]
    if isinstance(urls_node, str):
        return (urls_node
                .replace("fapi.binance.com", "demo-fapi.binance.com")
                .replace("testnet.binancefuture.com", "demo-fapi.binance.com"))
    return urls_node


def _build_exchange():
    ex = ccxt.binanceusdm({
        "apiKey": os.environ.get("BINANCE_API_KEY"),
        "secret": os.environ.get("BINANCE_API_SECRET"),
        "enableRateLimit": True,
    })
    if USE_TESTNET:
        try:
            ex.urls["api"] = _redirect_all_urls_to_demo(ex.urls["api"])
        except Exception as e:
            print(f"Demo-fapi URL override uygulanamadi: {e}")
        try:
            ex.options["fetchCurrencies"] = False
        except Exception:
            pass
    ex.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
    ex.options.setdefault("fetchOpenOrders", {})["warnWithoutSymbol"] = False
    return ex


exchange = _build_exchange()


def _reset_exchange_session():
    # TAMIRCI (2026-07-28): tekrarlanan aglayarn/timeout hatalarinda borsa
    # baglantisini (ccxt session) sifirdan kurar - eski session'in takilmis
    # olma ihtimaline karsi.
    global exchange
    try:
        exchange = _build_exchange()
        print("Tamirci: borsa oturumu sifirlandi.")
        return True
    except Exception as e:
        print(f"Tamirci: borsa oturumu sifirlanamadi ({e})")
        return False

# ============================================================
# TELEGRAM
# ============================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
_telegram_update_offset = None


def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[Telegram devre disi] {text}")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=15,
        )
    except Exception as e:
        print(f"Telegram mesaji gonderilemedi: {e}")


def process_telegram_updates():
    global _telegram_update_offset
    if not TELEGRAM_TOKEN:
        return
    try:
        params = {"timeout": 0}
        if _telegram_update_offset is not None:
            params["offset"] = _telegram_update_offset
        resp = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params=params, timeout=15,
        )
        updates = resp.json().get("result", [])
    except Exception as e:
        print(f"Telegram guncellemeleri cekilemedi: {e}")
        return

    for update in updates:
        _telegram_update_offset = update["update_id"] + 1
        message = update.get("message")
        if not message:
            continue
        text = (message.get("text") or "").strip().lower()
        if text.startswith("/stats"):
            send_telegram_message(build_stats_message())
        elif text.startswith("/positions"):
            send_telegram_message(build_positions_message())
        elif text.startswith("/orders"):
            send_telegram_message(build_orders_message())
        elif text.startswith("/regime"):
            send_telegram_message(build_regime_message())


def _cancel_all_open_orders_for_symbol(symbol: str) -> int:
    """Bir sembol icin borsadaki TUM acik emirleri iptal eder. Kac emir
    iptal edildigini dondurur (basarisiz iptaller sessizce atlanir)."""
    cancelled = 0
    try:
        orders = exchange.fetch_open_orders(symbol)
    except Exception as e:
        print(f"{symbol}: acik emirler cekilemedi ({e})")
        return 0
    for o in orders:
        try:
            exchange.cancel_order(o["id"], symbol)
            cancelled += 1
        except Exception:
            pass
    return cancelled


# ============================================================
# 2. KATMAN: TAMIRCI (Auto-Healer / Self-Healing)
# ============================================================
# DERS (2026-07-28, Gemini ile birlikte tasarlandi): Dedektif bir hata
# bulunca sistem sadece uyarip beklemesin - Tamirci once otomatik onarim
# dener, basarili olursa akisa kaldigi yerden devam edilir. Sadece Tamirci
# ayni sorunu CIRCUIT_BREAKER_MAX_FAILURES turda cozemezse mevcut devre
# kesici (update_positions icinde) pozisyonu kapatir - Tamirci bu mekanizmayi
# DEGISTIRMIYOR, sadece her turda once bir onarim sansi ekliyor.
def _is_network_error(e: Exception) -> bool:
    text = str(e).lower()
    return (
        isinstance(e, (ccxt.NetworkError, ccxt.RequestTimeout))
        or "timeout" in text or "timed out" in text
        or "connection" in text or "network" in text
    )


def _is_stop_limit_error(e: Exception) -> bool:
    text = str(e).lower()
    return "-4045" in text or "max stop order" in text


def tamirci_attempt_repair(symbol: str, e: Exception) -> bool:
    """Hatanin turune gore otomatik onarim dener. Onarim denendiyse True
    doner (basarili olup olmadigi degil, bir eylem alindigi anlamina gelir);
    cagiran taraf onarimdan sonra islemi bir kez daha denemelidir."""
    if _is_stop_limit_error(e):
        cancelled = _cancel_all_open_orders_for_symbol(symbol)
        print(f"Tamirci: {symbol} icin {cancelled} hayalet/eski emir temizlendi (-4045 onarim denemesi).")
        return True
    if _is_network_error(e):
        return _reset_exchange_session()
    return False


# ============================================================
# 1. KATMAN: SISTEM DEDEKTIFI
# ============================================================
_ERROR_HINTS = {
    "-4045": "Hesapta cok fazla acik emir birikmis olabilir (max stop order limit).",
    "-2019": "Marjin yetersiz - referans bakiye/risk ayari gercek bakiyeyle uyumsuz olabilir.",
    "-1122": "Bu sembol su an islem gormuyor (askida/gecersiz durum).",
    "-2027": "Kaldirac/pozisyon limiti asildi.",
    "-4164": "Minimum islem tutarinin altinda kaldi (notional cok kucuk).",
    "429": "Borsa/Gemini API istek limitine takildi (rate limit) - gecici olmali.",
    "insufficient": "Bakiye/marjin yetersiz olabilir.",
    "timeout": "Baglanti zaman asimina ugradi - gecici bir ag sorunu olabilir.",
    "no such file or directory": "DATA_DIR/Volume yolu henuz olusturulmamis veya yanlis ayarlanmis olabilir.",
}


def _guess_root_cause(exc: Exception) -> str:
    text = str(exc).lower()
    for key, hint in _ERROR_HINTS.items():
        if key.lower() in text:
            return hint
    return "Bilinen bir kaliba uymuyor - detaylari incele."


def track_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            tb = traceback.extract_tb(sys.exc_info()[2])
            last_frame = tb[-1] if tb else None
            file_line = f"{os.path.basename(last_frame.filename)}:{last_frame.lineno}" if last_frame else "bilinmiyor"
            symbol_guess = "N/A"
            for a in list(args) + list(kwargs.values()):
                if isinstance(a, str) and "/" in a and "USDT" in a.upper():
                    symbol_guess = a
                    break
            healed_note = ""
            if _is_network_error(e):
                if _reset_exchange_session():
                    healed_note = "\n🛠️ Tamirci: borsa oturumu sıfırlandı, bir sonraki turda tekrar denenecek."
            send_telegram_message(
                f"🚨 [DEDEKTİF RAPORU]\n"
                f"Fonksiyon: {func.__name__} | Konum: {file_line} | Coin: {symbol_guess}\n"
                f"Hata: {e}\n"
                f"Tahmini kök neden: {_guess_root_cause(e)}{healed_note}"
            )
            print(f"[DEDEKTIF] {func.__name__} ({file_line}, {symbol_guess}): {e}")
            raise
    return wrapper


# ============================================================
# AYARLAR
# ============================================================
DATA_DIR = os.environ.get("DATA_DIR", ".")
# DERS (2026-07-28, Dedektif'in yakaladigi ilk gercek hata): DATA_DIR bir
# Railway Volume yoluna (orn. /data) ayarlanmissa ama bu klasor henuz
# olusturulmamissa, pozisyon dosyasina yazarken "No such file or directory"
# hatasi aliniyordu. Baslarken klasoru garanti altina aliyoruz.
os.makedirs(DATA_DIR, exist_ok=True)
CHECK_INTERVAL_MINUTES = int(os.environ.get("CHECK_INTERVAL_MINUTES", "5"))
MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", "3"))
NEW_TRADES_HALTED = os.environ.get("NEW_TRADES_HALTED", "false").lower() == "true"
ATR_PERIOD = 14

TP_H1_TIMEFRAME = os.environ.get("TP_H1_TIMEFRAME", "1h")
TP_M15_TIMEFRAME = os.environ.get("TP_M15_TIMEFRAME", "15m")
TP_EMA200_PERIOD = int(os.environ.get("TP_EMA200_PERIOD", "200"))
TP_SUPERTREND_PERIOD = int(os.environ.get("TP_SUPERTREND_PERIOD", "10"))
TP_SUPERTREND_MULT = float(os.environ.get("TP_SUPERTREND_MULT", "3.0"))
TP_EMA_FAST = int(os.environ.get("TP_EMA_FAST", "20"))
TP_EMA_SLOW = int(os.environ.get("TP_EMA_SLOW", "50"))
TP_RSI_PERIOD = int(os.environ.get("TP_RSI_PERIOD", "14"))
TP_RSI_LONG_MAX = float(os.environ.get("TP_RSI_LONG_MAX", "45"))
TP_RSI_SHORT_MIN = float(os.environ.get("TP_RSI_SHORT_MIN", "55"))
TP_PULLBACK_TOLERANCE_PCT = float(os.environ.get("TP_PULLBACK_TOLERANCE_PCT", "0.3"))
TP_SWING_LOOKBACK = int(os.environ.get("TP_SWING_LOOKBACK", "20"))

TP_LEVERAGE = int(os.environ.get("TP_LEVERAGE", "10"))
TP_POSITION_PCT_OF_BALANCE = float(os.environ.get("TP_POSITION_PCT_OF_BALANCE", "20"))  # DERS (2026-07-28): risk-bazli miktar hesabinin, dar stop mesafelerinde asiri buyuk notional/marjin talep etmesini onleyen tavan
TP_RISK_PER_TRADE_PCT = float(os.environ.get("TP_RISK_PER_TRADE_PCT", "2.0"))
TP_RISK_RANGING_PCT = float(os.environ.get("TP_RISK_RANGING_PCT", "0.5"))
TP_PARTIAL_CLOSE_PCT = float(os.environ.get("TP_PARTIAL_CLOSE_PCT", "50"))
TP_PARTIAL_TP_R_MULT = float(os.environ.get("TP_PARTIAL_TP_R_MULT", "1.5"))
TP_POST_BREAKEVEN_TRAIL_MULT = float(os.environ.get("TP_POST_BREAKEVEN_TRAIL_MULT", "2.0"))

CIRCUIT_BREAKER_MAX_FAILURES = int(os.environ.get("CIRCUIT_BREAKER_MAX_FAILURES", "3"))

# ANA PORTFÖY BEYNİ (Master Portfolio Brain) ayarları (2026-07-28, Gemini'nin talebi)
PORTFOLIO_DAILY_LOSS_LIMIT_PCT = float(os.environ.get("PORTFOLIO_DAILY_LOSS_LIMIT_PCT", "5.0"))
PORTFOLIO_DAILY_HALT_HOURS = float(os.environ.get("PORTFOLIO_DAILY_HALT_HOURS", "24"))
PORTFOLIO_CONSEC_LOSS_LIMIT = int(os.environ.get("PORTFOLIO_CONSEC_LOSS_LIMIT", "3"))
PORTFOLIO_COOLDOWN_HOURS = float(os.environ.get("PORTFOLIO_COOLDOWN_HOURS", "2"))
PORTFOLIO_RISK_PENALTY_MULT = float(os.environ.get("PORTFOLIO_RISK_PENALTY_MULT", "0.5"))
# Gemini'nin talebi "%50 risk azalt VEYA 2 saat cooldown" seklinde iki secenek
# sunuyordu, hangisi net degildi - daha korumaci olan ikisini birlikte
# uyguluyoruz: 3. arka arkaya STOP'ta hem 2 saatlik giris donmasi (cooldown)
# hem de cooldown bitince risk %50 dusuk devam eder (bir sonraki kazançlı
# islem gelene ya da PORTFOLIO_RISK_PENALTY_HOURS dolana kadar).
PORTFOLIO_RISK_PENALTY_HOURS = float(os.environ.get("PORTFOLIO_RISK_PENALTY_HOURS", "24"))
PORTFOLIO_PROFIT_RISK_PCT = float(os.environ.get("PORTFOLIO_PROFIT_RISK_PCT", "2.0"))
PORTFOLIO_LOSS_RISK_PCT = float(os.environ.get("PORTFOLIO_LOSS_RISK_PCT", "0.5"))
RESET_ON_START = os.environ.get("RESET_ON_START", "false").lower() == "true"

REGIME_ADX_TREND_THRESHOLD = float(os.environ.get("REGIME_ADX_TREND_THRESHOLD", "25"))
REGIME_ADX_RANGE_THRESHOLD = float(os.environ.get("REGIME_ADX_RANGE_THRESHOLD", "20"))
REGIME_BENCHMARK_SYMBOL = os.environ.get("REGIME_BENCHMARK_SYMBOL", "BTC/USDT:USDT")
REGIME_CHECK_EVERY_N_CYCLES = int(os.environ.get("REGIME_CHECK_EVERY_N_CYCLES", "3"))

REFERENCE_BALANCE_FILE = os.path.join(DATA_DIR, "reference_balance.txt")
POSITIONS_FILE = os.path.join(DATA_DIR, "positions.csv")
CLOSED_TRADES_FILE = os.path.join(DATA_DIR, "closed_trades.csv")
PORTFOLIO_STATE_FILE = os.path.join(DATA_DIR, "portfolio_state.json")

POSITION_FIELDNAMES = [
    "symbol", "direction", "entry_price", "stop_price", "extreme_price", "entry_time",
    "exchange_stop_order_id", "partial_tp_price", "partial_tp_taken", "original_qty",
    "exchange_tp_order_id", "stop_missing_count",
]
CLOSED_TRADE_FIELDNAMES = ["timestamp", "symbol", "direction", "entry_price", "exit_price", "pct_change", "reason"]

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

try:
    exchange.load_markets()

    def _is_tradeable(sym):
        m = exchange.markets.get(sym)
        if m is None:
            return False
        if m.get("active") is False:
            return False
        info_status = (m.get("info") or {}).get("status")
        if info_status and info_status != "TRADING":
            return False
        return True

    _dropped = [s for s in WATCHLIST if not _is_tradeable(s)]
    WATCHLIST = [s for s in WATCHLIST if _is_tradeable(s)]
    if _dropped:
        print(f"WATCHLIST temizlendi: {len(_dropped)} sembol yok/aktif degil, cikarildi.")
except Exception as e:
    print(f"Piyasa listesi dogrulanamadi ({e}), WATCHLIST oldugu gibi kullanilacak")

_current_regime = "TREND"
_current_risk_pct = TP_RISK_PER_TRADE_PCT
_regime_cycle_counter = 0


# ============================================================
# INDIKATORLER
# ============================================================
def _compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_supertrend(df: pd.DataFrame, period: int, mult: float) -> pd.Series:
    atr = _compute_atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + mult * atr
    lower_band = hl2 - mult * atr

    direction = pd.Series(index=df.index, dtype=object)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i]):
            direction.iloc[i] = None
            continue
        if df["close"].iloc[i - 1] <= final_upper.iloc[i - 1]:
            final_upper.iloc[i] = min(upper_band.iloc[i], final_upper.iloc[i - 1])
        else:
            final_upper.iloc[i] = upper_band.iloc[i]
        if df["close"].iloc[i - 1] >= final_lower.iloc[i - 1]:
            final_lower.iloc[i] = max(lower_band.iloc[i], final_lower.iloc[i - 1])
        else:
            final_lower.iloc[i] = lower_band.iloc[i]

        if df["close"].iloc[i] > final_upper.iloc[i - 1]:
            direction.iloc[i] = "YESIL"
        elif df["close"].iloc[i] < final_lower.iloc[i - 1]:
            direction.iloc[i] = "KIRMIZI"
        else:
            direction.iloc[i] = direction.iloc[i - 1] if i > 0 else None
    return direction


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def fetch_df(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ============================================================
# 3. KATMAN: PIYASA BEYNI
# ============================================================
@track_errors
def update_market_regime():
    global _current_regime, _current_risk_pct

    try:
        df = fetch_df(REGIME_BENCHMARK_SYMBOL, TP_H1_TIMEFRAME, 100)
        df["adx"] = _compute_adx(df, 14)
        latest_adx = df.iloc[-2]["adx"]
    except Exception as e:
        print(f"Piyasa Beyni: ADX hesaplanamadi ({e}), onceki rejim korunuyor")
        return

    if pd.isna(latest_adx):
        return

    sample = WATCHLIST[:30] if len(WATCHLIST) > 30 else WATCHLIST
    above, below, checked = 0, 0, 0
    for sym in sample:
        try:
            d = fetch_df(sym, TP_H1_TIMEFRAME, TP_EMA200_PERIOD + 5)
            d["ema200"] = d["close"].ewm(span=TP_EMA200_PERIOD, adjust=False).mean()
            row = d.iloc[-2]
            if pd.isna(row["ema200"]):
                continue
            checked += 1
            if row["close"] > row["ema200"]:
                above += 1
            else:
                below += 1
        except Exception:
            continue
    breadth_pct = (max(above, below) / checked * 100) if checked else 50

    old_regime = _current_regime
    if latest_adx > REGIME_ADX_TREND_THRESHOLD:
        _current_regime = "TREND"
    elif latest_adx < REGIME_ADX_RANGE_THRESHOLD:
        _current_regime = "YATAY"

    _current_risk_pct = TP_RISK_PER_TRADE_PCT if _current_regime == "TREND" else TP_RISK_RANGING_PCT

    if old_regime != _current_regime:
        send_telegram_message(
            f"🧠 [Piyasa Beyni] Rejim değişti: {old_regime} → {_current_regime}\n"
            f"BTC H1 ADX: {latest_adx:.1f} | Yön uyumu (breadth): %{breadth_pct:.0f}\n"
            f"Yeni risk/işlem: %{_current_risk_pct}"
        )
    print(f"[Piyasa Beyni] Rejim: {_current_regime} | ADX: {latest_adx:.1f} | Breadth: %{breadth_pct:.0f} | Risk: %{_current_risk_pct}")


def build_regime_message():
    return (f"🧠 Güncel piyasa rejimi: {_current_regime}\n"
            f"Aktif risk/işlem: %{_current_risk_pct}")


# ============================================================
# 4. KATMAN: STRATEJI MOTORU
# ============================================================
@track_errors
def get_h1_trend_bias(symbol: str):
    try:
        df = fetch_df(symbol, TP_H1_TIMEFRAME, TP_EMA200_PERIOD + 50)
    except Exception:
        return None
    if len(df) < TP_EMA200_PERIOD + 5:
        return None
    df["ema200"] = df["close"].ewm(span=TP_EMA200_PERIOD, adjust=False).mean()
    df["supertrend"] = _compute_supertrend(df, TP_SUPERTREND_PERIOD, TP_SUPERTREND_MULT)
    row = df.iloc[-2]
    if pd.isna(row["ema200"]) or row["supertrend"] is None:
        return None
    if row["close"] > row["ema200"] and row["supertrend"] == "YESIL":
        return "LONG"
    if row["close"] < row["ema200"] and row["supertrend"] == "KIRMIZI":
        return "SHORT"
    return None


@track_errors
def detect_pullback_entry(symbol: str, bias: str):
    try:
        df = fetch_df(symbol, TP_M15_TIMEFRAME, max(TP_EMA_SLOW, TP_SWING_LOOKBACK) + 50)
    except Exception:
        return None
    if len(df) < TP_EMA_SLOW + 5:
        return None
    df["ema_fast"] = df["close"].ewm(span=TP_EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=TP_EMA_SLOW, adjust=False).mean()
    df["rsi"] = _compute_rsi(df["close"], TP_RSI_PERIOD)
    df["atr14"] = _compute_atr(df, ATR_PERIOD)

    row = df.iloc[-2]
    if pd.isna(row["ema_fast"]) or pd.isna(row["ema_slow"]) or pd.isna(row["rsi"]) or pd.isna(row["atr14"]):
        return None

    tol = row["close"] * (TP_PULLBACK_TOLERANCE_PCT / 100)
    touched_fast = abs(row["low"] - row["ema_fast"]) <= tol or abs(row["high"] - row["ema_fast"]) <= tol
    touched_slow = abs(row["low"] - row["ema_slow"]) <= tol or abs(row["high"] - row["ema_slow"]) <= tol
    touched_ema = touched_fast or touched_slow

    if bias == "LONG":
        confirm_candle = row["close"] > row["open"]
        rsi_ok = row["rsi"] < TP_RSI_LONG_MAX
        if touched_ema and rsi_ok and confirm_candle:
            swing_low = df.iloc[-2 - TP_SWING_LOOKBACK:-2]["low"].min()
            return row["close"], row["atr14"], swing_low
    else:
        confirm_candle = row["close"] < row["open"]
        rsi_ok = row["rsi"] > TP_RSI_SHORT_MIN
        if touched_ema and rsi_ok and confirm_candle:
            swing_high = df.iloc[-2 - TP_SWING_LOOKBACK:-2]["high"].max()
            return row["close"], row["atr14"], swing_high
    return None


# ============================================================
# POZISYON KAYDI
# ============================================================
def _read_positions():
    if not os.path.isfile(POSITIONS_FILE):
        return []
    with open(POSITIONS_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("partial_tp_taken", "0")
        r.setdefault("stop_missing_count", "0")
    return rows


def _write_positions(rows):
    with open(POSITIONS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POSITION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 0. KATMAN: ANA PORTFÖY BEYNİ (Master Portfolio Brain)
# ============================================================
# DERS (2026-07-28, Gemini'nin talebi): kullanicinin risk yoneticisi rolunu
# ustlenen ust katman. Piyasa Beyni'nin (ADX rejimine gore) belirledigi
# riskin USTUNE, hesabin GENEL SAGLIGINA gore ek kisitlamalar/carpanlar
# uygular - rejim riskini DEGISTIRMEZ, sadece daha da kisar gerekirse.
_PORTFOLIO_STATE_DEFAULT = {
    "day_start_date": None, "day_start_balance": None, "halt_until": None,
    "consecutive_losses": 0, "cooldown_until": None, "risk_penalty_until": None,
    "cumulative_pnl_pct": 0.0,
}


def _load_portfolio_state():
    if os.path.isfile(PORTFOLIO_STATE_FILE):
        try:
            with open(PORTFOLIO_STATE_FILE) as f:
                state = json.load(f)
            merged = dict(_PORTFOLIO_STATE_DEFAULT)
            merged.update(state)
            return merged
        except Exception as e:
            print(f"Portföy Beyni: state okunamadi, sifirdan basliyor ({e})")
    return dict(_PORTFOLIO_STATE_DEFAULT)


def _save_portfolio_state(state):
    try:
        with open(PORTFOLIO_STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"Portföy Beyni: state kaydedilemedi ({e})")


def _portfolio_note_trade_closed(pct_change: float):
    """Her pozisyon kapanisinda cagrilir - ardisik kayip ve kumulatif PnL takibi."""
    state = _load_portfolio_state()
    state["cumulative_pnl_pct"] = state.get("cumulative_pnl_pct", 0.0) + pct_change
    if pct_change < 0:
        state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
        if state["consecutive_losses"] >= PORTFOLIO_CONSEC_LOSS_LIMIT:
            now = datetime.now()
            state["cooldown_until"] = (now + timedelta(hours=PORTFOLIO_COOLDOWN_HOURS)).isoformat()
            state["risk_penalty_until"] = (now + timedelta(hours=PORTFOLIO_RISK_PENALTY_HOURS)).isoformat()
            state["consecutive_losses"] = 0
            send_telegram_message(
                f"🧠 [Ana Portföy Beyni] Arka arkaya {PORTFOLIO_CONSEC_LOSS_LIMIT} işlem STOP oldu.\n"
                f"⏸️ {PORTFOLIO_COOLDOWN_HOURS} saat yeni işlem açılmayacak (cooldown).\n"
                f"⚠️ Sonrasında da {PORTFOLIO_RISK_PENALTY_HOURS} saat boyunca risk %{PORTFOLIO_RISK_PENALTY_MULT*100:.0f} azaltılmış devam edecek."
            )
    else:
        state["consecutive_losses"] = 0
    _save_portfolio_state(state)


def update_portfolio_brain():
    """Her cycle basi cagrilir - gunluk kayip limiti kontrolu + gun donusu."""
    state = _load_portfolio_state()
    try:
        balance = exchange.fetch_balance()
        current_balance = balance.get("USDT", {}).get("total") or balance.get("total", {}).get("USDT") or 0
    except Exception as e:
        print(f"Portföy Beyni: bakiye cekilemedi ({e})")
        return state

    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("day_start_date") != today:
        state["day_start_date"] = today
        state["day_start_balance"] = current_balance
        # Yeni gunde, onceki gunden kalma bir gunluk-kayip donmasi varsa
        # (ve suresi zaten dolmussa) burada dogal olarak temizlenmis olur;
        # aktif bir halt suresi hala devam ediyorsa (24 saat dolmadiysa)
        # asagidaki now < halt_until kontrolu zaten onu koruyacaktir.
        _save_portfolio_state(state)
        return state

    day_start = state.get("day_start_balance") or current_balance
    if day_start:
        daily_pnl_pct = (current_balance - day_start) / day_start * 100
        already_halted = bool(state.get("halt_until"))
        if daily_pnl_pct <= -PORTFOLIO_DAILY_LOSS_LIMIT_PCT and not already_halted:
            halt_until = datetime.now() + timedelta(hours=PORTFOLIO_DAILY_HALT_HOURS)
            state["halt_until"] = halt_until.isoformat()
            send_telegram_message(
                f"🚨 [Ana Portföy Beyni] Günlük kayıp limiti (%{PORTFOLIO_DAILY_LOSS_LIMIT_PCT}) aşıldı "
                f"(bugün: %{daily_pnl_pct:.2f}).\n"
                f"⛔ Yeni işlem açılışları {PORTFOLIO_DAILY_HALT_HOURS} saat boyunca DONDURULDU."
            )
    _save_portfolio_state(state)
    return state


def _portfolio_trading_allowed():
    state = _load_portfolio_state()
    now = datetime.now()
    halt_until = state.get("halt_until")
    if halt_until:
        try:
            if now < datetime.fromisoformat(halt_until):
                return False, "günlük kayıp limiti donması aktif"
        except Exception:
            pass
    cooldown_until = state.get("cooldown_until")
    if cooldown_until:
        try:
            if now < datetime.fromisoformat(cooldown_until):
                return False, "ardışık kayıp cooldown'u aktif"
        except Exception:
            pass
    return True, ""


def _portfolio_risk_multiplier():
    """Piyasa Beyni'nin belirledigi riske ek olarak uygulanacak carpan (<=1.0)."""
    state = _load_portfolio_state()
    mult = 1.0
    now = datetime.now()
    risk_penalty_until = state.get("risk_penalty_until")
    if risk_penalty_until:
        try:
            if now < datetime.fromisoformat(risk_penalty_until):
                mult = min(mult, PORTFOLIO_RISK_PENALTY_MULT)
        except Exception:
            pass
    # Genel portfoy durumu: kumulatif PnL negatifse muhafazakar tavana cek.
    cumulative = state.get("cumulative_pnl_pct", 0.0)
    if cumulative < 0 and PORTFOLIO_LOSS_RISK_PCT < PORTFOLIO_PROFIT_RISK_PCT:
        capped = PORTFOLIO_LOSS_RISK_PCT / PORTFOLIO_PROFIT_RISK_PCT
        mult = min(mult, capped)
    return mult


def _effective_risk_pct():
    return _current_risk_pct * _portfolio_risk_multiplier()


def log_closed_trade(symbol, direction, entry_price, exit_price, pct_change, reason):
    is_new = not os.path.isfile(CLOSED_TRADES_FILE)
    with open(CLOSED_TRADES_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CLOSED_TRADE_FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(), "symbol": symbol, "direction": direction,
            "entry_price": entry_price, "exit_price": exit_price,
            "pct_change": f"{pct_change:.3f}", "reason": reason,
        })
    try:
        _portfolio_note_trade_closed(pct_change)
    except Exception as e:
        print(f"Portföy Beyni: kapanan islem islenemedi ({e})")


def _get_reference_balance():
    env_override = os.environ.get("TP_REFERENCE_BALANCE")
    if env_override:
        return float(env_override)
    if os.path.isfile(REFERENCE_BALANCE_FILE):
        with open(REFERENCE_BALANCE_FILE) as f:
            return float(f.read().strip())
    balance = exchange.fetch_balance()
    free_usdt = balance.get("USDT", {}).get("free") or balance.get("free", {}).get("USDT") or 0
    with open(REFERENCE_BALANCE_FILE, "w") as f:
        f.write(str(free_usdt))
    return float(free_usdt)


def _compute_position_size(symbol: str, entry_price: float, stop_price: float) -> float:
    reference_balance = _get_reference_balance()
    risk_amount = reference_balance * (_effective_risk_pct() / 100)
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0
    qty_by_risk = risk_amount / stop_distance

    # DERS (2026-07-28, asil kok neden): risk-bazli qty TEK BASINA, stop
    # mesafesi cok dar oldugunda (ozellikle YATAY rejimde M15 dalgalanmalar
    # kucuk oluyor) devasa buyuklukte pozisyon miktari uretebiliyordu - kucuk
    # bir dolar riski ($55 gibi) hedeflerken bile marjini asan bir notional
    # cikabiliyordu. Eski dosyada bu ikinci bir tavanla (bakiye/kaldirac
    # bazli marjin sinirlamasi) engelleniyordu, sifirdan yazimda unutulmustu.
    # Iki adayin KUCUGUNU aliyoruz - hangisi daha kisitlayiciysa o gecerli olur.
    try:
        balance = exchange.fetch_balance()
        free_usdt = balance.get("USDT", {}).get("free") or balance.get("free", {}).get("USDT") or 0
    except Exception:
        free_usdt = reference_balance
    safe_balance_for_margin = min(reference_balance, free_usdt)
    max_notional = safe_balance_for_margin * (TP_POSITION_PCT_OF_BALANCE / 100) * TP_LEVERAGE
    qty_by_margin_cap = max_notional / entry_price

    qty = min(qty_by_risk, qty_by_margin_cap)

    # Borsanin izin verdigi MAKSIMUM emir miktarini da ayrica kontrol et
    # (dusuk fiyatli/yuksek arzli coinlerde -4005 hatasina yol aciyordu).
    try:
        market = exchange.markets.get(symbol, {})
        max_amount = (market.get("limits", {}) or {}).get("amount", {}).get("max")
        if max_amount and qty > max_amount:
            qty = max_amount
    except Exception:
        pass

    try:
        qty = float(exchange.amount_to_precision(symbol, qty))
    except Exception:
        pass
    return qty


def _set_leverage_safe(symbol: str):
    try:
        exchange.set_leverage(TP_LEVERAGE, symbol)
    except Exception as e:
        print(f"Kaldirac ayarlama hatasi ({symbol}): {e}")


def _close_position(symbol: str, direction: str, qty: float) -> str:
    try:
        positions = exchange.fetch_positions([symbol])
        live_qty, live_direction = qty, direction
        found = False
        for p in positions:
            contracts = abs(p.get("contracts") or 0)
            if contracts > 0:
                live_qty = contracts
                live_direction = "LONG" if (p.get("side") == "long") else "SHORT"
                found = True
                break
        if not found:
            return ""
    except Exception:
        live_qty, live_direction = qty, direction

    try:
        close_side = "sell" if live_direction == "LONG" else "buy"
        exchange.create_order(symbol, type="market", side=close_side, amount=live_qty, params={"reduceOnly": True})
        try:
            exchange.cancel_all_orders(symbol)
        except Exception:
            pass
        return ""
    except Exception as e:
        return str(e)


# DERS (2026-07-28): hesap genelindeki borsa stop emri limiti (-4045) dolunca,
# ayni tur icinde denenecek SONRAKI coinler de ayni sebeple basarisiz olur.
# Pozisyonu artik kapatmiyoruz (bkz. open_position), ama scan_for_entries'in
# bosuna yeni girisim denemeyi birakmasi icin hafif bir tur-ici sinyal.
_stop_limit_reached_this_cycle = False


def _mark_stop_limit_reached():
    global _stop_limit_reached_this_cycle
    _stop_limit_reached_this_cycle = True


# ============================================================
# INFAZ KATMANI: pozisyon acma
# ============================================================
@track_errors
def open_position(symbol: str, direction: str, entry_price: float, atr14: float, swing_stop: float):
    _set_leverage_safe(symbol)
    side = "buy" if direction == "LONG" else "sell"

    qty = _compute_position_size(symbol, entry_price, swing_stop)
    if qty <= 0:
        return None

    order = exchange.create_order(symbol, type="market", side=side, amount=qty)
    real_entry_price = order.get("average") or order.get("price")
    if not real_entry_price:
        try:
            real_entry_price = exchange.fetch_ticker(symbol)["last"]
        except Exception:
            real_entry_price = entry_price

    stop_price = swing_stop
    stop_side = "sell" if direction == "LONG" else "buy"
    stop_order_id = ""
    stop_missing_at_entry = False
    try:
        stop_order = exchange.create_order(
            symbol, type="STOP_MARKET", side=stop_side, amount=qty,
            params={"stopPrice": stop_price, "reduceOnly": True},
        )
        stop_order_id = stop_order.get("id", "")
    except Exception as e:
        # TAMIRCI: kapatmadan once bir onarim + tek seferlik yeniden deneme sansi.
        repaired = tamirci_attempt_repair(symbol, e)
        if repaired:
            try:
                stop_order = exchange.create_order(
                    symbol, type="STOP_MARKET", side=stop_side, amount=qty,
                    params={"stopPrice": stop_price, "reduceOnly": True},
                )
                stop_order_id = stop_order.get("id", "")
                send_telegram_message(f"🛠️ [TAMİRCİ DEVREYE GİRDİ] {symbol}: stop emri sorunu giderildi, işleme devam ediliyor.")
            except Exception as e2:
                e = e2  # asagidaki mesaj/kayit icin son hatayi kullan

        if not stop_order_id:
            # DERS (2026-07-28): girişte borsa stop emri konulamadigi anda pozisyonu
            # ANINDA kapatmak asiri tepkiydi - -4045 gibi hatalar Demo Trading'in
            # bilinen tutarsiz/gecici platform kusurlarindan biri (bkz. proje notlari)
            # ve cogunlukla tek seferlik. update_positions() zaten devam eden
            # pozisyonlar icin TAM da bu senaryoya karsi bir hibrit desen icermektedir:
            # her turda borsa stop'unu yeniden koymayi dener (Tamirci onariyla
            # birlikte), fiyat stop seviyesini gectiginde yazilimsal olarak (borsa
            # emri olsun olmasin) kapatir, ve sadece 3 tur ust uste basarisiz olursa
            # devre kesiciyle zorla kapatir (missing_count / CIRCUIT_BREAKER_MAX_FAILURES).
            # Girişte de ayni desene uyup pozisyonu ACIK BIRAKIYORUZ.
            stop_missing_at_entry = True
            print(f"{symbol}: giriste borsa stop emri konulamadi ({e}), Tamirci onarimi da yetersiz kaldi - pozisyon acik tutuluyor, yazilimsal yedek devrede.")
            if _is_stop_limit_error(e):
                # Hesap genelindeki stop emri limiti dolu olabilir - bu turda
                # denenecek SONRAKI coinler de ayni sebeple basarisiz olur, o yuzden
                # scan_for_entries'e bu turu erken kesmesi icin sinyal veriyoruz.
                # (Bu pozisyonu kapatmiyor, sadece taramayi durduruyor.)
                _mark_stop_limit_reached()

    stop_distance = abs(real_entry_price - stop_price)
    partial_tp_price = (real_entry_price + stop_distance * TP_PARTIAL_TP_R_MULT if direction == "LONG"
                         else real_entry_price - stop_distance * TP_PARTIAL_TP_R_MULT)
    partial_qty = qty * (TP_PARTIAL_CLOSE_PCT / 100)
    try:
        partial_qty = float(exchange.amount_to_precision(symbol, partial_qty))
    except Exception:
        pass
    # DERS (2026-07-28, Gemini'nin nihai mimari karari): TP emri artik borsaya
    # HIC KOYULMUYOR. Zaten update_positions() her cycle'da partial_tp_price'i
    # fiyatla karsilastirip gerekirse market emriyle kapatiyordu (asagida
    # degismedi) - borsadaki TAKE_PROFIT_MARKET emri fiilen hic kullanilmayan,
    # sadece koşullu emir kapasitesini tuketen fazladan bir emirdi. Kaldirilmasi
    # -4045'in ana nedeni olan hesap-geneli koşullu emir sayisini yariya
    # indiriyor (MAX_OPEN_POSITIONS ile birlikte, borsada sadece stop emirleri
    # kalir). TP guvenlik-kritik degil - bir cycle kacirilsa bile en kotu
    # ihtimalle kar kilitlenmesi birkac dakika gecikir, pozisyon risk altina
    # girmez (stop hala borsada/yazilimsal yedekte).
    tp_order_id = ""

    rows = _read_positions()
    rows.append({
        "symbol": symbol, "direction": direction,
        "entry_price": real_entry_price, "stop_price": stop_price, "extreme_price": real_entry_price,
        "entry_time": datetime.now().isoformat(),
        "exchange_stop_order_id": stop_order_id,
        "partial_tp_price": partial_tp_price, "partial_tp_taken": "0",
        "original_qty": qty, "exchange_tp_order_id": tp_order_id,
        "stop_missing_count": "1" if stop_missing_at_entry else "0",
    })
    _write_positions(rows)

    send_telegram_message(
        f"📈 {symbol} {direction} pozisyon açıldı (H1 trend + M15 pullback) [Rejim: {_current_regime}].\n"
        f"Giriş: {real_entry_price:.6f} | Stop: {stop_price:.6f} | "
        f"Parsiyel TP (%{TP_PARTIAL_CLOSE_PCT}, {TP_PARTIAL_TP_R_MULT}R): {partial_tp_price:.6f}\n"
        f"{'✅ Borsa stop aktif' if stop_order_id else '⚠️ Borsa stop KONULAMADI'} | "
        f"🖥️ TP yazılımsal takip ediliyor\n"
        f"Kaldıraç: {TP_LEVERAGE}x | Risk: %{_effective_risk_pct():.2f}"
    )
    return qty


# ============================================================
# INFAZ KATMANI: pozisyon izleme
# ============================================================
@track_errors
def update_positions():
    rows = _read_positions()
    if not rows:
        return
    still_open = []
    for r in rows:
        symbol = r["symbol"]
        direction = r["direction"]
        entry_price = float(r["entry_price"])
        stop_price = float(r["stop_price"])
        extreme_price = float(r["extreme_price"])
        partial_tp_taken = r.get("partial_tp_taken", "0") == "1"
        partial_tp_price = float(r["partial_tp_price"]) if r.get("partial_tp_price") else None
        original_qty = float(r["original_qty"]) if r.get("original_qty") else None

        try:
            positions = exchange.fetch_positions([symbol])
            live_qty = 0
            for p in positions:
                if abs(p.get("contracts") or 0) > 0:
                    live_qty = abs(p["contracts"])
                    break
        except Exception as e:
            print(f"{symbol}: pozisyon kontrolu basarisiz ({e})")
            still_open.append(r)
            continue

        if live_qty == 0:
            try:
                exit_price = exchange.fetch_ticker(symbol)["last"]
            except Exception:
                exit_price = entry_price
            raw_pct = (exit_price - entry_price) / entry_price * 100
            pct_change = raw_pct if direction == "LONG" else -raw_pct
            reason = "Breakeven/Trailing (borsa tetikledi)" if partial_tp_taken else "Stop/TP (borsa tetikledi)"
            log_closed_trade(symbol, direction, entry_price, exit_price, pct_change, reason)
            send_telegram_message(
                f"🔔 {symbol} {direction} pozisyon KAPANDI ({reason}).\n"
                f"Giriş: {entry_price:.6f} | Çıkış (yaklaşık): {exit_price:.6f} | Değişim: {pct_change:+.2f}%"
            )
            continue

        try:
            price = exchange.fetch_ticker(symbol)["last"]
        except Exception:
            still_open.append(r)
            continue

        if not partial_tp_taken and partial_tp_price is not None:
            tp_hit = (price >= partial_tp_price) if direction == "LONG" else (price <= partial_tp_price)
            if tp_hit:
                partial_qty = original_qty * (TP_PARTIAL_CLOSE_PCT / 100) if original_qty else live_qty * 0.5
                if live_qty > partial_qty * 0.9:
                    try:
                        close_side = "sell" if direction == "LONG" else "buy"
                        exchange.create_order(symbol, type="market", side=close_side,
                                               amount=min(partial_qty, live_qty), params={"reduceOnly": True})
                    except Exception as e:
                        print(f"{symbol}: parsiyel TP kapatma basarisiz ({e})")
                        still_open.append(r)
                        continue

                r["partial_tp_taken"] = "1"
                r["stop_price"] = str(entry_price)
                r["extreme_price"] = str(entry_price)
                try:
                    open_orders = exchange.fetch_open_orders(symbol)
                    for o in open_orders:
                        if o.get("id") in (r.get("exchange_stop_order_id"), r.get("exchange_tp_order_id")):
                            exchange.cancel_order(o["id"], symbol)
                    remaining_qty = max(live_qty - partial_qty, 0)
                    stop_side = "sell" if direction == "LONG" else "buy"
                    new_stop = exchange.create_order(
                        symbol, type="STOP_MARKET", side=stop_side, amount=remaining_qty,
                        params={"stopPrice": entry_price, "reduceOnly": True},
                    )
                    r["exchange_stop_order_id"] = new_stop.get("id", "")
                    r["exchange_tp_order_id"] = ""
                    r["stop_missing_count"] = "0"
                except Exception as e:
                    print(f"{symbol}: breakeven stop guncellenemedi ({e})")

                raw_pct = (price - entry_price) / entry_price * 100
                pct_change = raw_pct if direction == "LONG" else -raw_pct
                send_telegram_message(
                    f"💰 {symbol} {direction}: %{TP_PARTIAL_CLOSE_PCT} parsiyel TP alındı "
                    f"({TP_PARTIAL_TP_R_MULT}R, {pct_change:+.2f}%). Stop girişe (breakeven) çekildi."
                )
                still_open.append(r)
                continue

        if partial_tp_taken:
            try:
                df = fetch_df(symbol, TP_M15_TIMEFRAME, 60)
                df["atr14"] = _compute_atr(df, ATR_PERIOD)
                latest_atr = df.iloc[-2]["atr14"]
            except Exception as e:
                print(f"{symbol}: ATR guncellenemedi ({e})")
                still_open.append(r)
                continue

            if not pd.isna(latest_atr):
                if direction == "LONG":
                    new_extreme = max(extreme_price, price)
                    new_stop = max(stop_price, new_extreme - latest_atr * TP_POST_BREAKEVEN_TRAIL_MULT)
                else:
                    new_extreme = min(extreme_price, price)
                    new_stop = min(stop_price, new_extreme + latest_atr * TP_POST_BREAKEVEN_TRAIL_MULT)
                if new_stop != stop_price:
                    r["stop_price"] = str(new_stop)
                    r["extreme_price"] = str(new_extreme)
                    stop_price = new_stop
                    try:
                        open_orders = exchange.fetch_open_orders(symbol)
                        for o in open_orders:
                            if o.get("id") == r.get("exchange_stop_order_id"):
                                exchange.cancel_order(o["id"], symbol)
                        stop_side = "sell" if direction == "LONG" else "buy"
                        updated_stop = exchange.create_order(
                            symbol, type="STOP_MARKET", side=stop_side, amount=live_qty,
                            params={"stopPrice": new_stop, "reduceOnly": True},
                        )
                        r["exchange_stop_order_id"] = updated_stop.get("id", "")
                    except Exception as e:
                        print(f"{symbol}: trailing stop borsa guncellemesi basarisiz ({e})")

        missing_count = int(r.get("stop_missing_count", "0"))
        try:
            open_orders = exchange.fetch_open_orders(symbol)
            has_stop = any(o.get("id") == r.get("exchange_stop_order_id") for o in open_orders) if r.get("exchange_stop_order_id") else False
            if not has_stop:
                missing_count += 1
                r["stop_missing_count"] = str(missing_count)
                if missing_count >= CIRCUIT_BREAKER_MAX_FAILURES:
                    send_telegram_message(
                        f"🚨 {symbol}: borsa stop emri {missing_count} tur üst üste sağlanamadı — "
                        f"DEVRE KESİCİ devreye girdi, pozisyon HEMEN kapatılıyor."
                    )
                    close_err = _close_position(symbol, direction, live_qty)
                    if not close_err:
                        raw_pct = (price - entry_price) / entry_price * 100
                        pct_change = raw_pct if direction == "LONG" else -raw_pct
                        log_closed_trade(symbol, direction, entry_price, price, pct_change, "Devre kesici (stop garantilenemedi)")
                        continue
                    else:
                        still_open.append(r)
                        continue
                stop_side = "sell" if direction == "LONG" else "buy"
                try:
                    new_stop_order = exchange.create_order(
                        symbol, type="STOP_MARKET", side=stop_side, amount=live_qty,
                        params={"stopPrice": stop_price, "reduceOnly": True},
                    )
                    r["exchange_stop_order_id"] = new_stop_order.get("id", "")
                    send_telegram_message(f"🛡️ {symbol}: borsa stop emri bulunamadı, yeniden koyuldu ({missing_count}/{CIRCUIT_BREAKER_MAX_FAILURES}).")
                except Exception as e:
                    # TAMIRCI: dogrudan yeniden koyma basarisiz oldu - once onarim
                    # dene (hayalet emirleri temizle / oturumu sifirla), sonra
                    # bir kez daha dene. Bu da basarisiz olursa mevcut devre
                    # kesici sayaci (missing_count, yukarida zaten arttirildi)
                    # normal akisinda ilerlemeye devam eder.
                    repaired = tamirci_attempt_repair(symbol, e)
                    if repaired:
                        try:
                            new_stop_order = exchange.create_order(
                                symbol, type="STOP_MARKET", side=stop_side, amount=live_qty,
                                params={"stopPrice": stop_price, "reduceOnly": True},
                            )
                            r["exchange_stop_order_id"] = new_stop_order.get("id", "")
                            r["stop_missing_count"] = "0"
                            send_telegram_message(f"🛠️ [TAMİRCİ DEVREYE GİRDİ] {symbol}: stop emri sorunu giderildi, işleme devam ediliyor.")
                        except Exception as e2:
                            print(f"{symbol}: Tamirci onarimindan sonra da stop konulamadi ({e2})")
                    else:
                        print(f"{symbol}: stop yeniden koyulamadi, Tamirci bu hata turu icin onarim uygulamadi ({e})")
            else:
                r["stop_missing_count"] = "0"
        except Exception as e:
            print(f"{symbol}: borsa stop kontrolu basarisiz ({e})")

        breached = (price <= stop_price) if direction == "LONG" else (price >= stop_price)
        if breached:
            close_err = _close_position(symbol, direction, live_qty)
            raw_pct = (price - entry_price) / entry_price * 100
            pct_change = raw_pct if direction == "LONG" else -raw_pct
            if not close_err:
                reason = "Breakeven/Trailing" if partial_tp_taken else "Stop (yazılımsal)"
                log_closed_trade(symbol, direction, entry_price, price, pct_change, reason)
                send_telegram_message(
                    f"📉 {symbol} {direction} pozisyon kapandı ({reason}). "
                    f"Giriş: {entry_price:.6f} | Çıkış: {price:.6f} | Değişim: {pct_change:+.2f}%"
                )
            else:
                still_open.append(r)
            continue

        still_open.append(r)

    _write_positions(still_open)


def cleanup_orphaned_orders():
    rows = _read_positions()
    tracked_symbols = {r["symbol"] for r in rows}
    # DERS (2026-07-28, Tamirci katmani): eskiden sadece TAKIP EDILMEYEN
    # semboller temizleniyordu. Ama takip edilen bir sembolde de, eski/iptal
    # edilmemis bir stop/TP emri (ornegin trailing guncellemesi sirasinda
    # iptal cagrisi sessizce basarisiz olduysa) borsa tarafinda birikip
    # hesabin stop emri limitine katkida bulunabilir. Artik takip edilen
    # semboller icin de, o pozisyonun KAYITLI stop/TP emir ID'leriyle
    # eslesmeyen HER emri hayalet sayip iptal ediyoruz.
    known_ids_by_symbol = {}
    for r in rows:
        ids = {r.get("exchange_stop_order_id"), r.get("exchange_tp_order_id")}
        known_ids_by_symbol.setdefault(r["symbol"], set()).update(i for i in ids if i)

    for sym in WATCHLIST:
        try:
            orders = exchange.fetch_open_orders(sym)
        except Exception:
            continue
        known_ids = known_ids_by_symbol.get(sym, set()) if sym in tracked_symbols else set()
        for o in orders:
            if o.get("id") in known_ids:
                continue
            try:
                exchange.cancel_order(o["id"], sym)
            except Exception:
                pass


@track_errors
def scan_for_entries():
    global _stop_limit_reached_this_cycle
    _stop_limit_reached_this_cycle = False
    allowed, reason = _portfolio_trading_allowed()
    if not allowed:
        print(f"Ana Portföy Beyni: yeni işlem açılışı engellendi ({reason}).")
        return
    open_symbols = {r["symbol"] for r in _read_positions()}
    if len(open_symbols) >= MAX_OPEN_POSITIONS:
        return
    scanned = 0
    trend_found = 0
    entries = 0
    margin_exhausted = False
    for symbol in WATCHLIST:
        if symbol in open_symbols:
            continue
        if len(open_symbols) >= MAX_OPEN_POSITIONS:
            break
        if margin_exhausted or _stop_limit_reached_this_cycle:
            # Bir kere marjin yetersiz ya da borsa stop limiti dolu cikinca,
            # ayni turda denenen SONRAKI her coin de ayni sebeple basarisiz
            # olur - tek tek deneyip her birinde ayri Dedektif raporu
            # gondermek yerine, bu turu burada kesiyoruz (2026-07-28).
            break
        scanned += 1
        try:
            bias = get_h1_trend_bias(symbol)
            if bias is None:
                continue
            trend_found += 1
            result = detect_pullback_entry(symbol, bias)
            if result is None:
                continue
            entry_price, atr14, swing_stop = result
            opened_qty = open_position(symbol, bias, entry_price, atr14, swing_stop)
            if opened_qty:
                entries += 1
                open_symbols.add(symbol)
        except Exception as e:
            print(f"{symbol}: tarama hatasi ({e})")
            if "-2019" in str(e) or "insufficient" in str(e).lower():
                margin_exhausted = True
                print("Marjin tukendigi tespit edildi - bu turda baska yeni islem denenmeyecek.")

    print(f"Tur özeti: rejim={_current_regime} | taranan={scanned} | H1 trend bulundu={trend_found} | açılan={entries}")


# ============================================================
# TELEGRAM KOMUT MESAJLARI
# ============================================================
def build_positions_message():
    rows = _read_positions()
    if not rows:
        return "📋 Şu an açık pozisyon yok."
    lines = [f"📋 {len(rows)} açık pozisyon:"]
    for r in rows:
        lines.append(
            f"  {r['symbol']} {r['direction']} | Giriş: {r['entry_price']} | Stop: {r['stop_price']} | "
            f"Parsiyel TP alındı: {'Evet' if r.get('partial_tp_taken') == '1' else 'Hayır'}"
        )
    return "\n".join(lines)


def build_orders_message():
    tracked_symbols = {r["symbol"] for r in _read_positions()}
    all_orders = []
    for sym in tracked_symbols:
        try:
            all_orders.extend(exchange.fetch_open_orders(sym))
        except Exception:
            pass
    if not all_orders:
        return "📋 Takip edilen pozisyonlarda açık emir yok."
    lines = [f"📋 Toplam {len(all_orders)} açık emir:"]
    for o in all_orders:
        lines.append(f"  {o.get('symbol')}: {o.get('type')} @ {o.get('stopPrice') or o.get('price')}")
    return "\n".join(lines)


def build_stats_message():
    if not os.path.isfile(CLOSED_TRADES_FILE):
        return "📊 Henüz kapanan işlem yok."
    with open(CLOSED_TRADES_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "📊 Henüz kapanan işlem yok."
    total = len(rows)
    wins = sum(1 for r in rows if float(r["pct_change"]) > 0)
    total_pct = sum(float(r["pct_change"]) for r in rows)
    lines = [
        f"📊 İstatistik (tüm zamanlar)",
        f"Toplam kapanan işlem: {total}",
        f"Kazanan: {wins} | Kaybeden: {total - wins} | İsabet: %{wins/total*100:.1f}",
        f"Toplam net: %{total_pct:+.2f} | Ort./işlem: %{total_pct/total:+.3f}",
        "",
        "Son 5 işlem:",
    ]
    for r in rows[-5:]:
        lines.append(f"  {r['symbol']} {r['direction']}: %{float(r['pct_change']):+.2f} ({r['reason']})")
    return "\n".join(lines)


# ============================================================
# ANA DONGU
# ============================================================
def scan_once():
    global _regime_cycle_counter
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Tarama basliyor...")
    try:
        cleanup_orphaned_orders()
    except Exception as e:
        print(f"Basibos emir temizligi basarisiz ({e})")

    if _regime_cycle_counter % REGIME_CHECK_EVERY_N_CYCLES == 0:
        try:
            update_market_regime()
        except Exception as e:
            print(f"Piyasa Beyni guncellenemedi ({e})")
    _regime_cycle_counter += 1

    try:
        update_portfolio_brain()
    except Exception as e:
        print(f"Ana Portföy Beyni guncellenemedi ({e})")

    update_positions()

    if not NEW_TRADES_HALTED:
        scan_for_entries()


def _perform_full_reset():
    """RESET_ON_START=true oldugunda, run_forever() baslamadan ONCE bir kez
    cagrilir. Gemini'nin talebi: butun duzeltmeleri (yazilimsal TP, Tamirci,
    Portfoy Beyni) adil ve temiz sartlarda test etmek icin '0 km' reset.
    1) Hesaptaki TUM acik pozisyonlari (WATCHLIST disinda kalsa bile) market
       emriyle kapatir, 2) TUM acik emirleri iptal eder, 3) positions.csv /
       closed_trades.csv / portfolio_state.json'i sifirlar (veri kaybini
       onlemek icin eski dosyalari SILMEK yerine zaman damgali yedege tasir),
       4) referans bakiyeyi 10.000 USDT'ye sifirlar."""
    print("=== TAM RESET basliyor (RESET_ON_START=true) ===")
    closed = 0
    try:
        positions = exchange.fetch_positions()
    except Exception as e:
        positions = []
        print(f"Reset: acik pozisyonlar cekilemedi ({e})")
    for p in positions:
        try:
            contracts = abs(p.get("contracts") or 0)
            if contracts <= 0:
                continue
            sym = p["symbol"]
            side = "sell" if (p.get("side") == "long") else "buy"
            exchange.create_order(sym, type="market", side=side, amount=contracts, params={"reduceOnly": True})
            closed += 1
        except Exception as e:
            print(f"Reset: {p.get('symbol')} kapatilamadi ({e})")

    cancelled_orders = 0
    symbols_to_check = set(WATCHLIST) | {p.get("symbol") for p in positions if p.get("symbol")}
    for sym in symbols_to_check:
        try:
            for o in exchange.fetch_open_orders(sym):
                try:
                    exchange.cancel_order(o["id"], sym)
                    cancelled_orders += 1
                except Exception:
                    pass
        except Exception:
            pass

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for f in (POSITIONS_FILE, CLOSED_TRADES_FILE, PORTFOLIO_STATE_FILE):
        if os.path.isfile(f):
            try:
                os.rename(f, f"{f}.reset_{stamp}.bak")
            except Exception as e:
                print(f"Reset: {f} yedeklenemedi ({e})")

    try:
        with open(REFERENCE_BALANCE_FILE, "w") as f:
            f.write("10000")
    except Exception as e:
        print(f"Reset: referans bakiye yazilamadi ({e})")

    env_override_note = ""
    if os.environ.get("TP_REFERENCE_BALANCE"):
        env_override_note = (
            "\n⚠️ TP_REFERENCE_BALANCE env değişkeni hâlâ ayarlı — dosyadaki 10.000 "
            "yerine onu kullanmaya devam edecek, Railway'den kaldırman gerekebilir."
        )

    send_telegram_message(
        f"🔄 [TAM RESET TAMAMLANDI]\n"
        f"Kapatılan pozisyon: {closed} | İptal edilen emir: {cancelled_orders}\n"
        f"positions.csv / closed_trades.csv / portfolio_state.json sıfırlandı (eskileri .bak olarak saklandı).\n"
        f"Referans bakiye: 10.000 USDT.{env_override_note}\n"
        f"🧠 Ana Portföy Beyni: gün başı bakiyesi, ardışık kayıp sayacı ve cooldown sıfırlandı.\n"
        f"Sistem '0 km' olarak yeniden başlıyor."
    )
    print(f"=== TAM RESET tamamlandi: {closed} pozisyon kapatildi, {cancelled_orders} emir iptal edildi ===")


def run_forever():
    if RESET_ON_START:
        try:
            _perform_full_reset()
        except Exception as e:
            print(f"TAM RESET basarisiz oldu: {e}")
            send_telegram_message(f"🚨 TAM RESET başarısız oldu: {e}\nBot yine de normal başlatılıyor, açık pozisyonları manuel kontrol et.")

    try:
        update_market_regime()
    except Exception as e:
        print(f"Ilk rejim olcumu basarisiz ({e})")

    recovered = _read_positions()
    halt_note = ""
    if NEW_TRADES_HALTED:
        halt_note = "\n\n🛑 YENİ İŞLEM AÇMA YASAĞI AKTİF (NEW_TRADES_HALTED=true)."
    send_telegram_message(
        f"🚀 Trend-Pullback botu (5 KATMANLI, TAMİRCİ EKLENDİ) başlatıldı.\n"
        f"1️⃣ Sistem Dedektifi | 2️⃣ Tamirci (Auto-Healer) | 3️⃣ Devre Kesiciler | 4️⃣ Piyasa Beyni | 5️⃣ Trend-Pullback\n"
        f"{len(WATCHLIST)} coin taranıyor. Güncel rejim: {_current_regime} (risk: %{_current_risk_pct})\n"
        f"Hibrit çıkış: %{TP_PARTIAL_CLOSE_PCT} parsiyel TP ({TP_PARTIAL_TP_R_MULT}R, 🖥️ tamamen yazılımsal) → breakeven → "
        f"kalan %{100-TP_PARTIAL_CLOSE_PCT} için {TP_POST_BREAKEVEN_TRAIL_MULT}×ATR trailing.\n"
        f"Borsada sadece stop emri tutulur (max {MAX_OPEN_POSITIONS} pozisyon → en fazla {MAX_OPEN_POSITIONS} koşullu emir).\n"
        f"Kaldıraç: {TP_LEVERAGE}x | Devre kesici: {CIRCUIT_BREAKER_MAX_FAILURES} deneme\n"
        f"{len(recovered)} açık pozisyon geri yüklendi.{halt_note}"
    )
    while True:
        try:
            scan_once()
        except Exception as e:
            print(f"Tarama sirasinda beklenmeyen hata: {e}")
        elapsed = 0
        poll_interval = 5
        while elapsed < CHECK_INTERVAL_MINUTES * 60:
            process_telegram_updates()
            time.sleep(poll_interval)
            elapsed += poll_interval


if __name__ == "__main__":
    run_forever()
