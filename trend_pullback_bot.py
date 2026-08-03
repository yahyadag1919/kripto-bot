"""
trend_pullback_bot.py
======================
SIFIRDAN TEMIZ KURULUM v2 (2026-07-27, Gemini'nin 4 katmanli mimarisi)
+ TAMIRCI KATMANI (2026-07-28, Gemini ile birlikte, mevcut kod uzerine eklendi)
+ ANA PORTFOY BEYNI (2026-07-28, Gemini'nin talebi)
+ ZERO EXCHANGE ORDERS (2026-07-28, Gemini'nin NIHAI karari): -4045 devre kesici
  faciasini kalici olarak bitirmek icin borsaya STOP_MARKET/TAKE_PROFIT_MARKET
  DAHIL hicbir kosullu emir gonderilmiyor. Stop VE TP tamamen yazilimsal
  (her cycle'da fiyat kontrolu + market emriyle kapatma). Eski "Devre
  Kesiciler" katmani (borsa stop'u kayipsa 3 turda zorla kapatma) bu
  degisiklikle birlikte anlamsizlasti ve kaldirildi - artik tek koruma
  katmani yazilimsal fiyat kontrolu.

0. KATMAN: ANA PORTFÖY BEYNİ (günlük kayıp limiti, ardışık kayıp cooldown'u)
1. KATMAN: SISTEM DEDEKTIFI (Error Tracker & Profiler)
2. KATMAN: TAMIRCI (Auto-Healer / Self-Healing — ağ/oturum hataları için)
3. KATMAN: PIYASA BEYNI (Dynamic Market Allocator)
4. KATMAN: CEKIRDEK STRATEJI MOTORU (Trend-Pullback, %100 yazılımsal stop+TP)
"""

import os
import re
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
        elif text.startswith("/rapor"):
            send_telegram_message(build_stop_analysis_message())


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
# DERS (2026-08-02, Gemini): pozisyon basi risk artik rejime gore
# degismiyor, SABIT %0.5. Piyasa Beyni'nin gorevi risk seviyesini degil
# hangi MOTORUN calisacagini secmek. Ana Portfoy Beyni bu tabani hala
# asagi cekebilir (ardisik kayipta yariya indirme) - ama asla yukari cikmaz.
TP_RISK_PER_TRADE_PCT = float(os.environ.get("TP_RISK_PER_TRADE_PCT", "0.5"))
TP_RISK_RANGING_PCT = TP_RISK_PER_TRADE_PCT
# ------------------------------------------------------------
# ÇIKIŞ KURALLARI (2026-08-02, Gemini'nin nihai karari)
# ------------------------------------------------------------
# Breakeven mantigi TAMAMEN KALDIRILDI. Gerekce: M15 gurultusunde stop
# girise cekildikten sonra fiyat bir kez girise dokunup pozisyonu sifira
# yakin kapatiyordu; kazanan islemler boylece erken olduruluyor ama
# kaybedenler tam stop yiyordu - asimetri kasayi eritti.
# Yerine: SABIT 1:2 R:R. Her islem ya 1R kaybeder ya 2R kazanir.
# Parsiyel TP ve ATR trailing de bu kararla birlikte kaldirildi.
TP_RR_RATIO = float(os.environ.get("TP_RR_RATIO", "2.0"))

# ------------------------------------------------------------
# 3 MOTORLU STRATEJI PARAMETRELERI (2026-08-02)
# ------------------------------------------------------------
TP_H4_TIMEFRAME = os.environ.get("TP_H4_TIMEFRAME", "4h")

# A) BREAKOUT MOTORU - Bollinger sikismasi + hacimli kirilim
BB_PERIOD = int(os.environ.get("BB_PERIOD", "20"))
BB_STD_MULT = float(os.environ.get("BB_STD_MULT", "2.0"))
# Bant genisligi (bandwidth) son N mumun en dar %X'inde ise "sikisma" sayilir
BB_SQUEEZE_LOOKBACK = int(os.environ.get("BB_SQUEEZE_LOOKBACK", "50"))
BB_SQUEEZE_PERCENTILE = float(os.environ.get("BB_SQUEEZE_PERCENTILE", "25"))
BREAKOUT_VOLUME_MULT = float(os.environ.get("BREAKOUT_VOLUME_MULT", "1.5"))
BREAKOUT_VOLUME_MA = int(os.environ.get("BREAKOUT_VOLUME_MA", "20"))

# B) LIKIDITE AVCISI - kanal disina atilan igne + kanala geri kapanis
LIQ_RANGE_LOOKBACK = int(os.environ.get("LIQ_RANGE_LOOKBACK", "20"))
# Igne, kanal sinirini en az bu kadar (ATR carpani) asmali ki "tuzak" sayilsin
LIQ_WICK_MIN_ATR = float(os.environ.get("LIQ_WICK_MIN_ATR", "0.3"))
# Igne, mumun toplam boyunun en az bu orani olmali (govde degil igne olsun)
LIQ_WICK_MIN_RATIO = float(os.environ.get("LIQ_WICK_MIN_RATIO", "0.5"))

# C) H4 SWING TREND - M15 gurultusu tamamen yok sayilir
H4_EMA_PERIOD = int(os.environ.get("H4_EMA_PERIOD", "50"))
H4_PULLBACK_TOLERANCE_PCT = float(os.environ.get("H4_PULLBACK_TOLERANCE_PCT", "1.0"))

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
    "symbol", "direction", "entry_price", "stop_price", "tp_price", "entry_time",
    "engine", "original_qty", "exchange_stop_order_id", "exchange_tp_order_id",
    "stop_missing_count",
]
# DERS (2026-08-03): eskiden sadece kapanis zamani kaydediliyordu; motor adi
# ise "reason" metninin icine gomuluydu. Bu yuzden "stoplar acilistan ne kadar
# sonra geliyor" ve "hangi motor kac stop yedi" sorulari GECMIS VERIDEN
# cevaplanamiyordu. Artik motor, giris zamani ve sure ayri sutunlar.
CLOSED_TRADE_FIELDNAMES = ["timestamp", "symbol", "direction", "engine",
                           "entry_time", "duration_min", "entry_price",
                           "exit_price", "pct_change", "reason"]

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


def _compute_bollinger(df: pd.DataFrame, period: int = BB_PERIOD, mult: float = BB_STD_MULT):
    """(orta, ust, alt, bant_genisligi) doner. Bant genisligi orta banda
    oranlanmis yuzde - farkli fiyat seviyelerindeki coinleri karsilastirmayi
    mumkun kilar (BTC ile SHIB ayni olcekte degerlendirilebilsin)."""
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = mid + std * mult
    lower = mid - std * mult
    width = (upper - lower) / mid.replace(0, np.nan) * 100
    return mid, upper, lower, width


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
# 4. KATMAN: 3 MOTORLU STRATEJI (2026-08-02, Gemini'nin karari)
# ============================================================
# Eski tek strateji (H1 trend + M15 pullback) M15 gurultusunde kasa
# eritdigi icin kaldirildi. Yerine, Piyasa Beyni'nin rejime gore
# tetikledigi 3 bagimsiz motor geldi. Her motor ayni sozlesmeyi doner:
#   (direction, entry_price, stop_price, engine_adi)  ya da  None
# Stop'u her motor KENDI mantigina gore belirler; TP her zaman
# open_position icinde 1:2 R:R olarak hesaplanir (motorlar TP belirlemez).


def _engine_result(direction, entry, stop, name):
    """Ortak dogrulama: stop yanlis tarafta ya da sifir mesafedeyse sinyali
    iptal et. Bu kontrol olmadan bozuk bir stop, 1:2 R:R hesabini ve
    pozisyon boyutunu sacmalastirirdi (sifira bolme / devasa miktar)."""
    if entry is None or stop is None:
        return None
    if direction == "LONG" and stop >= entry:
        return None
    if direction == "SHORT" and stop <= entry:
        return None
    if abs(entry - stop) / entry < 0.0005:  # %0.05'ten dar stop = gurultu
        return None
    return direction, float(entry), float(stop), name


# ------------------------------------------------------------
# A) BREAKOUT MOTORU - Sikisma Patlamasi
# ------------------------------------------------------------
@track_errors
def engine_breakout(symbol: str):
    """Bollinger bantlari daraldiginda (squeeze) aktif olur; fiyat bandi
    ortalamanin BREAKOUT_VOLUME_MULT katı hacimle kirdigi yone girer."""
    try:
        df = fetch_df(symbol, TP_M15_TIMEFRAME, BB_SQUEEZE_LOOKBACK + BB_PERIOD + 30)
    except Exception:
        return None
    if len(df) < BB_SQUEEZE_LOOKBACK + BB_PERIOD + 5:
        return None

    mid, upper, lower, width = _compute_bollinger(df)
    df["bb_mid"], df["bb_up"], df["bb_low"], df["bb_width"] = mid, upper, lower, width
    df["vol_ma"] = df["volume"].rolling(BREAKOUT_VOLUME_MA).mean()
    df["atr14"] = _compute_atr(df, ATR_PERIOD)

    # -2: son KAPANMIS mum (canli mum -1'de, ona asla guvenmiyoruz)
    row = df.iloc[-2]
    if any(pd.isna(row[c]) for c in ("bb_up", "bb_low", "bb_width", "vol_ma", "atr14")):
        return None

    # SIKISMA: kirilim mumunun ONCESINDEKI bant genisligi dar olmali.
    # Kirilim mumunun kendi genisligine bakmak yanlis olurdu - patlama
    # aninda bantlar zaten aciliyor.
    prior_width = df["bb_width"].iloc[-(BB_SQUEEZE_LOOKBACK + 2):-2]
    if prior_width.isna().all():
        return None
    threshold = np.nanpercentile(prior_width.dropna(), BB_SQUEEZE_PERCENTILE)
    was_squeezed = df["bb_width"].iloc[-3] <= threshold
    if not was_squeezed:
        return None

    if row["volume"] < row["vol_ma"] * BREAKOUT_VOLUME_MULT:
        return None

    atr = float(row["atr14"])
    # KRITIK: kirilimi, kirilim mumunun KENDI bandiyla degil, BIR ONCEKI
    # mumun bandiyla karsilastiriyoruz. Aksi halde patlama mumunun kendi
    # oynakligi bandi genisletir ve fiyat kendi bandini neredeyse hicbir
    # zaman asamaz - kosul pratikte hic tetiklenmezdi.
    prev_up = df["bb_up"].iloc[-3]
    prev_low = df["bb_low"].iloc[-3]
    if pd.isna(prev_up) or pd.isna(prev_low):
        return None

    if row["close"] > prev_up:
        # Stop: kirilim mumunun dibinin biraz altina (yanlis kirilim korumasi)
        return _engine_result("LONG", row["close"], row["low"] - atr * 0.5, "BREAKOUT")
    if row["close"] < prev_low:
        return _engine_result("SHORT", row["close"], row["high"] + atr * 0.5, "BREAKOUT")
    return None


# ------------------------------------------------------------
# B) LIKIDITE AVCISI - Fakeout / Igne Avcisi
# ------------------------------------------------------------
@track_errors
def engine_liquidity_hunt(symbol: str):
    """Yatay piyasada, kanal disina atilan igneden sonra fiyat kanal icine
    geri kapandiysa TUZAK YONUNDE (ignenin tersine) girer."""
    try:
        df = fetch_df(symbol, TP_M15_TIMEFRAME, LIQ_RANGE_LOOKBACK + 60)
    except Exception:
        return None
    if len(df) < LIQ_RANGE_LOOKBACK + 10:
        return None

    df["atr14"] = _compute_atr(df, ATR_PERIOD)
    row = df.iloc[-2]
    if pd.isna(row["atr14"]):
        return None
    atr = float(row["atr14"])

    # Kanal, IGNE MUMUNDAN ONCEKI mumlarla hesaplanir - igne mumunu dahil
    # etmek kanali kendi kendine genisletir ve tuzagi gorunmez yapardi.
    window = df.iloc[-(LIQ_RANGE_LOOKBACK + 2):-2]
    if window.empty:
        return None
    range_high = float(window["high"].max())
    range_low = float(window["low"].min())
    if range_high <= range_low:
        return None

    candle_range = float(row["high"] - row["low"])
    if candle_range <= 0:
        return None

    # Ust igne: kanal tepesinin uzerine tasip ICERIDE kapanmis -> SHORT tuzagi
    upper_wick = float(row["high"] - max(row["close"], row["open"]))
    if (row["high"] > range_high + atr * LIQ_WICK_MIN_ATR
            and row["close"] < range_high
            and upper_wick / candle_range >= LIQ_WICK_MIN_RATIO):
        return _engine_result("SHORT", row["close"], row["high"] + atr * 0.2, "LIKIDITE")

    # Alt igne: kanal dibinin altina tasip ICERIDE kapanmis -> LONG tuzagi
    lower_wick = float(min(row["close"], row["open"]) - row["low"])
    if (row["low"] < range_low - atr * LIQ_WICK_MIN_ATR
            and row["close"] > range_low
            and lower_wick / candle_range >= LIQ_WICK_MIN_RATIO):
        return _engine_result("LONG", row["close"], row["low"] - atr * 0.2, "LIKIDITE")
    return None


# ------------------------------------------------------------
# C) H4 SWING TREND MOTORU
# ------------------------------------------------------------
@track_errors
def engine_h4_swing(symbol: str):
    """Guclu trendde aktif. M15'e HIC BAKMAZ - yon H4'ten, giris zamanlamasi
    H1'den gelir. Amac M15 gurultusunden tamamen kacinmak."""
    try:
        h4 = fetch_df(symbol, TP_H4_TIMEFRAME, H4_EMA_PERIOD + 40)
    except Exception:
        return None
    if len(h4) < H4_EMA_PERIOD + 5:
        return None

    h4["ema"] = h4["close"].ewm(span=H4_EMA_PERIOD, adjust=False).mean()
    h4["supertrend"] = _compute_supertrend(h4, TP_SUPERTREND_PERIOD, TP_SUPERTREND_MULT)
    h4_row = h4.iloc[-2]
    if pd.isna(h4_row["ema"]):
        return None

    if h4_row["close"] > h4_row["ema"] and h4_row["supertrend"] == "YESIL":
        bias = "LONG"
    elif h4_row["close"] < h4_row["ema"] and h4_row["supertrend"] == "KIRMIZI":
        bias = "SHORT"
    else:
        return None

    # Giris: H1'de ana trend yonunde EMA'ya geri cekilme + teyit mumu
    try:
        h1 = fetch_df(symbol, TP_H1_TIMEFRAME, TP_EMA_SLOW + 60)
    except Exception:
        return None
    if len(h1) < TP_EMA_SLOW + 5:
        return None

    h1["ema_fast"] = h1["close"].ewm(span=TP_EMA_FAST, adjust=False).mean()
    h1["atr14"] = _compute_atr(h1, ATR_PERIOD)
    row = h1.iloc[-2]
    if pd.isna(row["ema_fast"]) or pd.isna(row["atr14"]):
        return None

    atr = float(row["atr14"])
    tol = float(row["close"]) * (H4_PULLBACK_TOLERANCE_PCT / 100)
    near_ema = abs(float(row["low"]) - float(row["ema_fast"])) <= tol or \
               abs(float(row["high"]) - float(row["ema_fast"])) <= tol

    if bias == "LONG" and near_ema and row["close"] > row["open"]:
        swing_low = float(h1.iloc[-(TP_SWING_LOOKBACK + 2):-2]["low"].min())
        return _engine_result("LONG", row["close"], min(swing_low, float(row["low"])) - atr * 0.3, "H4_SWING")
    if bias == "SHORT" and near_ema and row["close"] < row["open"]:
        swing_high = float(h1.iloc[-(TP_SWING_LOOKBACK + 2):-2]["high"].max())
        return _engine_result("SHORT", row["close"], max(swing_high, float(row["high"])) + atr * 0.3, "H4_SWING")
    return None


def active_engines():
    """Piyasa Beyni'nin rejimine gore hangi motorlarin calisacagini secer.
    BREAKOUT her rejimde aday - cunku sikisma tespiti coin BAZINDA yapilir,
    global rejimden bagimsizdir (BTC trenddeyken bir altcoin sikisiyor
    olabilir). Diger ikisi birbirinin zitti oldugu icin rejime baglidir."""
    if _current_regime == "TREND":
        return [engine_h4_swing, engine_breakout]
    if _current_regime == "YATAY":
        return [engine_liquidity_hunt, engine_breakout]
    return [engine_breakout]


# ============================================================
# POZISYON KAYDI
# ============================================================
def _read_positions():
    if not os.path.isfile(POSITIONS_FILE):
        return []
    with open(POSITIONS_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("stop_missing_count", "0")
        r.setdefault("engine", "?")
        # Eski semada (parsiyel TP'li donem) yazilmis satirlar tp_price
        # icermiyor - bunlari 1:2 R:R'ye gore tamamliyoruz ki eski bir
        # pozisyon acikken yapilan deploy sonrasi kod cokmesin.
        if not r.get("tp_price"):
            try:
                entry = float(r["entry_price"])
                stop = float(r["stop_price"])
                dist = abs(entry - stop)
                r["tp_price"] = str(entry + dist * TP_RR_RATIO if r["direction"] == "LONG"
                                    else entry - dist * TP_RR_RATIO)
            except Exception:
                r["tp_price"] = ""
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


def log_closed_trade(symbol, direction, entry_price, exit_price, pct_change, reason,
                     engine="?", entry_time=""):
    duration_min = ""
    if entry_time:
        try:
            delta = datetime.now() - datetime.fromisoformat(entry_time)
            duration_min = f"{delta.total_seconds() / 60:.1f}"
        except Exception:
            pass
    is_new = not os.path.isfile(CLOSED_TRADES_FILE)
    with open(CLOSED_TRADES_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CLOSED_TRADE_FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(), "symbol": symbol, "direction": direction,
            "engine": engine, "entry_time": entry_time, "duration_min": duration_min,
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
def open_position(symbol: str, direction: str, entry_price: float, stop_price: float, engine: str):
    _set_leverage_safe(symbol)
    side = "buy" if direction == "LONG" else "sell"

    qty = _compute_position_size(symbol, entry_price, stop_price)
    if qty <= 0:
        return None

    order = exchange.create_order(symbol, type="market", side=side, amount=qty)
    real_entry_price = order.get("average") or order.get("price")
    if not real_entry_price:
        try:
            real_entry_price = exchange.fetch_ticker(symbol)["last"]
        except Exception:
            real_entry_price = entry_price

    # DERS (2026-07-28, Gemini'nin NIHAI karari - onceki "borsa stop'unu koru"
    # kararinin YERINE gecti): borsa tarafina artik HICBIR kosullu emir
    # (STOP_MARKET dahil) GONDERILMIYOR. -4045 devre kesici faciasi (her
    # pozisyonun 3 turda bir zorla kapatilmasi) stratejiyi test edilemez hale
    # getirdigi icin, stop da TP gibi tamamen yazilimsal hale getirildi:
    # update_positions() her cycle'da fiyati stop_price ile karsilastirip
    # gerekirse market emriyle kapatiyor (asagida degismedi). BILINCLI KABUL
    # EDILEN RISK: artik ne borsa ne yazilim koruma orta katmani var - bot
    # cakilir/gecikirse/Railway yeniden baslarsa, o sure boyunca pozisyon
    # HICBIR koruma altinda degildir. Bu tradeoff Gemini'ye acikca sorulup
    # onaylanmisti (rapor 2, soru 3).
    # SABIT 1:2 R:R (2026-08-02): stop mesafesi motorun belirledigi seviyeden
    # gelir, TP her zaman bunun TP_RR_RATIO kati. Breakeven / parsiyel TP /
    # trailing YOK - pozisyon ya stop'a ya TP'ye gider.
    # ONEMLI: R mesafesini GERCEKLESEN giris fiyatina gore degil, sinyal
    # anindaki planlanan stop mesafesine gore olcuyoruz olsaydik slippage
    # R:R'yi bozardi; bu yuzden gerceklesen giristen yeniden hesapliyoruz.
    stop_distance = abs(real_entry_price - stop_price)
    tp_price = (real_entry_price + stop_distance * TP_RR_RATIO if direction == "LONG"
                else real_entry_price - stop_distance * TP_RR_RATIO)

    rows = _read_positions()
    rows.append({
        "symbol": symbol, "direction": direction,
        "entry_price": real_entry_price, "stop_price": stop_price,
        "tp_price": tp_price,
        "entry_time": datetime.now().isoformat(),
        "engine": engine, "original_qty": qty,
        "exchange_stop_order_id": "", "exchange_tp_order_id": "",
        "stop_missing_count": "0",
    })
    _write_positions(rows)

    send_telegram_message(
        f"📈 {symbol} {direction} pozisyon açıldı — motor: {engine} [Rejim: {_current_regime}].\n"
        f"Giriş: {real_entry_price:.6f} | 🛑 Stop: {stop_price:.6f} | 🎯 TP (1:{TP_RR_RATIO:g}): {tp_price:.6f}\n"
        f"🖥️ Stop VE TP tamamen yazılımsal takip ediliyor (borsaya koşullu emir gönderilmiyor)\n"
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
        engine = r.get("engine", "?")
        try:
            tp_price = float(r["tp_price"]) if r.get("tp_price") else None
        except (TypeError, ValueError):
            tp_price = None

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
            log_closed_trade(symbol, direction, entry_price, exit_price, pct_change,
                             f"Borsa tetikledi ({engine})",
                             engine=engine, entry_time=r.get("entry_time", ""))
            send_telegram_message(
                f"🔔 {symbol} {direction} pozisyon KAPANDI (borsa tetikledi, motor: {engine}).\n"
                f"Giriş: {entry_price:.6f} | Çıkış (yaklaşık): {exit_price:.6f} | Değişim: {pct_change:+.2f}%"
            )
            continue

        try:
            price = exchange.fetch_ticker(symbol)["last"]
        except Exception:
            still_open.append(r)
            continue

        # SABIT 1:2 R:R (2026-08-02): breakeven / parsiyel TP / trailing
        # mantiginin TAMAMI kaldirildi. Pozisyon ya stop'a ya TP'ye gider.
        # STOP once kontrol edilir: ayni cycle'da her iki seviye de asilmis
        # gorunuyorsa (fiyat sicramasi / bot gecikmesi) kotu senaryoyu
        # varsaymak dogru olan - aksi halde gercekte stop olmus bir islemi
        # kazanc olarak kaydedebilirdik.
        stop_hit = (price <= stop_price) if direction == "LONG" else (price >= stop_price)
        tp_hit = (tp_price is not None and
                  ((price >= tp_price) if direction == "LONG" else (price <= tp_price)))

        if stop_hit or tp_hit:
            close_err = _close_position(symbol, direction, live_qty)
            raw_pct = (price - entry_price) / entry_price * 100
            pct_change = raw_pct if direction == "LONG" else -raw_pct
            if not close_err:
                if stop_hit:
                    reason = f"Stop (yazılımsal, {engine})"
                    emoji, baslik = "🛑", "STOP"
                else:
                    reason = f"TP 1:{TP_RR_RATIO:g} (yazılımsal, {engine})"
                    emoji, baslik = "🎯", "HEDEF (TP)"
                log_closed_trade(symbol, direction, entry_price, price, pct_change, reason,
                                 engine=engine, entry_time=r.get("entry_time", ""))
                send_telegram_message(
                    f"{emoji} {symbol} {direction} pozisyon kapandı — {baslik} | motor: {engine}\n"
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
            # Motorlar sirayla denenir; ILK sinyal veren kazanir. Sira
            # active_engines() tarafindan belirlenir - rejime en uygun motor
            # basta. Ayni coinde iki motoru birden acmak, ayni riski iki kez
            # almak demek olurdu.
            result = None
            for engine_fn in active_engines():
                result = engine_fn(symbol)
                if result:
                    break
            if not result:
                continue
            trend_found += 1
            direction, entry_price, stop_price, engine_name = result
            opened_qty = open_position(symbol, direction, entry_price, stop_price, engine_name)
            if opened_qty:
                entries += 1
                open_symbols.add(symbol)
        except Exception as e:
            print(f"{symbol}: tarama hatasi ({e})")
            if "-2019" in str(e) or "insufficient" in str(e).lower():
                margin_exhausted = True
                print("Marjin tukendigi tespit edildi - bu turda baska yeni islem denenmeyecek.")

    engines = ", ".join(f.__name__ for f in active_engines())
    print(f"Tur özeti: rejim={_current_regime} | aktif motorlar={engines} | "
          f"taranan={scanned} | sinyal={trend_found} | açılan={entries}")


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
            f"  {r['symbol']} {r['direction']} [{r.get('engine', '?')}] | Giriş: {r['entry_price']} | "
            f"🛑 {r['stop_price']} | 🎯 {r.get('tp_price', '?')}"
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


def build_stop_analysis_message():
    """STOP ANALITIK RAPORU (/rapor) — yon / motor / zamanlama kirilimi.
    Kaynak closed_trades.csv'dir; positions.csv DEGIL (o sadece hala ACIK
    pozisyonlari tutar, kapanan islem oraya hic yazilmaz)."""
    if not os.path.isfile(CLOSED_TRADES_FILE):
        return "📊 Henüz kapanmış işlem kaydı yok."
    with open(CLOSED_TRADES_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "📊 Henüz kapanmış işlem kaydı yok."

    def engine_of(r):
        eng = (r.get("engine") or "").strip()
        if eng and eng != "?":
            return eng
        # Eski kayitlarda motor adi ayri sutun degil, reason metninin icinde.
        m = re.search(r"(BREAKOUT|LIKIDITE|H4_SWING)", r.get("reason", ""))
        return m.group(1) if m else "BİLİNMİYOR"

    def outcome_of(r):
        # Kapanis SEBEBINE bakiyoruz, pct isaretine degil - zaman asimi veya
        # borsa tetiklemesi de kar/zarar uretebilir, onlari stop saymak yanlis.
        reason = (r.get("reason") or "").lower()
        if "stop" in reason:
            return "STOP"
        if "tp" in reason or "hedef" in reason:
            return "TP"
        return "DİĞER"

    def pctf(a, b):
        return f"%{a / b * 100:.1f}" if b else "-"

    def net_of(rs):
        total = 0.0
        for r in rs:
            try:
                total += float(r["pct_change"])
            except (TypeError, ValueError):
                pass
        return total

    stops = [r for r in rows if outcome_of(r) == "STOP"]
    tps = [r for r in rows if outcome_of(r) == "TP"]
    decided = len(stops) + len(tps)
    lines = ["📊 [STOP ANALİTİK RAPORU]", ""]
    lines.append(f"Toplam: {len(rows)} işlem | TP: {len(tps)} | STOP: {len(stops)} | "
                 f"diğer: {len(rows) - decided}")
    lines.append(f"İsabet: {pctf(len(tps), decided)} | Net: {net_of(rows):+.2f}%")
    if decided:
        wr = len(tps) / decided
        lines.append(f"Beklenti: {wr * TP_RR_RATIO - (1 - wr):+.3f}R/işlem "
                     f"(1:{TP_RR_RATIO:g} R:R'de başabaş isabet: "
                     f"%{100 / (1 + TP_RR_RATIO):.1f})")

    def group_block(title, keyfn):
        groups = {}
        for r in rows:
            groups.setdefault(keyfn(r), []).append(r)
        out = ["", title]
        for k in sorted(groups):
            g = groups[k]
            s = sum(1 for r in g if outcome_of(r) == "STOP")
            t = sum(1 for r in g if outcome_of(r) == "TP")
            out.append(f"  {k}: {len(g)} işlem | TP {t} | STOP {s} | "
                       f"isabet {pctf(t, s + t)} | net {net_of(g):+.2f}%")
        return out, groups

    dir_lines, _ = group_block("1️⃣ YÖN", lambda r: r["direction"])
    lines += dir_lines
    ls = sum(1 for r in stops if r["direction"] == "LONG")
    lines.append(f"  → Stop olan {len(stops)} işlemin {ls}'i LONG, {len(stops) - ls}'i SHORT.")

    eng_lines, eng_groups = group_block("2️⃣ MOTOR", engine_of)
    lines += eng_lines
    if eng_groups:
        worst = min(eng_groups, key=lambda k: net_of(eng_groups[k]))
        lines.append(f"  → En çok zarar yazan: {worst} ({net_of(eng_groups[worst]):+.2f}%)")

    lines.append("")
    lines.append("3️⃣ STOP ZAMANLAMASI")
    durs = []
    for r in stops:
        d = (r.get("duration_min") or "").strip()
        if d:
            try:
                durs.append(float(d))
            except ValueError:
                pass
    if not durs:
        lines.append("  ⚠️ Süre verisi yok — eski kayıtlarda giriş zamanı")
        lines.append("  tutulmuyordu. Bundan sonraki kapanışlarda dolacak.")
    else:
        durs.sort()
        quick = sum(1 for d in durs if d <= 30)
        lines.append(f"  Ortanca: {durs[len(durs) // 2]:.0f} dk | "
                     f"En hızlı: {durs[0]:.0f} | En yavaş: {durs[-1]:.0f}")
        lines.append(f"  ≤30 dk: {quick}/{len(durs)} → iğne stopu göstergesi")
        lines.append(f"  >30 dk: {len(durs) - quick}/{len(durs)} → yön hatası göstergesi")

    if len(rows) < 30:
        lines.append("")
        lines.append(f"ℹ️ Sadece {len(rows)} işlem var — motor bazlı sonuçlar "
                     f"şans eseri çıkmış olabilir, güvenilir değil.")
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
        f"🚀 Kripto bot (3 MOTORLU MİMARİ + ZERO EXCHANGE ORDERS) başlatıldı.\n"
        f"0️⃣ Portföy Beyni | 1️⃣ Sistem Dedektifi | 2️⃣ Tamirci | 3️⃣ Piyasa Beyni (motor seçici) | 4️⃣ 3 Strateji Motoru\n"
        f"{len(WATCHLIST)} coin taranıyor. Güncel rejim: {_current_regime}\n"
        f"⚙️ Motorlar: A) Breakout (BB sıkışma + {BREAKOUT_VOLUME_MULT}× hacim) | "
        f"B) Likidite Avcısı (iğne tuzağı, YATAY'da) | C) H4 Swing Trend (TREND'de, M15 yok sayılır)\n"
        f"🎯 Çıkış: SABİT 1:{TP_RR_RATIO:g} R:R — breakeven/parsiyel TP/trailing YOK.\n"
        f"💰 Sabit risk: %{TP_RISK_PER_TRADE_PCT} (Portföy Beyni gerekirse daha da kısar)\n"
        f"🖥️ Stop VE TP tamamen yazılımsal — borsaya HİÇBİR koşullu emir gönderilmiyor (max {MAX_OPEN_POSITIONS} pozisyon).\n"
        f"Kaldıraç: {TP_LEVERAGE}x\n"
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
