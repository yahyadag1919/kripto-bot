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
# STRATEJI: VWAP Sapmasi (10. tur / uzun tutus turnuvasinda RSI+VWAP'tan
# daha fazla sinyal ve daha iyi ort. net getiri veren sistem: %76.4 isabet,
# 4796 sinyal, +%0.193 ort. net, +%926.4 toplam)
#
# Fiyat, kayan VWAP'tan (hacim agirlikli ortalama fiyat) belirgin sekilde
# sapmisken tersine (bounce) giris yapilir. RSI artik giris sarti degil,
# sadece bilgi/teyit amacli gosteriliyor.
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
# NOT: sinyaller zaten 15dk'lik KAPANMIS mumlara bakiyor, yani yeni bilgi en
# fazla 15dk'da bir olusuyor - bunu daha sik kontrol etmek yeni sinyal
# yaratmaz. Ama mum kapandigi an ile botun bunu FARK ETTIGI an arasindaki
# gecikmeyi (kotu senaryoda ~15dk'ya kadar cikabiliyordu) azaltmak icin
# tarama sikligini yukselttik - varsayilan artik 2dk, ayni kapanmis mum
# kontrol ediliyor olsa bile en gec ~2dk icinde fark edilip islem aciliyor.
CHECK_INTERVAL_MINUTES = int(os.environ.get("CHECK_INTERVAL_MINUTES", "2"))

RSI_PERIOD = 14
ATR_PERIOD = 14
VWAP_WINDOW = 96          # ~24 saat (15m mumla) - kayan VWAP penceresi

# RSI artik giris sarti degil, sadece breakdown mesajinda bilgi amacli
RSI_LONG_MAX = 30
RSI_SHORT_MIN = 70
# Eski sabit esik (artik kullanilmiyor, referans icin birakildi)
VWAP_DEV_LONG_MAX = -2.0
VWAP_DEV_SHORT_MIN = 2.0
# Yeni: coin'in kendi ATR'sine gore dinamik VWAP sapma esigi (bkz. compute_indicators)
DYNAMIC_ATR_MULT = 2.5

# Ikinci sinyal kolu: Hacim Z-Skor (10. turda 570 sinyal, %72.8 isabet, +%0.371 ort. net)
VOLUME_ZSCORE_THRESHOLD = 2.0

# --- Trend + Funding filtresi (backtest ile dogrulandi) ---
# Filtresiz VWAP: 2054 sinyal, %75.0 isabet, +%0.200 ort net
# Trend+Funding filtreli VWAP: 354 sinyal, %79.9 isabet, +%0.470 ort net (2.3x)
# Filtresiz Hacim Z-Skor CANLIDA ZARARLIYDI (-%0.071); Trend+Funding filtreli: +%0.223, %73.1 isabet
TREND_FUNDING_FILTER_ENABLED = True
TREND_TIMEFRAME = "4h"
TREND_EMA_PERIOD = 200
_trend_cache = {}  # symbol -> (ema200_deger, hesaplandigi_zaman) - her taramada yeniden cekmemek icin

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

exchange = ccxt.binanceusdm({
    "apiKey": os.environ.get("BINANCE_API_KEY"),
    "secret": os.environ.get("BINANCE_API_SECRET"),
    "enableRateLimit": True,
})
# fetch_open_orders() sembolsuz cagrilinca (hesap capinda tum emirleri
# cekmek icin, cleanup_orphaned_orders'ta kullaniliyor) ccxt bir uyari
# firlatiyor ve bu, cagriyi BASARISIZ gibi gorunduruyordu (except bloguna
# dusup fonksiyon hic calismadan cikiyordu). Bu satir uyariyi kabul edip
# gercek veriyi almasini sagliyor.
exchange.options["warnWithoutSymbol"] = False

# TESTNET=true iken sahte parayla Binance'in test ortaminda calisir - gercek
# paraya gecmeden once BUNUNLA test et. Railway'de TESTNET degiskenini "false"
# yapinca gercek hesaba baglanir.
USE_TESTNET = os.environ.get("TESTNET", "true").lower() == "true"


def _redirect_all_urls_to_demo(urls_node):
    """
    ccxt'nin urls['api'] yapisi, futures icin onlarca alt-uc-nokta (fapiPublic,
    fapiPrivate, fapiPrivateV2, fapiPrivateV3, fapiData, vs.) icerir - bunlarin
    hepsini tek tek elle yazmak yerine, iceride gecen 'fapi.binance.com' adresini
    (nerede gecerse gecsin, ic ice sozluk/liste farketmeksizin) 'demo-fapi.binance.com'
    ile degistiriyoruz. Boylece ccxt surumu/ic yapisi degisse bile calismaya devam eder.
    """
    if isinstance(urls_node, dict):
        return {k: _redirect_all_urls_to_demo(v) for k, v in urls_node.items()}
    if isinstance(urls_node, list):
        return [_redirect_all_urls_to_demo(v) for v in urls_node]
    if isinstance(urls_node, str):
        return (urls_node
                .replace("fapi.binance.com", "demo-fapi.binance.com")
                .replace("testnet.binancefuture.com", "demo-fapi.binance.com"))
    return urls_node


if USE_TESTNET:
    # NOT: ccxt, binanceusdm icin set_sandbox_mode()'u ARTIK DESTEKLEMIYOR (deprecated,
    # bkz. https://t.me/ccxt_announcements/92) - o cagriyi kullanmiyoruz. Bunun yerine
    # Binance'in yeni "Demo Trading" sistemine (demo.binance.com uzerinden olusturulan
    # key'ler) ait demo-fapi adresine TUM ic uc-noktalari (fapiPublic, fapiPrivate,
    # fapiPrivateV2/V3, fapiData vs.) kapsayacak sekilde yonlendiriyoruz.
    try:
        exchange.urls["api"] = _redirect_all_urls_to_demo(exchange.urls["api"])
    except Exception as e:
        print(f"Demo-fapi URL override uygulanamadi (ccxt surumu farkli olabilir): {e}")
    # ccxt, API key varsa piyasalari yuklerken ekstra bir "para birimi detaylari"
    # cagrisi yapip gercek (canli) spot sunucusuna (api.binance.com/sapi/...) gidebiliyor -
    # bu cagri futures islemleri icin gereksiz, demo key'le orada hata veriyordu. Kapatiyoruz.
    try:
        exchange.options["fetchCurrencies"] = False
    except Exception:
        pass

# Otomatik islem ayarlari
AUTO_TRADING_ENABLED = os.environ.get("AUTO_TRADING_ENABLED", "false").lower() == "true"
FULL_AUTO_TRADING = os.environ.get("FULL_AUTO_TRADING", "false").lower() == "true"
POSITION_PCT_OF_BALANCE = float(os.environ.get("POSITION_PCT_OF_BALANCE", "2"))  # bakiyenin yuzde kaci (marj ust siniri)
LEVERAGE = int(os.environ.get("LEVERAGE", "20"))
CONFIRM_TIMEOUT_MINUTES = 15

# --- Gemini ile birlikte degerlendirilen risk modeli fikirleri ---
# 1) R-risk modeli: her islemde riske edilecek DOLAR miktari sabittir (bakiyenin
#    RISK_PER_TRADE_PCT'i), stop mesafesi ise coin'in kendi oynakligina (ATR) gore
#    belirlenir - boylece BTC'nin %3 hareketiyle oynak bir altcoin'in %3 hareketi
#    ayni "risk birimi" sayilmaz, pozisyon buyuklugu buna gore kuculur/buyur.
RISK_PER_TRADE_PCT = float(os.environ.get("RISK_PER_TRADE_PCT", "1"))  # bakiyenin yuzde kaci riske edilecek (R)

# --- Yon tersine cevirme + basit sabit TP ---
# Backtest edilen 4 farkli yon-tahmini yaklasimi (VWAP, Hacim Z-Skor, Donchian
# trend-takip, mum momentum) walk-forward testte HICBIRI gercek pozitif edge
# gostermedi. Kullanicinin talebiyle: sinyal yonu TERSINE cevriliyor (LONG
# sinyali SHORT olarak aciliyor, SHORT sinyali LONG olarak aciliyor) - test
# edilmeden, canlida (testnet) denenip gozlemlenecek.
REVERSE_SIGNALS = os.environ.get("REVERSE_SIGNALS", "true").lower() == "true"
# TP artik stop mesafesiyle ORANTILI degil (TP_RISK_REWARD_RATIO KULLANILMIYOR) -
# kullanicinin istedigi gibi sadece komisyonun hemen ustunde, sabit kucuk bir
# kar hedefi. tp_target_pct = ROUNDTRIP_COMMISSION_PCT + SIMPLE_PROFIT_TARGET_PCT
SIMPLE_PROFIT_TARGET_PCT = float(os.environ.get("SIMPLE_PROFIT_TARGET_PCT", "0.3"))
# Stop da ayni sekilde basitlestirildi: eskiden ATR14*1.5 / gecersizlik / sabit %3
# tavanindan EN SIKI olani seciliyordu, ama bazi coinlerde bu hala cok genis
# kalabiliyordu (kucuk TP'ye kiyasla). Artik SABIT ve DAR bir yuzde - TP'ye
# yakin buyuklukte, kullanicinin "kucuk kar hedefi + dar stop" mantigina uygun.
SIMPLE_STOP_PCT = float(os.environ.get("SIMPLE_STOP_PCT", "0.6"))
# 2) Global pozisyon limiti: piyasa tek yone sert kirildiginda botun art arda
#    onlarca coin'de ayni yonde pozisyon acip kasayi tek yone kilitlemesini onler.
MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", "5"))

# --- DONCHIAN TREND-TAKIP MODU ---
# Kullanicinin talebiyle: su ana kadarki TUM testlerde en kotu (en dusuk isabet
# orani %28.1, en buyuk ort. kayip -%1.9/islem) cikan strateji - Donchian(55)
# kirilim + EMA200 trend filtresi + ATR chandelier trailing stop - AYNEN
# ORIJINAL (TERSINE CEVRILMEDEN) haliyle canliya deniyor. Amac: "testlere
# guvenmiyorum, en kotu cikani oldugu gibi canlida gozlemleyelim" talebi.
# ETKINLESTIRILINCE VWAP Sapmasi + Hacim Z-Skor sinyalleri DEVRE DISI kalir,
# bunun yerine SADECE Donchian sinyali calisir - REVERSE_SIGNALS bu moddan
# ETKILENMEZ (kullanici acikca "tersine degil, orijinal" dedi).
DONCHIAN_MODE = os.environ.get("DONCHIAN_MODE", "false").lower() == "true"
DONCHIAN_TIMEFRAME = os.environ.get("DONCHIAN_TIMEFRAME", "15m")  # eskiden 4h - 15m'e cekince cok daha sik sinyal
DONCHIAN_PERIOD = int(os.environ.get("DONCHIAN_PERIOD", "55"))
DONCHIAN_TREND_EMA_PERIOD = int(os.environ.get("DONCHIAN_TREND_EMA_PERIOD_ENV", "200"))
CHANDELIER_MULT = 3.0        # trailing stop mesafesi = ATR14(4h) * bu katsayi
DONCHIAN_REVERSAL_EXIT_CANDLES = int(os.environ.get("DONCHIAN_REVERSAL_EXIT_CANDLES", "2"))
# Pozisyon KARDAYKEN, yeni zirve/dip yapmayi birakip ust uste bu kadar mum
# TERS yonde kapanirsa, ATR stop mesafesini beklemeden HEMEN piyasa emriyle
# kapatilir - "artis durup dususe gecerken kari kilitle" mantigi. Henuz
# kara gecmemis pozisyonlarda bu kural devreye girmez, ilk ATR stop korumasi
# aynen calismaya devam eder.
DONCHIAN_STATE_FILE = os.path.join(os.environ.get("DATA_DIR", "."), "donchian_positions.csv")
DONCHIAN_FIELDNAMES = ["symbol", "direction", "signal_direction", "entry_price", "stop_price", "extreme_price", "entry_time", "against_count"]
# Sinyal tespit MANTIGI (Donchian kirilim + trend filtresi) KESINLIKLE degismiyor -
# kullanicinin acikca istedigi gibi. Sadece GERCEK ISLEM ters yonde aciliyor,
# Telegram bildirimi ise sinyalin ORIJINAL yonunu gostermeye devam ediyor
# (yani LONG sinyali gelirse bildirim hala "LONG" yazar, ama borsada SHORT acilir).
DONCHIAN_INVERT_EXECUTION = os.environ.get("DONCHIAN_INVERT_EXECUTION", "true").lower() == "true"
# Kullanicinin istegiyle: stop'a takilirsa kaybedilecek DOLAR miktari dogrudan
# bakiyenin bu yuzdesi olacak sekilde pozisyon boyutlandiriliyor (ATR mesafesi
# ne olursa olsun sonuc hep ayni % kayip). Marjin tavani da buna gore genisletildi
# (asiri dar ATR mesafesinde bile hedeflenen %10 riske ulasilabilsin diye).
DONCHIAN_RISK_PER_TRADE_PCT = float(os.environ.get("DONCHIAN_RISK_PER_TRADE_PCT", "10"))
DONCHIAN_POSITION_PCT_OF_BALANCE = float(os.environ.get("DONCHIAN_POSITION_PCT_OF_BALANCE", "20"))
# Sinyal tespit edilince HEMEN islem acmak yerine, GERCEKTE acilacak yonde
# (ters cevrilmis yonde) fiyatin birkac mum boyunca dogrulanmasini bekler.
# Amac: sinyal LONG dese bile coin gercekten yukselmeye devam ediyorsa, bizim
# SHORT'umuzun hemen zarara girmesini onlemek - once dogrulama, sonra giris.
DONCHIAN_CONFIRMATION_CANDLES = int(os.environ.get("DONCHIAN_CONFIRMATION_CANDLES", "1"))
DONCHIAN_MAX_WAIT_CANDLES = int(os.environ.get("DONCHIAN_MAX_WAIT_CANDLES", "5"))
DONCHIAN_PENDING_STATE_FILE = os.path.join(os.environ.get("DATA_DIR", "."), "donchian_pending_signals.csv")
DONCHIAN_PENDING_FIELDNAMES = ["symbol", "signal_direction", "trade_direction", "reference_price", "atr14", "confirm_count", "candles_waited"]
# Kullanicinin acikca istedigi gibi: %10 risk, o anki DEGISKEN bakiyeden degil,
# hesaba YATIRILAN SABIT tutardan hesaplaniyor. Bu tutar bir dosyada saklanip
# bot her yeniden baslatildiginda ayni kaliyor (Railway redeploy'da bile).
# Railway'de DONCHIAN_REFERENCE_BALANCE env degiskeniyle elle de ayarlanabilir/
# duzeltilebilir (orn. gercek yatirilan tutar farkliysa).
DONCHIAN_REFERENCE_BALANCE_FILE = os.path.join(os.environ.get("DATA_DIR", "."), "donchian_reference_balance.txt")

# --- SIKISMA + KIRILIM MODU (SQUEEZE_MODE) ---
# Kullanicinin bir grafikte gozlemledigi oruntu: fiyat bir sure DAR bir bantta
# sikisiyor (hareketli ortalamalar birbirine yaklasiyor, dusuk oynaklik), sonra
# GUCLU bir mumla o bandi kirip sert bir yone gidiyor. Test EDILMEDEN, dogrudan
# canlida denenmesi istendi. SQUEEZE_MODE etkinlestirilince DONCHIAN_MODE'un
# ONUNE gecer (ikisi ayni anda calismaz, karisikligi onlemek icin).
SQUEEZE_MODE = os.environ.get("SQUEEZE_MODE", "false").lower() == "true"
SQUEEZE_TIMEFRAME = os.environ.get("SQUEEZE_TIMEFRAME", "15m")
SQUEEZE_BB_PERIOD = int(os.environ.get("SQUEEZE_BB_PERIOD", "20"))
SQUEEZE_BB_STD = float(os.environ.get("SQUEEZE_BB_STD", "2.0"))
SQUEEZE_BBW_LOOKBACK = int(os.environ.get("SQUEEZE_BBW_LOOKBACK", "100"))  # sikisma esigi bu kadar mum gerive bakarak belirlenir
SQUEEZE_PERCENTILE = float(os.environ.get("SQUEEZE_PERCENTILE", "0.20"))  # BBW bu yuzdelik dilimin altindaysa "sikisma"
SQUEEZE_WINDOW = int(os.environ.get("SQUEEZE_WINDOW", "20"))  # kirilim seviyesi bu kadar mumun en yuksek/dusugu
SQUEEZE_STRONG_BODY_MULT = float(os.environ.get("SQUEEZE_STRONG_BODY_MULT", "1.5"))
SQUEEZE_BODY_AVG_WINDOW = int(os.environ.get("SQUEEZE_BODY_AVG_WINDOW", "20"))
SQUEEZE_ATR_STOP_MULT = float(os.environ.get("SQUEEZE_ATR_STOP_MULT", "2.0"))  # ilk stop mesafesi = ATR14 * bu katsayi
SQUEEZE_REVERSAL_EXIT_CANDLES = int(os.environ.get("SQUEEZE_REVERSAL_EXIT_CANDLES", "2"))
SQUEEZE_TRAIL_MULT = float(os.environ.get("SQUEEZE_TRAIL_MULT", "2.0"))  # trailing stop mesafesi de ATR14 * bu katsayi
SQUEEZE_RISK_PER_TRADE_PCT = float(os.environ.get("SQUEEZE_RISK_PER_TRADE_PCT", "10"))
SQUEEZE_POSITION_PCT_OF_BALANCE = float(os.environ.get("SQUEEZE_POSITION_PCT_OF_BALANCE", "20"))
SQUEEZE_STATE_FILE = os.path.join(os.environ.get("DATA_DIR", "."), "squeeze_positions.csv")
SQUEEZE_FIELDNAMES = ["symbol", "direction", "signal_direction", "entry_price", "stop_price", "extreme_price", "entry_time", "against_count"]
SQUEEZE_REFERENCE_BALANCE_FILE = os.path.join(os.environ.get("DATA_DIR", "."), "squeeze_reference_balance.txt")
# Kullanicinin Donchian'da istedigi ayni mantik: SINYAL TESPITI (sikisma+kirilim)
# KESINLIKLE degismiyor. Sadece GERCEK ISLEM ters yonde aciliyor, Telegram
# bildirimi sinyalin ORIJINAL yonunu gostermeye devam ediyor (coin zaten dipteyken
# SHORT acilmasi gibi durumlari onlemek icin - kullanici XRPUSDT/ENJUSDT ornekleriyle
# bunu acikca istedi).
SQUEEZE_INVERT_EXECUTION = os.environ.get("SQUEEZE_INVERT_EXECUTION", "true").lower() == "true"

# --- Native TP (Take Profit) emri ---
# Binance'e, checkpoint dongusunun beklemesine gerek kalmadan aninda tetiklenecek
# bir TP emri de birakiyoruz (en yakin checkpoint hedefinde, 1sa/%0.3). Ama ciplak
# hedefe TP koyarsak, gidis-donus komisyonu dusuldugunde net kar SIFIRIN ALTINA
# inebilir - o yuzden TP fiyatina komisyon payini da ekliyoruz, boylece TP
# tetiklenince gercekten net kardayiz, sadece brut hedefte degil.
ROUNDTRIP_COMMISSION_PCT = float(os.environ.get("ROUNDTRIP_COMMISSION_PCT", "0.1"))  # Binance USDT-M taker x2 tahmini
# TP artik SABIT bir yuzdeye (en yakin checkpoint hedefi) degil, STOP MESAFESIYLE
# ORANTILI olarak yerlestiriliyor - eski yontemde TP (~%0.4) stop mesafesinden
# (ATR bazli, genelde %1-3) COK dar kaliyordu, bu da yuksek isabet oranina ragmen
# kucuk-kucuk-kazan-buyuk-kaybet orunusuyle kademeli zarara yol aciyordu (kotu
# risk/odul orani, gereken breakeven isabet oranini %80-88'e cikariyordu).

# GUVENLIK KILIDI: tam otomasyon + gercek hesap kombinasyonu, ayri bir onay
# degiskeni olmadan ASLA calismaz - yanlislikla gercek parayla insansiz
# otomasyona gecmeyi engellemek icin. Testnet'te bu kilit devreye girmez.
if FULL_AUTO_TRADING and not USE_TESTNET:
    if os.environ.get("CONFIRM_REAL_MONEY_FULL_AUTO", "false").lower() != "true":
        print(
            "UYARI: FULL_AUTO_TRADING=true ve TESTNET=false ama "
            "CONFIRM_REAL_MONEY_FULL_AUTO=true ayarlanmamis. Guvenlik icin tam "
            "otomasyon KAPATILDI, yari-otomatik (onay butonlu) moda dusuluyor."
        )
        FULL_AUTO_TRADING = False
        AUTO_TRADING_ENABLED = True

PENDING_CONFIRMATIONS = {}   # confirm_id -> {symbol, direction, entry_price, invalidation, created_at}
_telegram_update_offset = 0

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


def send_telegram_confirm(text: str, confirm_id: str):
    """Sinyal mesajini 'Ac'/'Gec' butonlariyla gonderir - buton basilmadan hicbir islem yapilmaz."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    keyboard = {
        "inline_keyboard": [[
            {"text": "✅ Aç", "callback_data": f"confirm:{confirm_id}"},
            {"text": "❌ Geç", "callback_data": f"reject:{confirm_id}"},
        ]]
    }
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "reply_markup": keyboard}, timeout=10)
    except Exception as e:
        print(f"Telegram onay mesaji gonderim hatasi: {e}")


# ---------------------------------------------------------------------------
# Yari-otomatik islem yurutme (Binance Futures)
# ---------------------------------------------------------------------------

def _set_leverage_safe(symbol: str):
    try:
        exchange.set_leverage(LEVERAGE, symbol)
    except Exception as e:
        print(f"Kaldirac ayarlama hatasi ({symbol}): {e}")


def _compute_final_stop_price(direction: str, entry_price: float, invalidation: float, atr14: float = None) -> float:
    """
    Basitlestirildi: eskiden 3 farkli aday (strateji-bazli gecersizlik, ATR14
    bazli, sabit %3 tavan) arasindan en sikisi seciliyordu, ama bu bazi
    coinlerde hala genis kalip kucuk TP'ye kiyasla oranti bozuyordu. Artik
    SABIT ve DAR bir yuzde (SIMPLE_STOP_PCT) kullaniliyor - TP ile ayni
    "basit sabit hedef" mantiginda.
    """
    return (entry_price * (1 - SIMPLE_STOP_PCT / 100) if direction == "LONG"
            else entry_price * (1 + SIMPLE_STOP_PCT / 100))


def _compute_position_size(symbol: str, entry_price: float, stop_price: float) -> float:
    """
    Gemini'nin onerdigi "R-risk" modeli: pozisyon buyuklugu, sabit bir DOLAR riskine
    (bakiyenin RISK_PER_TRADE_PCT'i) gore, stop mesafesine bolunerek hesaplanir - boylece
    stop'a takilirsa kaybedilen miktar her zaman ayni (riske edilen tutar) kalir, coin'in
    oynakligindan (ATR'sinden) bagimsiz olarak. POSITION_PCT_OF_BALANCE*LEVERAGE ise bir
    UST SINIR (tavan) olarak kalir - asiri dar bir stop'ta pozisyonun cok buyumesini onler.
    """
    balance = exchange.fetch_balance()
    free_usdt = balance.get("USDT", {}).get("free", 0)

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0

    risk_amount = free_usdt * (RISK_PER_TRADE_PCT / 100)
    risk_based_qty = risk_amount / stop_distance

    max_notional = free_usdt * (POSITION_PCT_OF_BALANCE / 100) * LEVERAGE
    max_qty_by_margin = max_notional / entry_price

    quantity = min(risk_based_qty, max_qty_by_margin)
    return float(exchange.amount_to_precision(symbol, quantity))


def execute_order(symbol: str, direction: str, entry_price: float, invalidation: float, atr14: float = None):
    """Once stop seviyesini (gecersizlik / ATR / sabit % - hangisi en siki ise) belirler,
    pozisyon buyuklugunu bu stop mesafesine gore (R-risk modeli) hesaplar, piyasa emriyle
    pozisyonu acar ve koruyucu stop emrini birakir. Stop emri HERHANGI bir sebeple
    basarisiz olursa (orn. fiyat zaten stop seviyesini gecmisse, '-2021 Order would
    immediately trigger' hatasi), pozisyonu KORUMASIZ birakmak yerine aninda piyasa
    emriyle kapatir.

    ONEMLI: entry_price parametresi SINYAL ANINDAKI (gecmis, kapanmis mum) fiyattir -
    tarama ile emrin borsaya ulasmasi arasinda gecen surede (en kotu ihtimalle ~15dk)
    fiyat kaymis olabilir. Emir doldurulduktan sonra GERCEK dolum fiyatini alip stop/TP'yi
    ona gore yeniden hesapliyoruz - aksi halde pozisyon "hemen zararda" acilabiliyor ve
    stop, gercek fiyata gore zaten gecilmis bir seviyede kalip -2021 hatasi verebiliyordu."""
    _set_leverage_safe(symbol)
    side = "buy" if direction == "LONG" else "sell"

    # sinyal-anindaki fiyatla kabaca stop/qty hesapla - sadece emri gonderebilmek icin
    provisional_stop = _compute_final_stop_price(direction, entry_price, invalidation, atr14)
    qty = _compute_position_size(symbol, entry_price, provisional_stop)
    if qty <= 0:
        raise ValueError("Hesaplanan pozisyon miktari sifir veya negatif - bakiyeni/stop mesafesini kontrol et.")

    order = exchange.create_order(symbol, type="market", side=side, amount=qty)

    # GERCEK dolum fiyatini al - bundan sonraki tum hesaplamalar buna gore yapilacak
    real_entry_price = order.get("average") or order.get("price")
    if not real_entry_price:
        try:
            real_entry_price = exchange.fetch_ticker(symbol)["last"]
        except Exception:
            real_entry_price = entry_price  # son care

    stop_price = _compute_final_stop_price(direction, real_entry_price, invalidation, atr14)

    stop_side = "sell" if direction == "LONG" else "buy"
    try:
        exchange.create_order(
            symbol, type="STOP_MARKET", side=stop_side, amount=qty,
            params={"stopPrice": stop_price, "reduceOnly": True},
        )
    except Exception as e:
        close_err = _close_position(symbol, direction, qty)
        if not close_err:
            send_telegram_message(
                f"⚠️ {symbol}: koruyucu stop emri başarısız oldu ({e}) — pozisyon KORUMASIZ kalmasın "
                f"diye anında piyasa emriyle kapatıldı."
            )
        else:
            send_telegram_message(
                f"🚨 {symbol}: hem koruyucu stop ({e}) HEM acil kapama ({close_err}) başarısız oldu! "
                f"Pozisyonu HEMEN manuel kontrol et!"
            )

    # Basitlestirilmis TP: stop mesafesiyle ORANTI YOK artik - sadece komisyonun
    # hemen ustunde, sabit kucuk bir kar hedefi (kullanicinin orijinal manuel
    # yontemine uygun: "komisyon disinda kucuk bir kar hedefine stop koyuyordum").
    tp_target_pct = ROUNDTRIP_COMMISSION_PCT + SIMPLE_PROFIT_TARGET_PCT
    tp_price = (real_entry_price * (1 + tp_target_pct / 100) if direction == "LONG"
                else real_entry_price * (1 - tp_target_pct / 100))
    tp_side = "sell" if direction == "LONG" else "buy"
    try:
        exchange.create_order(
            symbol, type="TAKE_PROFIT_MARKET", side=tp_side, amount=qty,
            params={"stopPrice": tp_price, "reduceOnly": True},
        )
    except Exception as e:
        # TP eklenemezse kritik degil - checkpoint dongusu zaten yedek olarak
        # calisiyor, sadece aninda tetiklenme avantajini kaybederiz.
        print(f"{symbol}: native TP emri eklenemedi ({e}), checkpoint dongusu yedek olarak calisacak")

    return order, qty, stop_price, real_entry_price


def process_telegram_updates():
    """Telegram'dan gelen buton tikla (callback_query) olaylarini isler."""
    global _telegram_update_offset
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    try:
        resp = requests.get(url, params={"offset": _telegram_update_offset, "timeout": 5}, timeout=10)
        data = resp.json()
    except Exception as e:
        print(f"Telegram guncelleme cekme hatasi: {e}")
        return

    for update in data.get("result", []):
        _telegram_update_offset = update["update_id"] + 1

        message = update.get("message")
        if message:
            text = (message.get("text") or "").strip().lower()
            if text.startswith("/stats"):
                parts = text.split()
                hours = None
                if len(parts) > 1:
                    try:
                        hours = int(parts[1])
                    except ValueError:
                        pass
                send_telegram_message(build_stats_message(hours))
            continue

        cq = update.get("callback_query")
        if not cq:
            continue

        data_str = cq.get("data", "")
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
                data={"callback_query_id": cq["id"]}, timeout=10,
            )
        except Exception:
            pass

        if ":" not in data_str:
            continue
        action, confirm_id = data_str.split(":", 1)
        info = PENDING_CONFIRMATIONS.pop(confirm_id, None)
        if not info:
            send_telegram_message("Bu sinyalin suresi dolmus ya da zaten islendi.")
            continue

        if action == "confirm":
            try:
                order, qty, stop_price, real_entry_price = execute_order(
                    info["symbol"], info["direction"], info["entry_price"], info["invalidation"], info.get("atr14")
                )
                actual_risk_usd = abs(real_entry_price - stop_price) * qty
                kayma_notu = ""
                if abs(real_entry_price - info["entry_price"]) / info["entry_price"] * 100 > 0.05:
                    kayma_notu = f" (sinyal anı: {info['entry_price']:.6f}, fiyat kaymış)"
                send_telegram_message(
                    f"✅ {info['symbol']} {info['direction']} pozisyonu açıldı.\n"
                    f"Miktar: {qty} | Gerçek giriş: {real_entry_price:.6f}{kayma_notu} | Stop: {stop_price:.6f}\n"
                    f"Stop'a takılırsa risk edilen: ~{actual_risk_usd:.2f} USDT (bakiyenin ~%{RISK_PER_TRADE_PCT})"
                )
                # bu sinyal icin daha once qty=0 ile yazilmis pending kaydini guncelle -
                # GERCEK dolum fiyatiyla, sinyal anindaki eski fiyatla degil, boylece
                # checkpoint yuzdeleri gercek giristen dogru hesaplanir
                pending_rows = _read_pending()
                for pr in pending_rows:
                    if (pr["symbol"] == info["symbol"] and pr["direction"] == info["direction"]
                            and pr.get("closed", "0") == "0" and float(pr.get("qty", 0) or 0) == 0):
                        pr["qty"] = qty
                        pr["entry_price"] = real_entry_price
                        break
                _write_pending(pending_rows)
            except Exception as e:
                send_telegram_message(f"❌ {info['symbol']} emri gönderilirken hata oluştu: {e}")
        elif action == "reject":
            send_telegram_message(f"{info['symbol']} {info['direction']} sinyali geçildi.")


def expire_old_confirmations():
    now = datetime.now()
    expired_ids = [
        cid for cid, info in PENDING_CONFIRMATIONS.items()
        if now - info["created_at"] > timedelta(minutes=CONFIRM_TIMEOUT_MINUTES)
    ]
    for cid in expired_ids:
        info = PENDING_CONFIRMATIONS.pop(cid)
        send_telegram_message(f"⏱ {info['symbol']} {info['direction']} sinyali {CONFIRM_TIMEOUT_MINUTES}dk içinde onaylanmadı, iptal edildi.")


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

    df["vol_std20"] = df["volume"].rolling(20).std()
    df["vol_zscore"] = (df["volume"] - df["vol_sma20"]) / df["vol_std20"].replace(0, np.nan)

    # Gemini/test onerisi: sabit %2 yerine, coin'in kendi ATR'sine gore
    # daralip genisleyen dinamik VWAP sapma esigi (12.5-20 gunluk backtest'te
    # buyuk orneklemde - 988-1122 sinyal - tutarli pozitif cikti)
    df["dynamic_vwap_threshold_pct"] = (df["atr14"] / df["close"]) * 100 * DYNAMIC_ATR_MULT

    return df


# ---------------------------------------------------------------------------
# RSI + VWAP birlesik teyit kapisi
# ---------------------------------------------------------------------------

def check_breakout_gate(df: pd.DataFrame):
    """
    Son KAPANMIS muma bakar (df.iloc[-2]).
    LONG: fiyat VWAP'in dinamik esigin altinda (RSI artik sart degil, bilgi amacli)
    SHORT: fiyat VWAP'in dinamik esigin ustunde (RSI artik sart degil, bilgi amacli)
    Esik artik sabit %2 degil, coin'in kendi ATR'sine gore daralip genisliyor
    (test: buyuk orneklemde - 988-1122 sinyal - tutarli pozitif sonuc verdi).
    """
    if len(df) < VWAP_WINDOW + 5:
        return None

    row = df.iloc[-2]
    if pd.isna(row["vwap_dev_pct"]) or pd.isna(row["rsi"]) or pd.isna(row["atr14"]) or pd.isna(row.get("dynamic_vwap_threshold_pct")):
        return None

    threshold = row["dynamic_vwap_threshold_pct"]
    if threshold <= 0:
        return None

    if row["vwap_dev_pct"] <= -threshold:
        return "LONG", row

    if row["vwap_dev_pct"] >= threshold:
        return "SHORT", row

    return None


def check_volume_zscore_gate(df: pd.DataFrame):
    """
    Son KAPANMIS muma bakar (df.iloc[-2]).
    Hacim, son 20 mumun ortalamasindan z-skor bazinda asiri sapmissa (klimaks hacim),
    mumun yonune ters bounce sinyali uretir:
    LONG: klimaks hacimli dusus mumu (satis tukenmesi)
    SHORT: klimaks hacimli yukselis mumu (alim tukenmesi)
    """
    if len(df) < 25:
        return None

    row = df.iloc[-2]
    if pd.isna(row.get("vol_zscore")) or pd.isna(row["atr14"]):
        return None

    if row["vol_zscore"] < VOLUME_ZSCORE_THRESHOLD:
        return None

    if row["close"] < row["open"]:
        return "LONG", row
    elif row["close"] > row["open"]:
        return "SHORT", row

    return None


# ---------------------------------------------------------------------------
# DONCHIAN TREND-TAKIP MODU - ayri, kendi kendine yeten bir alt-sistem.
# VWAP/Hacim Z-Skor'un checkpoint tabanli cikis mantigindan TAMAMEN bagimsiz -
# kendi CSV dosyasinda pozisyon takibi yapiyor, kendi trailing-stop dongusu var.
# ---------------------------------------------------------------------------

def fetch_donchian_ohlcv_df(symbol: str, limit: int = 300) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=DONCHIAN_TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def compute_donchian_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["donchian_high"] = df["high"].shift(1).rolling(DONCHIAN_PERIOD).max()
    df["donchian_low"] = df["low"].shift(1).rolling(DONCHIAN_PERIOD).min()
    df["ema_trend"] = df["close"].ewm(span=DONCHIAN_TREND_EMA_PERIOD, adjust=False).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(ATR_PERIOD).mean()
    return df


def check_donchian_gate(df: pd.DataFrame):
    """ORIJINAL yon - REVERSE_SIGNALS'tan etkilenmez, kullanicinin istegiyle
    ayni sekilde (tersine cevirmeden) test edilmesi icin."""
    row = df.iloc[-2]
    if pd.isna(row.get("donchian_high")) or pd.isna(row.get("ema_trend")) or pd.isna(row.get("atr14")):
        return None
    if row["close"] > row["donchian_high"] and row["close"] > row["ema_trend"]:
        return "LONG", row
    elif row["close"] < row["donchian_low"] and row["close"] < row["ema_trend"]:
        return "SHORT", row
    return None


def _read_donchian_positions():
    if not os.path.isfile(DONCHIAN_STATE_FILE):
        return []
    with open(DONCHIAN_STATE_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("against_count", "0")
    return rows


def _write_donchian_positions(rows):
    with open(DONCHIAN_STATE_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DONCHIAN_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def get_donchian_reference_balance() -> float:
    """Hesaba YATIRILAN SABIT tutari dondurur - o anki degisken bakiye DEGIL.
    Once Railway env degiskenini kontrol eder (elle ayarlanmis/duzeltilmis olabilir),
    yoksa diskteki kayitli degeri okur, o da yoksa ILK KEZ o anki bakiyeyi baz alip
    diske kaydeder (bundan sonra hep bu sabit deger kullanilir, redeploy'da bile degismez)."""
    env_override = os.environ.get("DONCHIAN_REFERENCE_BALANCE")
    if env_override:
        return float(env_override)

    if os.path.isfile(DONCHIAN_REFERENCE_BALANCE_FILE):
        with open(DONCHIAN_REFERENCE_BALANCE_FILE) as f:
            return float(f.read().strip())

    balance = exchange.fetch_balance()
    free_usdt = balance.get("USDT", {}).get("free", 0)
    with open(DONCHIAN_REFERENCE_BALANCE_FILE, "w") as f:
        f.write(str(free_usdt))
    return free_usdt


def _compute_donchian_position_size(symbol: str, entry_price: float, stop_price: float) -> float:
    """_compute_position_size ile ayni mantik (sabit DOLAR riski / stop mesafesi),
    ama Donchian moduna ozel: DONCHIAN_RISK_PER_TRADE_PCT (varsayilan %10) ve
    DONCHIAN_POSITION_PCT_OF_BALANCE (varsayilan %20) kullanir - boylece stop'a
    takilirsa kaybedilen TAM OLARAK bakiyenin %10'u olur (ATR mesafesinden bagimsiz).
    Risk tutari, o anki DEGISKEN bakiyeden degil, YATIRILAN SABIT tutardan hesaplanir -
    ama marjin tavani (asiri pozisyon acilmasin diye guvenlik siniri) hala o anki GERCEK
    kullanilabilir bakiyeyi de goz onunde bulunduruyor (yetersiz marjin hatasi almamak icin)."""
    reference_balance = get_donchian_reference_balance()

    balance = exchange.fetch_balance()
    free_usdt = balance.get("USDT", {}).get("free", 0)

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0

    risk_amount = reference_balance * (DONCHIAN_RISK_PER_TRADE_PCT / 100)
    risk_based_qty = risk_amount / stop_distance

    # marjin tavani: hem sabit referansa hem o anki gercek bakiyeye gore - hangisi
    # daha kucukse o kullanilir, boylece bakiye referansin altina dusmusse bile
    # gercekte olmayan bir marjini kullanmaya calisip hata almayiz
    safe_balance_for_margin = min(reference_balance, free_usdt) if free_usdt > 0 else reference_balance
    max_notional = safe_balance_for_margin * (DONCHIAN_POSITION_PCT_OF_BALANCE / 100) * LEVERAGE
    max_qty_by_margin = max_notional / entry_price

    return min(risk_based_qty, max_qty_by_margin)


def open_donchian_position(symbol: str, signal_direction: str, entry_price: float, atr14: float):
    """SINYAL TESPITI (Donchian kirilim + trend filtresi) hic degismiyor - signal_direction
    parametresi dogrudan check_donchian_gate'in urettigi ORIJINAL yondur, DOKUNULMUYOR.
    Sadece GERCEK ISLEM, DONCHIAN_INVERT_EXECUTION acikken TERS yonde aciliyor - Telegram
    bildirimi ise HER ZAMAN sinyalin orijinal yonunu gosterir (kullanicinin acik istegi:
    'bildirim aynı gelsin, ama tersine işlem açılsın')."""
    trade_direction = (("SHORT" if signal_direction == "LONG" else "LONG")
                        if DONCHIAN_INVERT_EXECUTION else signal_direction)

    _set_leverage_safe(symbol)
    side = "buy" if trade_direction == "LONG" else "sell"

    provisional_stop = (entry_price - atr14 * CHANDELIER_MULT if trade_direction == "LONG"
                         else entry_price + atr14 * CHANDELIER_MULT)
    qty = _compute_donchian_position_size(symbol, entry_price, provisional_stop)
    if qty <= 0:
        raise ValueError("Hesaplanan pozisyon miktari sifir veya negatif.")

    order = exchange.create_order(symbol, type="market", side=side, amount=qty)
    real_entry_price = order.get("average") or order.get("price")
    if not real_entry_price:
        try:
            real_entry_price = exchange.fetch_ticker(symbol)["last"]
        except Exception:
            real_entry_price = entry_price

    stop_price = (real_entry_price - atr14 * CHANDELIER_MULT if trade_direction == "LONG"
                  else real_entry_price + atr14 * CHANDELIER_MULT)
    stop_side = "sell" if trade_direction == "LONG" else "buy"
    try:
        exchange.create_order(
            symbol, type="STOP_MARKET", side=stop_side, amount=qty,
            params={"stopPrice": stop_price, "reduceOnly": True},
        )
    except Exception as e:
        close_err = _close_position(symbol, trade_direction, qty)
        send_telegram_message(
            f"⚠️ {symbol} (Donchian): koruyucu stop emri başarısız oldu ({e}) — "
            f"pozisyon {'kapatıldı' if not close_err else 'KAPATILAMADI, MANUEL KONTROL ET: ' + close_err}."
        )
        return None

    rows = _read_donchian_positions()
    rows.append({
        "symbol": symbol, "direction": trade_direction, "signal_direction": signal_direction,
        "entry_price": real_entry_price, "stop_price": stop_price, "extreme_price": real_entry_price,
        "entry_time": datetime.now().isoformat(), "against_count": "0",
    })
    _write_donchian_positions(rows)

    send_telegram_message(
        f"🐢 [Donchian Trend-Takip - ORİJİNAL yön] {symbol} {signal_direction} pozisyon açıldı.\n"
        f"Giriş: {real_entry_price:.6f} | İlk stop: {stop_price:.6f} (ATR×{CHANDELIER_MULT})\n"
        f"Bu strateji şu ana kadarki testlerde EN KÖTÜ çıkan sonuçtu (en düşük isabet, "
        f"en büyük kayıp) — kullanıcı isteğiyle tersine çevrilmeden, olduğu gibi deneniyor.\n"
        f"TP YOK — sadece trailing stop kazananın büyümesine izin veriyor."
    )
    return qty


def update_donchian_trailing_stops():
    """Her tarama dongusunde: acik Donchian pozisyonlarinin hala borsada acik olup
    olmadigini kontrol eder (stop'a takilip sessizce kapanmis olabilir), acik
    olanlarin trailing stop'unu (sadece kar yonunde) gunceller."""
    rows = _read_donchian_positions()
    if not rows:
        return
    still_open = []

    for r in rows:
        symbol = r["symbol"]
        direction = r["direction"]  # GERCEK islem yonu - pct hesap ve pozisyon sorgusu icin
        signal_direction = r.get("signal_direction", direction)  # bildirimde gosterilecek, sinyalin orijinal yonu
        entry_price = float(r["entry_price"])
        current_stop = float(r["stop_price"])
        extreme = float(r["extreme_price"])

        try:
            positions = exchange.fetch_positions([symbol])
            live_qty = sum(abs(float(p.get("contracts") or 0)) for p in positions if p.get("symbol") == symbol)
        except Exception as e:
            print(f"{symbol} (Donchian): pozisyon sorgulanamadi ({e}), bir sonraki tura birakildi")
            still_open.append(r)
            continue

        if live_qty <= 0:
            try:
                current_price = exchange.fetch_ticker(symbol)["last"]
            except Exception:
                current_price = current_stop
            raw_pct = (current_price - entry_price) / entry_price * 100
            pct_change = raw_pct if direction == "LONG" else -raw_pct
            sonuc = "TP (trailing, sessiz)" if pct_change > 0 else "SL (trailing, sessiz)"
            log_closed_trade(symbol, "Donchian Trend-Takip (orijinal)", signal_direction, entry_price, current_price, pct_change, sonuc)
            send_telegram_message(
                f"🐢 [Donchian] {symbol} {signal_direction} pozisyon kapanmış (trailing stop'a takılmış). "
                f"Giriş: {entry_price:.6f} | Şimdi: {current_price:.6f} | Değişim: {pct_change:+.2f}%"
            )
            continue

        # GUVENLIK AGI: pozisyon acik ama borsada AKTIF bir stop emri yoksa
        # (onceki bir hata, manuel mudahale, vs.) HEMEN yeniden koy - pozisyon
        # asla korumasiz kalmasin.
        try:
            open_orders = exchange.fetch_open_orders(symbol)
            has_stop_order = any(o.get("type", "").upper() in ("STOP_MARKET", "STOP") for o in open_orders)
            if not has_stop_order:
                stop_side = "sell" if direction == "LONG" else "buy"
                exchange.create_order(
                    symbol, type="STOP_MARKET", side=stop_side, amount=live_qty,
                    params={"stopPrice": current_stop, "reduceOnly": True},
                )
                send_telegram_message(
                    f"🛡️ [Donchian] {symbol}: AKTİF STOP EMRİ BULUNAMADI, güvenlik ağı devreye girdi — "
                    f"stop {current_stop:.6f} seviyesinde yeniden koyuldu."
                )
        except Exception as e:
            print(f"{symbol} (Donchian): stop emri kontrolu/yeniden koyma basarisiz ({e})")

        try:
            df = fetch_donchian_ohlcv_df(symbol, limit=DONCHIAN_TREND_EMA_PERIOD + 30)
            df = compute_donchian_indicators(df)
            last_row = df.iloc[-2]
            latest_close = last_row["close"]
            latest_atr = last_row["atr14"]
        except Exception as e:
            print(f"{symbol} (Donchian): 4h veri cekilemedi, trailing stop guncellenmedi ({e})")
            still_open.append(r)
            continue

        if pd.isna(latest_atr):
            still_open.append(r)
            continue

        if direction == "LONG":
            new_extreme = max(extreme, latest_close)
            candidate_stop = new_extreme - latest_atr * CHANDELIER_MULT
            new_stop = max(current_stop, candidate_stop)
            in_profit = latest_close > entry_price
        else:
            new_extreme = min(extreme, latest_close)
            candidate_stop = new_extreme + latest_atr * CHANDELIER_MULT
            new_stop = min(current_stop, candidate_stop)
            in_profit = latest_close < entry_price

        # KAR KİLİTLEME: yeni zirve/dip yapmayi biraktiysa (extreme guncellenmedi)
        # VE pozisyon kardaysa, ust uste kac mumdur TERS gittigini say. Esige
        # ulasinca ATR stop mesafesini beklemeden HEMEN kapat (kullanicinin
        # istegi: "artis durup dususe geciyorken kari al, gevis vermeden cik").
        against_count = int(r.get("against_count", "0"))
        if new_extreme == extreme and in_profit:
            against_count += 1
        else:
            against_count = 0
        r["against_count"] = str(against_count)

        if in_profit and against_count >= DONCHIAN_REVERSAL_EXIT_CANDLES:
            close_err = _close_position(symbol, direction, live_qty)
            current_price = latest_close
            raw_pct = (current_price - entry_price) / entry_price * 100
            pct_change = raw_pct if direction == "LONG" else -raw_pct
            if not close_err:
                log_closed_trade(symbol, "Donchian Trend-Takip (orijinal)", signal_direction, entry_price, current_price, pct_change, "TP (momentum döndü)")
                send_telegram_message(
                    f"✅ [Donchian] {symbol} {signal_direction} pozisyon KAR KİLİTLENDİ (momentum döndü, "
                    f"{against_count} mum üst üste geri gitti). Giriş: {entry_price:.6f} | Çıkış: {current_price:.6f} | "
                    f"Değişim: {pct_change:+.2f}%"
                )
            else:
                print(f"{symbol} (Donchian): kar kilitleme kapatma basarisiz ({close_err}), pozisyon acik birakildi")
                still_open.append(r)
            continue

        if new_stop != current_stop:
            try:
                stop_side = "sell" if direction == "LONG" else "buy"
                # ONCE mevcut emirlerin ID'lerini not al (fiyat karsilastirmasi
                # YAPMIYORUZ artik - float hassasiyeti yuzunden eslesme
                # basarisiz olup emirlerin birikmesine, sonunda Binance'in
                # "max stop order limit" hatasina yol acmisti).
                try:
                    old_order_ids = [o["id"] for o in exchange.fetch_open_orders(symbol)]
                except Exception:
                    old_order_ids = []

                exchange.create_order(
                    symbol, type="STOP_MARKET", side=stop_side, amount=live_qty,
                    params={"stopPrice": new_stop, "reduceOnly": True},
                )
                # Yeni emir basariyla koyuldu - simdi ONCEDEN NOT ALINAN ESKI
                # emirleri (ID'ye gore, fiyata gore DEGIL) iptal et.
                for old_id in old_order_ids:
                    try:
                        exchange.cancel_order(old_id, symbol)
                    except Exception as cancel_err:
                        print(f"{symbol} (Donchian): eski emir {old_id} iptal edilemedi ({cancel_err})")
                r["stop_price"] = str(new_stop)
                r["extreme_price"] = str(new_extreme)
                print(f"{symbol} (Donchian): trailing stop güncellendi {current_stop:.6f} -> {new_stop:.6f}")
            except Exception as e:
                print(f"{symbol} (Donchian): trailing stop güncellenemedi ({e}), ESKİ STOP HALA AKTİF (korumasız kalmadı)")

        still_open.append(r)

    _write_donchian_positions(still_open)


def _read_donchian_pending():
    if not os.path.isfile(DONCHIAN_PENDING_STATE_FILE):
        return []
    with open(DONCHIAN_PENDING_STATE_FILE, newline="") as f:
        return list(csv.DictReader(f))


def _write_donchian_pending(rows):
    with open(DONCHIAN_PENDING_STATE_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DONCHIAN_PENDING_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def process_donchian_pending_signals():
    """Bekleyen (henuz acilmamis) sinyalleri kontrol eder: GERCEKTE acilacak yonde
    (trade_direction) yeterli sayida mum kapanmissa (DONCHIAN_CONFIRMATION_CANDLES)
    pozisyonu simdi acar. Cok uzun beklerse (DONCHIAN_MAX_WAIT_CANDLES) sinyali iptal eder."""
    pending = _read_donchian_pending()
    if not pending:
        return
    still_pending = []

    for r in pending:
        symbol = r["symbol"]
        signal_direction = r["signal_direction"]
        trade_direction = r["trade_direction"]
        confirm_count = int(r["confirm_count"])
        candles_waited = int(r["candles_waited"])
        atr14 = float(r["atr14"])

        try:
            df = fetch_donchian_ohlcv_df(symbol, limit=30)
        except Exception as e:
            print(f"{symbol} (Donchian bekleyen sinyal): veri cekilemedi ({e}), bir sonraki tura birakildi")
            still_pending.append(r)
            continue

        last_closed = df.iloc[-2]
        candle_favors_trade = ((last_closed["close"] > last_closed["open"]) if trade_direction == "LONG"
                                else (last_closed["close"] < last_closed["open"]))

        candles_waited += 1
        if candle_favors_trade:
            confirm_count += 1
        else:
            confirm_count = 0  # ters mum gelirse dogrulama sifirlanir, baştan saymaya baslar

        if confirm_count >= DONCHIAN_CONFIRMATION_CANDLES:
            try:
                open_donchian_position(symbol, signal_direction, last_closed["close"], atr14)
            except Exception as e:
                print(f"{symbol} (Donchian): dogrulanmis sinyal acilamadi ({e})")
            continue  # acildi ya da hata verdi - her iki durumda da bekleme listesinden cikar

        if candles_waited >= DONCHIAN_MAX_WAIT_CANDLES:
            print(f"{symbol} (Donchian): {DONCHIAN_MAX_WAIT_CANDLES} mum icinde dogrulanamadi, sinyal iptal edildi")
            continue  # listeden dusur, iptal

        r["confirm_count"] = str(confirm_count)
        r["candles_waited"] = str(candles_waited)
        still_pending.append(r)

    _write_donchian_pending(still_pending)


def scan_donchian_once():
    """VWAP/Hacim Z-Skor'dan bagimsiz, ayri bir tarama: Donchian kirilim sinyali
    arar. Bulursa HEMEN acmaz - GERCEKTE acilacak (ters cevrilmis) yonde fiyat
    hareketinin dogrulanmasini beklemek uzere bekleme listesine ekler."""
    open_symbols = {r["symbol"] for r in _read_donchian_positions()}
    pending_symbols = {r["symbol"] for r in _read_donchian_pending()}
    if len(open_symbols) >= MAX_OPEN_POSITIONS:
        return

    pending_rows = _read_donchian_pending()
    added = False

    for symbol in WATCHLIST:
        if symbol in _unsupported_symbols or symbol in open_symbols or symbol in pending_symbols:
            continue
        try:
            df = fetch_donchian_ohlcv_df(symbol, limit=DONCHIAN_TREND_EMA_PERIOD + 30)
            df = compute_donchian_indicators(df)
            result = check_donchian_gate(df)
            if not result:
                continue
            signal_direction, row = result
            trade_direction = (("SHORT" if signal_direction == "LONG" else "LONG")
                                if DONCHIAN_INVERT_EXECUTION else signal_direction)

            pending_rows.append({
                "symbol": symbol, "signal_direction": signal_direction, "trade_direction": trade_direction,
                "reference_price": str(row["close"]), "atr14": str(row["atr14"]),
                "confirm_count": "0", "candles_waited": "0",
            })
            added = True
            send_telegram_message(
                f"🔎 [Donchian] {symbol} {signal_direction} sinyali tespit edildi — "
                f"gerçek yönde ({DONCHIAN_CONFIRMATION_CANDLES} mum) doğrulama bekleniyor, "
                f"henüz pozisyon açılmadı."
            )
            if len(open_symbols) + len(pending_rows) >= MAX_OPEN_POSITIONS:
                break
        except Exception as e:
            if "does not have" in str(e).lower():
                _unsupported_symbols.add(symbol)
            else:
                print(f"{symbol} (Donchian) hata: {e}")

    if added:
        _write_donchian_pending(pending_rows)


# ---------------------------------------------------------------------------
# SIKISMA + KIRILIM MODU - Donchian'a benzer sekilde ayri, kendi kendine
# yeten bir alt-sistem. Test edilmeden dogrudan canlida denenmesi istendi.
# ---------------------------------------------------------------------------

def get_squeeze_reference_balance() -> float:
    env_override = os.environ.get("SQUEEZE_REFERENCE_BALANCE")
    if env_override:
        return float(env_override)
    if os.path.isfile(SQUEEZE_REFERENCE_BALANCE_FILE):
        with open(SQUEEZE_REFERENCE_BALANCE_FILE) as f:
            return float(f.read().strip())
    balance = exchange.fetch_balance()
    free_usdt = balance.get("USDT", {}).get("free", 0)
    with open(SQUEEZE_REFERENCE_BALANCE_FILE, "w") as f:
        f.write(str(free_usdt))
    return free_usdt


def fetch_squeeze_ohlcv_df(symbol: str, limit: int = 200) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=SQUEEZE_TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def compute_squeeze_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mid = df["close"].rolling(SQUEEZE_BB_PERIOD).mean()
    std = df["close"].rolling(SQUEEZE_BB_PERIOD).std()
    upper = mid + SQUEEZE_BB_STD * std
    lower = mid - SQUEEZE_BB_STD * std
    df["bbw"] = (upper - lower) / mid
    df["squeeze_threshold"] = df["bbw"].rolling(SQUEEZE_BBW_LOOKBACK).quantile(SQUEEZE_PERCENTILE)
    df["in_squeeze"] = df["bbw"] <= df["squeeze_threshold"]
    df["squeeze_high"] = df["high"].shift(1).rolling(SQUEEZE_WINDOW).max()
    df["squeeze_low"] = df["low"].shift(1).rolling(SQUEEZE_WINDOW).min()
    df["body"] = (df["close"] - df["open"]).abs()
    df["avg_body20"] = df["body"].rolling(SQUEEZE_BODY_AVG_WINDOW).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(ATR_PERIOD).mean()
    return df


def check_squeeze_gate(df: pd.DataFrame):
    """Bir onceki mum SIKISMADAYDI, simdiki mum GUCLU govdeyle sikisma araligini
    kiriyorsa -> kirilim yonunde sinyal (dogrudan bu yonde islem acilir, TERS
    CEVRILMEZ - kullanicinin gozlemledigi orneğe uygun)."""
    if len(df) < max(SQUEEZE_BBW_LOOKBACK, SQUEEZE_WINDOW, SQUEEZE_BODY_AVG_WINDOW) + 3:
        return None
    prev = df.iloc[-3]
    row = df.iloc[-2]
    if pd.isna(prev.get("in_squeeze")) or pd.isna(row.get("squeeze_high")) or pd.isna(row.get("avg_body20")) or pd.isna(row.get("atr14")) or row["avg_body20"] == 0:
        return None
    if not prev["in_squeeze"]:
        return None
    strong_body = row["body"] >= row["avg_body20"] * SQUEEZE_STRONG_BODY_MULT
    if not strong_body:
        return None
    if row["close"] > row["squeeze_high"]:
        return "LONG", row
    elif row["close"] < row["squeeze_low"]:
        return "SHORT", row
    return None


def _read_squeeze_positions():
    if not os.path.isfile(SQUEEZE_STATE_FILE):
        return []
    with open(SQUEEZE_STATE_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault("against_count", "0")
    return rows


def _write_squeeze_positions(rows):
    with open(SQUEEZE_STATE_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SQUEEZE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _compute_squeeze_position_size(entry_price: float, stop_price: float) -> float:
    """Donchian'daki bakiye-bazli mantigin ayni - stop'a takilirsa kaybedilen
    TAM OLARAK referans bakiyenin SQUEEZE_RISK_PER_TRADE_PCT'i olur."""
    reference_balance = get_squeeze_reference_balance()
    balance = exchange.fetch_balance()
    free_usdt = balance.get("USDT", {}).get("free", 0)

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0

    risk_amount = reference_balance * (SQUEEZE_RISK_PER_TRADE_PCT / 100)
    risk_based_qty = risk_amount / stop_distance

    max_notional = min(reference_balance, free_usdt) * (SQUEEZE_POSITION_PCT_OF_BALANCE / 100) * LEVERAGE
    max_qty_by_margin = max_notional / entry_price

    return min(risk_based_qty, max_qty_by_margin)


def open_squeeze_position(symbol: str, signal_direction: str, entry_price: float, atr14: float):
    """SINYAL TESPITI (sikisma+kirilim) hic degismiyor - signal_direction parametresi
    check_squeeze_gate'in urettigi ORIJINAL yondur, DOKUNULMUYOR. Sadece GERCEK ISLEM,
    SQUEEZE_INVERT_EXECUTION acikken TERS yonde aciliyor - Telegram bildirimi HER ZAMAN
    sinyalin orijinal yonunu gosterir (Donchian modundakiyle AYNI mantik)."""
    trade_direction = (("SHORT" if signal_direction == "LONG" else "LONG")
                        if SQUEEZE_INVERT_EXECUTION else signal_direction)

    _set_leverage_safe(symbol)
    side = "buy" if trade_direction == "LONG" else "sell"

    provisional_stop = (entry_price - atr14 * SQUEEZE_ATR_STOP_MULT if trade_direction == "LONG"
                         else entry_price + atr14 * SQUEEZE_ATR_STOP_MULT)
    qty = _compute_squeeze_position_size(entry_price, provisional_stop)
    if qty <= 0:
        raise ValueError("Hesaplanan pozisyon miktari sifir veya negatif.")

    order = exchange.create_order(symbol, type="market", side=side, amount=qty)
    real_entry_price = order.get("average") or order.get("price")
    if not real_entry_price:
        try:
            real_entry_price = exchange.fetch_ticker(symbol)["last"]
        except Exception:
            real_entry_price = entry_price

    stop_price = (real_entry_price - atr14 * SQUEEZE_ATR_STOP_MULT if trade_direction == "LONG"
                  else real_entry_price + atr14 * SQUEEZE_ATR_STOP_MULT)
    stop_side = "sell" if trade_direction == "LONG" else "buy"
    try:
        exchange.create_order(
            symbol, type="STOP_MARKET", side=stop_side, amount=qty,
            params={"stopPrice": stop_price, "reduceOnly": True},
        )
    except Exception as e:
        close_err = _close_position(symbol, trade_direction, qty)
        send_telegram_message(
            f"⚠️ {symbol} (Sıkışma+Kırılım): koruyucu stop emri başarısız oldu ({e}) — "
            f"pozisyon {'kapatıldı' if not close_err else 'KAPATILAMADI, MANUEL KONTROL ET: ' + close_err}."
        )
        return None

    rows = _read_squeeze_positions()
    rows.append({
        "symbol": symbol, "direction": trade_direction, "signal_direction": signal_direction,
        "entry_price": real_entry_price, "stop_price": stop_price, "extreme_price": real_entry_price,
        "entry_time": datetime.now().isoformat(), "against_count": "0",
    })
    _write_squeeze_positions(rows)

    send_telegram_message(
        f"🗜️ [Sıkışma+Kırılım] {symbol} {signal_direction} pozisyon açıldı (sıkışma sonrası güçlü kırılım).\n"
        f"Giriş: {real_entry_price:.6f} | İlk stop: {stop_price:.6f} (ATR×{SQUEEZE_ATR_STOP_MULT})\n"
        f"TP YOK — trailing stop kazananın büyümesine izin veriyor."
    )
    return qty


def update_squeeze_trailing_stops():
    rows = _read_squeeze_positions()
    if not rows:
        return
    still_open = []

    for r in rows:
        symbol = r["symbol"]
        direction = r["direction"]  # GERCEK islem yonu
        signal_direction = r.get("signal_direction", direction)  # bildirimde gosterilecek
        entry_price = float(r["entry_price"])
        current_stop = float(r["stop_price"])
        extreme = float(r["extreme_price"])

        try:
            positions = exchange.fetch_positions([symbol])
            live_qty = sum(abs(float(p.get("contracts") or 0)) for p in positions if p.get("symbol") == symbol)
        except Exception as e:
            print(f"{symbol} (Sıkışma): pozisyon sorgulanamadi ({e})")
            still_open.append(r)
            continue

        if live_qty <= 0:
            try:
                current_price = exchange.fetch_ticker(symbol)["last"]
            except Exception:
                current_price = current_stop
            raw_pct = (current_price - entry_price) / entry_price * 100
            pct_change = raw_pct if direction == "LONG" else -raw_pct
            sonuc = "TP (trailing, sessiz)" if pct_change > 0 else "SL (trailing, sessiz)"
            log_closed_trade(symbol, "Sıkışma+Kırılım", signal_direction, entry_price, current_price, pct_change, sonuc)
            send_telegram_message(
                f"🗜️ [Sıkışma+Kırılım] {symbol} {signal_direction} pozisyon kapanmış (trailing stop'a takılmış). "
                f"Giriş: {entry_price:.6f} | Şimdi: {current_price:.6f} | Değişim: {pct_change:+.2f}%"
            )
            continue

        # GUVENLIK AGI: pozisyon acik ama borsada AKTIF bir stop emri yoksa,
        # HEMEN yeniden koy - pozisyon asla korumasiz kalmasin.
        try:
            open_orders = exchange.fetch_open_orders(symbol)
            has_stop_order = any(o.get("type", "").upper() in ("STOP_MARKET", "STOP") for o in open_orders)
            if not has_stop_order:
                stop_side = "sell" if direction == "LONG" else "buy"
                exchange.create_order(
                    symbol, type="STOP_MARKET", side=stop_side, amount=live_qty,
                    params={"stopPrice": current_stop, "reduceOnly": True},
                )
                send_telegram_message(
                    f"🛡️ [Sıkışma] {symbol}: AKTİF STOP EMRİ BULUNAMADI, güvenlik ağı devreye girdi — "
                    f"stop {current_stop:.6f} seviyesinde yeniden koyuldu."
                )
        except Exception as e:
            print(f"{symbol} (Sıkışma): stop emri kontrolu/yeniden koyma basarisiz ({e})")

        try:
            df = fetch_squeeze_ohlcv_df(symbol, limit=SQUEEZE_BBW_LOOKBACK + 30)
            df = compute_squeeze_indicators(df)
            last_row = df.iloc[-2]
            latest_close = last_row["close"]
            latest_atr = last_row["atr14"]
        except Exception as e:
            print(f"{symbol} (Sıkışma): veri cekilemedi ({e})")
            still_open.append(r)
            continue

        if pd.isna(latest_atr):
            still_open.append(r)
            continue

        if direction == "LONG":
            new_extreme = max(extreme, latest_close)
            candidate_stop = new_extreme - latest_atr * SQUEEZE_TRAIL_MULT
            new_stop = max(current_stop, candidate_stop)
            in_profit = latest_close > entry_price
        else:
            new_extreme = min(extreme, latest_close)
            candidate_stop = new_extreme + latest_atr * SQUEEZE_TRAIL_MULT
            new_stop = min(current_stop, candidate_stop)
            in_profit = latest_close < entry_price

        against_count = int(r.get("against_count", "0"))
        if new_extreme == extreme and in_profit:
            against_count += 1
        else:
            against_count = 0
        r["against_count"] = str(against_count)

        if in_profit and against_count >= SQUEEZE_REVERSAL_EXIT_CANDLES:
            close_err = _close_position(symbol, direction, live_qty)
            current_price = latest_close
            raw_pct = (current_price - entry_price) / entry_price * 100
            pct_change = raw_pct if direction == "LONG" else -raw_pct
            if not close_err:
                log_closed_trade(symbol, "Sıkışma+Kırılım", signal_direction, entry_price, current_price, pct_change, "TP (momentum döndü)")
                send_telegram_message(
                    f"✅ [Sıkışma] {symbol} {signal_direction} pozisyon KAR KİLİTLENDİ (momentum döndü, "
                    f"{against_count} mum üst üste geri gitti). Giriş: {entry_price:.6f} | Çıkış: {current_price:.6f} | "
                    f"Değişim: {pct_change:+.2f}%"
                )
            else:
                print(f"{symbol} (Sıkışma): kar kilitleme kapatma basarisiz ({close_err}), pozisyon acik birakildi")
                still_open.append(r)
            continue

        if new_stop != current_stop:
            try:
                stop_side = "sell" if direction == "LONG" else "buy"
                try:
                    old_order_ids = [o["id"] for o in exchange.fetch_open_orders(symbol)]
                except Exception:
                    old_order_ids = []

                exchange.create_order(
                    symbol, type="STOP_MARKET", side=stop_side, amount=live_qty,
                    params={"stopPrice": new_stop, "reduceOnly": True},
                )
                for old_id in old_order_ids:
                    try:
                        exchange.cancel_order(old_id, symbol)
                    except Exception as cancel_err:
                        print(f"{symbol} (Sıkışma): eski emir {old_id} iptal edilemedi ({cancel_err})")
                r["stop_price"] = str(new_stop)
                r["extreme_price"] = str(new_extreme)
            except Exception as e:
                print(f"{symbol} (Sıkışma): trailing stop güncellenemedi ({e}), ESKİ STOP HALA AKTİF (korumasız kalmadı)")

        still_open.append(r)

    _write_squeeze_positions(still_open)


def scan_squeeze_once():
    open_symbols = {r["symbol"] for r in _read_squeeze_positions()}
    if len(open_symbols) >= MAX_OPEN_POSITIONS:
        return

    for symbol in WATCHLIST:
        if symbol in _unsupported_symbols or symbol in open_symbols:
            continue
        try:
            df = fetch_squeeze_ohlcv_df(symbol, limit=SQUEEZE_BBW_LOOKBACK + 30)
            df = compute_squeeze_indicators(df)
            result = check_squeeze_gate(df)
            if not result:
                continue
            direction, row = result
            open_squeeze_position(symbol, direction, row["close"], row["atr14"])
            if len(_read_squeeze_positions()) >= MAX_OPEN_POSITIONS:
                break
        except Exception as e:
            if "does not have" in str(e).lower():
                _unsupported_symbols.add(symbol)
            else:
                print(f"{symbol} (Sıkışma) hata: {e}")



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


# ---------------------------------------------------------------------------
# Trend + Funding filtresi (backtest ile dogrulandi - bkz. yukaridaki not)
# 4sa/8sa'da bir degisen degerler oldugu icin saatte bir yenilenen basit bir
# onbellek kullaniyoruz - 172 coin icin her taramada tekrar cekmek gereksiz
# API yuku olustururdu.
# ---------------------------------------------------------------------------

_TREND_FUNDING_CACHE_MINUTES = 60


def _cache_get_or_fetch(cache_dict, symbol, fetch_fn):
    cached = cache_dict.get(symbol)
    if cached and (datetime.now() - cached[1]).total_seconds() < _TREND_FUNDING_CACHE_MINUTES * 60:
        return cached[0]
    try:
        value = fetch_fn()
        cache_dict[symbol] = (value, datetime.now())
        return value
    except Exception as e:
        print(f"{symbol}: trend/funding verisi cekilemedi: {e}")
        return cached[0] if cached else None


def get_trend_ema(symbol: str):
    def _fetch():
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TREND_TIMEFRAME, limit=TREND_EMA_PERIOD + 20)
        closes = pd.Series([c[4] for c in ohlcv])
        return closes.ewm(span=TREND_EMA_PERIOD, adjust=False).mean().iloc[-1]
    return _cache_get_or_fetch(_trend_cache, symbol, _fetch)


_funding_cache = {}


def get_funding_rate(symbol: str):
    def _fetch():
        fr = exchange.fetch_funding_rate(symbol)
        return fr.get("fundingRate")
    return _cache_get_or_fetch(_funding_cache, symbol, _fetch)


def passes_trend_funding_filter(symbol: str, direction: str, current_price: float) -> tuple:
    """Ikisi de gecmeli: 4sa 200 EMA trend yonu + funding rate isareti (backtest'te dogrulanan kombinasyon)."""
    if not TREND_FUNDING_FILTER_ENABLED:
        return True, "filtre kapali"

    ema = get_trend_ema(symbol)
    funding = get_funding_rate(symbol)
    if ema is None or funding is None:
        return False, "trend/funding verisi alinamadi, guvenli tarafta kalindi"

    trend_ok = (current_price > ema) if direction == "LONG" else (current_price < ema)
    funding_ok = (funding < 0) if direction == "LONG" else (funding > 0)

    if trend_ok and funding_ok:
        return True, f"trend+funding uyumlu (4sa EMA200={ema:.4f}, funding={funding:.5f})"
    return False, f"trend/funding uyumsuz (4sa EMA200={ema:.4f}, funding={funding:.5f})"


def compute_invalidation(direction: str, row) -> float:
    atr = row["atr14"] if pd.notna(row["atr14"]) else 0
    buffer = atr * INVALIDATION_ATR_BUFFER
    if direction == "LONG":
        return row["close"] - buffer
    return row["close"] + buffer


# ---------------------------------------------------------------------------
# Loglama
# ---------------------------------------------------------------------------

# DATA_DIR bir Railway Volume'e (kalici disk) isaret ederse, bu CSV'ler her
# deploy'da SIFIRLANMAZ - acik pozisyon takibi/checkpoint durumu korunur.
# DATA_DIR ayarlanmazsa eskisi gibi calisir (gecici, deploy'da sifirlanir).
DATA_DIR = os.environ.get("DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)

SIGNAL_LOG_FILE = os.path.join(DATA_DIR, "signal_history.csv")
PENDING_FILE = os.path.join(DATA_DIR, "pending_signals.csv")
OUTCOME_FILE = os.path.join(DATA_DIR, "signal_outcomes.csv")
# Tum kapanan pozisyonlarin TEK bir yerde toplandigi log - sessiz stop/TP
# (native emirle borsada kapananlar) DAHIL, kapanis sebebi ne olursa olsun.
# /stats komutu buradan okuyor. OUTCOME_FILE'dan farkli: OUTCOME_FILE sadece
# checkpoint dongusunun kendi tetikledigi kapanislari yaziyordu, native
# STOP_MARKET/TAKE_PROFIT_MARKET emriyle sessizce kapanan (gercek otomatik
# islemlerin cogunlugu bu sekilde kapaniyor) pozisyonlar hic loglanmiyordu.
CLOSED_LOG_FILE = os.path.join(DATA_DIR, "closed_trades.csv")


def log_closed_trade(symbol, strategy, direction, entry_price, exit_price, pct_change, sonuc):
    file_exists = os.path.isfile(CLOSED_LOG_FILE)
    with open(CLOSED_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "symbol", "strategy", "direction", "entry_price",
                "exit_price", "pct_change", "sonuc"
            ])
        writer.writerow([
            datetime.now().isoformat(), symbol, strategy, direction, entry_price,
            exit_price if exit_price is not None else "",
            f"{pct_change:.3f}" if pct_change is not None else "",
            sonuc,
        ])


def build_stats_message(hours=None):
    if not os.path.isfile(CLOSED_LOG_FILE):
        return "Henüz kapanmış işlem kaydı yok."
    with open(CLOSED_LOG_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    if hours:
        cutoff = datetime.now() - timedelta(hours=hours)
        rows = [r for r in rows if datetime.fromisoformat(r["timestamp"]) >= cutoff]

    baslik = f"📊 İstatistik (son {hours}sa)" if hours else "📊 İstatistik (tüm zamanlar)"
    if not rows:
        return f"{baslik}\nBu aralıkta kapanmış işlem yok."

    total = len(rows)
    tp_rows = [r for r in rows if r["sonuc"].startswith("TP")]
    sl_rows = [r for r in rows if not r["sonuc"].startswith("TP")]
    pct_values = [float(r["pct_change"]) for r in rows if r.get("pct_change")]
    total_pct = sum(pct_values)
    avg_pct = total_pct / len(pct_values) if pct_values else 0.0
    win_rate = len(tp_rows) / total * 100

    open_count = len([r for r in _read_pending() if r.get("closed", "0") != "1"])

    strategy_breakdown = {}
    for r in rows:
        s = r.get("strategy", "?")
        d = strategy_breakdown.setdefault(s, {"tp": 0, "sl": 0, "pct": 0.0})
        if r["sonuc"].startswith("TP"):
            d["tp"] += 1
        else:
            d["sl"] += 1
        if r.get("pct_change"):
            d["pct"] += float(r["pct_change"])

    lines = [
        baslik,
        f"Toplam kapanan işlem: {total}",
        f"TP: {len(tp_rows)} | SL/süre-doldu: {len(sl_rows)} | İsabet: %{win_rate:.1f}",
        f"Toplam net: %{total_pct:+.2f} | Ort./işlem: %{avg_pct:+.3f}",
        f"Şu an açık pozisyon: {open_count}",
        "",
        "Strateji bazında:",
    ]
    for s, d in strategy_breakdown.items():
        t = d["tp"] + d["sl"]
        wr = (d["tp"] / t * 100) if t else 0
        lines.append(f"- {s}: {t} işlem, %{wr:.0f} isabet, toplam %{d['pct']:+.2f}")

    return "\n".join(lines)


def log_signal(symbol: str, strategy: str, direction: str, row, breakdown: list):
    file_exists = os.path.isfile(SIGNAL_LOG_FILE)
    with open(SIGNAL_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "strategy", "direction", "price", "rsi", "breakdown"])
        writer.writerow([
            datetime.now().isoformat(), symbol, strategy, direction, row["close"], row["rsi"], " | ".join(breakdown)
        ])


PENDING_FIELDNAMES = ["symbol", "strategy", "direction", "entry_price", "entry_time", "invalidation", "qty"] + [
    f"checked_{label}" for _, _, label in CHECKPOINTS
] + ["closed"]


def log_pending(symbol: str, strategy: str, direction: str, entry_price: float, entry_time: datetime,
                 invalidation: float, qty: float = 0):
    file_exists = os.path.isfile(PENDING_FILE)
    with open(PENDING_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(PENDING_FIELDNAMES)
        row = [symbol, strategy, direction, entry_price, entry_time.isoformat(), invalidation, qty]
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


def log_outcome(symbol, strategy, direction, entry_price, entry_time, minutes, label, target_pct,
                 current_price, pct_change, success):
    file_exists = os.path.isfile(OUTCOME_FILE)
    with open(OUTCOME_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "symbol", "strategy", "direction", "entry_price", "entry_time", "minutes_after", "checkpoint",
                "target_pct", "price_now", "pct_change", "success"
            ])
        writer.writerow([
            symbol, strategy, direction, entry_price, entry_time, minutes, label,
            target_pct, current_price, f"{pct_change:.3f}", success
        ])


def _close_position(symbol: str, direction: str, qty: float) -> str:
    """
    Pozisyonu kapatmadan once borsadan GERCEK pozisyon miktarini/yonunu sorar
    (kendi kaydina korukoru guvenmek yerine) - boylece "ReduceOnly Order is
    rejected" (-2022) hatasi (kayitli miktar borsadaki gercek miktarla
    uyusmuyorsa olur) onlenir. Pozisyon zaten kapanmissa (miktar 0), bosuna
    kapatma emri denemez, sadece kalan acik emirleri (orn. eski stop) temizler.
    Basarili olursa bos string, basarisiz olursa hata metnini dondurur.
    """
    live_qty = qty
    live_direction = direction
    try:
        positions = exchange.fetch_positions([symbol])
        found = 0
        for p in positions:
            contracts = abs(p.get("contracts") or 0)
            if contracts > 0:
                found = contracts
                live_direction = "LONG" if p.get("side") == "long" else "SHORT"
                break
        live_qty = found
    except Exception as e:
        print(f"{symbol}: gercek pozisyon miktari sorgulanamadi, kayitli miktara guveniliyor: {e}")

    if live_qty <= 0:
        # pozisyon zaten kapanmis (baska bir yoldan) - kapatma emrine gerek yok
        try:
            exchange.cancel_all_orders(symbol)
        except Exception:
            pass
        return ""

    close_side = "sell" if live_direction == "LONG" else "buy"
    try:
        exchange.create_order(symbol, type="market", side=close_side, amount=live_qty, params={"reduceOnly": True})
    except Exception as e:
        return str(e)

    try:
        exchange.cancel_all_orders(symbol)
    except Exception as e:
        print(f"{symbol}: kapanistan sonra kalan emirler iptal edilemedi: {e}")

    return ""


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
        strategy = r.get("strategy", "?")
        direction = r["direction"]
        qty = float(r.get("qty", 0) or 0)
        closed = False

        # Gercek pozisyon acikken (qty>0), borsada hala var mi diye sor - stop-loss
        # sessizce tetiklenip pozisyonu kapatmis olabilir, bot bunu fark etmeden
        # checkpoint'leri beklemeye devam ederse kullanici hicbir bildirim gormez.
        if qty > 0:
            try:
                positions = exchange.fetch_positions([symbol])
                still_open = any(abs(p.get("contracts") or 0) > 0 for p in positions)
            except Exception as e:
                print(f"{symbol}: canli pozisyon kontrolu basarisiz, checkpoint kontrolune devam ediliyor: {e}")
                still_open = True  # emin olamadigimizda checkpoint akisina birak, yanlislikla "durduruldu" demeyelim

            if not still_open:
                try:
                    current_price = exchange.fetch_ticker(symbol)["last"]
                    raw_pct = (current_price - entry_price) / entry_price * 100
                    pct_change = raw_pct if direction == "LONG" else -raw_pct
                except Exception:
                    current_price = None
                    pct_change = None

                try:
                    exchange.cancel_all_orders(symbol)
                except Exception:
                    pass

                detay = f"Şimdi: {current_price:.4f} | Değişim: {pct_change:+.2f}%" if current_price is not None else "(fiyat bilgisi alınamadı)"
                # pct_change pozitifse muhtemelen TP tetiklenmis (kar), negatifse stop (zarar) -
                # kesin degil (baska bir yerden de kapanmis olabilir) ama en olasi aciklama bu.
                if pct_change is not None and pct_change > 0:
                    baslik = f"🎯 [{strategy}] {symbol} {direction} - pozisyon muhtemelen TP'ye takılıp KARLA kapanmış."
                    sonuc = "TP (sessiz/native)"
                else:
                    baslik = f"🛑 [{strategy}] {symbol} {direction} - pozisyon muhtemelen stop-loss'a takılıp ZARARLA kapanmış."
                    sonuc = "SL (sessiz/native)"
                send_telegram_message(f"{baslik}\nGiriş: {entry_price:.4f} | {detay}")
                log_closed_trade(symbol, strategy, direction, entry_price, current_price, pct_change, sonuc)
                r["closed"] = "1"
                continue  # bu satir icin checkpoint dongusune hic girme, zaten kapanmis

        for minutes, target_pct, label in CHECKPOINTS:
            flag_key = f"checked_{label}"
            if r.get(flag_key, "0") == "1":
                continue

            time_elapsed = now >= entry_time + timedelta(minutes=minutes)

            try:
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker["last"]
                raw_pct_change = (current_price - entry_price) / entry_price * 100
                pct_change = raw_pct_change if direction == "LONG" else -raw_pct_change
                success = pct_change >= target_pct
            except Exception as e:
                print(f"{symbol} sonuc kontrolu hatasi: {e}")
                break

            if success:
                # Hedefe ulasildi - checkpoint'in zamani gelmemis olsa bile HEMEN kar al,
                # zaman siniri gelene kadar beklemek kari geri verme riski tasir.
                log_outcome(symbol, strategy, direction, entry_price, r["entry_time"], minutes, label,
                            target_pct, current_price, pct_change, success)
                r[flag_key] = "1"
                close_err = _close_position(symbol, direction, qty)
                erken_not = "" if time_elapsed else " (hedef sureden ONCE tutuldu, erken kar alindi)"
                msg = (
                    f"🎯 [{strategy}] {symbol} {direction} - {label} hedefte tutturuldu{erken_not}\n"
                    f"Giriş: {entry_price:.4f} | Şimdi: {current_price:.4f}\n"
                    f"Değişim: {pct_change:+.2f}% (hedef: %{target_pct})\n\n"
                    + (f"✅ Pozisyon otomatik kapatıldı."
                       if qty > 0 and not close_err
                       else (f"⚠️ Pozisyon kapatma emri başarısız: {close_err}\nManuel kapatmayi unutma!"
                             if close_err else "Öneri: kârı realize etmeyi değerlendir."))
                )
                send_telegram_message(msg)
                log_closed_trade(symbol, strategy, direction, entry_price, current_price, pct_change, "TP (checkpoint)")
                r["closed"] = "1"
                closed = True
                break

            if not time_elapsed:
                # hedef henuz tutmadi VE bu checkpoint'in suresi de dolmadi - bir sonraki
                # taramada tekrar denenecek, simdilik bekle
                break

            # hedef tutmadi AMA suresi doldu - bu checkpoint'i "denendi" olarak isaretle,
            # bir sonraki (daha gevsek) checkpoint'e gec
            log_outcome(symbol, strategy, direction, entry_price, r["entry_time"], minutes, label,
                        target_pct, current_price, pct_change, success)
            r[flag_key] = "1"

            if label == CHECKPOINTS[-1][2]:
                close_err = _close_position(symbol, direction, qty)
                msg = (
                    f"⏱ [{strategy}] {symbol} {direction} - 24sa sonunda hiçbir checkpoint'te hedef tutmadı\n"
                    f"Giriş: {entry_price:.4f} | Şimdi: {current_price:.4f}\n"
                    f"Son değişim: {pct_change:+.2f}%\n\n"
                    + (f"Sinyal geçersiz sayıldı, pozisyon otomatik kapatıldı."
                       if qty > 0 and not close_err
                       else (f"⚠️ Pozisyon kapatma emri başarısız: {close_err}\nManuel kapatmayi unutma!"
                             if close_err else "Sinyal geçersiz sayılıyor.")))
                send_telegram_message(msg)
                log_closed_trade(symbol, strategy, direction, entry_price, current_price, pct_change, "SL (24sa süre doldu)")
                r["closed"] = "1"
                closed = True
                break
            # son checkpoint degilse dongu devam eder, bir sonraki (daha gevsek) hedefi kontrol eder

        if not closed:
            still_pending.append(r)

    _write_pending(still_pending)


# ---------------------------------------------------------------------------
# Ana tarama
# ---------------------------------------------------------------------------

def _emit_signal(symbol: str, strategy: str, strategy_desc: str, direction: str, row, breakdown: list):
    log_signal(symbol, strategy, direction, row, breakdown)

    invalidation = compute_invalidation(direction, row)
    entry_price = row["close"]
    atr14 = row["atr14"] if pd.notna(row.get("atr14")) else None
    yon_emoji = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
    breakdown_text = "\n".join(f"- {b}" for b in breakdown)

    checkpoint_text = " / ".join(f"{label}(%{target})" for _, target, label in CHECKPOINTS)
    msg = (
        f"{yon_emoji} {symbol} - {strategy} sinyali (bounce)\n"
        f"({strategy_desc})\n\n"
        f"Giriş fiyatı: {entry_price:.4f}\n"
        f"Geçersizlik seviyesi: {invalidation:.4f}\n\n"
        f"⏱ Checkpoint hedefleri: {checkpoint_text}\n"
        f"İlk tutan hedefte pozisyon kapanmış sayılır, en geç 24sa'da değerlendirme gelir.\n\n"
        f"Teyit detayları:\n{breakdown_text}"
    )
    print(msg)

    executed_qty = 0
    logged_entry_price = entry_price  # gercek doldurma olmazsa (sinyal-amacli/basarisiz) sinyal fiyatina duser
    if FULL_AUTO_TRADING:
        try:
            order, executed_qty, stop_price, real_entry_price = execute_order(symbol, direction, entry_price, invalidation, atr14)
            logged_entry_price = real_entry_price
            actual_risk_usd = abs(real_entry_price - stop_price) * executed_qty
            kayma_notu = ""
            if abs(real_entry_price - entry_price) / entry_price * 100 > 0.05:
                kayma_notu = f" (sinyal anı: {entry_price:.4f}, fiyat kaymış)"
            msg += (
                f"\n\n🤖 TAM OTOMATİK: pozisyon açıldı.\n"
                f"Miktar: {executed_qty} | Gerçek giriş: {real_entry_price:.6f}{kayma_notu} | Stop: {stop_price:.6f}\n"
                f"Stop'a takılırsa risk edilen: ~{actual_risk_usd:.2f} USDT (bakiyenin ~%{RISK_PER_TRADE_PCT})"
            )
        except Exception as e:
            msg += f"\n\n❌ TAM OTOMATİK emir başarısız oldu: {e}"
        send_telegram_message(msg)
    elif AUTO_TRADING_ENABLED:
        confirm_id = f"{symbol.replace('/', '').replace(':', '')}-{int(time.time())}"
        PENDING_CONFIRMATIONS[confirm_id] = {
            "symbol": symbol, "direction": direction, "entry_price": entry_price,
            "invalidation": invalidation, "atr14": atr14, "created_at": datetime.now(),
        }
        msg += f"\n\n⚠️ {CONFIRM_TIMEOUT_MINUTES}dk içinde onaylamazsan otomatik iptal olur."
        send_telegram_confirm(msg, confirm_id)
    else:
        send_telegram_message(msg)

    log_pending(symbol, strategy, direction, logged_entry_price, datetime.now(), invalidation, qty=executed_qty)


def cleanup_orphaned_orders():
    """Her tarama turunde calisir: hesapta acik olan TUM emirleri ceker, ve
    su an Donchian/Sikisma tarafindan takip edilen bir pozisyonu OLMAYAN
    herhangi bir sembolde emir varsa iptal eder. cleanup_duplicate_stop_orders
    sadece baslangicta ve sadece halen takip edilen semboller icin calisiyordu -
    o yuzden gecmisten (eski VWAP/Hacim sistemi, kapanmis pozisyonlar, manuel
    islemler, vs.) kalma basibos emirler hic temizlenmeden birikip Binance'in
    hesap capindaki 'max stop order limit' hatasina (-4045) yol aciyordu, bu da
    YENI pozisyon acilirken stop konulamamasina ve pozisyonun hemen zorla
    kapatilmasina sebep oluyordu."""
    tracked_symbols = set()
    for r in _read_donchian_positions():
        tracked_symbols.add(r["symbol"])
    for r in _read_squeeze_positions():
        tracked_symbols.add(r["symbol"])

    try:
        all_open_orders = exchange.fetch_open_orders()
    except Exception as e:
        print(f"Basibos emir taramasi basarisiz (tum emirler cekilemedi): {e}")
        return

    orphan_symbols = {}
    for o in all_open_orders:
        sym = o.get("symbol")
        if sym and sym not in tracked_symbols:
            orphan_symbols.setdefault(sym, []).append(o)

    for sym, orders in orphan_symbols.items():
        for o in orders:
            try:
                exchange.cancel_order(o["id"], sym)
            except Exception as e:
                print(f"{sym}: basibos emir {o['id']} iptal edilemedi ({e})")
        print(f"{sym}: takip edilmeyen {len(orders)} basibos emir temizlendi")


def scan_once():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Tarama basliyor...")

    if DONCHIAN_MODE or SQUEEZE_MODE:
        try:
            cleanup_orphaned_orders()
        except Exception as e:
            print(f"Basibos emir temizligi basarisiz ({e})")

    if SQUEEZE_MODE:
        # Sikisma+Kirilim modu digerlerinin (VWAP/Hacim/Donchian) ONUNE gecer -
        # test edilmeden dogrudan canlida denenmesi istendi.
        update_squeeze_trailing_stops()
        scan_squeeze_once()
        return

    if DONCHIAN_MODE:
        # VWAP/Hacim Z-Skor sinyalleri bu modda DEVRE DISI - kullanicinin
        # istegiyle sadece Donchian trend-takip (orijinal yon) calisiyor.
        update_donchian_trailing_stops()
        process_donchian_pending_signals()
        scan_donchian_once()
        return

    check_pending_outcomes()

    # Ayni coin'de zaten acik/bekleyen bir pozisyon varsa tekrar sinyal
    # uretip ustune emir yigmamak icin - Open Orders'in sismesinin asil sebebi buydu.
    already_open_symbols = {r["symbol"] for r in _read_pending() if r.get("closed", "0") != "1"}

    # Gemini - global risk kilidi: piyasa tek yone sert kirildiginda botun art arda
    # onlarca coin'de ayni yonde pozisyon acip kasayi tek yone kilitlemesini onler.
    if FULL_AUTO_TRADING and len(already_open_symbols) >= MAX_OPEN_POSITIONS:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] MAX_OPEN_POSITIONS limiti "
            f"({MAX_OPEN_POSITIONS}) dolu ({len(already_open_symbols)} acik pozisyon) - "
            f"bu tur yeni sinyal aranmadi, mevcut pozisyonlar takip edilmeye devam ediyor."
        )
        return

    closest_long = None   # (vwap_dev_pct, symbol) - en negatif (LONG esigine en yakin)
    closest_short = None  # (vwap_dev_pct, symbol) - en pozitif (SHORT esigine en yakin)
    scanned = 0

    for symbol in WATCHLIST:
        if symbol in _unsupported_symbols:
            continue
        if symbol in already_open_symbols:
            print(f"{symbol}: zaten acik/bekleyen pozisyon var, tekrar sinyal uretilmedi")
            continue
        try:
            df = fetch_ohlcv_df(symbol, TIMEFRAME, limit=100)
            df = compute_indicators(df)
            row = df.iloc[-2]

            if pd.notna(row.get("vwap_dev_pct")):
                dev = row["vwap_dev_pct"]
                scanned += 1
                if closest_long is None or dev < closest_long[0]:
                    closest_long = (dev, symbol)
                if closest_short is None or dev > closest_short[0]:
                    closest_short = (dev, symbol)

            fired = False

            vwap_result = check_breakout_gate(df)
            if vwap_result:
                direction, vrow = vwap_result
                if REVERSE_SIGNALS:
                    direction = "SHORT" if direction == "LONG" else "LONG"
                filter_ok, filter_note = passes_trend_funding_filter(symbol, direction, vrow["close"])
                if not filter_ok:
                    print(f"{symbol}: VWAP sinyali tespit edildi ama trend/funding filtresine takildi ({filter_note})")
                else:
                    breakdown = [
                        f"✅ VWAP sapması (dinamik eşik): %{vrow['vwap_dev_pct']:+.2f} "
                        f"(eşik: ±%{vrow['dynamic_vwap_threshold_pct']:.2f})"
                        + (" [YÖN TERSİNE ÇEVRİLDİ]" if REVERSE_SIGNALS else ""),
                        f"✅ Trend+Funding: {filter_note}",
                        f"ℹ️ RSI: {vrow['rsi']:.1f} (bilgi amaçlı, şart değil)",
                        f"✅ Hacim {vrow['volume']/vrow['vol_sma20']:.2f}x ortalama" if pd.notna(vrow.get('vol_sma20')) and vrow.get('vol_sma20') else "➖ Hacim verisi yetersiz",
                    ]
                    ob_support, ob_note = score_orderbook(symbol, direction)
                    breakdown.append(f"{'✅' if ob_support else '➖'} Order book: {ob_note}")
                    _emit_signal(
                        symbol, "VWAP Sapması (dinamik+filtreli)",
                        "Genisletilmis backtest: Trend+Funding filtreli VWAP, 354 sinyal, %79.9 isabet, +%0.470 ort. net (filtresize gore 2.3x)",
                        direction, vrow, breakdown,
                    )
                    fired = True

            zscore_result = check_volume_zscore_gate(df)
            if zscore_result:
                direction, zrow = zscore_result
                if REVERSE_SIGNALS:
                    direction = "SHORT" if direction == "LONG" else "LONG"
                filter_ok, filter_note = passes_trend_funding_filter(symbol, direction, zrow["close"])
                if not filter_ok:
                    print(f"{symbol}: Hacim Z-Skor sinyali tespit edildi ama trend/funding filtresine takildi ({filter_note})")
                else:
                    breakdown = [
                        f"✅ Hacim Z-Skor: {zrow['vol_zscore']:.2f} (giriş şartı, eşik: {VOLUME_ZSCORE_THRESHOLD})"
                        + (" [YÖN TERSİNE ÇEVRİLDİ]" if REVERSE_SIGNALS else ""),
                        f"✅ Trend+Funding: {filter_note}",
                        f"ℹ️ Mum yönü: {'düşüş (klimaks satış)' if direction == 'SHORT' else 'yükseliş (klimaks alım)'}",
                        f"ℹ️ RSI: {zrow['rsi']:.1f} (bilgi amaçlı, şart değil)",
                    ]
                    ob_support, ob_note = score_orderbook(symbol, direction)
                    breakdown.append(f"{'✅' if ob_support else '➖'} Order book: {ob_note}")
                    _emit_signal(
                        symbol, "Hacim Z-Skor (filtreli)",
                        "Genisletilmis backtest: Trend+Funding filtreli Hacim Z-Skor, 412 sinyal, %73.1 isabet, +%0.223 ort. net (filtresiz haliyle canlida zararliydi)",
                        direction, zrow, breakdown,
                    )
                    fired = True

            if not fired:
                print(f"{symbol}: kriter yok")

        except Exception as e:
            if "does not have" in str(e).lower():
                _unsupported_symbols.add(symbol)
                print(f"{symbol}: bu borsada islem gormuyor, listeden cikarildi")
            else:
                print(f"{symbol} hata: {e}")

    if closest_long and closest_short:
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Tarama bitti - {scanned} coin. "
            f"Esige en yakin -> LONG: {closest_long[1]} (%{closest_long[0]:+.2f}) | "
            f"SHORT: {closest_short[1]} (%{closest_short[0]:+.2f}) "
            f"(esik artik dinamik/coin bazli, sabit degil)"
        )


def cleanup_duplicate_stop_orders():
    """Onceki bir hatadan (float fiyat esleme basarisiz oldugu icin eski stop
    emirleri iptal edilemeyip birikmisti) kalma FAZLADAN emirleri temizler.
    Her acik Donchian/Sikisma pozisyonu icin: birden fazla emir varsa,
    EN SON konulani birakip digerlerini iptal eder."""
    open_symbols = set()
    for r in _read_donchian_positions():
        open_symbols.add(r["symbol"])
    for r in _read_squeeze_positions():
        open_symbols.add(r["symbol"])

    for symbol in open_symbols:
        try:
            open_orders = exchange.fetch_open_orders(symbol)
            if len(open_orders) <= 1:
                continue
            # id'ler genelde artan sirada olusuyor - en buyugu (en yeni) haric hepsini iptal et
            open_orders_sorted = sorted(open_orders, key=lambda o: o.get("timestamp") or 0)
            to_cancel = open_orders_sorted[:-1]
            for o in to_cancel:
                try:
                    exchange.cancel_order(o["id"], symbol)
                except Exception as e:
                    print(f"{symbol}: fazladan emir {o['id']} iptal edilemedi ({e})")
            if to_cancel:
                print(f"{symbol}: {len(to_cancel)} fazladan/eski emir temizlendi")
        except Exception as e:
            print(f"{symbol}: emir temizligi basarisiz ({e})")


def run_forever():
    if DONCHIAN_MODE or SQUEEZE_MODE:
        # Onceki bir hatadan kalma birikmis fazladan stop emirlerini temizle -
        # bu emirler Binance'in "max stop order limit" hatasina sebep olup
        # pozisyonlarin zorla kapanmasina yol acmisti.
        try:
            cleanup_duplicate_stop_orders()
        except Exception as e:
            print(f"Baslangic emir temizligi basarisiz ({e})")

    checkpoint_text = " / ".join(f"{label}(%{target})" for _, target, label in CHECKPOINTS)
    if FULL_AUTO_TRADING:
        mode_text = (
            f"🤖 TAM OTOMATİK MOD AÇIK — sinyaller ONAY BEKLEMEDEN Binance Futures'ta gerçek emir açar "
            f"(bakiyenin %{POSITION_PCT_OF_BALANCE} | {LEVERAGE}x kaldıraç | sabit %{SIMPLE_STOP_PCT} stop'u). "
            f"{'⚠️ TESTNET (sahte para)' if USE_TESTNET else '🔴 GERÇEK HESAP - GERÇEK PARA'}"
        )
    elif AUTO_TRADING_ENABLED:
        mode_text = (
            f"⚡ YARI-OTOMATİK MOD AÇIK — sinyaller Telegram'dan onay bekleyecek, onaylarsan "
            f"Binance Futures'ta gerçek emir açılır (bakiyenin %{POSITION_PCT_OF_BALANCE} | {LEVERAGE}x kaldıraç | "
            f"sabit %{SIMPLE_STOP_PCT} stop'u). "
            f"{'⚠️ TESTNET (sahte para)' if USE_TESTNET else '🔴 GERÇEK HESAP - GERÇEK PARA'}"
        )
    else:
        mode_text = "Sadece sinyal modu — otomatik işlem kapalı."

    recovered = [r for r in _read_pending() if r.get("closed", "0") != "1"]
    persistence_note = (
        f"💾 Kalıcı depolama AKTİF (DATA_DIR={DATA_DIR}) — yeniden başlatmada "
        f"{len(recovered)} açık pozisyon takibi geri yüklendi."
        if DATA_DIR != "."
        else "⚠️ Kalıcı depolama KAPALI (DATA_DIR ayarlanmamış) — bu deploy'daki açık "
             "pozisyon takibi bir sonraki deploy'da/restart'ta silinecek."
    )

    if SQUEEZE_MODE:
        recovered_squeeze = _read_squeeze_positions()
        send_telegram_message(
            "🗜️ Kripto botu (SIKIŞMA + KIRILIM) başlatıldı.\n"
            f"{len(WATCHLIST)} coin taranıyor, {SQUEEZE_TIMEFRAME} mumla.\n\n"
            f"Strateji: Bollinger Band genişliği son {SQUEEZE_BBW_LOOKBACK} mumun en dar "
            f"%{int(SQUEEZE_PERCENTILE*100)}'ine düşünce SIKIŞMA sayılır; ardından güçlü bir mum "
            f"(gövde ort.×{SQUEEZE_STRONG_BODY_MULT}) son {SQUEEZE_WINDOW} mumun en yüksek/en düşüğünü "
            f"kırınca sinyal (kullanıcının grafikte gözlemlediği örüntü).\n"
            + (f"🔄 YÖN TERSİNE ÇEVRİLDİ: bildirim sinyal yönünü gösterir ama gerçek işlem TERSİ yönde açılır "
               f"(ör. LONG bildirimi gelir, gerçekte SHORT açılır).\n" if SQUEEZE_INVERT_EXECUTION else "")
            + f"ATR×{SQUEEZE_TRAIL_MULT} chandelier trailing stop, TP YOK.\n"
            f"💰 Stop'a takılırsa kayıp = YATIRILAN SABİT bakiyenin %{SQUEEZE_RISK_PER_TRADE_PCT}'i "
            f"(referans: {get_squeeze_reference_balance():.2f} USDT, SQUEEZE_REFERENCE_BALANCE ile düzeltilebilir).\n\n"
            f"{mode_text}\n\n"
            f"💾 {len(recovered_squeeze)} açık Sıkışma pozisyonu geri yüklendi (varsa)."
        )
    elif DONCHIAN_MODE:
        recovered_donchian = _read_donchian_positions()
        send_telegram_message(
            "🐢 Kripto botu (DONCHIAN TREND-TAKİP - ORİJİNAL yön) başlatıldı.\n"
            f"{len(WATCHLIST)} coin taranıyor, {DONCHIAN_TIMEFRAME} mumla.\n\n"
            f"Strateji: Donchian({DONCHIAN_PERIOD}) kırılım + EMA{DONCHIAN_TREND_EMA_PERIOD} trend filtresi, "
            f"ATR×{CHANDELIER_MULT} chandelier trailing stop (kâr büyüdükçe stop yukarı çekilir, asla gevşemez). "
            f"TP YOK — kazananın büyümesine izin veriliyor.\n"
            f"💰 Stop'a takılırsa kayıp = YATIRILAN SABİT bakiyenin %{DONCHIAN_RISK_PER_TRADE_PCT}'i "
            f"(referans: {get_donchian_reference_balance():.2f} USDT — bu, o anki değişken bakiye değil, "
            f"sabit bir referans; yanlışsa Railway'de DONCHIAN_REFERENCE_BALANCE env değişkeniyle düzelt).\n"
            f"🔎 Sinyal gelince hemen açılmıyor — gerçek yönde {DONCHIAN_CONFIRMATION_CANDLES} mum "
            f"doğrulama bekleniyor (en fazla {DONCHIAN_MAX_WAIT_CANDLES} mum, doğrulanmazsa iptal).\n\n"
            "⚠️ Bu strateji, şu ana kadarki TÜM testlerde EN KÖTÜ çıkan sonuçtu (isabet %28.1, "
            "ort. -%1.9/işlem, 4h mumla test edilmişti) — kullanıcı isteğiyle tersine çevrilmeden, "
            "olduğu gibi canlıda deneniyor. Daha sık sinyal için zaman dilimi 4h'den 15m'e kısaltıldı, "
            "bu yüzden gerçek sonuç orijinal backtest'ten biraz farklı çıkabilir.\n\n"
            f"{mode_text}\n\n"
            f"💾 {len(recovered_donchian)} açık Donchian pozisyonu geri yüklendi (varsa)."
        )
    else:
        send_telegram_message(
            "Kripto botu (VWAP Sapması + Hacim Z-Skor) başlatıldı.\n"
            f"{len(WATCHLIST)} coin taranıyor.\n\n"
            "İki bağımsız sinyal kolu çalışıyor:\n"
            f"1) VWAP Sapması: fiyat kayan VWAP'tan %2+ sapmış\n"
            f"2) Hacim Z-Skor: hacim, son 20 mumun ortalamasından z-skor≥{VOLUME_ZSCORE_THRESHOLD} sapmış (klimaks hacim)\n\n"
            + (f"🔄 YÖN TERSİNE ÇEVRİLDİ: sinyal LONG derse SHORT, SHORT derse LONG açılıyor.\n\n" if REVERSE_SIGNALS else "")
            + f"🎯 TP: sabit, komisyon + %{SIMPLE_PROFIT_TARGET_PCT} (basit, stop mesafesiyle orantı yok)\n\n"
            f"Checkpoint hedefleri: {checkpoint_text}\n"
            f"En fazla {MAX_HOLD_MINUTES // 60}sa tutuş, her checkpoint'te otomatik durum bildirimi gelecek.\n\n"
            f"{mode_text}\n\n"
            f"{persistence_note}"
        )
    while True:
        scan_once()
        # bir sonraki taramaya kadar Telegram buton tikla olaylarini sik sik kontrol et
        # (tam otomatik modda buton yok ama surec ayni kalsin diye dongu korunuyor)
        elapsed = 0
        poll_interval = 5
        while elapsed < CHECK_INTERVAL_MINUTES * 60:
            # /stats gibi komutlari her modda (tam otomatik dahil) dinle -
            # eskiden sadece yari-otomatik modda calisiyordu, tam otomatik
            # modda /stats hicbir cevap vermiyordu.
            process_telegram_updates()
            if AUTO_TRADING_ENABLED and not FULL_AUTO_TRADING:
                expire_old_confirmations()
            time.sleep(poll_interval)
            elapsed += poll_interval


if __name__ == "__main__":
    run_forever()
