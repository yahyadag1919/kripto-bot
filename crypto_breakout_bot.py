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
CHECK_INTERVAL_MINUTES = 15

RSI_PERIOD = 14
ATR_PERIOD = 14
VWAP_WINDOW = 96          # ~24 saat (15m mumla) - kayan VWAP penceresi

# RSI artik giris sarti degil, sadece breakdown mesajinda bilgi amacli
RSI_LONG_MAX = 30
RSI_SHORT_MIN = 70
# Tek giris sarti: VWAP sapmasi (10. turda test edilen esik)
VWAP_DEV_LONG_MAX = -2.0   # VWAP'in en az %2 altinda
VWAP_DEV_SHORT_MIN = 2.0   # VWAP'in en az %2 ustunde

# Ikinci sinyal kolu: Hacim Z-Skor (10. turda 570 sinyal, %72.8 isabet, +%0.371 ort. net)
VOLUME_ZSCORE_THRESHOLD = 2.0

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
POSITION_PCT_OF_BALANCE = float(os.environ.get("POSITION_PCT_OF_BALANCE", "2"))  # bakiyenin yuzde kaci
LEVERAGE = int(os.environ.get("LEVERAGE", "20"))
CONFIRM_TIMEOUT_MINUTES = 15
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", "3"))  # sabit maks. zarar yuzdesi (fiyat bazinda, kaldiracsiz)

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


def _compute_position_size(symbol: str, entry_price: float) -> float:
    balance = exchange.fetch_balance()
    free_usdt = balance.get("USDT", {}).get("free", 0)
    position_value = free_usdt * (POSITION_PCT_OF_BALANCE / 100) * LEVERAGE
    quantity = position_value / entry_price
    return float(exchange.amount_to_precision(symbol, quantity))


def _compute_final_stop_price(direction: str, entry_price: float, invalidation: float) -> float:
    """
    Iki aday stop seviyesinden (strateji bazli 'gecersizlik' ve sabit STOP_LOSS_PCT)
    hangisi girisin daha yakininda ise (yani zarari daha kucuk tutuyorsa) onu secer.
    Boylece STOP_LOSS_PCT her zaman bir "maksimum zarar tavani" gibi calisir.
    """
    pct_stop = entry_price * (1 - STOP_LOSS_PCT / 100) if direction == "LONG" else entry_price * (1 + STOP_LOSS_PCT / 100)
    if direction == "LONG":
        return max(invalidation, pct_stop)  # ikisi de giristen asagida - buyuk olan (giristen yakin) daha sikidir
    else:
        return min(invalidation, pct_stop)  # ikisi de giristen yukarida - kucuk olan (giristen yakin) daha sikidir


def execute_order(symbol: str, direction: str, entry_price: float, invalidation: float):
    """Piyasa emriyle pozisyon acar + (gecersizlik seviyesi ile sabit %STOP_LOSS_PCT'den hangisi
    daha siki ise) koruyucu stop emri birakir - boylece maksimum zarar her zaman sinirli olur.
    Stop emri HERHANGI bir sebeple basarisiz olursa (orn. fiyat zaten stop seviyesini gecmisse,
    '-2021 Order would immediately trigger' hatasi), pozisyonu KORUMASIZ birakmak yerine
    aninda piyasa emriyle kapatir."""
    _set_leverage_safe(symbol)
    side = "buy" if direction == "LONG" else "sell"
    qty = _compute_position_size(symbol, entry_price)
    if qty <= 0:
        raise ValueError("Hesaplanan pozisyon miktari sifir veya negatif - bakiyeni kontrol et.")

    order = exchange.create_order(symbol, type="market", side=side, amount=qty)

    stop_price = _compute_final_stop_price(direction, entry_price, invalidation)
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

    return order, qty, stop_price


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
                order, qty, stop_price = execute_order(info["symbol"], info["direction"], info["entry_price"], info["invalidation"])
                send_telegram_message(
                    f"✅ {info['symbol']} {info['direction']} pozisyonu açıldı.\n"
                    f"Miktar: {qty} | Giriş: ~{info['entry_price']:.4f} | Stop: {stop_price:.4f} (maks. %{STOP_LOSS_PCT} zarar)"
                )
                # bu sinyal icin daha once qty=0 ile yazilmis pending kaydini guncelle,
                # boylece checkpoint sistemi bu gercek pozisyonu daha sonra kapatabilsin
                pending_rows = _read_pending()
                for pr in pending_rows:
                    if (pr["symbol"] == info["symbol"] and pr["direction"] == info["direction"]
                            and pr.get("closed", "0") == "0" and float(pr.get("qty", 0) or 0) == 0):
                        pr["qty"] = qty
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

    return df


# ---------------------------------------------------------------------------
# RSI + VWAP birlesik teyit kapisi
# ---------------------------------------------------------------------------

def check_breakout_gate(df: pd.DataFrame):
    """
    Son KAPANMIS muma bakar (df.iloc[-2]).
    LONG: fiyat VWAP'in belirgin altinda (RSI artik sart degil, bilgi amacli)
    SHORT: fiyat VWAP'in belirgin ustunde (RSI artik sart degil, bilgi amacli)
    """
    if len(df) < VWAP_WINDOW + 5:
        return None

    row = df.iloc[-2]
    if pd.isna(row["vwap_dev_pct"]) or pd.isna(row["rsi"]) or pd.isna(row["atr14"]):
        return None

    if row["vwap_dev_pct"] <= VWAP_DEV_LONG_MAX:
        return "LONG", row

    if row["vwap_dev_pct"] >= VWAP_DEV_SHORT_MIN:
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
    Pozisyonu piyasa emriyle kapatir (reduceOnly) VE o sembole ait kalan acik
    emirleri (orn. artik sahipsiz kalan koruyucu stop emri) iptal eder - aksi
    halde pozisyon kapansa bile stop emri borsada sahipsiz asili kalir ve
    "Open Orders" listesi zamanla sismeye devam eder. Basarili olursa bos
    string, basarisiz olursa hata metnini dondurur.
    """
    if qty <= 0:
        return ""  # gercek pozisyon yok (sinyal-amacli veya onaysiz), kapatacak bir sey yok
    close_side = "sell" if direction == "LONG" else "buy"
    try:
        exchange.create_order(symbol, type="market", side=close_side, amount=qty, params={"reduceOnly": True})
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

                log_outcome(symbol, strategy, direction, entry_price, r["entry_time"], minutes, label,
                            target_pct, current_price, pct_change, success)
                r[flag_key] = "1"

                if success:
                    close_err = _close_position(symbol, direction, qty)
                    msg = (
                        f"🎯 [{strategy}] {symbol} {direction} - {label} checkpoint'te hedef tutturuldu\n"
                        f"Giriş: {entry_price:.4f} | Şimdi: {current_price:.4f}\n"
                        f"Değişim: {pct_change:+.2f}% (hedef: %{target_pct})\n\n"
                        + (f"✅ Pozisyon otomatik kapatıldı."
                           if qty > 0 and not close_err
                           else (f"⚠️ Pozisyon kapatma emri başarısız: {close_err}\nManuel kapatmayi unutma!"
                                 if close_err else "Öneri: kârı realize etmeyi değerlendir."))
                    )
                    send_telegram_message(msg)
                    r["closed"] = "1"
                    closed = True
                    break
                elif label == CHECKPOINTS[-1][2]:
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

def _emit_signal(symbol: str, strategy: str, strategy_desc: str, direction: str, row, breakdown: list):
    log_signal(symbol, strategy, direction, row, breakdown)

    invalidation = compute_invalidation(direction, row)
    entry_price = row["close"]
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
    if FULL_AUTO_TRADING:
        try:
            order, executed_qty, stop_price = execute_order(symbol, direction, entry_price, invalidation)
            msg += (
                f"\n\n🤖 TAM OTOMATİK: pozisyon açıldı.\n"
                f"Miktar: {executed_qty} | Stop: {stop_price:.4f} (maks. %{STOP_LOSS_PCT} zarar)"
            )
        except Exception as e:
            msg += f"\n\n❌ TAM OTOMATİK emir başarısız oldu: {e}"
        send_telegram_message(msg)
    elif AUTO_TRADING_ENABLED:
        confirm_id = f"{symbol.replace('/', '').replace(':', '')}-{int(time.time())}"
        PENDING_CONFIRMATIONS[confirm_id] = {
            "symbol": symbol, "direction": direction, "entry_price": entry_price,
            "invalidation": invalidation, "created_at": datetime.now(),
        }
        msg += f"\n\n⚠️ {CONFIRM_TIMEOUT_MINUTES}dk içinde onaylamazsan otomatik iptal olur."
        send_telegram_confirm(msg, confirm_id)
    else:
        send_telegram_message(msg)

    log_pending(symbol, strategy, direction, entry_price, datetime.now(), invalidation, qty=executed_qty)


def scan_once():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Tarama basliyor...")

    check_pending_outcomes()

    closest_long = None   # (vwap_dev_pct, symbol) - en negatif (LONG esigine en yakin)
    closest_short = None  # (vwap_dev_pct, symbol) - en pozitif (SHORT esigine en yakin)
    scanned = 0

    for symbol in WATCHLIST:
        if symbol in _unsupported_symbols:
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
                breakdown = [
                    f"✅ VWAP sapması: %{vrow['vwap_dev_pct']:+.2f} (giriş şartı)",
                    f"ℹ️ RSI: {vrow['rsi']:.1f} (bilgi amaçlı, şart değil)",
                    f"✅ Hacim {vrow['volume']/vrow['vol_sma20']:.2f}x ortalama" if pd.notna(vrow.get('vol_sma20')) and vrow.get('vol_sma20') else "➖ Hacim verisi yetersiz",
                ]
                ob_support, ob_note = score_orderbook(symbol, direction)
                breakdown.append(f"{'✅' if ob_support else '➖'} Order book: {ob_note}")
                _emit_signal(
                    symbol, "VWAP Sapması",
                    "10. tur / uzun tutuş turnuvasında en çok sinyal + en iyi ort. net getiriyi veren sistem",
                    direction, vrow, breakdown,
                )
                fired = True

            zscore_result = check_volume_zscore_gate(df)
            if zscore_result:
                direction, zrow = zscore_result
                breakdown = [
                    f"✅ Hacim Z-Skor: {zrow['vol_zscore']:.2f} (giriş şartı, eşik: {VOLUME_ZSCORE_THRESHOLD})",
                    f"ℹ️ Mum yönü: {'düşüş (klimaks satış)' if direction == 'LONG' else 'yükseliş (klimaks alım)'}",
                    f"ℹ️ RSI: {zrow['rsi']:.1f} (bilgi amaçlı, şart değil)",
                ]
                ob_support, ob_note = score_orderbook(symbol, direction)
                breakdown.append(f"{'✅' if ob_support else '➖'} Order book: {ob_note}")
                _emit_signal(
                    symbol, "Hacim Z-Skor",
                    "10. tur turnuvasında 570 sinyal, %72.8 isabet, +%0.371 ort. net getiren ikinci sistem",
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
            f"Esige en yakin -> LONG: {closest_long[1]} (%{closest_long[0]:+.2f}, esik: %{VWAP_DEV_LONG_MAX}) | "
            f"SHORT: {closest_short[1]} (%{closest_short[0]:+.2f}, esik: %{VWAP_DEV_SHORT_MIN})"
        )


def run_forever():
    checkpoint_text = " / ".join(f"{label}(%{target})" for _, target, label in CHECKPOINTS)
    if FULL_AUTO_TRADING:
        mode_text = (
            f"🤖 TAM OTOMATİK MOD AÇIK — sinyaller ONAY BEKLEMEDEN Binance Futures'ta gerçek emir açar "
            f"(bakiyenin %{POSITION_PCT_OF_BALANCE} | {LEVERAGE}x kaldıraç | maks. %{STOP_LOSS_PCT} zarar stop'u). "
            f"{'⚠️ TESTNET (sahte para)' if USE_TESTNET else '🔴 GERÇEK HESAP - GERÇEK PARA'}"
        )
    elif AUTO_TRADING_ENABLED:
        mode_text = (
            f"⚡ YARI-OTOMATİK MOD AÇIK — sinyaller Telegram'dan onay bekleyecek, onaylarsan "
            f"Binance Futures'ta gerçek emir açılır (bakiyenin %{POSITION_PCT_OF_BALANCE} | {LEVERAGE}x kaldıraç | "
            f"maks. %{STOP_LOSS_PCT} zarar stop'u). "
            f"{'⚠️ TESTNET (sahte para)' if USE_TESTNET else '🔴 GERÇEK HESAP - GERÇEK PARA'}"
        )
    else:
        mode_text = "Sadece sinyal modu — otomatik işlem kapalı."

    send_telegram_message(
        "Kripto botu (VWAP Sapması + Hacim Z-Skor) başlatıldı.\n"
        f"{len(WATCHLIST)} coin taranıyor.\n\n"
        "İki bağımsız sinyal kolu çalışıyor:\n"
        f"1) VWAP Sapması: fiyat kayan VWAP'tan %2+ sapmış\n"
        f"2) Hacim Z-Skor: hacim, son 20 mumun ortalamasından z-skor≥{VOLUME_ZSCORE_THRESHOLD} sapmış (klimaks hacim)\n\n"
        f"Checkpoint hedefleri: {checkpoint_text}\n"
        f"En fazla {MAX_HOLD_MINUTES // 60}sa tutuş, her checkpoint'te otomatik durum bildirimi gelecek.\n\n"
        f"{mode_text}"
    )
    while True:
        scan_once()
        # bir sonraki taramaya kadar Telegram buton tikla olaylarini sik sik kontrol et
        # (tam otomatik modda buton yok ama surec ayni kalsin diye dongu korunuyor)
        elapsed = 0
        poll_interval = 5
        while elapsed < CHECK_INTERVAL_MINUTES * 60:
            if AUTO_TRADING_ENABLED and not FULL_AUTO_TRADING:
                process_telegram_updates()
                expire_old_confirmations()
            time.sleep(poll_interval)
            elapsed += poll_interval


if __name__ == "__main__":
    run_forever()
