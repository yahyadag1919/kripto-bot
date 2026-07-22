"""
walk_forward_test.py

AMAC
----
Su ana kadarki TUM backtest turlari (round 1-10, filtre testleri, next-gen
testleri) AYNI veri seti uzerinde onlarca strateji varyasyonu deneyip icinden
"en iyi" sonucu vereni sectiler - ve basarisini YINE AYNI veri uzerinde
raporladilar. Bu, "in-sample" (veri-ici) test denir ve ciddi bir tuzak icerir:
60-100 farkli varyasyon denersen, hicbirinde gercek bir edge (kenar/avantaj)
olmasa bile, sirf sans eseri bazilari iyi sonuc verir. Sectigimiz "kazananlar"
bu rastlantisal iyi gorunumun kurbani olmus olabilir.

Bu script bunu test ediyor: veriyi ZAMAN SIRASINA GORE ikiye boluyor.
  - TRAIN (ilk ~%60): burada hicbir yeni secim/optimizasyon YAPMIYORUZ -
    su an canlida calisan iki stratejiyi (VWAP Sapmasi + Hacim Z-Skor,
    Trend+Funding filtreli) AYNEN canli bottaki parametrelerle calistirip
    referans olarak raporluyoruz.
  - TEST (son ~%40): AYNI parametrelerle, botun hic gormedigi/uzerinde hic
    ayarlama yapilmamis donemde calistiriyoruz.

YORUM
-----
  - TRAIN ve TEST sonuclari (isabet orani, ort. net %) birbirine yakinsa
    -> edge muhtemelen gercek, piyasa kosuluna/rastlantiya bagli degil
  - TEST, TRAIN'e gore belirgin kotulesiyorsa (ozellikle ort. net % negatife
    donuyorsa) -> TRAIN'deki basari buyuk ihtimalle rastlantisal/overfit,
    canlida gordugumuz kayiplarin sebebi muhtemelen budur

ONEMLI KISIT
------------
Bu script'i CLAUDE calistiramiyor - sandbox'in internet erisimi kapali.
Bunu SENIN calistirman, ciktisini (konsol veya CSV) paylasman gerekiyor.
Calistirmak icin: pip install ccxt pandas numpy (API key GEREKMEZ, sadece
public/genel veri cekiliyor).

    python walk_forward_test.py

Varsayilan olarak 25 coin, 15m mumla, son ~45 gunluk veriyi ceker (Binance
public API rate limitine takilmamak icin coin sayisini/suresini asagidan
degistirebilirsin).
"""

import os
import time
from datetime import datetime, timedelta

import ccxt
import numpy as np
import pandas as pd
import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def send_telegram_message(text: str):
    """TELEGRAM_TOKEN/TELEGRAM_CHAT_ID ayarliysa (canli bottakiyle AYNI degiskenler,
    Railway'de zaten tanimli) sonucu Telegram'a da gonderir - telefondan takip
    edebilmek icin. Ayarli degilse sadece konsola yazar (sessizce atlar)."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
    except Exception as e:
        print(f"Telegram gonderim hatasi: {e}")

# ---------------------------------------------------------------------------
# Ayarlar - canli bottaki degerlerle AYNI tutuldu, burada YENI bir optimizasyon
# YAPILMIYOR. Amac secilenlerin dogrulugunu test etmek, yeni secim yapmak degil.
# ---------------------------------------------------------------------------

COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "SUI",
    "DOT", "TRX", "ATOM", "NEAR", "TON", "LTC", "BCH", "ETC", "FIL", "APT",
    "ARB", "OP", "INJ", "SEI", "ICP",
]
WATCHLIST = [f"{c}/USDT:USDT" for c in COINS]

TIMEFRAME = "15m"
DAYS_OF_HISTORY = 45          # ne kadar geriye gidilecek
TRAIN_FRACTION = 0.6          # ilk %60 TRAIN, son %40 TEST

RSI_PERIOD = 14
ATR_PERIOD = 14
VWAP_WINDOW = 96
DYNAMIC_ATR_MULT = 2.5
VOLUME_ZSCORE_THRESHOLD = 2.0

TREND_EMA_PERIOD = 200        # 4h mumla
INVALIDATION_ATR_BUFFER = 1.0
ATR_STOP_MULTIPLIER = 1.5
TP_RISK_REWARD_RATIO = 1.5
ROUNDTRIP_COMMISSION_PCT = 0.1

CHECKPOINTS = [
    (60, 0.3, "1sa"),
    (240, 0.6, "4sa"),
    (720, 1.0, "12sa"),
    (1440, 1.5, "24sa"),
]
CANDLES_PER_CHECKPOINT = [(m // 15, t, l) for m, t, l in CHECKPOINTS]  # 15m mum sayisina cevir
MAX_HOLD_CANDLES = CANDLES_PER_CHECKPOINT[-1][0]

exchange = ccxt.binanceusdm({"enableRateLimit": True})


# ---------------------------------------------------------------------------
# Veri cekme
# ---------------------------------------------------------------------------

def fetch_full_history(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Binance'ten sayfalayarak (limit=1500/istek) tam gecmisi ceker."""
    ms_per_candle = {"15m": 15 * 60 * 1000, "4h": 4 * 60 * 60 * 1000}[timeframe]
    since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    all_rows = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1500)
        if not batch:
            break
        all_rows += batch
        last_ts = batch[-1][0]
        if last_ts <= since or len(batch) < 1500:
            since = last_ts + ms_per_candle
            if len(batch) < 1500:
                break
        else:
            since = last_ts + ms_per_candle
        time.sleep(exchange.rateLimit / 1000)
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df


def fetch_funding_history(symbol: str, days: int) -> pd.DataFrame:
    since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    all_rows = []
    while True:
        batch = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
        if not batch:
            break
        all_rows += batch
        last_ts = batch[-1]["timestamp"]
        if last_ts <= since or len(batch) < 1000:
            break
        since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)
    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "fundingRate"])
    df = pd.DataFrame([{"timestamp": r["timestamp"], "fundingRate": r["fundingRate"]} for r in all_rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Indikatorler - canli bottaki compute_indicators ile AYNI mantik
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vol_sma20"] = df["volume"].rolling(20).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(ATR_PERIOD).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]
    df["vwap"] = pv.rolling(VWAP_WINDOW).sum() / df["volume"].rolling(VWAP_WINDOW).sum()
    df["vwap_dev_pct"] = (df["close"] - df["vwap"]) / df["vwap"] * 100

    df["vol_std20"] = df["volume"].rolling(20).std()
    df["vol_zscore"] = (df["volume"] - df["vol_sma20"]) / df["vol_std20"].replace(0, np.nan)

    # dinamik ATR-bazli VWAP sapma esigi (canli bottaki gibi)
    df["dynamic_vwap_threshold_pct"] = (df["atr14"] / df["close"]) * 100 * DYNAMIC_ATR_MULT

    df["ema200_4h"] = np.nan  # asagida ayri hesaplanip birlestirilecek
    return df


def attach_4h_trend(df_15m: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    df_15m = df_15m.drop(columns=["ema200_4h"], errors="ignore")  # compute_indicators'daki bos placeholder'i sil, cakisma yaratmasin
    df_4h = df_4h.copy()
    df_4h["ema200"] = df_4h["close"].ewm(span=TREND_EMA_PERIOD, adjust=False).mean()
    df_4h = df_4h[["timestamp", "ema200"]].sort_values("timestamp")
    df_15m = pd.merge_asof(df_15m.sort_values("timestamp"), df_4h, on="timestamp", direction="backward")
    df_15m = df_15m.rename(columns={"ema200": "ema200_4h"})
    return df_15m


def attach_funding(df_15m: pd.DataFrame, df_funding: pd.DataFrame) -> pd.DataFrame:
    if df_funding.empty:
        df_15m["funding"] = np.nan
        return df_15m
    df_15m = pd.merge_asof(df_15m.sort_values("timestamp"), df_funding, on="timestamp", direction="backward")
    return df_15m.rename(columns={"fundingRate": "funding"})


# ---------------------------------------------------------------------------
# Sinyal + sonuc simulasyonu - canli bottaki check_breakout_gate /
# check_volume_zscore_gate / passes_trend_funding_filter / checkpoint
# dongusu ile AYNI mantik, backtest icin vektorize degil, satir satir.
# ---------------------------------------------------------------------------

def simulate(df: pd.DataFrame, strategy: str) -> list:
    """df: compute_indicators + trend + funding eklenmis, TEK bir donemin (train ya da test)
    verisi. Donen: her biri bir kapanan islemi temsil eden dict listesi."""
    trades = []
    n = len(df)
    i = VWAP_WINDOW  # yeterli isinma suresi

    while i < n - 1:
        row = df.iloc[i]
        if pd.isna(row["vwap_dev_pct"]) or pd.isna(row["atr14"]) or pd.isna(row["ema200_4h"]) or pd.isna(row["funding"]):
            i += 1
            continue

        direction = None
        if strategy == "vwap":
            thr = row["dynamic_vwap_threshold_pct"]
            if row["vwap_dev_pct"] <= -thr:
                direction = "LONG"
            elif row["vwap_dev_pct"] >= thr:
                direction = "SHORT"
        elif strategy == "zscore":
            if pd.notna(row["vol_zscore"]) and row["vol_zscore"] >= VOLUME_ZSCORE_THRESHOLD:
                direction = "LONG" if row["close"] < row["open"] else "SHORT"

        if direction is None:
            i += 1
            continue

        # Trend + funding filtresi (AND mantigi, canli botla ayni)
        trend_ok = (row["close"] > row["ema200_4h"]) if direction == "LONG" else (row["close"] < row["ema200_4h"])
        funding_ok = (row["funding"] < 0) if direction == "LONG" else (row["funding"] > 0)
        if not (trend_ok and funding_ok):
            i += 1
            continue

        entry_price = row["close"]
        atr = row["atr14"]
        invalidation = entry_price - atr * INVALIDATION_ATR_BUFFER if direction == "LONG" else entry_price + atr * INVALIDATION_ATR_BUFFER
        atr_stop = entry_price - atr * ATR_STOP_MULTIPLIER if direction == "LONG" else entry_price + atr * ATR_STOP_MULTIPLIER
        stop_price = max(invalidation, atr_stop) if direction == "LONG" else min(invalidation, atr_stop)
        stop_distance_pct = abs(entry_price - stop_price) / entry_price * 100

        tp_target_pct = max(stop_distance_pct * TP_RISK_REWARD_RATIO, CHECKPOINTS[0][1]) + ROUNDTRIP_COMMISSION_PCT

        outcome = None
        pct_change = None
        cp_idx = 0

        for j in range(i + 1, min(i + 1 + MAX_HOLD_CANDLES, n)):
            candle = df.iloc[j]
            elapsed_candles = j - i

            if direction == "LONG":
                # stop once (candle low ile) - once stop mu vurdu bak
                if candle["low"] <= stop_price:
                    pct_change = -stop_distance_pct - ROUNDTRIP_COMMISSION_PCT
                    outcome = "SL"
                    break
                if candle["high"] >= entry_price * (1 + tp_target_pct / 100):
                    pct_change = tp_target_pct
                    outcome = "TP"
                    break
            else:
                if candle["high"] >= stop_price:
                    pct_change = -stop_distance_pct - ROUNDTRIP_COMMISSION_PCT
                    outcome = "SL"
                    break
                if candle["low"] <= entry_price * (1 - tp_target_pct / 100):
                    pct_change = tp_target_pct
                    outcome = "TP"
                    break

            # checkpoint hedefleri (opportunistik, kucukten buyuge)
            while cp_idx < len(CANDLES_PER_CHECKPOINT) and elapsed_candles >= CANDLES_PER_CHECKPOINT[cp_idx][0]:
                cp_idx += 1

        if outcome is None:
            # 24sa doldu, hicbir hedef/stop tutmadi - son fiyatla kapat
            last_j = min(i + MAX_HOLD_CANDLES, n - 1)
            last_close = df.iloc[last_j]["close"]
            raw = (last_close - entry_price) / entry_price * 100
            pct_change = (raw if direction == "LONG" else -raw) - ROUNDTRIP_COMMISSION_PCT
            outcome = "SURE_DOLDU"

        trades.append({
            "timestamp": row["timestamp"], "direction": direction,
            "entry": entry_price, "outcome": outcome, "pct_change": pct_change,
        })

        i += 1  # bir sonraki mumdan taramaya devam (ayni coin uzerinde ust uste sinyal olabilir)

    return trades


def summarize(trades: list, label: str) -> dict:
    if not trades:
        return {"label": label, "n": 0, "hit_rate": None, "avg_net": None, "total_net": None}
    n = len(trades)
    wins = sum(1 for t in trades if t["pct_change"] > 0)
    total_net = sum(t["pct_change"] for t in trades)
    avg_net = total_net / n
    return {
        "label": label, "n": n, "hit_rate": wins / n * 100,
        "avg_net": avg_net, "total_net": total_net,
    }


def print_summary(s: dict):
    if s["n"] == 0:
        print(f"  {s['label']}: sinyal yok")
        return
    print(
        f"  {s['label']}: {s['n']} işlem | isabet %{s['hit_rate']:.1f} | "
        f"ort. net %{s['avg_net']:+.3f} | toplam net %{s['total_net']:+.2f}"
    )


# ---------------------------------------------------------------------------
# Ana akis
# ---------------------------------------------------------------------------

def main():
    print(f"Walk-forward test basliyor - {len(COINS)} coin, {DAYS_OF_HISTORY} gun, TRAIN/TEST bolme: %{int(TRAIN_FRACTION*100)}/%{int((1-TRAIN_FRACTION)*100)}\n")
    send_telegram_message(
        f"🔬 Walk-forward test başladı ({len(COINS)} coin, {DAYS_OF_HISTORY} gün). "
        f"Bitince sonuç buraya gelecek, biraz sürebilir..."
    )

    results = {"vwap": {"train": [], "test": []}, "zscore": {"train": [], "test": []}}

    for symbol in WATCHLIST:
        print(f"--- {symbol} ---")
        try:
            df15 = fetch_full_history(symbol, TIMEFRAME, DAYS_OF_HISTORY)
            df4h = fetch_full_history(symbol, "4h", DAYS_OF_HISTORY + 40)  # EMA200 isinmasi icin fazladan gecmis
            dff = fetch_funding_history(symbol, DAYS_OF_HISTORY)
        except Exception as e:
            print(f"  veri cekilemedi: {e}")
            continue

        if len(df15) < VWAP_WINDOW * 2:
            print("  yeterli veri yok, atlaniyor")
            continue

        df15 = compute_indicators(df15)
        df15 = attach_4h_trend(df15, df4h)
        df15 = attach_funding(df15, dff)

        split_idx = int(len(df15) * TRAIN_FRACTION)
        train_df = df15.iloc[:split_idx].reset_index(drop=True)
        test_df = df15.iloc[split_idx:].reset_index(drop=True)

        for strategy in ["vwap", "zscore"]:
            train_trades = simulate(train_df, strategy)
            test_trades = simulate(test_df, strategy)
            results[strategy]["train"] += train_trades
            results[strategy]["test"] += test_trades

    print("\n================= SONUC (tum coinler birlesik) =================\n")
    lines = ["📊 Walk-forward test SONUÇ (tüm coinler birleşik):\n"]
    for strategy, label in [("vwap", "VWAP Sapması"), ("zscore", "Hacim Z-Skor")]:
        print(f"{label}:")
        train_s = summarize(results[strategy]["train"], "TRAIN (ilk %60)")
        test_s = summarize(results[strategy]["test"], "TEST  (son %40, hic gorulmemis)")
        print_summary(train_s)
        print_summary(test_s)
        print()

        lines.append(f"{label}:")
        for s in (train_s, test_s):
            if s["n"] == 0:
                lines.append(f"  {s['label']}: sinyal yok")
            else:
                lines.append(
                    f"  {s['label']}: {s['n']} işlem | isabet %{s['hit_rate']:.1f} | "
                    f"ort. net %{s['avg_net']:+.3f} | toplam net %{s['total_net']:+.2f}"
                )
        lines.append("")

    lines.append(
        "Yorum: TEST, TRAIN'e yakınsa edge gerçek olabilir. TEST'te ort. net "
        "% ciddi düşüyorsa (özellikle eksiye dönüyorsa) TRAIN'deki başarı "
        "muhtemelen overfit/rastlantısal."
    )
    send_telegram_message("\n".join(lines))

    print("YORUM: TEST sonuclari TRAIN'e yakinsa edge gercek olabilir.")
    print("TEST'te ort. net % TRAIN'e gore ciddi dusuyorsa (ozellikle negatife")
    print("donuyorsa) simdiye kadarki 'basarili' sonuclar buyuk ihtimalle")
    print("overfit/rastlantisal - canlidaki surekli kaybin sebebi muhtemelen bu.")


if __name__ == "__main__":
    main()
