    price_below_ema20 = row["close"] < row["ema20"]

    if volume_ok and body_ok and consecutive_bull and trend_up and price_above_ema20:
        return "LONG", row
    if volume_ok and body_ok and consecutive_bear and trend_down and price_below_ema20:
        return "SHORT", row

    return None


SIGNAL_LOG_FILE = "signal_history.csv"


def log_signal(symbol: str, direction: str, row, exhausted: bool):
    file_exists = os.path.isfile(SIGNAL_LOG_FILE)
    with open(SIGNAL_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "symbol", "direction", "price", "volume",
                "vol_sma15", "atr14", "ema20", "ema50", "rsi", "exhausted"
            ])
        writer.writerow([
            datetime.now().isoformat(), symbol, direction, row["close"],
            row["volume"], row["vol_sma15"], row["atr14"], row["ema20"],
            row["ema50"], row["rsi"], exhausted
        ])


def scan_once():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Tarama basliyor...")
    for symbol in WATCHLIST:
        try:
            df = fetch_data(symbol)
            df = compute_indicators(df)
            result = check_breakout(df)

            if result:
                direction, row = result
                exhausted = check_exhaustion(direction, row)
                log_signal(symbol, direction, row, exhausted)

                msg = (
                    f"{symbol} - {direction} sinyali\n"
                    f"Fiyat: {row['close']:.4f}\n"
                    f"Hacim: {row['volume']:.0f} (SMA15: {row['vol_sma15']:.0f})\n"
                    f"ATR14: {row['atr14']:.4f}\n"
                    f"EMA20/50: {row['ema20']:.4f} / {row['ema50']:.4f}\n"
                    f"RSI({EXHAUSTION_RSI_PERIOD}): {row['rsi']:.1f}\n"
                    f"Zaman dilimi: {TIMEFRAME}"
                )
                if exhausted:
                    ters_yon = "LONG" if direction == "SHORT" else "SHORT"
                    msg += (
                        f"\n\n⚠️ Olası tükenme belirtisi.\n"
                        f"Hareket zaten ilerlemiş olabilir, {ters_yon} yönünde "
                        f"tepki (bounce) ihtimali göz önünde bulundurulabilir."
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
