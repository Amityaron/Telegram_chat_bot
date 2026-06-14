import os
import html
import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from tabulate import tabulate

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes


# ============================================================
# Load Telegram API Key from .env
# ============================================================

load_dotenv()

TELEGRAM_API_KEY = os.getenv("Telegram_API_KEY")

if not TELEGRAM_API_KEY:
    raise ValueError(
        "Telegram_API_KEY was not found. "
        "Create a .env file in the same folder and add:\n"
        "Telegram_API_KEY=your_token_here"
    )


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


# ============================================================
# Default tickers
# ============================================================

DEFAULT_ETFS = [
    "UGL",
    "USD",
    "QLD",
    "LUMI.TA",
    "ROM",
    "BTC-USD",
    "SSO",
    "UWM",
    "URSP",
    "UYG",
    "TA35.TA",
    "SVIX",
]


# ============================================================
# Helper: extract Close series safely from yfinance result
# ============================================================

def get_close_series(raw_data: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Extract a clean Close price Series from yfinance data.
    Handles regular columns and MultiIndex columns.
    """

    if raw_data is None or raw_data.empty:
        raise ValueError("No data returned from Yahoo Finance")

    # Case 1: regular columns
    if "Close" in raw_data.columns:
        close_data = raw_data["Close"]

        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]

        close_series = pd.to_numeric(close_data, errors="coerce").dropna()

        if close_series.empty:
            raise ValueError("Close column exists but has no valid values")

        return close_series

    # Case 2: MultiIndex columns
    if isinstance(raw_data.columns, pd.MultiIndex):
        close_columns = [
            col for col in raw_data.columns
            if any(str(level).lower() == "close" for level in col)
        ]

        if not close_columns:
            raise ValueError("'Close' column was not found in MultiIndex data")

        close_data = raw_data[close_columns[0]]

        if isinstance(close_data, pd.DataFrame):
            close_data = close_data.iloc[:, 0]

        close_series = pd.to_numeric(close_data, errors="coerce").dropna()

        if close_series.empty:
            raise ValueError("Close column exists but has no valid values")

        return close_series

    raise ValueError("'Close' column was not found")


# ============================================================
# Finance calculations
# ============================================================

def calculate_ticker_stats(ticker: str, days_back: int = 120) -> Dict[str, Any]:
    """
    Download data from Yahoo Finance and calculate:

    - Current price
    - Price Z Score over last 22 trading days
    - Diff Z Score based on 22-day price difference
    """

    ticker = ticker.strip().upper()

    end_date = pd.Timestamp.now(tz=None)
    start_date = end_date - pd.DateOffset(days=days_back)

    raw_data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    data = get_close_series(raw_data, ticker)

    if len(data) < 44:
        raise ValueError("Not enough data to calculate 22-day indicators")

    current_price = float(data.iloc[-1])

    # ========================================================
    # 1. Price Z Score 22 trading days
    # ========================================================

    last_22_prices = data.tail(22)

    mean_price_22 = float(last_22_prices.mean())
    std_price_22 = float(last_22_prices.std())

    if std_price_22 != 0:
        price_z_22d = round((current_price - mean_price_22) / std_price_22, 2)
    else:
        price_z_22d = 0

    # ========================================================
    # 2. Diff Z Score 22 trading days
    # Current diff = price today - price 22 days ago
    # Then Z-score that diff relative to historical 22-day diffs
    # ========================================================

    diff_22 = data - data.shift(22)
    diff_22 = diff_22.dropna()

    if diff_22.empty:
        raise ValueError("Could not calculate 22-day diff")

    current_diff_22 = float(diff_22.iloc[-1])
    mean_diff_22 = float(diff_22.mean())
    std_diff_22 = float(diff_22.std())

    if std_diff_22 != 0:
        diff_z_22d = round((current_diff_22 - mean_diff_22) / std_diff_22, 2)
    else:
        diff_z_22d = 0

    return {
        "Ticker": ticker,
        "Price": round(current_price, 2),
        "Price_Z": price_z_22d,
        "Diff_Z": diff_z_22d,
    }


def build_stats_table(tickers: List[str]) -> pd.DataFrame:
    """
    Build a DataFrame for all tickers.
    If a ticker fails, keep it in the result with an Error message.
    """

    results = []

    for ticker in tickers:
        ticker = ticker.strip().upper()

        if not ticker:
            continue

        try:
            stats = calculate_ticker_stats(ticker)
            stats["Error"] = ""
            results.append(stats)

        except Exception as e:
            results.append(
                {
                    "Ticker": ticker,
                    "Price": np.nan,
                    "Price_Z": np.nan,
                    "Diff_Z": np.nan,
                    "Error": str(e),
                }
            )

    df = pd.DataFrame(results)

    if not df.empty and "Price_Z" in df.columns:
        df = df.sort_values(
            by="Price_Z",
            ascending=True,
            na_position="last",
        )

    return df


# ============================================================
# Telegram table formatting
# ============================================================

def dataframe_to_telegram_text(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to clean Telegram HTML table.
    Uses <pre> block for monospace formatting.
    """

    if df.empty:
        return "No results to display."

    ok_df = df[df["Error"].eq("")].copy()
    error_df = df[df["Error"].ne("")].copy()

    text_parts = []

    if not ok_df.empty:
        ok_df = ok_df[["Ticker", "Price", "Price_Z", "Diff_Z"]]

        table = tabulate(
            ok_df,
            headers="keys",
            tablefmt="github",
            showindex=False,
            floatfmt=".2f",
        )

        text_parts.append("📊 Finance Z-Score Table")
        text_parts.append("")
        text_parts.append("<pre>" + html.escape(table) + "</pre>")

    if not error_df.empty:
        text_parts.append("")
        text_parts.append("⚠️ Errors:")

        for _, row in error_df.iterrows():
            ticker = html.escape(str(row["Ticker"]))
            error = html.escape(str(row["Error"]))
            text_parts.append(f"- <b>{ticker}</b>: {error}")

    return "\n".join(text_parts)


async def send_long_message(update: Update, text: str) -> None:
    """
    Telegram has a message length limit.
    Split long messages if needed.
    """

    max_len = 3900

    if len(text) <= max_len:
        await update.message.reply_text(text, parse_mode="HTML")
        return

    for i in range(0, len(text), max_len):
        chunk = text[i:i + max_len]
        await update.message.reply_text(chunk, parse_mode="HTML")


# ============================================================
# Telegram commands
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = """
שלום 👋

זה בוט ניתוח Yahoo Finance.

פקודות:

/table
מציג טבלה על רשימת ברירת המחדל.

/table NVDA QQQ SMH GLD BTC-USD
מציג טבלה לפי טיקרים שאתה שולח.

/stock NVDA
מציג ניתוח לטיקר בודד.

העמודות:

Price
מחיר אחרון.

Price_Z
Z Score של המחיר האחרון מול 22 ימי המסחר האחרונים.

Diff_Z
Z Score של שינוי 22 ימים מול כל תקופת המדגם.
"""
    await update.message.reply_text(message)


async def table_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /table
    /table NVDA QQQ SMH GLD BTC-USD
    """

    tickers = context.args if context.args else DEFAULT_ETFS

    await update.message.reply_text("מוריד נתונים מ-Yahoo Finance ומחשב טבלה...")

    df = build_stats_table(tickers)
    final_text = dataframe_to_telegram_text(df)

    await send_long_message(update, final_text)


async def stock_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /stock NVDA
    """

    if not context.args:
        await update.message.reply_text("תכתוב טיקר. לדוגמה:\n/stock NVDA")
        return

    ticker = context.args[0].strip().upper()

    await update.message.reply_text(f"בודק את {ticker}...")

    try:
        stats = calculate_ticker_stats(ticker)

        text = f"""
📊 ניתוח עבור <b>{html.escape(stats["Ticker"])}</b>

מחיר אחרון: <b>{stats["Price"]}</b>

Price Z Score 22D: <b>{stats["Price_Z"]}</b>
Diff Z Score 22D: <b>{stats["Diff_Z"]}</b>
"""
        await update.message.reply_text(text, parse_mode="HTML")

    except Exception as e:
        await update.message.reply_text(
            f"שגיאה בניתוח {html.escape(ticker)}:\n{html.escape(str(e))}",
            parse_mode="HTML",
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_API_KEY).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("table", table_command))
    app.add_handler(CommandHandler("stock", stock_command))

    print("Telegram Finance Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()