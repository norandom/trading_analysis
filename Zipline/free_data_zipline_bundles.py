import os
import shutil
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from zipline.data import bundles
from zipline.data.bundles import register
from zipline.data.bundles.csvdir import csvdir_equities
from zipline.data.data_portal import DataPortal
from zipline.utils.calendar_utils import get_calendar
from zipline.utils.cli import maybe_show_progress

# Disable pandas PerformanceWarnings and FutureWarnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def safe_float_to_uint32(value):
    """Safely convert float to uint32, clamping values if necessary."""
    return np.uint32(max(0, min(value, np.iinfo(np.uint32).max)))


def process_symbol_data(df, trading_days, missing_dates):
    """
    Processes the given dataframe (df) for a specific symbol,
    adjusts the data, and fills missing dates.

    Parameters:
    df (pandas DataFrame): The input dataframe containing the symbol's data.
    trading_days (list): A list of trading days for the specified symbol.
    missing_dates (list): A list of missing dates for the specified symbol.

    Returns:
    pandas DataFrame: A processed dataframe with adjusted data and missing dates filled.

    The function first sets the column names of the dataframe. It then drops any rows with missing values.
    An 'adj_factor' is calculated by dividing the 'adj_close' by the 'close' column.
    The 'open', 'high', 'low', and 'close' columns are then multiplied by the 'adj_factor'.
    The 'adj_close' and 'adj_factor' columns are then dropped from the dataframe.
    The index of the dataframe is renamed to 'date'.
    The 'dividend' and 'split' columns are initialized with 0.0 and 1.0 respectively.
    The dataframe is then reindexed to match the trading_days list.
    The dataframe is filled forward (ffill) to fill any missing values.
    For each missing date in the missing_dates list, if it's not already in the dataframe, the previous date's values are copied to that missing date.
    The dataframe is then sorted by the index (date).
    The 'open', 'high', 'low', and 'close' columns are then rounded to the nearest integer and converted to uint32 data type.
    The 'volume' column is filled with 0 if it's missing and clipped to ensure it's a non-negative integer. It's then converted to uint32 data type.

    The processed dataframe is then returned.
    """


def main():
    symbols = ['AAPL', 'NFLX', 'NVDA', 'JPM', 'SPY',
               "GC=F", "SI=F", "CL=F", "ZW=F", "PL=F", "ZC=F", "ZS=F", "KC=F",
               "CC=F", "^GDAXI", "^GSPC"]

    START_DATE = "2010-01-04"
    END_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")  # Use yesterday's date

    print(f"Downloading data for {len(symbols)} symbols from {START_DATE} to {END_DATE}")
    stocks_df = yf.download(tickers=symbols, start=START_DATE, end=END_DATE, group_by='ticker')
    print("Data download completed")

    calendar = get_calendar('XNYS')
    start_session = pd.Timestamp(START_DATE)
    end_session = pd.Timestamp(END_DATE)

    # Use pandas date_range instead of calendar.sessions_window
    all_days = pd.date_range(start=start_session, end=end_session, freq='D')
    trading_days = [day for day in all_days if calendar.is_session(day)]

    print(f"Total NYSE trading days: {len(trading_days)}")

    base_dir = os.path.expanduser('~/.zipline/custom_data/daily')
    os.makedirs(base_dir, exist_ok=True)

    missing_dates = [pd.Timestamp('2011-01-03'), pd.Timestamp('2016-10-10'), pd.Timestamp('2016-11-11')]

    for symbol in symbols:
        if symbol in stocks_df.columns.get_level_values(0):
            df = process_symbol_data(stocks_df[symbol].copy(), trading_days, missing_dates)
            print(f"\nSymbol: {symbol}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            print(f"Number of rows: {len(df)}")
            print(f"Unique dates: {df.index.nunique()}")
            print("Sample of data:")
            print(df.head())
            csv_file_path = os.path.join(base_dir, f'{symbol}.csv')
            df.to_csv(csv_file_path)
            print(f"CSV file saved: {csv_file_path}")

    print("\nAll CSV files have been created")

    bundle_path = os.path.expanduser("~/.zipline/data/custom_data")
    if os.path.exists(bundle_path):
        shutil.rmtree(bundle_path)
        print(f"Removed existing bundle data: {bundle_path}")

    CSV_FOLDER_PATH = os.path.expanduser("~/.zipline/custom_data")

    print("\nRegistering the bundle")
    register(
        'custom_data',
        csvdir_equities(['daily'], CSV_FOLDER_PATH),
        calendar_name='XNYS',
    )

    print("Ingesting the bundle")
    try:
        bundles.ingest('custom_data')
        print("Bundle ingestion completed successfully")
    except Exception as e:
        print(f"Error during bundle ingestion: {str(e)}")
        raise

    print("\nLoading the bundle")
    try:
        bundle = bundles.load('custom_data')
        print("Bundle loaded successfully")
    except Exception as e:
        print(f"Error loading bundle: {str(e)}")
        raise

    assets = bundle.asset_finder.retrieve_all(bundle.asset_finder.sids)
    print(f"Bundle contains {len(assets)} assets")

    print("\nCreating DataPortal for data verification")
    try:
        data_portal = DataPortal(
            asset_finder=bundle.asset_finder,
            trading_calendar=calendar,
            first_trading_day=bundle.equity_daily_bar_reader.first_trading_day,
            equity_daily_reader=bundle.equity_daily_bar_reader,
            adjustment_reader=bundle.adjustment_reader
        )
        print("DataPortal created successfully")
    except Exception as e:
        print(f"Error creating DataPortal: {str(e)}")
        raise

    print("\nRetrieving historical data")
    try:
        bar_count = len(trading_days)

        panel = data_portal.get_history_window(
            assets=assets,
            end_dt=end_session,
            bar_count=bar_count,
            frequency='1d',
            field='close',
            data_frequency='daily'
        )
        print("Historical data retrieved successfully")
        print(f"\nData retrieved for {len(assets)} assets")
        print(f"Date range: {panel.index[0]} to {panel.index[-1]}")
        print(f"Number of trading days: {len(panel)}")
        print("\nSample of retrieved data:")
        print(panel.head())
    except Exception as e:
        print(f"Error retrieving historical data: {str(e)}")
        raise

    print("\nScript completed successfully")


if __name__ == "__main__":
    main()