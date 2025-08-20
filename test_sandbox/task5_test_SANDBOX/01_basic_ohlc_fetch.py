"""
基本的なOHLCデータ取得テスト - MT5から履歴データを取得して表示
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from datetime import datetime, timedelta
import polars as pl
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
import time

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.ohlc_fetcher import HistoricalDataFetcher
from src.common.config import BaseConfig
from utils.ohlc_display_helpers import (
    print_success, print_error, print_warning, print_info,
    print_section, create_ohlc_table, create_statistics_panel,
    create_candlestick_chart, format_timeframe
)

console = Console()

def main():
    """メインテスト関数"""
    print_section("Basic OHLC Data Fetch Test")
    
    # 設定
    symbol = "EURJPY"
    timeframe = "M5"  # 5分足
    days_back = 3  # 過去3日分
    
    print_info(f"Symbol: {symbol}")
    print_info(f"Timeframe: {format_timeframe(timeframe)}")
    print_info(f"Period: Last {days_back} days")
    
    # 日付範囲を設定
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    try:
        # MT5設定を作成
        config = BaseConfig()
        
        # MT5クライアント用の設定を辞書形式で作成
        mt5_config = {
            "account": config.mt5_login,
            "password": config.mt5_password.get_secret_value() if config.mt5_password else None,
            "server": config.mt5_server,
            "timeout": config.mt5_timeout
        }
        
        # MT5クライアントを作成
        print_info("Creating MT5 client...")
        mt5_client = MT5ConnectionManager(mt5_config)
        
        # HistoricalDataFetcherを作成
        print_info("Initializing Historical Data Fetcher...")
        fetcher_config = {
            "batch_size": 1000,
            "max_workers": 1  # シンプルなテストなので並列処理は使わない
        }
        fetcher = HistoricalDataFetcher(
            mt5_client=mt5_client,
            config=fetcher_config
        )
        
        # 接続
        print_info("Connecting to MT5...")
        if not fetcher.connect():
            print_error("Failed to connect to MT5")
            return
        
        print_success("Connected to MT5 successfully!")
        
        # データ取得
        print_info(f"Fetching OHLC data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        
        start_fetch = time.time()
        result = fetcher.fetch_ohlc_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        fetch_time = time.time() - start_fetch
        
        # LazyFrameをDataFrameに変換
        if result is not None:
            df = result.collect()
            print_success(f"Data fetched successfully in {fetch_time:.2f} seconds!")
            print_info(f"Total records: {len(df)}")
            
            # データ表示用のレイアウトを作成
            layout = Layout()
            layout.split_column(
                Layout(name="chart", size=22),
                Layout(name="table", size=25),
                Layout(name="stats", size=12)
            )
            
            # チャートを作成
            chart = create_candlestick_chart(df)
            layout["chart"].update(Panel(chart, title=f"{symbol} - {format_timeframe(timeframe)} Chart", border_style="green"))
            
            # テーブルを作成
            table = create_ohlc_table(df, title=f"Recent {symbol} Data", max_rows=15)
            layout["table"].update(table)
            
            # 統計パネルを作成
            stats = create_statistics_panel(df, symbol, timeframe)
            layout["stats"].update(stats)
            
            # 表示
            console.print(layout)
            
            # パターン検出の例
            print_section("Pattern Detection Example")
            
            # 簡単な分析
            if len(df) > 1:
                last_close = df["close"][-1]
                prev_close = df["close"][-2]
                change = last_close - prev_close
                change_pct = (change / prev_close) * 100
                
                color = "green" if change >= 0 else "red"
                print_info(f"Last Close: {last_close:.5f}")
                print_info(f"Previous Close: {prev_close:.5f}")
                console.print(f"Change: [{color}]{change:+.5f} ({change_pct:+.2f}%)[/{color}]")
                
                # ボラティリティ
                high_low_range = df["high"] - df["low"]
                avg_range = high_low_range.mean()
                print_info(f"Average High-Low Range: {avg_range:.5f}")
                
                # 最高値・最安値
                print_info(f"Period High: {df['high'].max():.5f}")
                print_info(f"Period Low: {df['low'].min():.5f}")
            
            # データ品質チェック
            print_section("Data Quality Check")
            
            # カラム名を確認
            time_col = "timestamp" if "timestamp" in df.columns else "time"
            
            # 時間の連続性をチェック
            time_diffs = df[time_col].diff()
            expected_interval = 5 * 60  # 5分（秒単位）
            
            # polarsで時間差を秒に変換
            time_diffs_seconds = time_diffs.dt.total_seconds()
            
            # 異常な間隔を検出
            anomalies = time_diffs_seconds.filter(
                (time_diffs_seconds > expected_interval * 1.5) | 
                (time_diffs_seconds < expected_interval * 0.5)
            )
            
            if len(anomalies) > 0:
                print_warning(f"Found {len(anomalies)} time gaps or anomalies")
            else:
                print_success("No time anomalies detected")
            
            # 価格の妥当性チェック
            invalid_prices = df.filter(
                (pl.col("high") < pl.col("low")) |
                (pl.col("open") > pl.col("high")) |
                (pl.col("open") < pl.col("low")) |
                (pl.col("close") > pl.col("high")) |
                (pl.col("close") < pl.col("low"))
            )
            
            if len(invalid_prices) > 0:
                print_error(f"Found {len(invalid_prices)} bars with invalid price relationships")
            else:
                print_success("All price relationships are valid (Low ≤ Open/Close ≤ High)")
            
            # メモリ使用量を表示
            memory_usage = df.estimated_size("mb")
            print_info(f"DataFrame memory usage: {memory_usage:.2f} MB")
            
        else:
            print_error("Failed to fetch data")
        
    except Exception as e:
        print_error(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 切断
        if 'fetcher' in locals():
            fetcher.disconnect()
            print_info("Disconnected from MT5")

if __name__ == "__main__":
    main()