"""
複数時間枠表示テスト - 同じシンボルの異なるタイムフレームを比較
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
from rich.table import Table
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.ohlc_fetcher import HistoricalDataFetcher
from src.common.config import BaseConfig
from utils.ohlc_display_helpers import (
    print_success, print_error, print_warning, print_info,
    print_section, create_ohlc_table, create_statistics_panel,
    create_comparison_table, format_timeframe, create_fetch_progress
)

console = Console()

class MultiTimeframeAnalyzer:
    """複数タイムフレーム分析クラス"""
    
    def __init__(self, symbol: str, timeframes: list, days_back: int = 7):
        self.symbol = symbol
        self.timeframes = timeframes
        self.days_back = days_back
        self.data = {}
        self.fetch_times = {}
        
    def fetch_timeframe_data(self, fetcher: HistoricalDataFetcher, timeframe: str) -> tuple:
        """特定のタイムフレームのデータを取得"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        
        start_time = time.time()
        try:
            result = fetcher.fetch_ohlc_data(
                symbol=self.symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            
            if result is not None:
                df = result.collect()
                fetch_time = time.time() - start_time
                return timeframe, df, fetch_time, None
            else:
                return timeframe, None, 0, "Failed to fetch data"
                
        except Exception as e:
            return timeframe, None, 0, str(e)
    
    def analyze_correlation(self) -> Table:
        """タイムフレーム間の相関を分析"""
        table = Table(title="Timeframe Correlation Analysis", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="yellow")
        
        for tf in self.timeframes:
            table.add_column(format_timeframe(tf), justify="right")
        
        # 各メトリックを計算
        metrics = {
            "Trend Direction": [],
            "Volatility": [],
            "Volume Ratio": [],
            "Price Range": [],
            "Candle Type": []
        }
        
        for tf in self.timeframes:
            if tf not in self.data or self.data[tf] is None or len(self.data[tf]) < 2:
                for metric in metrics:
                    metrics[metric].append("N/A")
                continue
            
            df = self.data[tf]
            
            # トレンド方向
            trend = "↑" if df["close"][-1] > df["close"][0] else "↓"
            metrics["Trend Direction"].append(trend)
            
            # ボラティリティ（平均範囲）
            avg_range = (df["high"] - df["low"]).mean()
            metrics["Volatility"].append(f"{avg_range:.5f}")
            
            # ボリューム比率（平均に対する最新）
            if "tick_volume" in df.columns:
                avg_volume = df["tick_volume"].mean()
                last_volume = df["tick_volume"][-1]
                volume_ratio = last_volume / avg_volume if avg_volume > 0 else 0
                metrics["Volume Ratio"].append(f"{volume_ratio:.2f}x")
            else:
                metrics["Volume Ratio"].append("N/A")
            
            # 価格範囲
            price_range = df["high"].max() - df["low"].min()
            metrics["Price Range"].append(f"{price_range:.5f}")
            
            # 最新キャンドルタイプ
            last_bar = df[-1]
            if abs(last_bar["close"] - last_bar["open"]) < (last_bar["high"] - last_bar["low"]) * 0.1:
                candle_type = "Doji"
            elif last_bar["close"] > last_bar["open"]:
                candle_type = "Bullish"
            else:
                candle_type = "Bearish"
            metrics["Candle Type"].append(candle_type)
        
        # テーブルに行を追加
        for metric, values in metrics.items():
            table.add_row(metric, *values)
        
        return table
    
    def create_summary_layout(self) -> Layout:
        """サマリーレイアウトを作成"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="comparison", size=15),
            Layout(name="correlation", size=10),
            Layout(name="details")
        )
        
        # ヘッダー
        header_text = f"[bold cyan]{self.symbol} - Multi-Timeframe Analysis[/bold cyan]\n"
        header_text += f"Period: Last {self.days_back} days | Timeframes: {', '.join(self.timeframes)}"
        layout["header"].update(Panel(header_text, border_style="cyan"))
        
        # 比較テーブル
        comparison = create_comparison_table(self.data, self.symbol)
        layout["comparison"].update(comparison)
        
        # 相関分析
        correlation = self.analyze_correlation()
        layout["correlation"].update(correlation)
        
        # 詳細パネル
        details_text = "Fetch Times:\n"
        for tf, fetch_time in self.fetch_times.items():
            details_text += f"  {format_timeframe(tf)}: {fetch_time:.2f}s\n"
        
        details_text += f"\nTotal Fetch Time: {sum(self.fetch_times.values()):.2f}s"
        layout["details"].update(Panel(details_text, title="Performance", border_style="green"))
        
        return layout

def main():
    """メインテスト関数"""
    print_section("Multi-Timeframe Display Test")
    
    # 設定
    symbol = "EURJPY"
    timeframes = ["M1", "M5", "M15", "H1", "H4", "D1"]  # 1分足から日足まで
    days_back = 7
    
    print_info(f"Symbol: {symbol}")
    print_info(f"Timeframes: {', '.join([format_timeframe(tf) for tf in timeframes])}")
    print_info(f"Period: Last {days_back} days")
    
    analyzer = MultiTimeframeAnalyzer(symbol, timeframes, days_back)
    
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
            "max_workers": 3  # 並列処理を使用
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
        
        # プログレスバーで進捗表示
        progress = create_fetch_progress()
        
        with progress:
            task = progress.add_task(
                "[cyan]Fetching data for all timeframes...",
                total=len(timeframes)
            )
            
            # 並列でデータ取得
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(
                        analyzer.fetch_timeframe_data, 
                        fetcher, 
                        tf
                    ): tf for tf in timeframes
                }
                
                for future in as_completed(futures):
                    tf, df, fetch_time, error = future.result()
                    
                    if error:
                        print_error(f"{format_timeframe(tf)}: {error}")
                    else:
                        analyzer.data[tf] = df
                        analyzer.fetch_times[tf] = fetch_time
                        print_success(f"{format_timeframe(tf)}: {len(df)} records fetched in {fetch_time:.2f}s")
                    
                    progress.update(task, advance=1)
        
        # 結果表示
        print_section("Analysis Results")
        
        # サマリーレイアウトを表示
        layout = analyzer.create_summary_layout()
        console.print(layout)
        
        # 個別のタイムフレーム詳細を表示
        print_section("Individual Timeframe Details")
        
        for tf in timeframes[:3]:  # 最初の3つのタイムフレームの詳細を表示
            if tf in analyzer.data and analyzer.data[tf] is not None:
                df = analyzer.data[tf]
                
                print(f"\n[bold cyan]{format_timeframe(tf)}[/bold cyan]")
                
                # 最新のデータを表示
                table = create_ohlc_table(df, title=f"{symbol} - {format_timeframe(tf)}", max_rows=5)
                console.print(table)
                
                # 簡単な統計
                if len(df) > 0:
                    print_info(f"  Records: {len(df)}")
                    print_info(f"  Latest Close: {df['close'][-1]:.5f}")
                    print_info(f"  Average Volume: {df['tick_volume'].mean():.0f}" if 'tick_volume' in df.columns else "  Volume: N/A")
        
        # タイムフレーム間の整合性チェック
        print_section("Data Consistency Check")
        
        # M1とM5の整合性をチェック（M5の1本 = M1の5本）
        if "M1" in analyzer.data and "M5" in analyzer.data:
            m1_df = analyzer.data["M1"]
            m5_df = analyzer.data["M5"]
            
            if m1_df is not None and m5_df is not None and len(m1_df) >= 5 and len(m5_df) >= 1:
                # 最新のM5バーに対応するM1バーを取得
                m5_last = m5_df[-1]
                m5_time = m5_last["time"]
                
                # M1データから対応する5本を探す
                m1_corresponding = m1_df.filter(
                    (pl.col("time") >= m5_time - timedelta(minutes=4)) &
                    (pl.col("time") <= m5_time)
                )
                
                if len(m1_corresponding) == 5:
                    # 集計して比較
                    m1_high = m1_corresponding["high"].max()
                    m1_low = m1_corresponding["low"].min()
                    m1_volume = m1_corresponding["tick_volume"].sum() if "tick_volume" in m1_corresponding.columns else 0
                    
                    m5_high = m5_last["high"]
                    m5_low = m5_last["low"]
                    m5_volume = m5_last["tick_volume"] if "tick_volume" in m5_df.columns else 0
                    
                    print_info("M1 vs M5 Consistency:")
                    print_info(f"  M5 High: {m5_high:.5f}, M1 Aggregated High: {m1_high:.5f}")
                    print_info(f"  M5 Low: {m5_low:.5f}, M1 Aggregated Low: {m1_low:.5f}")
                    
                    if abs(m5_high - m1_high) < 0.0001 and abs(m5_low - m1_low) < 0.0001:
                        print_success("  ✓ Price consistency verified")
                    else:
                        print_warning("  ⚠ Price inconsistency detected")
                else:
                    print_warning(f"Could not find matching M1 bars for consistency check")
        
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