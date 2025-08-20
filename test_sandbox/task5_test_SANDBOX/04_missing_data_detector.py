"""
欠損データ検出テスト - マーケット休場時間とデータギャップの視覚化
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from datetime import datetime, timedelta
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
import time

from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
from src.mt5_data_acquisition.ohlc_fetcher import HistoricalDataFetcher
from src.common.config import BaseConfig
from utils.ohlc_display_helpers import (
    print_success, print_error, print_warning, print_info,
    print_section, format_timestamp
)

console = Console()

class MissingDataDetector:
    """欠損データ検出クラス"""
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.missing_periods = []
        self.market_closed_periods = []
        self.data_gaps = []
        
    def analyze_data(self, df: pl.DataFrame) -> dict:
        """データを分析して欠損を検出"""
        if df.is_empty() or len(df) < 2:
            return {"error": "Insufficient data for analysis"}
        
        # タイムフレームに応じた期待される間隔（秒）
        expected_intervals = {
            "M1": 60,
            "M5": 300,
            "M15": 900,
            "M30": 1800,
            "H1": 3600,
            "H4": 14400,
            "D1": 86400,
            "W1": 604800,
            "MN1": 2592000  # 約30日
        }
        
        expected_interval = expected_intervals.get(self.timeframe, 60)
        
        # 時間差を計算
        time_diffs = df["time"].diff()
        time_diffs_seconds = time_diffs.dt.total_seconds()
        
        # 統計を計算
        stats = {
            "total_records": len(df),
            "time_range": f"{df['time'].min()} to {df['time'].max()}",
            "expected_interval": expected_interval,
            "actual_intervals": {
                "min": time_diffs_seconds.min(),
                "max": time_diffs_seconds.max(),
                "mean": time_diffs_seconds.mean(),
                "median": time_diffs_seconds.median()
            }
        }
        
        # ギャップを検出（期待される間隔の1.5倍以上）
        gap_threshold = expected_interval * 1.5
        gaps = []
        
        for i in range(1, len(df)):
            diff_seconds = (df["time"][i] - df["time"][i-1]).total_seconds()
            
            if diff_seconds > gap_threshold:
                gap_info = {
                    "start": df["time"][i-1],
                    "end": df["time"][i],
                    "duration_seconds": diff_seconds,
                    "duration_str": self._format_duration(diff_seconds),
                    "expected_bars": int(diff_seconds / expected_interval),
                    "is_weekend": self._is_weekend_gap(df["time"][i-1], df["time"][i]),
                    "is_market_closed": self._is_market_closed_time(df["time"][i-1], df["time"][i])
                }
                gaps.append(gap_info)
        
        # ギャップを分類
        for gap in gaps:
            if gap["is_weekend"]:
                self.market_closed_periods.append(gap)
            elif gap["is_market_closed"]:
                self.market_closed_periods.append(gap)
            else:
                self.data_gaps.append(gap)
        
        stats["gaps_found"] = len(gaps)
        stats["weekend_gaps"] = sum(1 for g in gaps if g["is_weekend"])
        stats["market_closed_gaps"] = sum(1 for g in gaps if g["is_market_closed"] and not g["is_weekend"])
        stats["data_gaps"] = len(self.data_gaps)
        
        return stats
    
    def _format_duration(self, seconds: float) -> str:
        """期間を読みやすい形式にフォーマット"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"
    
    def _is_weekend_gap(self, start: datetime, end: datetime) -> bool:
        """週末のギャップかどうかを判定"""
        # 金曜日の夜から月曜日の朝
        if start.weekday() == 4 and end.weekday() == 0:  # 金曜から月曜
            return True
        # 土日を含む
        if start.weekday() >= 5 or end.weekday() >= 5:
            return True
        return False
    
    def _is_market_closed_time(self, start: datetime, end: datetime) -> bool:
        """マーケット休場時間かどうかを判定"""
        # 簡単な実装：夜間（22:00-00:00）をマーケット休場とする
        if start.hour >= 22 or end.hour <= 1:
            return True
        return False
    
    def create_gap_visualization(self) -> Layout:
        """ギャップの視覚化レイアウトを作成"""
        layout = Layout()
        layout.split_column(
            Layout(name="summary", size=8),
            Layout(name="gaps", size=20),
            Layout(name="timeline")
        )
        
        # サマリー
        summary_text = f"[bold cyan]Missing Data Analysis - {self.symbol} ({self.timeframe})[/bold cyan]\n\n"
        summary_text += f"Total Gaps Found: {len(self.data_gaps) + len(self.market_closed_periods)}\n"
        summary_text += f"  • Data Gaps (Potential Issues): [red]{len(self.data_gaps)}[/red]\n"
        summary_text += f"  • Market Closed Periods: [yellow]{len(self.market_closed_periods)}[/yellow]\n"
        
        layout["summary"].update(Panel(summary_text, border_style="cyan"))
        
        # ギャップテーブル
        gap_table = Table(title="Detected Gaps", show_header=True, header_style="bold magenta")
        gap_table.add_column("Type", style="cyan")
        gap_table.add_column("Start Time")
        gap_table.add_column("End Time")
        gap_table.add_column("Duration", justify="right")
        gap_table.add_column("Missing Bars", justify="right")
        gap_table.add_column("Status")
        
        # データギャップ（問題の可能性）
        for gap in self.data_gaps[:10]:  # 最初の10個
            gap_table.add_row(
                "[red]DATA GAP[/red]",
                format_timestamp(gap["start"]),
                format_timestamp(gap["end"]),
                gap["duration_str"],
                str(gap["expected_bars"]),
                "[red]⚠ Check Required[/red]"
            )
        
        # マーケット休場
        for gap in self.market_closed_periods[:5]:  # 最初の5個
            gap_type = "WEEKEND" if gap["is_weekend"] else "MARKET CLOSED"
            gap_table.add_row(
                f"[yellow]{gap_type}[/yellow]",
                format_timestamp(gap["start"]),
                format_timestamp(gap["end"]),
                gap["duration_str"],
                str(gap["expected_bars"]),
                "[green]✓ Expected[/green]"
            )
        
        layout["gaps"].update(gap_table)
        
        # タイムライン視覚化
        timeline = self._create_timeline_visualization()
        layout["timeline"].update(Panel(timeline, title="Gap Timeline", border_style="blue"))
        
        return layout
    
    def _create_timeline_visualization(self) -> str:
        """タイムラインの視覚化を作成"""
        if not self.data_gaps and not self.market_closed_periods:
            return "No gaps to visualize"
        
        # 簡単なASCIIタイムライン
        timeline = []
        timeline.append("Data Coverage Timeline (█ = Data, ░ = Market Closed, ✗ = Missing):")
        timeline.append("")
        
        # 最近の10日間を表示
        all_gaps = sorted(self.data_gaps + self.market_closed_periods, key=lambda x: x["start"])
        
        if all_gaps:
            # 日ごとにグループ化
            days = {}
            for gap in all_gaps[-20:]:  # 最後の20個のギャップ
                day = gap["start"].date()
                if day not in days:
                    days[day] = []
                days[day].append(gap)
            
            for day in sorted(days.keys())[-10:]:  # 最後の10日
                day_str = day.strftime("%Y-%m-%d")
                gaps_in_day = days[day]
                
                # その日のギャップの種類を判定
                has_data_gap = any(g in self.data_gaps for g in gaps_in_day)
                has_weekend = any(g["is_weekend"] for g in gaps_in_day)
                
                if has_data_gap:
                    timeline.append(f"{day_str}: {'█' * 20}{'✗' * 5}{'█' * 15} [red]Data Gap[/red]")
                elif has_weekend:
                    timeline.append(f"{day_str}: {'█' * 20}{'░' * 20} [yellow]Weekend[/yellow]")
                else:
                    timeline.append(f"{day_str}: {'█' * 35}{'░' * 5} [green]Normal[/green]")
        
        return "\n".join(timeline)

def main():
    """メインテスト関数"""
    print_section("Missing Data Detection Test")
    
    # 設定
    symbol = "EURJPY"
    timeframe = "M5"  # 5分足
    days_back = 14  # 2週間分（週末を含む）
    
    print_info(f"Symbol: {symbol}")
    print_info(f"Timeframe: {timeframe}")
    print_info(f"Period: Last {days_back} days (including weekends)")
    
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
        fetcher = HistoricalDataFetcher(
            mt5_client=mt5_client,
            config={}
        )
        
        # 接続
        print_info("Connecting to MT5...")
        if not fetcher.connect():
            print_error("Failed to connect to MT5")
            return
        
        print_success("Connected to MT5 successfully!")
        
        # 日付範囲（週末を含む）
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # データ取得
        print_info(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        
        result = fetcher.fetch_ohlc_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if result is not None:
            df = result.collect()
            print_success(f"Data fetched: {len(df)} records")
            
            # 欠損データ検出器を作成
            detector = MissingDataDetector(symbol, timeframe)
            
            # データ分析
            print_info("Analyzing data for gaps and missing periods...")
            stats = detector.analyze_data(df)
            
            # 統計情報を表示
            print_section("Analysis Statistics")
            
            stats_table = Table(show_header=False, box=None)
            stats_table.add_column("Metric", style="yellow")
            stats_table.add_column("Value", justify="right")
            
            stats_table.add_row("Total Records", str(stats["total_records"]))
            stats_table.add_row("Time Range", stats["time_range"])
            stats_table.add_row("Expected Interval", f"{stats['expected_interval']}s")
            stats_table.add_row("Total Gaps", str(stats["gaps_found"]))
            stats_table.add_row("Weekend Gaps", f"[yellow]{stats['weekend_gaps']}[/yellow]")
            stats_table.add_row("Market Closed", f"[yellow]{stats['market_closed_gaps']}[/yellow]")
            stats_table.add_row("Data Gaps", f"[red]{stats['data_gaps']}[/red]")
            
            console.print(Panel(stats_table, title="Statistics", border_style="blue"))
            
            # ギャップの視覚化
            print_section("Gap Visualization")
            layout = detector.create_gap_visualization()
            console.print(layout)
            
            # 欠損データ検出機能を使用
            print_section("Using Built-in Missing Period Detection")
            
            missing_periods = fetcher.detect_missing_periods(df, timeframe)
            
            if missing_periods:
                print_warning(f"Found {len(missing_periods)} missing periods using built-in detection")
                
                # 最初の5つを表示
                for i, period in enumerate(missing_periods[:5], 1):
                    print_info(f"  {i}. {period['start']} to {period['end']} ({period['missing_count']} bars)")
                    if period.get('is_market_closed'):
                        print_info(f"     → Market was closed")
                    elif period.get('is_gap'):
                        print_info(f"     → Potential data gap")
            else:
                print_success("No missing periods detected by built-in function")
            
            # データ品質スコア
            print_section("Data Quality Score")
            
            total_expected_bars = (end_date - start_date).total_seconds() / stats["expected_interval"]
            actual_bars = stats["total_records"]
            coverage_ratio = actual_bars / total_expected_bars * 100
            
            # 品質スコアを計算
            quality_score = 100
            quality_score -= stats["data_gaps"] * 10  # データギャップごとに-10点
            quality_score = max(0, quality_score)
            
            quality_panel = f"""
[bold]Data Quality Assessment[/bold]

Coverage Ratio: {coverage_ratio:.1f}%
Quality Score: {quality_score}/100

Rating: {self._get_quality_rating(quality_score)}

Recommendations:
"""
            
            if stats["data_gaps"] > 0:
                quality_panel += "• [red]Investigate data gaps - possible connection issues[/red]\n"
            
            if coverage_ratio < 70:
                quality_panel += "• [yellow]Low coverage - consider checking data source[/yellow]\n"
            
            if quality_score >= 90:
                quality_panel += "• [green]Excellent data quality - suitable for analysis[/green]\n"
            
            console.print(Panel(quality_panel, title="Quality Assessment", border_style="green"))
            
        else:
            print_error("Failed to fetch data")
            
    except Exception as e:
        print_error(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'fetcher' in locals():
            fetcher.disconnect()
            print_info("Disconnected from MT5")
    
    def _get_quality_rating(self, score: int) -> str:
        """品質評価を取得"""
        if score >= 90:
            return "[green]★★★★★ Excellent[/green]"
        elif score >= 70:
            return "[green]★★★★☆ Good[/green]"
        elif score >= 50:
            return "[yellow]★★★☆☆ Fair[/yellow]"
        elif score >= 30:
            return "[yellow]★★☆☆☆ Poor[/yellow]"
        else:
            return "[red]★☆☆☆☆ Very Poor[/red]"

if __name__ == "__main__":
    detector = MissingDataDetector("", "")  # ダミーインスタンス
    detector._get_quality_rating = lambda self, score: (
        "[green]★★★★★ Excellent[/green]" if score >= 90 else
        "[green]★★★★☆ Good[/green]" if score >= 70 else
        "[yellow]★★★☆☆ Fair[/yellow]" if score >= 50 else
        "[yellow]★★☆☆☆ Poor[/yellow]" if score >= 30 else
        "[red]★☆☆☆☆ Very Poor[/red]"
    )
    main()