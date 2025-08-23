"""
Task7テスト共通表示ヘルパー関数

このモジュールには、Task7のテストファイル群で共通して使用される
表示・可視化・ユーティリティ関数が含まれています。
"""

import time
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich import box
import polars as pl

console = Console()

class PerformanceTracker:
    """パフォーマンス追跡クラス"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_snapshots: List[float] = []
        self.processing_times: List[float] = []
        self.stage_times: Dict[str, float] = {}
        
    def start_tracking(self):
        """追跡開始"""
        self.start_time = time.time()
        self.take_memory_snapshot()
        
    def end_tracking(self):
        """追跡終了"""
        self.end_time = time.time()
        self.take_memory_snapshot()
        
    def take_memory_snapshot(self):
        """メモリスナップショットを取得"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        self.memory_snapshots.append(memory_mb)
        
    def start_stage(self, stage_name: str):
        """段階開始"""
        self.stage_times[stage_name] = time.time()
        
    def end_stage(self, stage_name: str):
        """段階終了"""
        if stage_name in self.stage_times:
            elapsed = time.time() - self.stage_times[stage_name]
            self.processing_times.append(elapsed)
            self.stage_times[stage_name] = elapsed
        self.take_memory_snapshot()
        
    def get_total_time(self) -> float:
        """総処理時間を取得"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
        
    def get_average_processing_time(self) -> float:
        """平均処理時間を取得"""
        if self.processing_times:
            return sum(self.processing_times) / len(self.processing_times)
        return 0.0
        
    def get_peak_memory(self) -> float:
        """ピークメモリ使用量を取得"""
        if self.memory_snapshots:
            return max(self.memory_snapshots)
        return 0.0
        
    def get_memory_trend(self) -> str:
        """メモリ使用量の傾向を取得"""
        if len(self.memory_snapshots) < 2:
            return "➡️"
        
        recent = sum(self.memory_snapshots[-3:]) / len(self.memory_snapshots[-3:])
        initial = sum(self.memory_snapshots[:3]) / min(3, len(self.memory_snapshots))
        
        if recent > initial * 1.2:
            return "📈"
        elif recent < initial * 0.8:
            return "📉"
        else:
            return "➡️"

def get_system_info() -> Dict[str, Any]:
    """システム情報を取得"""
    process = psutil.Process()
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent()
    
    system_memory = psutil.virtual_memory()
    
    return {
        'process_memory_mb': memory_info.rss / (1024 * 1024),
        'process_memory_percent': process.memory_percent(),
        'cpu_percent': cpu_percent,
        'system_memory_total_mb': system_memory.total / (1024 * 1024),
        'system_memory_available_mb': system_memory.available / (1024 * 1024),
        'system_memory_percent': system_memory.percent
    }

def format_number(value: Union[int, float], precision: int = 2) -> str:
    """数値を適切にフォーマット"""
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return f"{int(value):,}"
    else:
        return f"{value:,.{precision}f}"

def format_bytes(bytes_value: int) -> str:
    """バイト数を適切な単位でフォーマット"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f}TB"

def format_duration(seconds: float) -> str:
    """秒数を適切な単位でフォーマット"""
    if seconds < 1.0:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60.0:
        return f"{seconds:.3f}s"
    elif seconds < 3600.0:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes}m"

def create_status_indicator(value: float, thresholds: Dict[str, float], 
                          indicators: Dict[str, str] = None) -> str:
    """値に基づくステータスインジケーターを作成"""
    if indicators is None:
        indicators = {"good": "✅", "warning": "⚠️", "critical": "❌"}
    
    if "critical" in thresholds and value >= thresholds["critical"]:
        return indicators["critical"]
    elif "warning" in thresholds and value >= thresholds["warning"]:
        return indicators["warning"]
    else:
        return indicators["good"]

def create_performance_table(tracker: PerformanceTracker, title: str = "パフォーマンス統計") -> Table:
    """パフォーマンス統計テーブルを作成"""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("項目", style="cyan")
    table.add_column("値", style="green")
    table.add_column("状態", style="yellow")
    
    total_time = tracker.get_total_time()
    avg_time = tracker.get_average_processing_time()
    peak_memory = tracker.get_peak_memory()
    memory_trend = tracker.get_memory_trend()
    
    # 処理時間
    table.add_row("総処理時間", format_duration(total_time), 
                  create_status_indicator(total_time, {"warning": 60, "critical": 300}))
    
    # 平均処理時間
    table.add_row("平均処理時間", format_duration(avg_time),
                  create_status_indicator(avg_time, {"warning": 1.0, "critical": 5.0}))
    
    # ピークメモリ
    table.add_row("ピークメモリ", f"{peak_memory:.2f}MB",
                  create_status_indicator(peak_memory, {"warning": 200, "critical": 500}))
    
    # メモリ傾向
    table.add_row("メモリ傾向", memory_trend, "")
    
    # 処理段階数
    table.add_row("処理段階数", str(len(tracker.processing_times)), "")
    
    return table

def create_system_status_table(title: str = "システム状態") -> Table:
    """システム状態テーブルを作成"""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("項目", style="cyan")
    table.add_column("値", style="green")
    table.add_column("状態", style="yellow")
    
    info = get_system_info()
    
    # プロセスメモリ
    table.add_row("プロセスメモリ", f"{info['process_memory_mb']:.2f}MB",
                  create_status_indicator(info['process_memory_mb'], {"warning": 200, "critical": 500}))
    
    # CPU使用率
    table.add_row("CPU使用率", f"{info['cpu_percent']:.1f}%",
                  create_status_indicator(info['cpu_percent'], {"warning": 70, "critical": 90}))
    
    # システムメモリ
    table.add_row("システムメモリ", f"{info['system_memory_percent']:.1f}%",
                  create_status_indicator(info['system_memory_percent'], {"warning": 80, "critical": 95}))
    
    # 利用可能メモリ
    table.add_row("利用可能メモリ", f"{info['system_memory_available_mb']:.0f}MB", "")
    
    return table

def create_dataframe_info_table(df: pl.DataFrame, title: str = "データフレーム情報") -> Table:
    """DataFrameの情報テーブルを作成"""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("項目", style="cyan")
    table.add_column("値", style="green")
    
    if df.is_empty():
        table.add_row("状態", "空のデータフレーム")
        return table
    
    # 基本情報
    table.add_row("行数", format_number(df.height))
    table.add_row("列数", format_number(df.width))
    table.add_row("メモリサイズ", format_bytes(df.estimated_size()))
    
    # 列の型情報
    dtypes_str = ", ".join([f"{name}: {dtype}" for name, dtype in df.dtypes])
    if len(dtypes_str) > 50:
        dtypes_str = dtypes_str[:47] + "..."
    table.add_row("データ型", dtypes_str)
    
    # null値の情報
    null_counts = df.null_count()
    total_nulls = sum(null_counts.row(0))
    table.add_row("null値数", format_number(total_nulls))
    
    return table

def create_processing_summary_panel(
    input_rows: int, 
    output_rows: int, 
    processing_time: float,
    memory_before: float,
    memory_after: float,
    title: str = "処理サマリー"
) -> Panel:
    """処理サマリーパネルを作成"""
    
    # 処理効率
    processing_rate = input_rows / processing_time if processing_time > 0 else 0
    memory_change = memory_after - memory_before
    
    content = Text()
    content.append(f"📊 入力: ", style="cyan")
    content.append(f"{format_number(input_rows)}行\n", style="green")
    
    content.append(f"📤 出力: ", style="cyan")
    content.append(f"{format_number(output_rows)}行\n", style="green")
    
    content.append(f"⏱️  処理時間: ", style="cyan")
    content.append(f"{format_duration(processing_time)}\n", style="green")
    
    content.append(f"⚡ 処理速度: ", style="cyan")
    content.append(f"{format_number(processing_rate, 1)}行/秒\n", style="green")
    
    content.append(f"🧠 メモリ変化: ", style="cyan")
    memory_color = "red" if memory_change > 0 else "green"
    content.append(f"{memory_change:+.2f}MB", style=memory_color)
    
    return Panel(content, title=title, box=box.ROUNDED)

def print_section_header(title: str, emoji: str = "🔹"):
    """セクションヘッダーを表示"""
    console.print(f"\n{emoji} [bold blue]{title}[/bold blue]")
    console.print("─" * (len(title) + 3))

def print_success(message: str):
    """成功メッセージを表示"""
    console.print(f"[green]✅ {message}[/green]")

def print_warning(message: str):
    """警告メッセージを表示"""
    console.print(f"[yellow]⚠️ {message}[/yellow]")

def print_error(message: str):
    """エラーメッセージを表示"""
    console.print(f"[red]❌ {message}[/red]")

def print_info(message: str):
    """情報メッセージを表示"""
    console.print(f"[blue]ℹ️ {message}[/blue]")

def create_progress_bar(description: str, total: int) -> Progress:
    """進行状況バーを作成"""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    )

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全な除算（ゼロ除算を回避）"""
    return numerator / denominator if denominator != 0 else default

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """パーセント変化を計算"""
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

def format_percentage_change(change: float) -> str:
    """パーセント変化をフォーマット"""
    sign = "+" if change > 0 else ""
    return f"{sign}{change:.2f}%"

class TimestampFormatter:
    """タイムスタンプフォーマッター"""
    
    @staticmethod
    def format_datetime(dt: datetime, format_type: str = "full") -> str:
        """日時をフォーマット"""
        if format_type == "time_only":
            return dt.strftime("%H:%M:%S")
        elif format_type == "date_only":
            return dt.strftime("%Y-%m-%d")
        elif format_type == "compact":
            return dt.strftime("%m/%d %H:%M")
        else:  # full
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def format_elapsed(start_time: datetime, end_time: datetime = None) -> str:
        """経過時間をフォーマット"""
        if end_time is None:
            end_time = datetime.now()
        
        elapsed = end_time - start_time
        total_seconds = elapsed.total_seconds()
        
        return format_duration(total_seconds)

def create_comparison_table(
    data: List[Dict[str, Any]], 
    title: str = "比較結果"
) -> Table:
    """比較テーブルを作成"""
    if not data:
        return Table(title=title)
    
    table = Table(title=title, box=box.ROUNDED)
    
    # 列を動的に作成
    keys = list(data[0].keys())
    for key in keys:
        table.add_column(key, style="cyan" if key == keys[0] else "green")
    
    # データを追加
    for row_data in data:
        row_values = [str(row_data.get(key, "-")) for key in keys]
        table.add_row(*row_values)
    
    return table

def analyze_dataframe_quality(df: pl.DataFrame) -> Dict[str, Any]:
    """DataFrameの品質分析"""
    if df.is_empty():
        return {"error": "Empty DataFrame"}
    
    analysis = {
        "total_rows": df.height,
        "total_columns": df.width,
        "memory_usage_bytes": df.estimated_size(),
        "null_percentage": 0.0,
        "duplicate_rows": 0,
        "column_types": df.dtypes,
        "quality_score": 100.0
    }
    
    # null値の割合
    if df.height > 0:
        null_counts = df.null_count()
        total_cells = df.height * df.width
        total_nulls = sum(null_counts.row(0))
        analysis["null_percentage"] = (total_nulls / total_cells) * 100
    
    # 重複行（概算）
    try:
        unique_rows = df.unique().height
        analysis["duplicate_rows"] = df.height - unique_rows
    except:
        analysis["duplicate_rows"] = 0  # エラーの場合は0とする
    
    # 品質スコア計算（簡易版）
    quality_score = 100.0
    quality_score -= analysis["null_percentage"]  # null値で減点
    quality_score -= (analysis["duplicate_rows"] / analysis["total_rows"]) * 10  # 重複で減点
    analysis["quality_score"] = max(0.0, quality_score)
    
    return analysis

def create_quality_report_table(analysis: Dict[str, Any], title: str = "データ品質レポート") -> Table:
    """データ品質レポートテーブルを作成"""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("項目", style="cyan")
    table.add_column("値", style="green")
    table.add_column("評価", style="yellow")
    
    if "error" in analysis:
        table.add_row("エラー", analysis["error"], "❌")
        return table
    
    # 基本統計
    table.add_row("総行数", format_number(analysis["total_rows"]), "")
    table.add_row("総列数", format_number(analysis["total_columns"]), "")
    table.add_row("メモリ使用量", format_bytes(analysis["memory_usage_bytes"]), "")
    
    # 品質指標
    null_status = create_status_indicator(analysis["null_percentage"], {"warning": 5, "critical": 20})
    table.add_row("null値割合", f"{analysis['null_percentage']:.2f}%", null_status)
    
    table.add_row("重複行数", format_number(analysis["duplicate_rows"]), "")
    
    quality_status = create_status_indicator(analysis["quality_score"], 
                                           {"warning": 70, "critical": 50}, 
                                           {"good": "🟢", "warning": "🟡", "critical": "🔴"})
    table.add_row("品質スコア", f"{analysis['quality_score']:.1f}/100", quality_status)
    
    return table