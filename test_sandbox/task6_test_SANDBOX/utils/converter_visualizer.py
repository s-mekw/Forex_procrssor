"""
変換プロセス可視化ヘルパー - TickToBarConverterの動作をアニメーション表示
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import time
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.align import Align
from collections import deque
import random

console = Console()

class ConverterVisualizer:
    """TickToBarConverter動作の可視化クラス"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.tick_buffer = deque(maxlen=20)
        self.bar_buffer = deque(maxlen=10)
        self.animation_frame = 0
        self.conversion_count = 0
        
    def create_tick_stream_animation(self, tick_count: int) -> str:
        """ティックストリームのアニメーション"""
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        current_frame = frames[self.animation_frame % len(frames)]
        self.animation_frame += 1
        
        stream = []
        stream.append(f"[cyan]Tick Stream {current_frame}[/cyan]")
        stream.append("")
        
        # ティックの流れを表現
        tick_flow = "→ " * min(tick_count % 10, 10)
        stream.append(f"[blue]{'●' * min(tick_count, 50)}[/blue] {tick_flow}")
        stream.append(f"Total: {tick_count} ticks received")
        
        return "\n".join(stream)
    
    def create_conversion_animation(self, is_converting: bool) -> str:
        """変換プロセスのアニメーション"""
        if not is_converting:
            return """
[dim]Converter Idle[/dim]
     ┌─────┐
     │     │
     │  ⏸  │
     │     │
     └─────┘
"""
        
        frames = [
            """
[yellow]Converting...[/yellow]
     ┌─────┐
  ●→ │ ⚙️  │ →█
     └─────┘
""",
            """
[yellow]Converting...[/yellow]
     ┌─────┐
 ●●→ │ ⚙️  │ →█
     └─────┘
""",
            """
[yellow]Converting...[/yellow]
     ┌─────┐
●●●→ │ ⚙️  │ →█
     └─────┘
"""
        ]
        
        return frames[self.animation_frame % len(frames)]
    
    def create_bar_output_animation(self, bar_count: int) -> str:
        """バー出力のアニメーション"""
        output = []
        output.append("[green]Bar Output[/green]")
        output.append("")
        
        # バーの表現
        bars = "█" * min(bar_count, 30)
        output.append(f"[green]{bars}[/green]")
        output.append(f"Total: {bar_count} bars completed")
        
        if bar_count > 0:
            output.append(f"[dim]Latest bar completed {self.conversion_count} ticks ago[/dim]")
        
        return "\n".join(output)
    
    def create_pipeline_view(self, tick_count: int, current_bar_ticks: int, 
                            completed_bars: int, is_active: bool) -> Panel:
        """パイプラインビューを作成"""
        content = []
        
        # ヘッダー
        content.append(f"[bold cyan]Symbol: {self.symbol}[/bold cyan]")
        content.append("")
        
        # ティック入力
        content.append(self.create_tick_stream_animation(tick_count))
        content.append("")
        
        # 変換プロセス
        content.append(self.create_conversion_animation(is_active))
        content.append("")
        
        # 現在のバー進捗
        if is_active and current_bar_ticks > 0:
            progress = min(current_bar_ticks / 60 * 100, 100)  # 60ティックで1バーと仮定
            bar_visual = "█" * int(progress / 10) + "░" * (10 - int(progress / 10))
            content.append(f"Current Bar Progress: [{bar_visual}] {progress:.1f}%")
            content.append(f"Ticks in current bar: {current_bar_ticks}")
            content.append("")
        
        # バー出力
        content.append(self.create_bar_output_animation(completed_bars))
        
        return Panel("\n".join(content), title="Tick → Bar Pipeline", border_style="cyan")
    
    def create_memory_usage_bar(self, used_bars: int, max_bars: int) -> str:
        """メモリ使用状況バー"""
        if max_bars == 0:
            return "[dim]Memory: Unlimited[/dim]"
        
        usage_percent = (used_bars / max_bars) * 100
        filled = int(usage_percent / 5)
        
        bar = f"[{'█' * filled}{'░' * (20 - filled)}]"
        
        if usage_percent >= 90:
            color = "red"
        elif usage_percent >= 70:
            color = "yellow"
        else:
            color = "green"
        
        return f"Memory: [{color}]{bar}[/{color}] {used_bars}/{max_bars} bars"
    
    def create_performance_metrics(self, metrics: Dict[str, Any]) -> Panel:
        """パフォーマンスメトリクスパネル"""
        lines = []
        
        # 処理速度
        ticks_per_sec = metrics.get('ticks_per_second', 0)
        if ticks_per_sec > 1000:
            speed_color = "green"
        elif ticks_per_sec > 100:
            speed_color = "yellow"
        else:
            speed_color = "red"
        
        lines.append(f"Processing Speed: [{speed_color}]{ticks_per_sec:.1f}[/{speed_color}] ticks/sec")
        
        # レイテンシー
        latency = metrics.get('avg_latency_ms', 0)
        if latency < 1:
            latency_color = "green"
        elif latency < 10:
            latency_color = "yellow"
        else:
            latency_color = "red"
        
        lines.append(f"Avg Latency: [{latency_color}]{latency:.2f}[/{latency_color}] ms")
        
        # CPU使用率（シミュレート）
        cpu_usage = metrics.get('cpu_usage', random.uniform(5, 25))
        lines.append(f"CPU Usage: {cpu_usage:.1f}%")
        
        # メモリ使用量
        memory_mb = metrics.get('memory_mb', random.uniform(50, 150))
        lines.append(f"Memory: {memory_mb:.1f} MB")
        
        return Panel("\n".join(lines), title="Performance", border_style="blue")

class AnimatedFlowDiagram:
    """アニメーション付きフロー図"""
    
    def __init__(self):
        self.frame = 0
        self.tick_positions = deque(maxlen=5)
        
    def create_flow_diagram(self, is_active: bool) -> str:
        """フロー図を作成"""
        self.frame += 1
        
        if is_active:
            # ティックの位置を更新
            if self.frame % 2 == 0:
                self.tick_positions.append(0)
            
            # 各ティックの位置を進める
            new_positions = []
            for pos in self.tick_positions:
                new_pos = pos + 1
                if new_pos <= 10:
                    new_positions.append(new_pos)
            self.tick_positions = deque(new_positions, maxlen=5)
        
        # フロー図の構築
        lines = []
        lines.append("  [bold]Data Flow Diagram[/bold]")
        lines.append("")
        
        # ティックソース
        lines.append("  ┌──────────┐")
        lines.append("  │  MT5 API │")
        lines.append("  └─────┬────┘")
        lines.append("        │")
        
        # ティックの流れ
        for i in range(3):
            if i in [p // 4 for p in self.tick_positions]:
                lines.append("        ●")
            else:
                lines.append("        │")
        
        # コンバーター
        lines.append("        ↓")
        lines.append("  ┌──────────┐")
        
        if is_active and self.frame % 4 < 2:
            lines.append("  │[yellow]Converter[/yellow]│")
        else:
            lines.append("  │Converter │")
        
        lines.append("  └─────┬────┘")
        lines.append("        │")
        
        # バー出力
        if is_active and any(p > 8 for p in self.tick_positions):
            lines.append("        █")
        else:
            lines.append("        │")
        
        lines.append("        ↓")
        lines.append("  ┌──────────┐")
        lines.append("  │   Bars   │")
        lines.append("  └──────────┘")
        
        return "\n".join(lines)

def create_visual_tick_to_bar(tick: Any, current_bar: Any) -> str:
    """ティックからバーへの変換を視覚的に表現"""
    visual = []
    
    # ティック情報
    visual.append("[cyan]New Tick:[/cyan]")
    visual.append(f"  Price: {float(tick.bid):.5f}")
    visual.append(f"  Time: {tick.timestamp.strftime('%H:%M:%S.%f')[:-3]}")
    visual.append("")
    visual.append("      ↓")
    visual.append("")
    
    # 現在のバー状態
    if current_bar:
        visual.append("[yellow]Current Bar Update:[/yellow]")
        visual.append(f"  O: {float(current_bar.open):.5f}")
        visual.append(f"  H: {float(current_bar.high):.5f} {'↑' if tick.bid > current_bar.high else ''}")
        visual.append(f"  L: {float(current_bar.low):.5f} {'↓' if tick.bid < current_bar.low else ''}")
        visual.append(f"  C: {float(current_bar.close):.5f} → {float(tick.bid):.5f}")
        visual.append(f"  Ticks: {current_bar.tick_count} → {current_bar.tick_count + 1}")
    else:
        visual.append("[green]New Bar Started:[/green]")
        visual.append(f"  O: {float(tick.bid):.5f}")
        visual.append(f"  H: {float(tick.bid):.5f}")
        visual.append(f"  L: {float(tick.bid):.5f}")
        visual.append(f"  C: {float(tick.bid):.5f}")
        visual.append(f"  Ticks: 1")
    
    return "\n".join(visual)

def create_gap_detection_visual(gap_seconds: float, threshold: float) -> Panel:
    """ギャップ検出の視覚化"""
    if gap_seconds <= threshold:
        status = "[green]Normal[/green]"
        color = "green"
    else:
        status = "[red]Gap Detected![/red]"
        color = "red"
    
    lines = []
    lines.append(f"Status: {status}")
    lines.append("")
    lines.append(f"Gap Duration: [{color}]{gap_seconds:.1f}[/{color}] seconds")
    lines.append(f"Threshold: {threshold:.1f} seconds")
    lines.append("")
    
    # ビジュアルバー
    if gap_seconds > 0:
        normalized = min(gap_seconds / (threshold * 2), 1.0)
        filled = int(normalized * 20)
        bar = "█" * filled + "░" * (20 - filled)
        lines.append(f"[{color}]{bar}[/{color}]")
    
    return Panel("\n".join(lines), title="Gap Detection", border_style=color)

def create_error_handling_visual(error_type: str, error_count: int) -> str:
    """エラーハンドリングの視覚化"""
    visuals = {
        "timestamp_reversal": """
[red]⚠ Timestamp Reversal Detected[/red]
  Past ← [red]●[/red] Current
        ↓
    [REJECTED]
""",
        "invalid_data": """
[red]⚠ Invalid Data Detected[/red]
  [red]✗[/red] Negative price
  [red]✗[/red] Zero volume
        ↓
    [REJECTED]
""",
        "none": """
[green]✓ All Checks Passed[/green]
  ✓ Valid timestamp
  ✓ Valid price
  ✓ Valid volume
        ↓
    [ACCEPTED]
"""
    }
    
    visual = visuals.get(error_type, visuals["none"])
    
    if error_count > 0:
        visual += f"\n[dim]Total errors handled: {error_count}[/dim]"
    
    return visual

class ProgressTracker:
    """進捗トラッカー"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.tick_times = deque(maxlen=100)
        self.bar_times = deque(maxlen=20)
        
    def add_tick(self):
        """ティック追加を記録"""
        self.tick_times.append(datetime.now())
        
    def add_bar(self):
        """バー完成を記録"""
        self.bar_times.append(datetime.now())
        
    def get_rates(self) -> Dict[str, float]:
        """レートを計算"""
        now = datetime.now()
        
        # ティックレート
        recent_ticks = [t for t in self.tick_times if (now - t).total_seconds() < 10]
        tick_rate = len(recent_ticks) / 10.0 if recent_ticks else 0
        
        # バーレート
        recent_bars = [t for t in self.bar_times if (now - t).total_seconds() < 300]
        bar_rate = len(recent_bars) / 5.0 if recent_bars else 0  # per minute
        
        return {
            'tick_rate': tick_rate,
            'bar_rate': bar_rate,
            'uptime': (now - self.start_time).total_seconds()
        }
    
    def create_rate_display(self) -> str:
        """レート表示を作成"""
        rates = self.get_rates()
        
        lines = []
        lines.append(f"Tick Rate: {rates['tick_rate']:.1f}/sec")
        lines.append(f"Bar Rate: {rates['bar_rate']:.1f}/min")
        lines.append(f"Uptime: {rates['uptime']:.0f}s")
        
        return "\n".join(lines)