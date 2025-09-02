"""
軸の範囲設定を確認
"""
import sys
from pathlib import Path
sys.path.append(str(Path('.').parent.parent))

from rci_multiframe_chart import RCIMultiframeChart
import MetaTrader5 as mt5
import plotly.offline as pyo

# チャートマネージャー作成
print("Initializing chart manager...")
chart_manager = RCIMultiframeChart()

# チャート作成
print("\nCreating chart...")
fig = chart_manager.create_chart()

# レイアウトの詳細を確認
print("\n=== Layout Analysis ===")
layout = fig.layout

# 各軸の設定を確認
axis_configs = {
    'xaxis': 'x軸 (row=1, col=1)',
    'xaxis2': 'x軸 (row=1, col=2)',
    'xaxis3': 'x軸 (row=2, col=1)',
    'xaxis4': 'x軸 (row=2, col=2)',
    'xaxis5': 'x軸 (row=3, col=1)',
    'xaxis6': 'x軸 (row=3, col=2)',
    'xaxis7': 'x軸 (row=4, col=1)',
    'xaxis8': 'x軸 (row=4, col=2)',
    'yaxis': 'y軸 (row=1, col=1)',
    'yaxis2': 'y軸 (row=1, col=2)',
    'yaxis3': 'y軸 (row=2, col=1)',
    'yaxis4': 'y軸 (row=2, col=2)',
    'yaxis5': 'y軸 (row=3, col=1)',
    'yaxis6': 'y軸 (row=3, col=2)',
    'yaxis7': 'y軸 (row=4, col=1)',
    'yaxis8': 'y軸 (row=4, col=2)',
}

print("\n=== Y-Axis Range Settings ===")
for axis_name, description in axis_configs.items():
    if axis_name.startswith('y'):
        axis = getattr(layout, axis_name, None)
        if axis:
            axis_range = getattr(axis, 'range', None)
            axis_title = getattr(axis, 'title', None)
            if axis_title:
                title_text = getattr(axis_title, 'text', 'No title')
            else:
                title_text = 'No title'
            
            print(f"{axis_name:8s} ({description:25s}): range={axis_range}, title='{title_text}'")

# M5チャートのデータ範囲を確認
print("\n=== M5 Data Ranges ===")
m5_ohlc_trace = None
m5_rci_traces = []

for trace in fig.data:
    if trace.name == "M5 OHLC":
        m5_ohlc_trace = trace
    elif "M5 RCI" in trace.name:
        m5_rci_traces.append(trace)

if m5_ohlc_trace:
    if hasattr(m5_ohlc_trace, 'high') and m5_ohlc_trace.high:
        high_values = list(m5_ohlc_trace.high)
        low_values = list(m5_ohlc_trace.low)
        print(f"M5 OHLC price range: {min(low_values):.2f} - {max(high_values):.2f}")

for trace in m5_rci_traces:
    if hasattr(trace, 'y') and trace.y:
        y_values = [v for v in trace.y if v is not None]
        if y_values:
            print(f"{trace.name}: {min(y_values):.2f} - {max(y_values):.2f}")

# サブプロットの設定を確認
print("\n=== Subplot Configuration ===")
if hasattr(fig, '_grid_ref'):
    print(f"Grid reference: {fig._grid_ref}")

# 特定の修正を提案
print("\n=== Potential Issues ===")
yaxis4 = getattr(layout, 'yaxis4', None)
if yaxis4:
    axis_range = getattr(yaxis4, 'range', None)
    if axis_range is None:
        print("⚠️  yaxis4 (M5 RCI subplot) has no explicit range set")
    elif axis_range[0] > 100 or axis_range[1] > 200:
        print(f"⚠️  yaxis4 range seems wrong for RCI: {axis_range}")
        print("    RCI should be in range [-105, 105]")

# クリーンアップ
if mt5.initialize():
    mt5.shutdown()

print("\n✅ Analysis completed")