"""
トレースの配置を詳細に分析
"""
import sys
from pathlib import Path
sys.path.append(str(Path('.').parent.parent))

from rci_multiframe_chart import RCIMultiframeChart
import MetaTrader5 as mt5
import json

# チャートマネージャー作成
print("Initializing chart manager...")
chart_manager = RCIMultiframeChart()

# チャート作成
print("\nCreating chart...")
fig = chart_manager.create_chart()

print("\n=== Detailed Trace Analysis ===")
print(f"Total traces: {len(fig.data)}")

# 各トレースの詳細情報を収集
trace_info = []
for i, trace in enumerate(fig.data):
    info = {
        'index': i,
        'name': trace.name,
        'type': type(trace).__name__,
        'xaxis': getattr(trace, 'xaxis', None),
        'yaxis': getattr(trace, 'yaxis', None),
    }
    
    # Candlestickトレースの場合、追加情報を取得
    if info['type'] == 'Candlestick':
        if hasattr(trace, 'x') and trace.x:
            info['data_points'] = len(trace.x)
            info['first_x'] = str(trace.x[0]) if trace.x else None
            info['last_x'] = str(trace.x[-1]) if trace.x else None
    
    trace_info.append(info)

# サブプロット別にグループ化
subplot_groups = {}
for info in trace_info:
    key = f"{info['xaxis']}, {info['yaxis']}"
    if key not in subplot_groups:
        subplot_groups[key] = []
    subplot_groups[key].append(info)

# 問題のあるサブプロットを特定
print("\n=== Subplot Analysis ===")
for subplot, traces in subplot_groups.items():
    print(f"\nSubplot ({subplot}):")
    has_candlestick = False
    has_scatter = False
    
    for trace in traces:
        print(f"  - {trace['type']:12s}: {trace['name']}")
        if trace['type'] == 'Candlestick':
            has_candlestick = True
        elif trace['type'] == 'Scatter':
            has_scatter = True
    
    # 問題を検出
    if has_candlestick and has_scatter:
        print(f"  ⚠️  WARNING: Both Candlestick and Scatter traces in same subplot!")

# 特にM5のrow=2, col=2（x4, y4）を詳しく確認
print("\n=== M5 Subwindow 1 (x4, y4) Detail ===")
for info in trace_info:
    if info['xaxis'] == 'x4' and info['yaxis'] == 'y4':
        print(f"Trace {info['index']:2d}: {info['type']:12s} - {info['name']}")
        if info['type'] == 'Candlestick':
            print(f"  ❌ PROBLEM: Candlestick should not be in RCI subplot!")
            print(f"  Data points: {info.get('data_points', 'N/A')}")

# M5のCandlestickトレースを詳しく確認
print("\n=== All Candlestick Traces ===")
for info in trace_info:
    if info['type'] == 'Candlestick':
        print(f"Trace {info['index']:2d}: {info['name']:15s} at ({info['xaxis']}, {info['yaxis']})")
        expected_axis = 'x2' if 'M5' in info['name'] else 'x'
        if info['xaxis'] != expected_axis:
            print(f"  ❌ Wrong axis! Expected {expected_axis}, got {info['xaxis']}")

# fig.dataの構造をJSON形式で出力（デバッグ用）
print("\n=== Figure Data Structure (first 2 traces) ===")
for i in range(min(2, len(fig.data))):
    trace = fig.data[i]
    print(f"\nTrace {i}: {trace.name}")
    # トレースの属性を確認
    attrs = ['xaxis', 'yaxis', 'type', 'showlegend', 'visible']
    for attr in attrs:
        if hasattr(trace, attr):
            print(f"  {attr}: {getattr(trace, attr)}")

# クリーンアップ
if mt5.initialize():
    mt5.shutdown()

print("\n✅ Analysis completed")