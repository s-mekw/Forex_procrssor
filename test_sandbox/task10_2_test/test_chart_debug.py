"""
チャートのデバッグ用スクリプト
M5のサブウィンドウ1の問題を調査
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

# トレース情報の詳細を出力
print(f"\n=== Chart Debug Information ===")
print(f"Total traces: {len(fig.data)}")
print(f"\n--- Trace Details ---")

for i, trace in enumerate(fig.data):
    xaxis = getattr(trace, 'xaxis', None)
    yaxis = getattr(trace, 'yaxis', None)
    print(f"Trace {i:2d}: {trace.name:20s} Type: {type(trace).__name__:12s} xaxis={xaxis}, yaxis={yaxis}")
    
    # サブプロットの位置を特定
    if xaxis and yaxis:
        x_num = int(xaxis[1:]) if len(xaxis) > 1 else 1
        y_num = int(yaxis[1:]) if len(yaxis) > 1 else 1
        
        # 行と列を計算（2列のレイアウト）
        col = 2 if x_num > 1 else 1
        row = (y_num + 1) // 2 if y_num > 1 else 1
        
        print(f"         -> Subplot position: row={row}, col={col}")

# HTMLファイルに保存
output_file = "debug_chart.html"
pyo.plot(fig, filename=output_file, auto_open=False)
print(f"\n✅ Chart saved to {output_file}")

# クリーンアップ
if mt5.initialize():
    mt5.shutdown()

print("\nDebug completed")