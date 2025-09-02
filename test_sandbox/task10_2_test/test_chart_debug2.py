"""
チャートのデバッグ用スクリプト（改良版）
サブプロットの位置を正しく計算
"""
import sys
from pathlib import Path
sys.path.append(str(Path('.').parent.parent))

from rci_multiframe_chart import RCIMultiframeChart
import MetaTrader5 as mt5
import plotly.offline as pyo

# サブプロットのインデックスマッピング
# 4行×2列のレイアウト
subplot_mapping = {
    'x': (1, 1), 'x2': (1, 2),
    'x3': (2, 1), 'x4': (2, 2),
    'x5': (3, 1), 'x6': (3, 2),
    'x7': (4, 1), 'x8': (4, 2),
    'y': (1, 1), 'y2': (1, 2),
    'y3': (2, 1), 'y4': (2, 2),
    'y5': (3, 1), 'y6': (3, 2),
    'y7': (4, 1), 'y8': (4, 2),
}

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
print(f"{'No':>3} {'Name':25} {'Type':12} {'xaxis':6} {'yaxis':6} {'Position'}")
print("-" * 70)

# エラーのあるトレースを記録
issues = []

for i, trace in enumerate(fig.data):
    xaxis = getattr(trace, 'xaxis', None)
    yaxis = getattr(trace, 'yaxis', None)
    
    # 正しい位置を取得
    if xaxis in subplot_mapping:
        row, col = subplot_mapping[xaxis]
    else:
        row, col = 1, 1  # デフォルト
    
    # トレース名から期待される位置を判定
    expected_col = 1 if trace.name.startswith('M1') else 2
    
    # 位置が正しいかチェック
    is_correct = col == expected_col
    status = "✅" if is_correct else "❌"
    
    print(f"{i:3} {trace.name:25} {type(trace).__name__:12} {xaxis:6} {yaxis:6} row={row}, col={col} {status}")
    
    if not is_correct:
        issues.append(f"Trace {i}: {trace.name} is at row={row}, col={col} but should be at col={expected_col}")

# 問題のあるトレースを強調表示
if issues:
    print(f"\n⚠️  ISSUES FOUND:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print(f"\n✅ All traces are correctly positioned")

# 特定のサブプロットの内容を確認
print(f"\n--- Subplot Contents ---")
subplot_contents = {}
for i, trace in enumerate(fig.data):
    xaxis = getattr(trace, 'xaxis', None)
    if xaxis in subplot_mapping:
        row, col = subplot_mapping[xaxis]
        key = f"row={row}, col={col}"
        if key not in subplot_contents:
            subplot_contents[key] = []
        subplot_contents[key].append(f"{trace.name} ({type(trace).__name__})")

for key in sorted(subplot_contents.keys()):
    print(f"\n{key}:")
    for content in subplot_contents[key]:
        print(f"  - {content}")

# 問題のあるサブプロット（M5のrow=2, col=2）を詳しく確認
print(f"\n--- Detailed Analysis of M5 Subwindow 1 (row=2, col=2) ---")
for i, trace in enumerate(fig.data):
    xaxis = getattr(trace, 'xaxis', None)
    if xaxis == 'x4':  # row=2, col=2
        print(f"  Trace {i}: {trace.name} ({type(trace).__name__})")
        if hasattr(trace, 'y') and trace.y is not None:
            print(f"    - Y values: {len(trace.y)} points, range: [{min(trace.y):.2f}, {max(trace.y):.2f}]")

# HTMLファイルに保存
output_file = "debug_chart2.html"
pyo.plot(fig, filename=output_file, auto_open=False)
print(f"\n✅ Chart saved to {output_file}")

# クリーンアップ
if mt5.initialize():
    mt5.shutdown()

print("\nDebug completed")