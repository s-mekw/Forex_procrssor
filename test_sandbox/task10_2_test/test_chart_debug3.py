"""
チャートのデバッグ - RCIデータの内容を確認
"""
import sys
from pathlib import Path
sys.path.append(str(Path('.').parent.parent))

from rci_multiframe_chart import RCIMultiframeChart
import MetaTrader5 as mt5

# チャートマネージャー作成
print("Initializing chart manager...")
chart_manager = RCIMultiframeChart()

# M5のRCIデータを直接確認
print("\n=== M5 RCI Data Analysis ===")
for period, values in chart_manager.m5_data.rci_data.items():
    print(f"\nM5 RCI[{period}]:")
    print(f"  - Total values: {len(values)}")
    
    # None値の数をカウント
    none_count = sum(1 for v in values if v is None)
    valid_count = sum(1 for v in values if v is not None)
    
    print(f"  - Valid values: {valid_count}")
    print(f"  - None values: {none_count}")
    
    # 有効な値の統計
    valid_values = [v for v in values if v is not None]
    if valid_values:
        print(f"  - Range: [{min(valid_values):.2f}, {max(valid_values):.2f}]")
        print(f"  - First 10 values: {values[:10]}")
    else:
        print(f"  - All values are None!")

# チャート作成
print("\n\nCreating chart...")
fig = chart_manager.create_chart()

# M5のサブウィンドウ1のトレースを詳しく確認
print("\n=== M5 Subwindow 1 Traces (row=2, col=2) ===")
for i, trace in enumerate(fig.data):
    xaxis = getattr(trace, 'xaxis', None)
    if xaxis == 'x4':  # row=2, col=2
        print(f"\nTrace {i}: {trace.name} ({type(trace).__name__})")
        if hasattr(trace, 'y') and trace.y is not None:
            y_list = list(trace.y)
            print(f"  - Y values count: {len(y_list)}")
            
            # None値をチェック
            none_count = sum(1 for v in y_list if v is None)
            valid_count = sum(1 for v in y_list if v is not None)
            print(f"  - Valid: {valid_count}, None: {none_count}")
            
            # 有効な値の範囲
            valid_values = [v for v in y_list if v is not None]
            if valid_values:
                print(f"  - Range: [{min(valid_values):.2f}, {max(valid_values):.2f}]")
                print(f"  - First 5 values: {y_list[:5]}")

# M1のRCIデータも確認
print("\n\n=== M1 RCI Data Analysis ===")
for period in [9, 13]:  # サブウィンドウ1のRCI期間
    if period in chart_manager.m1_data.rci_data:
        values = chart_manager.m1_data.rci_data[period]
        print(f"\nM1 RCI[{period}]:")
        print(f"  - Total values: {len(values)}")
        
        none_count = sum(1 for v in values if v is None)
        valid_count = sum(1 for v in values if v is not None)
        
        print(f"  - Valid values: {valid_count}")
        print(f"  - None values: {none_count}")
        
        valid_values = [v for v in values if v is not None]
        if valid_values:
            print(f"  - Range: [{min(valid_values):.2f}, {max(valid_values):.2f}]")

# クリーンアップ
if mt5.initialize():
    mt5.shutdown()

print("\n✅ Debug completed")