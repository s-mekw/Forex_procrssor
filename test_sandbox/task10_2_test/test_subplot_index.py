"""
Plotlyのサブプロットインデックスを確認
"""
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# サブプロットを作成
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=(
        "1-1", "1-2",
        "2-1", "2-2",
        "3-1", "3-2",
        "4-1", "4-2"
    ),
    specs=[
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}]
    ]
)

# 各サブプロットにトレースを追加
positions = [
    (1, 1), (1, 2),
    (2, 1), (2, 2),
    (3, 1), (3, 2),
    (4, 1), (4, 2)
]

for i, (row, col) in enumerate(positions):
    fig.add_trace(
        go.Scatter(
            x=[1, 2, 3],
            y=[i, i+1, i+2],
            name=f"Trace at {row}-{col}"
        ),
        row=row, col=col
    )

# トレースの配置を確認
print("=== Trace Positions ===")
for i, trace in enumerate(fig.data):
    xaxis = getattr(trace, 'xaxis', None)
    yaxis = getattr(trace, 'yaxis', None)
    print(f"Trace {i}: {trace.name} -> xaxis={xaxis}, yaxis={yaxis}")

# 期待される配置
print("\n=== Expected Positions ===")
print("row=1, col=1 -> x, y")
print("row=1, col=2 -> x2, y2")
print("row=2, col=1 -> x3, y3")
print("row=2, col=2 -> x4, y4")
print("row=3, col=1 -> x5, y5")
print("row=3, col=2 -> x6, y6")
print("row=4, col=1 -> x7, y7")
print("row=4, col=2 -> x8, y8")

# HTMLに保存
fig.write_html("subplot_test.html")
print("\n✅ Saved to subplot_test.html")