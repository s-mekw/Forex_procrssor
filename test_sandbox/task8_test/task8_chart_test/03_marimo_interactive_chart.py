"""
Marimo Interactive Chart Dashboard
BTCUSDのインタラクティブなリアルタイムチャート
使用方法: marimo edit 03_marimo_interactive_chart.py
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")

@app.cell
def __():
    import marimo as mo
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parents[3]))
    
    import polars as pl
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import MetaTrader5 as mt5
    from datetime import datetime, timedelta
    import asyncio
    from typing import Dict, List, Optional
    
    # プロジェクトのインポート
    from src.data_processing.indicators import TechnicalIndicatorEngine
    from src.mt5_data_acquisition.tick_to_bar import TickToBarConverter, Bar
    
    return mo, pl, np, go, make_subplots, mt5, datetime, timedelta, asyncio, Dict, List, Optional, TechnicalIndicatorEngine, TickToBarConverter, Path

@app.cell
def __(mo):
    mo.md("""
    # 📈 Real-time Chart with EMA Indicators
    
    Interactive dashboard for monitoring forex pairs with technical indicators
    """)
    return

@app.cell
def __(mo):
    # コントロールパネル
    symbol_select = mo.ui.dropdown(
        options=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
        value="EURUSD",
        label="Symbol"
    )
    
    timeframe_select = mo.ui.dropdown(
        options=["M1", "M5", "M15", "H1", "H4"],
        value="M1",
        label="Timeframe"
    )
    
    bars_slider = mo.ui.slider(
        start=100,
        stop=1000,
        step=100,
        value=500,
        label="Number of Bars"
    )
    
    # EMA期間設定
    ema1_input = mo.ui.number(
        value=20,
        start=5,
        stop=500,
        step=5,
        label="EMA Period 1"
    )
    
    ema2_input = mo.ui.number(
        value=50,
        start=5,
        stop=500,
        step=5,
        label="EMA Period 2"
    )
    
    ema3_input = mo.ui.number(
        value=200,
        start=5,
        stop=500,
        step=5,
        label="EMA Period 3"
    )
    
    # アクションボタン
    fetch_button = mo.ui.button(
        label="📥 Fetch Data",
        kind="success"
    )
    
    start_realtime_button = mo.ui.button(
        label="▶️ Start Real-time",
        kind="primary"
    )
    
    stop_realtime_button = mo.ui.button(
        label="⏹️ Stop Real-time",
        kind="danger"
    )
    
    control_panel = mo.vstack([
        mo.hstack([symbol_select, timeframe_select, bars_slider]),
        mo.hstack([ema1_input, ema2_input, ema3_input]),
        mo.hstack([fetch_button, start_realtime_button, stop_realtime_button])
    ])
    
    mo.md("## ⚙️ Control Panel")
    control_panel
    
    return symbol_select, timeframe_select, bars_slider, ema1_input, ema2_input, ema3_input, fetch_button, start_realtime_button, stop_realtime_button, control_panel

@app.cell
def __(mo, mt5, symbol_select, timeframe_select, bars_slider, ema1_input, ema2_input, ema3_input, fetch_button, pl, np, datetime, TechnicalIndicatorEngine):
    # データ管理
    ohlc_data = mo.state(None)
    ema_data = mo.state({})
    realtime_enabled = mo.state(False)
    last_update = mo.state(None)
    
    def fetch_mt5_data():
        """MT5からデータを取得"""
        if not mt5.initialize():
            return None, "MT5 initialization failed"
        
        symbol = symbol_select.value
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4
        }
        
        mt5_timeframe = timeframe_map.get(timeframe_select.value, mt5.TIMEFRAME_M1)
        
        # シンボル確認
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return None, f"Symbol {symbol} not found"
        
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)
        
        # データ取得
        rates = mt5.copy_rates_from_pos(
            symbol,
            mt5_timeframe,
            0,
            bars_slider.value
        )
        
        if rates is None or len(rates) == 0:
            return None, "Failed to fetch data"
        
        df = pl.DataFrame({
            "time": [datetime.fromtimestamp(r['time']) for r in rates],
            "open": np.array([r['open'] for r in rates], dtype=np.float32),
            "high": np.array([r['high'] for r in rates], dtype=np.float32),
            "low": np.array([r['low'] for r in rates], dtype=np.float32),
            "close": np.array([r['close'] for r in rates], dtype=np.float32),
            "volume": np.array([r['tick_volume'] for r in rates], dtype=np.float32)
        })
        
        return df, None
    
    def calculate_ema(df):
        """EMAを計算"""
        if df is None or df.is_empty():
            return {}
        
        periods = [ema1_input.value, ema2_input.value, ema3_input.value]
        periods = [p for p in periods if p > 0]  # 0以上の値のみ
        
        engine = TechnicalIndicatorEngine(ema_periods=periods)
        df_with_ema = engine.calculate_ema(df, price_column="close")
        
        ema_values = {}
        for period in periods:
            col_name = f"ema_{period}"
            if col_name in df_with_ema.columns:
                ema_values[period] = df_with_ema[col_name].to_list()
        
        return ema_values
    
    # データ取得処理
    if fetch_button.value:
        df, error = fetch_mt5_data()
        if error:
            mo.md(f"❌ Error: {error}")
        else:
            ohlc_data.set_value(df)
            ema_values = calculate_ema(df)
            ema_data.set_value(ema_values)
            last_update.set_value(datetime.now())
            mo.md(f"✅ Data fetched successfully: {len(df)} bars")
    
    return ohlc_data, ema_data, realtime_enabled, last_update, fetch_mt5_data, calculate_ema

@app.cell
def __(mo, ohlc_data, ema_data, symbol_select, timeframe_select, go, make_subplots, ema1_input, ema2_input, ema3_input):
    mo.md("## 📊 Interactive Chart")
    
    if ohlc_data.value is not None:
        df = ohlc_data.value
        ema_values = ema_data.value
        
        # チャート作成
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{symbol_select.value} - {timeframe_select.value}", "Volume")
        )
        
        # ローソク足
        fig.add_trace(
            go.Candlestick(
                x=df["time"].to_list(),
                open=df["open"].to_list(),
                high=df["high"].to_list(),
                low=df["low"].to_list(),
                close=df["close"].to_list(),
                name="OHLC",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350"
            ),
            row=1, col=1
        )
        
        # EMAライン
        ema_colors = {
            ema1_input.value: "blue",
            ema2_input.value: "green", 
            ema3_input.value: "red"
        }
        
        for period, values in ema_values.items():
            fig.add_trace(
                go.Scatter(
                    x=df["time"].to_list(),
                    y=values,
                    mode="lines",
                    name=f"EMA {period}",
                    line=dict(color=ema_colors.get(period, "gray"), width=2)
                ),
                row=1, col=1
            )
        
        # ボリューム
        colors = ["#26a69a" if c >= o else "#ef5350" 
                  for c, o in zip(df["close"].to_list(), df["open"].to_list())]
        
        fig.add_trace(
            go.Bar(
                x=df["time"].to_list(),
                y=df["volume"].to_list(),
                name="Volume",
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # レイアウト
        fig.update_layout(
            height=800,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )
        
        fig.update_xaxes(gridcolor="#333333")
        fig.update_yaxes(title_text="Price (USD)", gridcolor="#333333", row=1, col=1)
        fig.update_yaxes(title_text="Volume", gridcolor="#333333", row=2, col=1)
        
        mo.ui.plotly(fig)
    else:
        mo.md("📊 *Click 'Fetch Data' to load the chart*")
    
    return fig,

@app.cell
def __(mo, ohlc_data, ema_data, last_update, datetime):
    mo.md("## 📈 Market Statistics")
    
    if ohlc_data.value is not None:
        df = ohlc_data.value
        ema_values = ema_data.value
        
        # 最新価格
        latest_price = df["close"][-1]
        daily_high = df["high"].max()
        daily_low = df["low"].min()
        daily_range = daily_high - daily_low
        total_volume = df["volume"].sum()
        
        # 価格変動
        price_change = df["close"][-1] - df["open"][0]
        price_change_pct = (price_change / df["open"][0]) * 100
        
        # EMA情報
        ema_info = []
        for period in sorted(ema_values.keys()):
            if ema_values[period]:
                value = ema_values[period][-1]
                diff = latest_price - value
                ema_info.append({
                    "Period": f"EMA {period}",
                    "Value": f"${value:,.2f}",
                    "Difference": f"${diff:+,.2f}"
                })
        
        # 統計テーブル
        stats_data = {
            "Metric": [
                "Current Price",
                "Daily High",
                "Daily Low",
                "Daily Range",
                "Price Change",
                "Change %",
                "Total Volume",
                "Last Update"
            ],
            "Value": [
                f"${latest_price:,.2f}",
                f"${daily_high:,.2f}",
                f"${daily_low:,.2f}",
                f"${daily_range:,.2f}",
                f"${price_change:+,.2f}",
                f"{price_change_pct:+.2f}%",
                f"{total_volume:,.0f}",
                last_update.value.strftime("%H:%M:%S") if last_update.value else "N/A"
            ]
        }
        
        col1, col2 = mo.hstack([
            mo.ui.table(stats_data, label="Market Statistics"),
            mo.ui.table(ema_info, label="EMA Analysis") if ema_info else mo.md("*No EMA data*")
        ], justify="start", gap=2)
        
        mo.vstack([col1, col2])
    else:
        mo.md("*No data available*")
    
    return

@app.cell
def __(mo, ohlc_data):
    mo.md("## 📊 Candlestick Pattern Detection")
    
    if ohlc_data.value is not None:
        df = ohlc_data.value
        
        # パターン検出
        patterns = []
        
        for i in range(len(df) - 1):
            row = df.row(i, named=True)
            next_row = df.row(i + 1, named=True)
            
            body = abs(row["close"] - row["open"])
            upper_shadow = row["high"] - max(row["close"], row["open"])
            lower_shadow = min(row["close"], row["open"]) - row["low"]
            
            # Doji
            if body < (row["high"] - row["low"]) * 0.1:
                patterns.append({
                    "Time": str(row["time"]),
                    "Pattern": "Doji",
                    "Type": "Neutral"
                })
            
            # Hammer
            elif lower_shadow > body * 2 and upper_shadow < body * 0.5 and row["close"] > row["open"]:
                patterns.append({
                    "Time": str(row["time"]),
                    "Pattern": "Hammer",
                    "Type": "Bullish"
                })
            
            # Shooting Star
            elif upper_shadow > body * 2 and lower_shadow < body * 0.5 and row["close"] < row["open"]:
                patterns.append({
                    "Time": str(row["time"]),
                    "Pattern": "Shooting Star",
                    "Type": "Bearish"
                })
        
        if patterns:
            # 最新10件のパターンを表示
            mo.ui.table(patterns[-10:], label="Recent Candlestick Patterns")
        else:
            mo.md("*No significant patterns detected*")
    else:
        mo.md("*No data available for pattern detection*")
    
    return

@app.cell
def __(mo, start_realtime_button, stop_realtime_button, realtime_enabled, mt5, symbol_select, datetime, asyncio):
    mo.md("## ⚡ Real-time Updates")
    
    # リアルタイム更新のトグル
    if start_realtime_button.value:
        realtime_enabled.set_value(True)
        mo.md("✅ **Real-time updates started**")
    
    if stop_realtime_button.value:
        realtime_enabled.set_value(False)
        mo.md("🛑 **Real-time updates stopped**")
    
    # リアルタイムティック表示
    if realtime_enabled.value:
        tick = mt5.symbol_info_tick(symbol_select.value)
        if tick:
            tick_info = {
                "Time": datetime.fromtimestamp(tick.time).strftime("%H:%M:%S"),
                "Bid": f"${tick.bid:,.2f}",
                "Ask": f"${tick.ask:,.2f}",
                "Spread": f"{(tick.ask - tick.bid) * 10000:.1f} pips"
            }
            
            mo.ui.table([tick_info], label="Latest Tick")
        else:
            mo.md("*Waiting for tick data...*")
    else:
        mo.md("*Real-time updates disabled*")
    
    return

@app.cell
def __(mo):
    mo.md("""
    ## 📖 Instructions
    
    1. **Select Symbol**: Choose the trading symbol (BTCUSD by default)
    2. **Select Timeframe**: Choose the chart timeframe (M1 = 1 minute)
    3. **Set EMA Periods**: Configure up to 3 EMA periods
    4. **Fetch Data**: Click to load historical data
    5. **Start Real-time**: Enable live tick updates
    
    ### Features
    - 📊 Interactive candlestick chart with volume
    - 📈 Multiple EMA indicators with custom periods
    - 📉 Market statistics and analysis
    - 🕯️ Candlestick pattern detection
    - ⚡ Real-time tick updates
    
    Note: BTCUSD may not be available on all MT5 brokers. EURUSD is used as default.
    
    ### Color Legend
    - **Blue line**: EMA Period 1 (default: 20)
    - **Green line**: EMA Period 2 (default: 50)
    - **Red line**: EMA Period 3 (default: 200)
    - **Green candles**: Bullish (close > open)
    - **Red candles**: Bearish (close < open)
    """)
    return

@app.cell
def __():
    import marimo as mo
    return mo,

if __name__ == "__main__":
    app.run()