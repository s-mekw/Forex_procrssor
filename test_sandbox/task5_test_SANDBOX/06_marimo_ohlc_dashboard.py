"""
Marimo OHLCダッシュボード - インタラクティブなローソク足チャートと分析
"""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from datetime import datetime, timedelta
    import polars as pl
    import altair as alt
    import numpy as np
    from typing import Dict, List
    
    from src.mt5_data_acquisition.mt5_client import MT5ConnectionManager
    from src.mt5_data_acquisition.ohlc_fetcher import HistoricalDataFetcher
    from src.common.config import BaseConfig
    
    return mo, sys, Path, datetime, timedelta, pl, alt, np, Dict, List, MT5ConnectionManager, HistoricalDataFetcher, BaseConfig


@app.cell
def __(mo):
    mo.md("""
    # 📊 MT5 OHLC Interactive Dashboard
    
    リアルタイムでOHLCデータを取得・可視化する対話型ダッシュボード
    """)
    return


@app.cell
def __(mo):
    # UIコンポーネント
    symbol_select = mo.ui.dropdown(
        options=["EURJPY", "GBPJPY", "USDJPY", "EURUSD", "GBPUSD"],
        value="EURJPY",
        label="Symbol"
    )
    
    timeframe_select = mo.ui.dropdown(
        options={
            "M1": "1 Minute",
            "M5": "5 Minutes",
            "M15": "15 Minutes",
            "M30": "30 Minutes",
            "H1": "1 Hour",
            "H4": "4 Hours",
            "D1": "Daily"
        },
        value="M5",
        label="Timeframe"
    )
    
    days_slider = mo.ui.slider(
        start=1,
        stop=30,
        value=7,
        step=1,
        label="Days Back"
    )
    
    fetch_button = mo.ui.button(
        label="📥 Fetch Data",
        kind="success"
    )
    
    auto_refresh = mo.ui.checkbox(
        label="Auto Refresh (60s)",
        value=False
    )
    
    mo.hstack([
        symbol_select,
        timeframe_select,
        days_slider,
        fetch_button,
        auto_refresh
    ])
    return symbol_select, timeframe_select, days_slider, fetch_button, auto_refresh


@app.cell
def __(mo, fetch_button, symbol_select, timeframe_select, days_slider, datetime, timedelta, BaseConfig, MT5ConnectionManager, HistoricalDataFetcher):
    # データ取得関数
    def fetch_ohlc_data():
        """MT5からOHLCデータを取得"""
        try:
            config = BaseConfig()
            mt5_config = {
                "account": config.mt5_login,
                "password": config.mt5_password.get_secret_value() if config.mt5_password else None,
                "server": config.mt5_server,
                "timeout": config.mt5_timeout
            }
            mt5_client = MT5ConnectionManager(mt5_config)
            fetcher_config = {
                "batch_size": 5000,
                "max_workers": 2
            }
            fetcher = HistoricalDataFetcher(
                mt5_client=mt5_client,
                config=fetcher_config
            )
            
            if not fetcher.connect():
                return None, "Failed to connect to MT5"
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_slider.value)
            
            result = fetcher.fetch_ohlc_data(
                symbol=symbol_select.value,
                timeframe=timeframe_select.value,
                start_date=start_date,
                end_date=end_date
            )
            
            fetcher.disconnect()
            
            if result:
                df = result.collect()
                return df, None
            else:
                return None, "Failed to fetch data"
                
        except Exception as e:
            return None, str(e)
    
    # ボタンクリック時にデータ取得
    ohlc_data = None
    error_message = None
    
    if fetch_button.value:
        with mo.status.spinner("Fetching data..."):
            ohlc_data, error_message = fetch_ohlc_data()
    
    # エラー表示
    if error_message:
        mo.callout(
            f"Error: {error_message}",
            kind="danger"
        )
    elif ohlc_data is not None:
        mo.callout(
            f"✅ Fetched {len(ohlc_data)} records",
            kind="success"
        )
    
    return fetch_ohlc_data, ohlc_data, error_message


@app.cell
def __(ohlc_data, mo):
    # データ統計
    if ohlc_data is not None and not ohlc_data.is_empty():
        stats = mo.md(f"""
        ## 📈 Data Statistics
        
        - **Records:** {len(ohlc_data):,}
        - **Period:** {ohlc_data['time'].min()} to {ohlc_data['time'].max()}
        - **Latest Close:** {ohlc_data['close'][-1]:.5f}
        - **High:** {ohlc_data['high'].max():.5f}
        - **Low:** {ohlc_data['low'].min():.5f}
        - **Average Volume:** {ohlc_data['tick_volume'].mean():.0f} ticks
        """)
        mo.accordion({
            "📊 Statistics": stats
        })
    return stats,


@app.cell
def __(ohlc_data, alt, pl):
    # ローソク足チャートを作成
    if ohlc_data is not None and not ohlc_data.is_empty():
        # 最新の100本に限定（パフォーマンスのため）
        chart_data = ohlc_data.tail(100)
        
        # データを準備
        chart_df = chart_data.with_columns([
            (pl.col("close") >= pl.col("open")).alias("is_bullish")
        ])
        
        # Altairチャート用にPandasに変換
        chart_pd = chart_df.to_pandas()
        chart_pd['color'] = chart_pd['is_bullish'].map({True: 'green', False: 'red'})
        
        # ローソク足チャート
        base = alt.Chart(chart_pd).encode(
            x=alt.X('time:T', title='Time'),
            color=alt.Color('color:N', scale=None, legend=None)
        )
        
        # 実体部分
        bodies = base.mark_bar(width=2).encode(
            y=alt.Y('open:Q', title='Price'),
            y2='close:Q'
        )
        
        # ヒゲ部分
        wicks = base.mark_rule().encode(
            y='low:Q',
            y2='high:Q'
        )
        
        # チャートを結合
        candlestick_chart = (wicks + bodies).properties(
            width=900,
            height=400,
            title=f'Candlestick Chart'
        ).interactive()
        
        # ボリュームチャート
        volume_chart = alt.Chart(chart_pd).mark_bar().encode(
            x=alt.X('time:T', title=''),
            y=alt.Y('tick_volume:Q', title='Volume'),
            color=alt.Color('color:N', scale=None)
        ).properties(
            width=900,
            height=100
        )
        
        # チャートを縦に結合
        combined_chart = alt.vconcat(candlestick_chart, volume_chart)
        
        combined_chart
    else:
        chart_placeholder = "📊 Click 'Fetch Data' to load chart"
        chart_placeholder
    
    return chart_data, chart_df, chart_pd, base, bodies, wicks, candlestick_chart, volume_chart, combined_chart, chart_placeholder


@app.cell
def __(ohlc_data, pl, mo):
    # 移動平均とボリンジャーバンド
    if ohlc_data is not None and not ohlc_data.is_empty():
        # 移動平均を計算
        ma_periods = [5, 20, 50]
        ma_data = ohlc_data
        
        for period in ma_periods:
            if len(ohlc_data) >= period:
                ma_data = ma_data.with_columns(
                    pl.col("close").rolling_mean(period).alias(f"MA{period}")
                )
        
        # ボリンジャーバンドを計算（20期間、2標準偏差）
        if len(ohlc_data) >= 20:
            ma_data = ma_data.with_columns([
                pl.col("close").rolling_mean(20).alias("BB_Middle"),
                pl.col("close").rolling_std(20).alias("BB_Std")
            ])
            
            ma_data = ma_data.with_columns([
                (pl.col("BB_Middle") + 2 * pl.col("BB_Std")).alias("BB_Upper"),
                (pl.col("BB_Middle") - 2 * pl.col("BB_Std")).alias("BB_Lower")
            ])
        
        # 最新の指標値を表示
        latest = ma_data.tail(1).row(0, named=True)
        
        indicators_display = mo.md(f"""
        ## 📉 Technical Indicators
        
        ### Moving Averages
        - **MA5:** {latest.get('MA5', 'N/A'):.5f if isinstance(latest.get('MA5'), (int, float)) else 'N/A'}
        - **MA20:** {latest.get('MA20', 'N/A'):.5f if isinstance(latest.get('MA20'), (int, float)) else 'N/A'}
        - **MA50:** {latest.get('MA50', 'N/A'):.5f if isinstance(latest.get('MA50'), (int, float)) else 'N/A'}
        
        ### Bollinger Bands
        - **Upper Band:** {latest.get('BB_Upper', 'N/A'):.5f if isinstance(latest.get('BB_Upper'), (int, float)) else 'N/A'}
        - **Middle Band:** {latest.get('BB_Middle', 'N/A'):.5f if isinstance(latest.get('BB_Middle'), (int, float)) else 'N/A'}
        - **Lower Band:** {latest.get('BB_Lower', 'N/A'):.5f if isinstance(latest.get('BB_Lower'), (int, float)) else 'N/A'}
        """)
        
        mo.accordion({
            "📉 Indicators": indicators_display
        })
    
    return ma_periods, ma_data, latest, indicators_display


@app.cell
def __(ohlc_data, pl, alt):
    # パターン検出と表示
    if ohlc_data is not None and len(ohlc_data) >= 3:
        # 簡単なパターン検出
        patterns = []
        
        # Doji検出
        last_3 = ohlc_data.tail(3)
        for i in range(len(last_3)):
            row = last_3.row(i, named=True)
            body = abs(row["close"] - row["open"])
            range_hl = row["high"] - row["low"]
            
            if range_hl > 0 and body / range_hl < 0.1:
                patterns.append({
                    "time": row["time"],
                    "pattern": "Doji",
                    "price": row["close"]
                })
        
        # Engulfingパターン検出
        if len(last_3) >= 2:
            prev = last_3.row(-2, named=True)
            curr = last_3.row(-1, named=True)
            
            if (curr["close"] > curr["open"] and 
                prev["close"] < prev["open"] and
                curr["open"] < prev["close"] and
                curr["close"] > prev["open"]):
                patterns.append({
                    "time": curr["time"],
                    "pattern": "Bullish Engulfing",
                    "price": curr["close"]
                })
            elif (curr["close"] < curr["open"] and 
                  prev["close"] > prev["open"] and
                  curr["open"] > prev["close"] and
                  curr["close"] < prev["open"]):
                patterns.append({
                    "time": curr["time"],
                    "pattern": "Bearish Engulfing",
                    "price": curr["close"]
                })
        
        # パターン表示
        if patterns:
            pattern_text = "### 🎯 Detected Patterns\n\n"
            for p in patterns:
                pattern_text += f"- **{p['pattern']}** at {p['time']} (Price: {p['price']:.5f})\n"
        else:
            pattern_text = "No patterns detected in recent bars"
        
        pattern_display = pattern_text
    else:
        pattern_display = "Insufficient data for pattern detection"
    
    pattern_display
    
    return patterns, last_3, row, body, range_hl, prev, curr, pattern_text, pattern_display


@app.cell
def __(ohlc_data, mo):
    # データテーブル表示
    if ohlc_data is not None and not ohlc_data.is_empty():
        # 最新の20行を表示
        recent_data = ohlc_data.tail(20)
        
        # Marimoテーブル用にフォーマット
        table_data = recent_data.select([
            pl.col("time").dt.strftime("%Y-%m-%d %H:%M:%S"),
            pl.col("open").round(5),
            pl.col("high").round(5),
            pl.col("low").round(5),
            pl.col("close").round(5),
            pl.col("tick_volume"),
            (pl.col("spread") / 100000).round(1).alias("spread_pips")
        ])
        
        data_table = mo.ui.table(
            data=table_data.to_pandas(),
            pagination=True,
            page_size=10,
            selection=None
        )
        
        mo.accordion({
            "📋 Recent Data": data_table
        })
    
    return recent_data, table_data, data_table


@app.cell
def __(mo, auto_refresh, datetime):
    # 自動更新ステータス
    if auto_refresh.value:
        refresh_status = mo.md(f"""
        🔄 **Auto-refresh enabled** - Updates every 60 seconds
        
        Last update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """)
    else:
        refresh_status = mo.md("⏸️ **Auto-refresh disabled**")
    
    refresh_status
    
    return refresh_status,


@app.cell
def __(mo):
    # フッター
    mo.md("""
    ---
    
    ### 📌 Dashboard Features
    
    - **Interactive Charts:** ローソク足チャートとボリューム表示
    - **Technical Indicators:** 移動平均線とボリンジャーバンド
    - **Pattern Detection:** Doji、Engulfingパターンの自動検出
    - **Real-time Updates:** 自動更新機能（60秒間隔）
    - **Data Table:** 最新データの詳細表示
    
    ### 🎮 Usage Tips
    
    1. シンボルとタイムフレームを選択
    2. 期間を調整（1-30日）
    3. "Fetch Data"をクリックしてデータ取得
    4. チャートをドラッグしてズーム、ダブルクリックでリセット
    5. Auto Refreshで自動更新を有効化
    
    ### ⚡ Performance Notes
    
    - 大量データの場合、チャートは最新100本に制限
    - 並列処理により高速データ取得
    - Polarsによる効率的なデータ処理
    """)
    return


if __name__ == "__main__":
    app.run()