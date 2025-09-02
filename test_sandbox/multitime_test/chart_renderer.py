"""
Chart Renderer Module
チャート描画機能を提供するモジュール
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class ChartRenderer:
    """チャート描画クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.theme = config['theme']
        self.chart_config = config['chart']
        
    def create_multiframe_figure(
        self, 
        m1_data: Optional[pl.DataFrame] = None,
        m5_data: Optional[pl.DataFrame] = None,
        m1_current: Optional[Dict] = None,
        m5_current: Optional[Dict] = None,
        **kwargs
    ) -> go.Figure:
        """
        マルチタイムフレームチャートフィギュアを作成
        
        Args:
            m1_data: M1完成バーデータ
            m5_data: M5完成バーデータ
            m1_current: M1現在進行形バー
            m5_current: M5現在進行形バー
            **kwargs: その他のパラメータ
            
        Returns:
            Plotlyフィギュア
        """
        # サブプロット作成（1行2列）
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                "M1 - 1 Minute Chart",
                "M5 - 5 Minute Chart"
            ),
            horizontal_spacing=0.05
        )
        
        # M1チャート描画
        self._add_timeframe_chart(
            fig, 
            data=m1_data,
            current_bar=m1_current,
            timeframe="M1",
            display_bars=self.chart_config['display_bars_m1'],
            row=1, col=1
        )
        
        # M5チャート描画
        self._add_timeframe_chart(
            fig,
            data=m5_data,
            current_bar=m5_current,
            timeframe="M5",
            display_bars=self.chart_config['display_bars_m5'],
            row=1, col=2
        )
        
        # レイアウト設定を適用
        self._apply_layout(fig)
        
        return fig
    
    def _add_timeframe_chart(
        self,
        fig: go.Figure,
        data: Optional[pl.DataFrame],
        current_bar: Optional[Dict],
        timeframe: str,
        display_bars: int,
        row: int,
        col: int
    ):
        """
        単一タイムフレームのチャートを追加
        
        Args:
            fig: Plotlyフィギュア
            data: 完成バーデータ
            current_bar: 現在進行形バー
            timeframe: タイムフレーム名
            display_bars: 表示するバー数
            row: サブプロットの行
            col: サブプロットの列
        """
        # 完成バーを追加
        if data is not None and not data.is_empty():
            display_data = data.tail(display_bars)
            self._add_completed_bars(fig, display_data, timeframe, row, col)
            
            # 現在進行形バーを追加
            if current_bar is not None:
                last_bar_time = display_data["timestamp"].tail(1).to_list()[0] if not display_data.is_empty() else datetime.now()
                self._add_current_bar(fig, current_bar, timeframe, last_bar_time, row, col)
    
    def _add_completed_bars(
        self,
        fig: go.Figure,
        data: pl.DataFrame,
        timeframe: str,
        row: int,
        col: int
    ):
        """
        完成バーをチャートに追加
        
        Args:
            fig: Plotlyフィギュア
            data: バーデータ
            timeframe: タイムフレーム名
            row: サブプロットの行
            col: サブプロットの列
        """
        fig.add_trace(
            go.Candlestick(
                x=data["timestamp"].to_list(),
                open=data["open"].to_list(),
                high=data["high"].to_list(),
                low=data["low"].to_list(),
                close=data["close"].to_list(),
                name=timeframe,
                increasing_line_color=self.theme['bullish'],
                decreasing_line_color=self.theme['bearish']
            ),
            row=row, col=col
        )
    
    def _add_current_bar(
        self,
        fig: go.Figure,
        current_bar: Dict,
        timeframe: str,
        last_bar_time: datetime,
        row: int,
        col: int
    ):
        """
        現在進行形バーをチャートに追加（半透明で表示）
        
        Args:
            fig: Plotlyフィギュア
            current_bar: 現在進行形バーデータ
            timeframe: タイムフレーム名
            last_bar_time: 最後の完成バーの時刻
            row: サブプロットの行
            col: サブプロットの列
        """
        # タイムフレームに応じた時間間隔を計算
        if timeframe == "M1":
            interval_minutes = 1
        elif timeframe == "M5":
            interval_minutes = 5
        else:
            interval_minutes = 1  # デフォルト
        
        current_bar_time = last_bar_time + timedelta(minutes=interval_minutes)
        
        # 現在進行形バーを半透明で描画
        fig.add_trace(
            go.Candlestick(
                x=[current_bar_time],
                open=[current_bar['open']],
                high=[current_bar['high']],
                low=[current_bar['low']],
                close=[current_bar['close']],
                name=f"{timeframe} Current",
                increasing_line_color=self.theme['bullish'],
                decreasing_line_color=self.theme['bearish'],
                opacity=0.5,  # 半透明で表示
                showlegend=False
            ),
            row=row, col=col
        )
    
    def _apply_layout(self, fig: go.Figure):
        """
        チャートレイアウトを適用
        
        Args:
            fig: Plotlyフィギュア
        """
        # レイアウト設定
        fig.update_layout(
            showlegend=False,
            height=700,
            paper_bgcolor=self.theme['background'],
            plot_bgcolor=self.theme['background'],
            font={'color': self.theme['text']},
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False
        )
        
        # グリッド設定
        fig.update_xaxes(showgrid=True, gridcolor=self.theme['grid'])
        fig.update_yaxes(showgrid=True, gridcolor=self.theme['grid'])
    
    def create_price_display(
        self,
        current_price: float,
        last_update: datetime
    ) -> str:
        """
        価格表示テキストを作成
        
        Args:
            current_price: 現在価格
            last_update: 最終更新時刻
            
        Returns:
            価格表示テキスト
        """
        return f"Current Price: ${current_price:.2f} | Last Update: {last_update.strftime('%H:%M:%S')}"