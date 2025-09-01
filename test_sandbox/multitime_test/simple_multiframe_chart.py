"""
Simple MultiTimeframe Chart Display
M1とM5のチャートをリアルタイム表示するシンプルなテスト

config.tomlで設定されたシンボルを使用
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import logging
from datetime import datetime
import threading
import time
import toml
import MetaTrader5 as mt5
import polars as pl

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_processing.multiframe_manager import MultiTimeframeManager

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleMultiframeChart:
    """シンプルなマルチタイムフレームチャート"""
    
    def __init__(self, config_path: str = None):
        """初期化"""
        # configパスの解決
        if config_path is None:
            # スクリプトと同じディレクトリのconfig.tomlを使用
            from pathlib import Path
            script_dir = Path(__file__).parent
            config_path = script_dir / "config.toml"
        
        # 設定読み込み
        self.config = toml.load(config_path)
        logger.info(f"Config loaded from {config_path}")
        
        # 基本設定
        self.symbol = self.config['trading']['symbol']
        self.timeframes = self.config['trading']['timeframes']
        
        # MultiTimeframeManager
        self.manager = None
        
        # データ格納
        self.m1_data = None
        self.m5_data = None
        self.current_price = 0.0
        self.last_update = datetime.now()
        
        # スレッド制御
        self.is_running = False
        self.tick_thread = None
        
        # Dashアプリ
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def connect_mt5(self) -> bool:
        """MT5に接続"""
        try:
            mt5_config = self.config['mt5']
            
            # MT5初期化
            if not mt5.initialize(
                path=mt5_config['path'],
                login=mt5_config['account'],
                password=mt5_config['password'],
                server=mt5_config['server'],
                timeout=mt5_config['timeout']
            ):
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            logger.info(f"MT5 connected: Account {mt5_config['account']}")
            
            # シンボル確認
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                logger.error(f"Symbol {self.symbol} not found")
                return False
            
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(f"Failed to select symbol {self.symbol}")
                    return False
            
            logger.info(f"Symbol {self.symbol} selected")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def initialize_manager(self) -> bool:
        """MultiTimeframeManagerを初期化"""
        try:
            # マネージャー作成
            self.manager = MultiTimeframeManager(
                symbol=self.symbol,
                timeframes=self.timeframes,
                initial_bars=self.config['chart']['initial_bars'],
                max_bars=self.config['chart']['max_bars']
            )
            
            # 初期データ取得
            if not self.manager.initialize_data():
                logger.error("Failed to initialize manager data")
                return False
            
            # 初期データを保存
            self.m1_data = self.manager.get_completed_bars("M1")
            self.m5_data = self.manager.get_completed_bars("M5")
            
            # 現在価格を取得
            tick = mt5.symbol_info_tick(self.symbol)
            if tick:
                self.current_price = tick.bid
            
            logger.info("Manager initialized successfully")
            logger.info(f"M1 bars: {len(self.m1_data) if self.m1_data is not None else 0}")
            logger.info(f"M5 bars: {len(self.m5_data) if self.m5_data is not None else 0}")
            
            return True
            
        except Exception as e:
            logger.error(f"Manager initialization error: {e}")
            return False
    
    def tick_receiver_loop(self):
        """ティック受信ループ（別スレッド）"""
        logger.info("Tick receiver started")
        
        tick_count = 0
        m1_bars_completed = 0
        m5_bars_completed = 0
        
        while self.is_running:
            try:
                # ティック取得
                tick = mt5.symbol_info_tick(self.symbol)
                if tick is None:
                    time.sleep(0.1)
                    continue
                
                # ティック処理
                results = self.manager.process_tick(tick)
                tick_count += 1
                
                # 現在価格更新
                self.current_price = tick.bid
                self.last_update = datetime.now()
                
                # 新しいバーが完成したかチェック
                for tf_name, tf_result in results.items():
                    if tf_result.get("new_bar"):
                        if tf_name == "M1":
                            m1_bars_completed += 1
                            logger.info(f"✅ M1 bar completed #{m1_bars_completed} at {tf_result['timestamp']}")
                        elif tf_name == "M5":
                            m5_bars_completed += 1
                            logger.info(f"✅ M5 bar completed #{m5_bars_completed} at {tf_result['timestamp']}")
                
                # 最新データを取得
                self.m1_data = self.manager.get_completed_bars("M1")
                self.m5_data = self.manager.get_completed_bars("M5")
                
                # 10ティックごとにログ
                if tick_count % 10 == 0:
                    logger.debug(f"Processed {tick_count} ticks, M1 bars: {m1_bars_completed}, M5 bars: {m5_bars_completed}")
                
                time.sleep(0.1)  # CPU負荷軽減
                
            except Exception as e:
                logger.error(f"Tick receiver error: {e}")
                time.sleep(1)
        
        logger.info(f"Tick receiver stopped. Total ticks: {tick_count}, M1 bars: {m1_bars_completed}, M5 bars: {m5_bars_completed}")
    
    def start_tick_receiver(self):
        """ティック受信を開始"""
        if not self.is_running:
            self.is_running = True
            self.tick_thread = threading.Thread(target=self.tick_receiver_loop)
            self.tick_thread.daemon = True
            self.tick_thread.start()
            logger.info("Tick receiver thread started")
    
    def stop_tick_receiver(self):
        """ティック受信を停止"""
        if self.is_running:
            self.is_running = False
            if self.tick_thread:
                self.tick_thread.join(timeout=2)
            logger.info("Tick receiver stopped")
    
    def setup_layout(self):
        """Dashレイアウトを設定"""
        theme = self.config['theme']
        
        self.app.layout = html.Div([
            html.H1(
                f"{self.symbol} MultiTimeframe Chart - M1 & M5",
                style={'textAlign': 'center', 'color': theme['text']}
            ),
            
            html.Div(id='price-display', style={
                'textAlign': 'center',
                'fontSize': '24px',
                'color': theme['text'],
                'marginBottom': '20px'
            }),
            
            dcc.Graph(id='chart', style={'height': '800px'}),
            
            dcc.Interval(
                id='interval-component',
                interval=self.config['chart']['update_interval'],
                n_intervals=0
            )
        ], style={'backgroundColor': theme['background']})
    
    def setup_callbacks(self):
        """Dashコールバックを設定"""
        
        @self.app.callback(
            [Output('chart', 'figure'),
             Output('price-display', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_chart(n):
            """チャートを更新"""
            # 価格表示
            price_text = f"Current Price: ${self.current_price:.2f} | Last Update: {self.last_update.strftime('%H:%M:%S')}"
            
            # チャート作成
            fig = self.create_chart_figure()
            
            return fig, price_text
    
    def create_chart_figure(self) -> go.Figure:
        """チャートフィギュアを作成"""
        theme = self.config['theme']
        
        # サブプロット作成（1行2列）
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f"M1 - 1 Minute Chart",
                f"M5 - 5 Minute Chart"
            ),
            horizontal_spacing=0.05
        )
        
        # M1チャート
        if self.m1_data is not None and not self.m1_data.is_empty():
            display_bars = self.config['chart']['display_bars_m1']
            m1_display = self.m1_data.tail(display_bars)
            
            fig.add_trace(
                go.Candlestick(
                    x=m1_display["timestamp"].to_list(),
                    open=m1_display["open"].to_list(),
                    high=m1_display["high"].to_list(),
                    low=m1_display["low"].to_list(),
                    close=m1_display["close"].to_list(),
                    name="M1",
                    increasing_line_color=theme['bullish'],
                    decreasing_line_color=theme['bearish']
                ),
                row=1, col=1
            )
        
        # M5チャート
        if self.m5_data is not None and not self.m5_data.is_empty():
            display_bars = self.config['chart']['display_bars_m5']
            m5_display = self.m5_data.tail(display_bars)
            
            fig.add_trace(
                go.Candlestick(
                    x=m5_display["timestamp"].to_list(),
                    open=m5_display["open"].to_list(),
                    high=m5_display["high"].to_list(),
                    low=m5_display["low"].to_list(),
                    close=m5_display["close"].to_list(),
                    name="M5",
                    increasing_line_color=theme['bullish'],
                    decreasing_line_color=theme['bearish']
                ),
                row=1, col=2
            )
        
        # レイアウト設定
        fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            height=700,
            paper_bgcolor=theme['background'],
            plot_bgcolor=theme['background'],
            font={'color': theme['text']},
            xaxis_rangeslider_visible=False,
            xaxis2_rangeslider_visible=False
        )
        
        # グリッド設定
        fig.update_xaxes(showgrid=True, gridcolor=theme['grid'])
        fig.update_yaxes(showgrid=True, gridcolor=theme['grid'])
        
        return fig
    
    def run(self):
        """アプリケーションを実行"""
        try:
            # MT5接続
            if not self.connect_mt5():
                logger.error("Failed to connect to MT5")
                return
            
            # マネージャー初期化
            if not self.initialize_manager():
                logger.error("Failed to initialize manager")
                return
            
            # ティック受信開始
            self.start_tick_receiver()
            
            # Dashサーバー起動
            dash_config = self.config['dash']
            logger.info(f"Starting Dash server on http://{dash_config['host']}:{dash_config['port']}")
            
            self.app.run(
                host=dash_config['host'],
                port=dash_config['port'],
                debug=dash_config['debug']
            )
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            # クリーンアップ
            self.stop_tick_receiver()
            mt5.shutdown()
            logger.info("Application stopped")


def main():
    """メイン関数"""
    logger.info("=" * 60)
    logger.info("Simple MultiTimeframe Chart Test")
    logger.info("=" * 60)
    logger.info(f"Start time: {datetime.now()}")
    
    # アプリケーション実行
    app = SimpleMultiframeChart()
    app.run()


if __name__ == "__main__":
    main()