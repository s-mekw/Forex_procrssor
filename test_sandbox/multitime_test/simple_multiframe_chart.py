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
import toml

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# 新しいモジュールをインポート
from chart_renderer import ChartRenderer
from realtime_data_manager import RealtimeDataManager

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleMultiframeChart:
    """シンプルなマルチタイムフレームチャート（リファクタリング版）"""
    
    def __init__(self, config_path: str = None):
        """初期化"""
        # configパスの解決
        if config_path is None:
            from pathlib import Path
            script_dir = Path(__file__).parent
            config_path = script_dir / "config.toml"
        
        # 設定読み込み
        self.config = toml.load(config_path)
        logger.info(f"Config loaded from {config_path}")
        
        # 基本設定を取得（レイアウト用）
        self.symbol = self.config['trading']['symbol']
        
        # モジュールの初期化
        self.data_manager = RealtimeDataManager(self.config)
        self.chart_renderer = ChartRenderer(self.config)
        
        # Dashアプリ
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    
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
            # データマネージャーから最新データを取得
            data = self.data_manager.get_chart_data()
            
            # 価格表示テキストを作成
            price_text = self.chart_renderer.create_price_display(
                data['current_price'],
                data['last_update']
            )
            
            # チャートフィギュアを作成
            fig = self.chart_renderer.create_multiframe_figure(**data)
            
            return fig, price_text
    
    
    def run(self):
        """アプリケーションを実行"""
        try:
            # データマネージャーの初期化（MT5接続とマネージャー初期化）
            if not self.data_manager.connect_and_initialize():
                logger.error("Failed to initialize data manager")
                return
            
            # ティック受信開始
            self.data_manager.start_tick_receiver()
            
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
            self.data_manager.cleanup()
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