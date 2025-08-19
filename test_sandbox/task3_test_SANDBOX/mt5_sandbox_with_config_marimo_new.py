import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import sys
    import os
    import time
    import logging
    import json
    from typing import Dict, Any, Optional, List

    # プロジェクトルートをパスに追加
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    mo.md("# MT5接続サンドボックス（設定ファイル対応版）")
    return Any, Dict, List, Optional, logger, mo


@app.cell
def _(Any, Dict, List, Optional, logger, mo):
    from mt5_data_acquisition.mt5_client import MT5ConnectionManager
    from mt5_config_example import MT5ConfigHelper

    class MT5SandboxWithConfig:
        """設定ファイル対応MT5接続テスト用サンドボックスクラス"""

        def __init__(self, config_file: str = "mt5_config.json"):
            """初期化"""
            self.connection_manager = MT5ConnectionManager()
            self.is_connected = False
            self.config_helper = MT5ConfigHelper(config_file)
            self.config = {}

        def load_config(self) -> bool:
            """設定ファイルから設定を読み込み"""
            self.config = self.config_helper.load_config()

            if not self.config:
                logger.error("設定ファイルが読み込めませんでした")
                return False

            # 設定の検証
            if not self.config_helper.validate_config(self.config):
                logger.error("設定が無効です")
                return False

            logger.info("設定ファイルを正常に読み込みました")
            return True

        def connect_to_mt5(self) -> bool:
            """MT5への接続を試行"""
            if not self.config:
                logger.error("設定が読み込まれていません。load_config()を先に実行してください")
                return False

            logger.info("MT5接続を開始します...")

            try:
                # 接続実行
                success = self.connection_manager.connect(self.config)

                if success:
                    self.is_connected = True
                    logger.info("MT5接続に成功しました！")

                    # 接続情報を表示
                    self._display_connection_info()

                else:
                    logger.error("MT5接続に失敗しました")

                return success

            except Exception as e:
                logger.error(f"接続中にエラーが発生しました: {e}")
                return False

        def _display_connection_info(self):
            """接続情報を表示"""
            try:
                # ターミナル情報の表示
                terminal_info = self.connection_manager.terminal_info
                if terminal_info:
                    logger.info("=== ターミナル情報 ===")
                    logger.info(f"会社: {getattr(terminal_info, 'company', 'N/A')}")
                    logger.info(f"ビルド: {getattr(terminal_info, 'build', 'N/A')}")
                    logger.info(f"パス: {getattr(terminal_info, 'path', 'N/A')}")

                # アカウント情報の表示
                account_info = self.connection_manager.account_info
                if account_info:
                    logger.info("=== アカウント情報 ===")
                    logger.info(f"ログイン: {getattr(account_info, 'login', 'N/A')}")
                    logger.info(f"残高: {getattr(account_info, 'balance', 'N/A')}")
                    logger.info(f"レバレッジ: {getattr(account_info, 'leverage', 'N/A')}")
                    logger.info(f"通貨: {getattr(account_info, 'currency', 'N/A')}")
                    logger.info(f"サーバー: {getattr(account_info, 'server', 'N/A')}")

            except Exception as e:
                logger.warning(f"接続情報の表示中にエラーが発生しました: {e}")

        def get_symbols_info(self, symbols: List[str]) -> Dict[str, Any]:
            """シンボル情報を取得"""
            if not self.is_connected:
                logger.error("MT5に接続されていません")
                return {}

            try:
                import MetaTrader5 as mt5

                symbols_info = {}

                for symbol in symbols:
                    logger.info(f"シンボル情報を取得中: {symbol}")

                    # シンボル情報を取得
                    symbol_info = mt5.symbol_info(symbol)

                    if symbol_info is not None:
                        symbols_info[symbol] = {
                            'name': symbol_info.name,
                            'point': symbol_info.point,
                            'digits': symbol_info.digits,
                            'spread': symbol_info.spread,
                            'spread_float': symbol_info.spread_float,
                            'volume_min': symbol_info.volume_min,
                            'volume_max': symbol_info.volume_max,
                            'volume_step': symbol_info.volume_step,
                        }
                        logger.info(f"シンボル {symbol} の情報を取得しました")
                    else:
                        logger.warning(f"シンボル {symbol} の情報を取得できませんでした")
                        symbols_info[symbol] = None

                return symbols_info

            except Exception as e:
                logger.error(f"シンボル情報取得中にエラーが発生しました: {e}")
                return {}

        def get_ohlc_data(self, symbol: str, timeframe: int, count: int = 100) -> Optional[List[Dict[str, Any]]]:
            """OHLCデータを取得"""
            if not self.is_connected:
                logger.error("MT5に接続されていません")
                return None

            try:
                import MetaTrader5 as mt5

                logger.info(f"OHLCデータを取得中: {symbol}, タイムフレーム: {timeframe}, バー数: {count}")

                # OHLCデータを取得
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

                if rates is not None and len(rates) > 0:
                    # データを辞書のリストに変換
                    ohlc_data = []
                    for rate in rates:
                        ohlc_data.append({
                            'time': rate['time'],
                            'open': rate['open'],
                            'high': rate['high'],
                            'low': rate['low'],
                            'close': rate['close'],
                            'tick_volume': rate['tick_volume'],
                            'spread': rate['spread'],
                            'real_volume': rate['real_volume'],
                        })

                    logger.info(f"OHLCデータを取得しました: {len(ohlc_data)} バー")
                    return ohlc_data
                else:
                    logger.warning(f"シンボル {symbol} のOHLCデータを取得できませんでした")
                    return None

            except Exception as e:
                logger.error(f"OHLCデータ取得中にエラーが発生しました: {e}")
                return None

        def health_check(self) -> bool:
            """接続の健全性を確認"""
            if not self.is_connected:
                logger.warning("接続されていないため、ヘルスチェックをスキップします")
                return False

            logger.info("ヘルスチェックを実行中...")
            is_healthy = self.connection_manager.health_check()

            if is_healthy:
                logger.info("ヘルスチェック成功: 接続は正常です")
            else:
                logger.warning("ヘルスチェック失敗: 接続に問題があります")

            return is_healthy

        def disconnect(self):
            """MT5から切断"""
            if self.is_connected:
                logger.info("MT5から切断中...")
                self.connection_manager.disconnect()
                self.is_connected = False
                logger.info("MT5から切断しました")
            else:
                logger.info("既に切断されています")

    mo.md("## MT5SandboxWithConfigクラスを定義しました")
    return (MT5SandboxWithConfig,)


@app.cell
def _(MT5SandboxWithConfig, mo):
    mo.md("## サンドボックス初期化")

    # 設定ファイルを指定してサンドボックスを作成
    config_file = "test_sandbox/mt5_config.json"
    sandbox = MT5SandboxWithConfig(config_file)

    mo.md(f"✅ 設定ファイル対応サンドボックスを作成しました (設定ファイル: {config_file})")
    return (sandbox,)


@app.cell
def _(mo, sandbox):
    mo.md("## 設定ファイル読み込み")

    def load_config():
        success = sandbox.load_config()
        if success:
            return f"✅ 設定ファイルの読み込みに成功しました"
        else:
            return "❌ 設定ファイルの読み込みに失敗しました"

    load_button = mo.ui.button(
        label="設定ファイルを読み込み",
        on_click=lambda _: load_config(),
        kind="success"
    )

    load_button
    return (load_button,)


@app.cell
def _(load_button, mo, sandbox):
    mo.md("## 現在の設定表示")

    if load_button.value and sandbox.config:
        # 設定内容をテーブルで表示（パスワードはマスク）
        config_display = []
        for key, value in sandbox.config.items():
            if key == 'password':
                display_value = '*' * len(str(value)) if value else 'なし'
            else:
                display_value = str(value)

            config_display.append({
                "設定項目": key,
                "値": display_value
            })

        mo.ui.table(config_display, label="現在の設定")
    elif load_button.value:
        mo.md("❌ 設定が読み込まれていません")
    else:
        mo.md("上のボタンをクリックして設定ファイルを読み込んでください")
    return


@app.cell
def _(mo, sandbox):
    mo.md("## MT5接続")

    def connect_to_mt5():
        success = sandbox.connect_to_mt5()
        if success:
            return "✅ MT5接続に成功しました"
        else:
            return "❌ MT5接続に失敗しました"

    connect_button = mo.ui.button(
        label="MT5に接続",
        on_click=lambda _: connect_to_mt5(),
        kind="success"
    )

    connect_button
    return


@app.cell
def _(mo):
    mo.md("## テスト用シンボル設定")

    symbol_options = {
        "メジャー通貨ペア": ["EURUSD", "GBPUSD", "USDJPY"],
        "クロス通貨ペア": ["EURJPY", "GBPJPY", "AUDUSD"],
        "カスタム": ["EURUSD", "USDJPY"]
    }

    symbol_select = mo.ui.dropdown(
        options=symbol_options,
        label="テスト対象シンボルを選択:"
    )

    symbol_select
    return (symbol_select,)


@app.cell
def _(mo, sandbox, symbol_select):
    mo.md("## シンボル情報取得")

    def get_symbols_info():
        if symbol_select.value:
            symbols_info = sandbox.get_symbols_info(symbol_select.value)
            return symbols_info
        return {}

    symbols_button = mo.ui.button(
        label="シンボル情報を取得",
        on_click=lambda _: get_symbols_info(),
        kind="neutral"
    )

    symbols_button
    return get_symbols_info, symbols_button


@app.cell
def _(get_symbols_info, mo, symbols_button):
    mo.md("## シンボル情報表示")

    if symbols_button.value:
        symbols_info = get_symbols_info()
        if symbols_info:
            # シンボル情報をテーブル形式で表示
            symbols_table_data = []
            for sym, info in symbols_info.items():
                if info:
                    symbols_table_data.append({
                        "Symbol": sym,
                        "Point": info.get('point', 'N/A'),
                        "Digits": info.get('digits', 'N/A'),
                        "Spread": info.get('spread', 'N/A'),
                        "Min Volume": info.get('volume_min', 'N/A'),
                        "Max Volume": info.get('volume_max', 'N/A')
                    })
                else:
                    symbols_table_data.append({
                        "Symbol": sym,
                        "Point": "取得失敗",
                        "Digits": "-",
                        "Spread": "-",
                        "Min Volume": "-",
                        "Max Volume": "-"
                    })

            if symbols_table_data:
                mo.ui.table(symbols_table_data, label="シンボル情報")
            else:
                mo.md("シンボル情報を取得できませんでした")
        else:
            mo.md("シンボル情報を取得できませんでした")
    else:
        mo.md("上のボタンをクリックしてシンボル情報を取得してください")
    return


@app.cell
def _(mo, sandbox, symbol_select):
    mo.md("## OHLCデータ取得")

    def get_ohlc_data():
        if symbol_select.value and len(symbol_select.value) > 0:
            try:
                import MetaTrader5 as mt5
                selected_symbol = symbol_select.value[0]  # 最初のシンボル
                ohlc_data = sandbox.get_ohlc_data(selected_symbol, mt5.TIMEFRAME_H1, 10)
                return ohlc_data, selected_symbol
            except ImportError:
                print("MetaTrader5モジュールが利用できません")
                return None, None
        return None, None

    ohlc_button = mo.ui.button(
        label="OHLCデータを取得 (最初のシンボル、H1、10バー)",
        on_click=lambda _: get_ohlc_data(),
        kind="neutral"
    )

    ohlc_button
    return get_ohlc_data, ohlc_button


@app.cell
def _(get_ohlc_data, mo, ohlc_button):
    mo.md("## OHLCデータ表示")

    if ohlc_button.value:
        ohlc_data, selected_symbol = get_ohlc_data()
        if ohlc_data and selected_symbol:
            mo.md(f"### {selected_symbol} の最新10バー (H1)")

            # 最新5バーのみ表示
            ohlc_table_data = []
            for i, bar in enumerate(ohlc_data[-5:]):
                ohlc_table_data.append({
                    "時刻": bar['time'],
                    "Open": f"{bar['open']:.5f}",
                    "High": f"{bar['high']:.5f}",
                    "Low": f"{bar['low']:.5f}",
                    "Close": f"{bar['close']:.5f}",
                    "Volume": bar['tick_volume']
                })

            mo.ui.table(ohlc_table_data, label=f"{selected_symbol} OHLC データ (最新5バー)")
        else:
            mo.md("OHLCデータを取得できませんでした")
    else:
        mo.md("上のボタンをクリックしてOHLCデータを取得してください")
    return


@app.cell
def _(mo, sandbox):
    mo.md("## ヘルスチェック")

    def perform_health_check():
        return sandbox.health_check()

    health_button = mo.ui.button(
        label="ヘルスチェック実行",
        on_click=lambda _: perform_health_check(),
        kind="neutral"
    )

    health_button
    return


@app.cell
def _(mo, sandbox):
    mo.md("## 接続管理")

    def reconnect_mt5():
        return sandbox.connection_manager.reconnect()

    def disconnect_mt5():
        sandbox.disconnect()
        return True

    reconnect_button = mo.ui.button(
        label="再接続",
        on_click=lambda _: reconnect_mt5(),
        kind="warn"
    )

    disconnect_button = mo.ui.button(
        label="切断",
        on_click=lambda _: disconnect_mt5(),
        kind="danger"
    )

    connection_controls = mo.hstack([reconnect_button, disconnect_button], justify="start")
    connection_controls
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 使用方法

    1. **設定ファイル準備**: `mt5_config.json`を作成してください
    2. **設定読み込み**: 「設定ファイルを読み込み」ボタンで設定を読み込み
    3. **設定確認**: 現在の設定が正しく表示されることを確認
    4. **MT5接続**: 「MT5に接続」ボタンでMT5に接続
    5. **シンボル選択**: テスト対象の通貨ペアを選択
    6. **データ取得**: シンボル情報やOHLCデータを取得
    7. **ヘルスチェック**: 接続状態を確認
    8. **接続管理**: 再接続や切断を実行

    ### 設定ファイル形式 (mt5_config.json)
    ```json
    {
        "account": 12345678,
        "password": "your_password",
        "server": "YourBroker-Demo",
        "timeout": 60000,
        "path": "C:\\\\Program Files\\\\MetaTrader 5\\\\terminal64.exe"
    }
    ```

    ### 注意事項
    - 設定ファイル `test_sandbox/mt5_config.json` が必要です
    - `test_sandbox/mt5_config_example.py` を実行して設定ファイルを作成できます
    - デモアカウントでのテストを推奨します
    - MT5ターミナルが起動している必要があります
    """
    )
    return


if __name__ == "__main__":
    app.run()
