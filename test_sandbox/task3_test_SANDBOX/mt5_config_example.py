"""
MT5接続設定例

実際のMT5接続に必要な設定の例とヘルパー関数を提供します。
"""

import os
import json
from typing import Dict, Any, Optional


class MT5ConfigHelper:
    """MT5接続設定を管理するヘルパークラス"""
    
    def __init__(self, config_file: str = "mt5_config.json"):
        """初期化
        
        Args:
            config_file: 設定ファイルのパス
        """
        self.config_file = config_file
        self.config = {}
    
    def load_config(self) -> Dict[str, Any]:
        """設定ファイルから設定を読み込み
        
        Returns:
            Dict[str, Any]: 接続設定
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                print(f"設定ファイル '{self.config_file}' から設定を読み込みました")
            else:
                print(f"設定ファイル '{self.config_file}' が見つかりません")
                self.config = {}
        except Exception as e:
            print(f"設定ファイルの読み込み中にエラーが発生しました: {e}")
            self.config = {}
        
        return self.config
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """設定をファイルに保存
        
        Args:
            config: 保存する設定
        
        Returns:
            bool: 保存成功時True
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"設定を '{self.config_file}' に保存しました")
            self.config = config
            return True
        except Exception as e:
            print(f"設定ファイルの保存中にエラーが発生しました: {e}")
            return False
    
    def create_demo_config(self) -> Dict[str, Any]:
        """デモ用設定を作成
        
        Returns:
            Dict[str, Any]: デモ用設定
        """
        return {
            "account": 12345678,
            "password": "your_demo_password",
            "server": "YourBroker-Demo",
            "timeout": 60000,
            "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
        }
    
    def create_live_config(self) -> Dict[str, Any]:
        """ライブ用設定を作成
        
        Returns:
            Dict[str, Any]: ライブ用設定
        """
        return {
            "account": 87654321,
            "password": "your_live_password",
            "server": "YourBroker-Live",
            "timeout": 60000,
            "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """設定の妥当性を検証
        
        Args:
            config: 検証する設定
        
        Returns:
            bool: 妥当な場合True
        """
        required_fields = ['account', 'password', 'server']
        
        for field in required_fields:
            if field not in config:
                print(f"必須フィールド '{field}' が設定に含まれていません")
                return False
            
            if not config[field]:
                print(f"フィールド '{field}' が空です")
                return False
        
        # アカウント番号が数値であることを確認
        try:
            int(config['account'])
        except (ValueError, TypeError):
            print("アカウント番号は数値である必要があります")
            return False
        
        # タイムアウトが数値であることを確認
        if 'timeout' in config:
            try:
                int(config['timeout'])
            except (ValueError, TypeError):
                print("タイムアウトは数値である必要があります")
                return False
        
        print("設定の検証が完了しました")
        return True


def get_mt5_config_examples() -> Dict[str, Dict[str, Any]]:
    """MT5接続設定の例を取得
    
    Returns:
        Dict[str, Dict[str, Any]]: 設定例の辞書
    """
    return {
        "demo_example": {
            "account": 12345678,
            "password": "your_demo_password",
            "server": "YourBroker-Demo",
            "timeout": 60000,
            "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
        },
        "live_example": {
            "account": 87654321,
            "password": "your_live_password",
            "server": "YourBroker-Live",
            "timeout": 60000,
            "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
        },
        "minimal_example": {
            "account": 12345678,
            "password": "your_password",
            "server": "YourBroker-Server"
        }
    }


def print_config_examples():
    """設定例を表示"""
    examples = get_mt5_config_examples()
    
    print("=== MT5接続設定例 ===")
    print()
    
    for name, config in examples.items():
        print(f"【{name}】")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()


def create_config_file():
    """設定ファイルを作成"""
    helper = MT5ConfigHelper()
    
    print("MT5接続設定ファイルを作成します")
    print("以下の情報を入力してください:")
    print()
    
    # ユーザーから設定を取得
    account = input("アカウント番号: ").strip()
    password = input("パスワード: ").strip()
    server = input("サーバー名: ").strip()
    
    # オプション設定
    timeout = input("タイムアウト（ミリ秒、デフォルト: 60000）: ").strip()
    path = input("MT5のパス（オプション）: ").strip()
    
    # 設定を作成
    config = {
        "account": int(account) if account else 12345678,
        "password": password,
        "server": server,
        "timeout": int(timeout) if timeout else 60000
    }
    
    if path:
        config["path"] = path
    
    # 設定の検証
    if helper.validate_config(config):
        # 設定を保存
        if helper.save_config(config):
            print("設定ファイルが正常に作成されました")
            return config
        else:
            print("設定ファイルの作成に失敗しました")
            return None
    else:
        print("設定が無効です")
        return None


if __name__ == "__main__":
    print("MT5接続設定ヘルパー")
    print("=" * 50)
    print()
    
    # 設定例を表示
    print_config_examples()
    
    # 設定ファイルの作成
    print("設定ファイルを作成しますか？ (y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y':
        config = create_config_file()
        if config:
            print("設定ファイルが作成されました。mt5_connection_sandbox.pyで使用できます。")
    else:
        print("設定例を参考に、手動で設定ファイルを作成してください。")
