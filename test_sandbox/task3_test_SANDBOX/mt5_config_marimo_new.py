import marimo

__generated_with = "0.9.27"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import os
    import json
    from typing import Dict, Any
    
    mo.md("# MT5接続設定管理")
    return mo, os, json, Dict, Any


@app.cell
def __(mo, os, json, Dict, Any):
    class MT5ConfigHelper:
        """MT5接続設定を管理するヘルパークラス"""
        
        def __init__(self, config_file: str = "mt5_config.json"):
            self.config_file = config_file
            self.config = {}
        
        def load_config(self) -> Dict[str, Any]:
            """設定ファイルから設定を読み込み"""
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
            """設定をファイルに保存"""
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
            """デモ用設定を作成"""
            return {
                "account": 12345678,
                "password": "your_demo_password",
                "server": "YourBroker-Demo",
                "timeout": 60000,
                "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
            }
        
        def create_live_config(self) -> Dict[str, Any]:
            """ライブ用設定を作成"""
            return {
                "account": 87654321,
                "password": "your_live_password",
                "server": "YourBroker-Live",
                "timeout": 60000,
                "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
            }
        
        def validate_config(self, config: Dict[str, Any]) -> bool:
            """設定の妥当性を検証"""
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
    
    return MT5ConfigHelper,


@app.cell
def __(mo, MT5ConfigHelper):
    mo.md("## 設定ヘルパーの初期化")
    config_helper = MT5ConfigHelper(config_file="mt5_config.json")
    current_config = config_helper.load_config()
    return config_helper, current_config


@app.cell
def __(mo, config_helper):
    mo.md("## 設定テンプレート")
    
    demo_config = config_helper.create_demo_config()
    live_config = config_helper.create_live_config()
    
    mo.md(f"""
    ### デモ口座設定例
    ```json
    {demo_config}
    ```
    
    ### ライブ口座設定例
    ```json
    {live_config}
    ```
    """)
    return demo_config, live_config


@app.cell
def __(mo, demo_config, live_config, current_config):
    mo.md("## 設定テンプレートの選択")
    
    template_select = mo.ui.dropdown(
        options={
            "現在の設定": current_config,
            "デモ口座": demo_config,
            "ライブ口座": live_config,
            "空の設定": {}
        },
        label="読み込む設定テンプレートを選択:",
    )
    template_select
    return template_select,


@app.cell
def __(mo, template_select):
    mo.md("## 接続設定の編集")
    
    # 選択されたテンプレートから初期値を設定
    selected_config = template_select.value if template_select.value else {}
    
    # フォームフィールド
    account_field = mo.ui.number(
        label="アカウント番号:", 
        value=selected_config.get("account", 0),
        step=1,
    )
    
    password_field = mo.ui.text(
        label="パスワード:",
        value=selected_config.get("password", "")
    )
    
    server_field = mo.ui.text(
        label="サーバー名:",
        value=selected_config.get("server", "")
    )
    
    timeout_field = mo.ui.number(
        label="タイムアウト(ms):",
        value=selected_config.get("timeout", 60000),
        step=1000,
    )
    
    path_field = mo.ui.text(
        label="MT5 Path:",
        value=selected_config.get("path", "C:\\Program Files\\MetaTrader 5\\terminal64.exe")
    )
    
    return account_field, password_field, server_field, timeout_field, path_field, selected_config


@app.cell
def __(mo, account_field, password_field, server_field, timeout_field, path_field):
    mo.md("### フォーム入力")
    
    form_layout = mo.vstack([
        account_field,
        password_field, 
        server_field,
        timeout_field,
        path_field
    ])
    
    form_layout
    return form_layout,


@app.cell
def __(mo, account_field, password_field, server_field, timeout_field, path_field, config_helper):
    mo.md("## 設定の検証と保存")
    
    def get_current_form_config():
        return {
            "account": account_field.value,
            "password": password_field.value,
            "server": server_field.value,
            "timeout": timeout_field.value,
            "path": path_field.value,
        }
    
    def validate_current_config():
        current_form_config = get_current_form_config()
        return config_helper.validate_config(current_form_config)
    
    def save_current_config():
        current_form_config = get_current_form_config()
        if config_helper.validate_config(current_form_config):
            return config_helper.save_config(current_form_config)
        return False
    
    validate_button = mo.ui.button(
        label="設定を検証",
        on_click=lambda _: validate_current_config()
    )
    
    save_button = mo.ui.button(
        label="設定をファイルに保存",
        on_click=lambda _: save_current_config(),
        kind="success"
    )
    
    button_layout = mo.hstack([validate_button, save_button], justify="start")
    button_layout
    return validate_button, save_button, get_current_form_config, validate_current_config, save_current_config, button_layout


@app.cell
def __(mo, get_current_form_config):
    mo.md("## 現在のフォーム内容")
    
    current_form_data = get_current_form_config()
    mo.ui.table(
        [current_form_data],
        label="フォームに入力されている現在の設定値"
    )
    return current_form_data,


if __name__ == "__main__":
    app.run()