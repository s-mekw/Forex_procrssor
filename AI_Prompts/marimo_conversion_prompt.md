# Python to Marimo Conversion Prompt

## 概要
通常のPythonファイルを marimo ノートブック形式に変換する際のガイドライン

## 基本構造

### 1. ファイルヘッダー
```python
import marimo

__generated_with = "0.9.27"  # 現在のmarimoバージョン
app = marimo.App()
```

### 2. セル構造
```python
@app.cell
def __():  # 関数名は通常 __ (アンダースコア2つ)
    import marimo as mo
    # セルの内容
    return variable1, variable2  # セルから他のセルに渡す変数
```

### 3. ファイル終端
```python
if __name__ == "__main__":
    app.run()
```

## UI コンポーネントAPI（2025年版）

### テキスト入力
```python
# 正しい形式
text_field = mo.ui.text(
    value="default value",
    placeholder="Enter text...",
    kind="text",  # "text", "password", "email", "url"
    max_length=100,
    disabled=False
)

# 使用不可: mo.ui.text_input() (存在しない)
```

### 数値入力
```python
number_field = mo.ui.number(
    value=0,
    step=1,
    disabled=False
)
```

### ボタン
```python
button = mo.ui.button(
    label="Click me",
    kind="neutral",  # "neutral", "success", "warn", "danger"
    on_click=lambda _: print("clicked"),
    disabled=False
)
```

### ドロップダウン
```python
dropdown = mo.ui.dropdown(
    options={"Label 1": "value1", "Label 2": "value2"},
    value="Label 1",  # 初期値はオプションのキー（ラベル）を指定
    label="Select option:"
)

# ❌ 間違い - 値（value）を指定
dropdown = mo.ui.dropdown(
    options={"標準": 20, "多量": 50},
    value=20,  # エラー: 20は有効なオプション名ではない
    label="Select:"
)

# ✅ 正しい - キー（ラベル）を指定
dropdown = mo.ui.dropdown(
    options={"標準": 20, "多量": 50},
    value="標準",  # オプション名（キー）を指定
    label="Select:"
)
```

### レイアウト
```python
# 垂直レイアウト
mo.vstack([component1, component2])

# 水平レイアウト  
mo.hstack([component1, component2], justify="start")

# テーブル
mo.ui.table(data, label="Table title")
```

## セル間の変数依存関係

### 基本ルール
1. セルの引数は、前のセルから返された変数と一致する必要がある
2. 引数の順序は重要（戻り値の順序と一致）
3. `mo` は通常最初のセルでインポートし、他のセルで引数として受け取る

### 例
```python
@app.cell
def __():
    import marimo as mo
    data = {"key": "value"}
    return mo, data

@app.cell  
def __(mo, data):  # 引数の順序が重要
    mo.md(f"Data: {data}")
    result = process_data(data)
    return result,
```

## 一般的なエラーと修正

### 1. API名の間違い
```python
# ❌ 間違い
mo.output.write()  # 古いAPI
mo.ui.text_input() # 存在しない
mo.ui.refresh_on() # 存在しない

# ✅ 正しい
print()           # 標準のprint()を使用
mo.ui.text()      # 正しいAPI
# refresh機能は不要な場合が多い
```

### 2. 辞書の初期化
```python
# ❌ 間違い
return {{}}  # TypeError: unhashable type: 'dict'

# ✅ 正しい
return {}
```

### 3. 状態管理
```python
# シンプルな状態管理を推奨
@app.cell
def __(mo):
    # 複雑な状態管理は避ける
    config_data = {"default": "value"}
    return config_data,

@app.cell
def __(mo, config_data):
    # 直接値を使用
    display_value = config_data.get("key", "default")
    return display_value,
```

### 4. 変数名の重複エラー
```python
# ❌ 間違い - 複数セルで同じ変数名を使用
@app.cell
def cell1():
    symbol = "EURUSD"
    table_data = [...]
    return symbol, table_data

@app.cell  
def cell2():
    symbol = "GBPUSD"  # エラー: 'symbol' was defined by another cell
    table_data = [...]  # エラー: 'table_data' was defined by another cell
    return symbol, table_data

# ✅ 正しい - 一意な変数名を使用
@app.cell
def cell1():
    selected_symbol = "EURUSD"
    symbols_table_data = [...]
    return selected_symbol, symbols_table_data

@app.cell  
def cell2():
    ohlc_symbol = "GBPUSD"
    ohlc_table_data = [...]
    return ohlc_symbol, ohlc_table_data
```

## 変換手順

### Step 1: ファイル構造の準備
1. marimoヘッダーを追加
2. メイン実行部分を最後のif文に移動
3. クラス定義とインポートを最初のセルに配置

### Step 2: セル分割の戦略
1. **インポートセル**: ライブラリとmarimoのインポート
2. **クラス/関数定義セル**: 再利用可能なコンポーネント
3. **データ初期化セル**: 初期データや設定の作成
4. **UI作成セル**: インタラクティブ要素の作成
5. **表示セル**: 結果の表示

### Step 3: インタラクティブ要素の追加
1. 既存のprint()文をmo.md()に変換（必要に応じて）
2. ユーザー入力をUIコンポーネントに変換
3. ボタンやフォームを追加してインタラクション改善

## marimoの利点を活かす

### リアクティブ性
- セルの実行順序を気にせず、依存関係で自動実行
- UIコンポーネントの値変更で関連セルが自動更新

### データ可視化
```python
# インタラクティブなデータ表示
mo.ui.table(dataframe)  # 検索・フィルタ機能付き

# マークダウンでの動的表示
mo.md(f"## Current value: {variable}")
```

### フォーム統合
```python
form = mo.ui.form({
    "name": mo.ui.text(),
    "age": mo.ui.number(),
    "submit": mo.ui.button()
})
```

## トラブルシューティング

### よくある問題
1. **セル実行エラー**: 変数の依存関係を確認
2. **UI要素が動作しない**: API名を最新版と照合
3. **状態が保持されない**: mo.state()の使用は最小限に
4. **変数名重複エラー**: 複数セルで同じ変数名を使用している

### デバッグのコツ
1. シンプルなセルから始める
2. 一度に一つの機能を追加
3. エラーメッセージからAPI名を確認
4. 変数名は目的別にプレフィックスを付ける（例：`symbols_`, `ohlc_`, `config_`）
5. marimoが自動でファイルを整理することを考慮する

## まとめ

marimo変換時の重要ポイント:
- ✅ 正しいAPI名の使用 (`mo.ui.text`, `mo.ui.button`など)
- ✅ 適切なセル分割と変数依存関係の管理
- ✅ シンプルな状態管理
- ✅ marimoの標準構造の遵守
- ✅ 一意な変数名の使用（重複回避）
- ✅ 目的別変数命名（プレフィックス使用）
- ❌ 古いAPIや存在しない関数の使用回避
- ❌ 複雑な状態管理の回避
- ❌ セル間での変数名重複の回避

## marimoの自動整理機能への対応

### 自動変更される項目
marimoは保存時に以下を自動で整理します：
1. **関数名**: `__()` → `_()` に変更される場合がある
2. **バージョン**: `__generated_with` の値が最新バージョンに更新
3. **return文**: タプル形式に統一される（例：`return (var1, var2)`）
4. **インポート順序**: アルファベット順に整理

### 対応策
- 自動整理を前提とした命名戦略を採用
- 変数名の重複は事前に回避
- marimoが変更する箇所は手動修正しない
- バージョン番号は気にしない（自動更新される）