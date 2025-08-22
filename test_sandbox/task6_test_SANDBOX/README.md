# Task 6 - Tick to Bar Converter Visual E2E Tests

このディレクトリには、`src/mt5_data_acquisition/tick_to_bar.py`のTickToBarConverterエンジンの視覚的なE2Eテストが含まれています。

## ✅ 動作確認済み (2025-08-21)

- **MT5接続**: Axiory-Demo サーバーへの接続成功
- **リアルタイムティック受信**: EURUSD等のリアルタイムデータ取得確認
- **ティック→バー変換**: 1分足バーの生成を確認
- **視覚的表示**: Rich libraryによるCLI表示が正常動作

## 📁 ディレクトリ構造

```
task6_test_SANDBOX/
├── utils/
│   ├── bar_display_helpers.py      # 表示用ヘルパー関数（タイムゾーン修正済み）
│   └── converter_visualizer.py     # 変換プロセスの可視化
├── 01_basic_tick_to_bar.py        # ✅ 基本的な変換動作確認（MT5接続版）
├── 01_basic_tick_to_bar_demo.py   # ✅ デモ版（シミュレートデータ）
├── 01_basic_test_simple.py        # ✅ シンプル版（軽量実装）
├── 02_realtime_bar_builder.py     # リアルタイムバー生成の可視化
├── 03_multi_symbol_converter.py   # 複数通貨ペアの同時変換
├── 04_gap_detection_test.py       # ティック欠損検知テスト
├── 06_marimo_bar_dashboard.py     # インタラクティブダッシュボード
├── test_mt5_connection.py          # MT5接続テスト
├── test_tick_format.py             # ティック形式確認
├── test_tick_timezone.py           # タイムゾーン検証
└── README.md                       # このファイル
```

## 🚀 実行方法

### 前提条件
- MT5がインストールされ、デモアカウントでログイン可能
- Python環境に必要なパッケージがインストール済み
- `.env`ファイルにMT5認証情報が設定済み（FOREX_プレフィックス必須）
- **重要**: プロジェクトルートディレクトリから実行すること

### 環境変数設定（.env）
```env
# FOREX_プレフィックスが必須
FOREX_MT5_LOGIN=your_demo_login
FOREX_MT5_PASSWORD=your_demo_password
FOREX_MT5_SERVER=Axiory-Demo  # または使用するサーバー名
FOREX_MT5_TIMEOUT=60000
```

### 各テストの実行

#### 1. 基本的な変換動作確認（MT5接続版）
```bash
# プロジェクトルートから実行
uv run test_sandbox/task6_test_SANDBOX/01_basic_tick_to_bar.py
```
- MT5からリアルタイムティックを取得
- 1分バーへの変換プロセスを可視化
- OHLC値の更新とバー完成をリアルタイム表示
- **注意**: バッファサイズの調整が必要な場合があります

#### 1b. デモ版（MT5接続不要）
```bash
uv run test_sandbox/task6_test_SANDBOX/01_basic_tick_to_bar_demo.py
```
- シミュレートされたティックデータを使用
- MT5接続なしで動作確認可能
- 機能テストに最適

#### 2. リアルタイムバー生成の可視化
```bash
uv run test_sandbox/task6_test_SANDBOX/02_realtime_bar_builder.py
```
- バー形成過程をアニメーション表示
- パイプラインビューで変換フローを可視化
- パフォーマンスメトリクスの表示

#### 3. 複数通貨ペアの同時変換
```bash
uv run test_sandbox/task6_test_SANDBOX/03_multi_symbol_converter.py
```
- EURUSD、GBPUSD、USDJPYを同時処理
- 各通貨ペアの変換状況を並列表示
- CPU/メモリ使用状況のモニタリング

#### 4. ティック欠損検知テスト
```bash
uv run test_sandbox/task6_test_SANDBOX/04_gap_detection_test.py
```
- ギャップをシミュレートして検出機能をテスト
- ギャップイベントのタイムライン表示
- 警告ログと統計情報の可視化

#### 6. Marimoダッシュボード
```bash
uv run marimo edit test_sandbox/task6_test_SANDBOX/06_marimo_bar_dashboard.py
```
- Webベースのインタラクティブダッシュボード
- リアルタイムチャート（ティック、OHLC、ボリューム）
- パラメータをスライダーで動的に調整可能

## 🎯 主な機能

### TickToBarConverterの機能確認
- ✅ リアルタイムティック処理
- ✅ 未完成バーの継続更新
- ✅ バー完成時のコールバック通知
- ✅ ティック欠損検知（設定可能な閾値）
- ✅ エラーハンドリング（タイムスタンプ逆転、無効データ）
- ✅ メモリ管理（最大バー数制限）

### 視覚化要素
- 📊 Rich Liveによる動的ターミナル表示
- 📈 アスキーアートによる簡易チャート
- 🎨 カラーコーディング（価格変動、警告レベル）
- 📱 プログレスバーとアニメーション
- 🌐 Marimoによるインタラクティブweb UI

## 📊 表示される情報

### リアルタイムメトリクス
- **処理速度**: ティック/秒、バー/分
- **変換率**: 平均ティック数/バー
- **ギャップ統計**: 検出回数、最大ギャップ時間
- **エラー/警告**: リアルタイムカウント

### バー情報
- **OHLC値**: Open、High、Low、Close
- **ボリューム**: 累積取引量
- **ティック数**: バー内のティック数
- **平均スプレッド**: Bid-Askスプレッドの平均

## ⚠️ 注意事項

1. **MT5接続**: 実際のMT5接続が必要（デモアカウント推奨）
2. **市場時間**: FX市場が開いている時間帯でテスト推奨
3. **リソース使用**: 複数テスト同時実行は避ける
4. **データ保存**: テストデータは保存されません（表示のみ）

## 🔧 トラブルシューティング

### MT5接続エラー

#### `'BaseConfig' object has no attribute 'mt5'`
- **原因**: 設定フィールド名の誤り
- **解決**: `config.mt5_login`、`config.mt5_password`等を使用

#### `Invalid "login" argument`
- **原因**: MT5ConnectionManagerの引数キー誤り
- **解決**: `"account"` キーを使用（`"login"`ではない）
```python
mt5_config = {
    "account": config.mt5_login,  # "login"ではなく"account"
    "password": config.mt5_password.get_secret_value(),
    "server": config.mt5_server,
    "timeout": config.mt5_timeout,
}
```

#### 環境変数が読み込まれない
- **原因**: 実行ディレクトリが異なる
- **解決**: プロジェクトルートから実行、またはFOREX_プレフィックスを確認

### タイムゾーンエラー

#### `can't compare offset-naive and offset-aware datetimes`
- **原因**: MT5のティックがタイムゾーン付き、ローカルdatetimeがタイムゾーンなし
- **解決**: bar_display_helpers.pyで修正済み
```python
# タイムゾーンを統一
bar_time = current_bar.time.replace(tzinfo=None) if current_bar.time.tzinfo else current_bar.time
```

### Tickモデルの互換性問題

#### `'Tick' object is not subscriptable`
- **原因**: `common.models.Tick`と`tick_to_bar.Tick`の属性名が異なる
- **解決**: 変換処理を実装
```python
if hasattr(tick_data, 'timestamp'):
    tick = Tick(
        symbol=tick_data.symbol,
        time=tick_data.timestamp,  # timestampをtimeにマップ
        bid=Decimal(str(tick_data.bid)),
        ask=Decimal(str(tick_data.ask)),
        volume=Decimal(str(tick_data.volume)) if tick_data.volume else Decimal("1.0")
    )
```

### バッファオーバーフロー
- **症状**: `Buffer full - dropping ticks`警告
- **原因**: ティック消費速度が生成速度に追いつかない
- **解決**: 
  - バッファサイズを増やす: `buffer_size=5000`
  - 処理間隔を短くする: `await asyncio.sleep(0.1)`
  - 処理済みティックの追跡を実装

### 表示が更新されない
- ティックが受信されているか確認（市場時間外の可能性）
- シンボルが取引可能か確認
- ストリーミングが開始されているか確認

### Marimoダッシュボードが起動しない
```bash
# Marimoがインストールされているか確認
uv add marimo

# ブラウザが自動で開かない場合
uv run marimo edit test_sandbox/task6_test_SANDBOX/06_marimo_bar_dashboard.py --port 8080
```

## 📝 各テストの特徴

| テスト | 主な目的 | 表示方法 | 動作状況 | 特徴 |
|--------|----------|----------|----------|------|
| 01_basic | 基本動作確認 | Rich Terminal | ✅ 動作確認済み | MT5リアルタイム接続 |
| 01_basic_demo | デモ動作 | Rich Terminal | ✅ 完全動作 | シミュレートデータ |
| 01_basic_simple | 軽量版 | シンプル出力 | ✅ 動作確認済み | バッファ管理改善 |
| 02_realtime | プロセス可視化 | アニメーション | 🔧 要テスト | 動的な表示 |
| 03_multi | 並列処理 | 比較テーブル | 🔧 要テスト | パフォーマンス重視 |
| 04_gap | エラー処理 | タイムライン | 🔧 要テスト | 異常検知フォーカス |
| 06_marimo | 総合ダッシュボード | Web UI | 🔧 要テスト | インタラクティブ |

## 🎓 学習ポイント

これらのテストを通じて以下を確認できます：

1. **ティック→バー変換の仕組み**
   - タイムウィンドウによるバー区切り
   - OHLC値の更新ロジック
   - ボリューム累積

2. **リアルタイム処理**
   - 非同期ストリーミング
   - バッファ管理
   - レート制御

3. **エラー処理**
   - ティック欠損の検出
   - タイムスタンプ検証
   - 無効データの除外

4. **可視化技術**
   - Richによるターミナルグラフィック
   - Marimoによるインタラクティブダッシュボード
   - リアルタイムチャート更新

## 🎯 実装の成果

### 達成した機能
- ✅ MT5からのリアルタイムティック取得
- ✅ ティック→1分足バーへの変換
- ✅ バー完成時のコールバック通知
- ✅ Rich libraryによる視覚的表示
- ✅ タイムゾーン問題の解決
- ✅ Tickモデル互換性の実装

### 確認されたバー生成例
```
✅ Bar #1 completed at 07:29:00
   OHLC: 1.16452 / 1.16452 / 1.16450 / 1.16450
   Volume: 4.00, Ticks: 4
```

### 今後の改善点
- 📌 バッファ管理の最適化（高頻度ティック対応）
- 📌 タイムスタンプ逆転への対処
- 📌 複数シンボル同時処理の最適化

---

*これらのテストはTask 6のTickToBarConverterの動作を視覚的に確認し、実際の取引環境での振る舞いを体験するために設計されています。*

**最終更新**: 2025-08-21 | **動作確認済み環境**: Axiory-Demo (Login: 20046505)