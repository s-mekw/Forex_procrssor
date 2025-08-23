## Task 8: テクニカル指標計算エンジンの実装計画

### 📌 タスク概要
**要件: 2.2** of `../.kiro/specs/Forex_procrssor/requirements.md`
- polars-ta-extensionを使用したEMA（5、20、50、100、200期間）計算の実装
- 増分計算による効率的な更新メカニズムの追加
- 既存のPolarsProcessingEngineとの整合性を保った設計

### 📂 実装ファイル構成

#### 本体実装
- `src/data_processing/indicators.py` - TechnicalIndicatorEngineクラス

#### テスト実装
- `tests/unit/test_indicators.py` - ユニットテスト
- `tests/integration/test_indicators_integration.py` - 統合テスト（任意）

### 🔧 詳細実装計画

#### **Step 1: 環境セットアップ** ✅
```bash
# polars-ta-extensionのインストール
pip install polars-ta-extension
# pyproject.tomlへの追加
```

#### **Step 2: テストファイルの作成** ✅
`tests/unit/test_indicators.py`
- 基本構造: pytest形式
- テストケース:
  - EMA計算の精度検証（既知の値との比較）
  - 複数期間の同時計算
  - 空データ/異常値の処理
  - メモリ効率の検証
**実装完了**: 18個の包括的なテストケースが全て成功

#### **Step 3: 追加テクニカル指標の実装** 🚀

##### 3.1 RSI (Relative Strength Index) 実装
```python
def calculate_rsi(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """
    RSI計算の実装
    - 価格変化の計算
    - 上昇幅と下落幅の分離
    - RS (Relative Strength) = 平均上昇幅 / 平均下落幅
    - RSI = 100 - (100 / (1 + RS))
    """
```

##### 3.2 MACD 実装
```python
def calculate_macd(self, df: pl.DataFrame, 
                   fast_period: int = 12, 
                   slow_period: int = 26, 
                   signal_period: int = 9) -> pl.DataFrame:
    """
    MACD計算の実装
    - MACDライン = 12日EMA - 26日EMA
    - シグナルライン = MACDラインの9日EMA
    - MACDヒストグラム = MACDライン - シグナルライン
    """
```

##### 3.3 ボリンジャーバンド実装
```python
def calculate_bollinger_bands(self, df: pl.DataFrame, 
                              period: int = 20, 
                              std_dev: float = 2.0) -> pl.DataFrame:
    """
    ボリンジャーバンド計算
    - 中心線 = 20日SMA
    - 上バンド = 中心線 + (2 × 標準偏差)
    - 下バンド = 中心線 - (2 × 標準偏差)
    """
```

##### 3.4 メタデータ管理
```python
def _track_metadata(self, indicator_name: str, params: dict) -> None:
    """計算済み指標のメタデータ管理"""
    
ndef get_indicator_metadata(self) -> dict:
    """計算済み指標の情報取得"""
```

##### 3.5 バッチ処理最適化
```python
def calculate_all_indicators(self, df: pl.DataFrame, 
                             config: dict = None) -> pl.DataFrame:
    """
    全指標の一括計算
    - 並列処理の最適化
    - メモリ効率の考慮
    - エラーハンドリング
    """
```

#### **Step 4: 統合テストの実装**
`tests/integration/test_indicators_integration.py`

##### 4.1 複数指標の同時計算テスト
```python
def test_calculate_all_indicators():
    """全指標の一括計算テスト"""
```

##### 4.2 リアルタイムデータシミュレーション
```python
def test_realtime_update():
    """リアルタイム更新のテスト"""
```

##### 4.3 パフォーマンスベンチマーク
```python
def test_performance_benchmark():
    """大量データでのパフォーマンステスト"""
```

#### **Step 5: パイプライン統合**

##### 5.1 RealtimePipelineとの統合
```python
# src/pipeline/realtime.pyへの統合
from src.data_processing.indicators import TechnicalIndicatorEngine

class RealtimePipeline:
    def __init__(self):
        self.indicator_engine = TechnicalIndicatorEngine()
```

##### 5.2 StreamProcessorとの連携
```python
# ストリームデータの処理フロー
def process_stream_with_indicators(self, stream_data):
    # データ受信 -> 指標計算 -> 結果出力
```

##### 5.3 データフロー最適化
- メモリ効率的なバッファリング
- キャッシュ機能の実装
- 並列処理の最適化

#### **Step 6: エラーハンドリングとログ強化**
- 入力データ検証
- 計算エラーのキャッチとログ
- デフォルト値での継続処理
- パフォーマンスメトリクスの記録

### 🔗 参照ドキュメント
- 要件定義: `../.kiro/specs/Forex_procrssor/requirements.md` (要件2.2)
- 詳細設計: `../.kiro/specs/Forex_procrssor/design.md` (セクション2.2)
- 既存実装: `src/data_processing/processor.py` (PolarsProcessingEngine)
- 共通モデル: `src/common/models.py` (データ構造)

### ✅ 受け入れ条件（要件2.2より）
1. ✅ WHEN EMA計算を実行する THEN 5、20、50、100、200期間のEMAを同時計算する
2. ✅ WHEN Polars機能を使用する THEN ネイティブPolars表現で高速計算を実現する
3. ✅ WHEN 指標更新が要求される THEN 新しいデータポイントのみで増分計算を実行する
4. ✅ WHEN 複数指標を組み合わせる THEN 効率的なパイプライン処理で一括計算する
5. ✅ IF 計算エラーが発生 THEN エラー詳細をログに記録し、デフォルト値で継続する

### 🎯 Step 3 以降の目標
- [ ] RSI、MACD、ボリンジャーバンドの実装
- [ ] メタデータ管理機能の追加
- [ ] バッチ処理最適化
- [ ] 統合テストの実装
- [ ] パイプラインとの統合

### 🚀 次の実装優先順位 (Step 3以降)
1. **必須**: RSI、MACD、ボリンジャーバンドの基本実装（Step 3.1-3.3）
2. **重要**: メタデータ管理とバッチ処理（Step 3.4-3.5）
3. **推奨**: 統合テストとパフォーマンス検証（Step 4）
4. **任意**: パイプライン統合と最適化（Step 5-6）

### 📝 技術的な決定事項
- **ライブラリ**: Polars組み込み機能 (ewm_mean, rolling 等)
- **データ型**: Float32統一（メモリ効率）
- **並列処理**: Polarsのネイティブ並列処理を活用
- **メモリ管理**: 既存のPolarsProcessingEngineのメモリ管理機能を継承
- **テストファースト**: テストを先に作成してから実装を進める

### ⚠️ 注意事項
- **変更**: polars-ta-extensionの代わりにPolars組み込み機能を使用
- 既存のprocessor.pyとの一貫性を保つ
- Float32型への変換を忘れない
- ログレベルの統一（既存コードと同じ規約）
- メモリ効率を意識した実装