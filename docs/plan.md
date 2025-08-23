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

#### **Step 2: テストファイルの作成**
`tests/unit/test_indicators.py`
- 基本構造: pytest形式
- テストケース:
  - EMA計算の精度検証（既知の値との比較）
  - 複数期間の同時計算
  - 空データ/異常値の処理
  - メモリ効率の検証

#### **Step 3: TechnicalIndicatorEngineクラスの基本実装**
```python
class TechnicalIndicatorEngine:
    def __init__(self):
        # Float32最適化の設定
        # ログ設定
    
    def calculate_ema(self, df: pl.DataFrame, periods: list[int]) -> pl.DataFrame:
        # EMA計算のメインメソッド
```

#### **Step 4: EMA計算ロジック**
- polars-ta-extensionのEMA関数を使用
- 期間: [5, 20, 50, 100, 200]
- Float32型での処理（メモリ効率）
- 並列計算の実装

#### **Step 5: 増分計算メカニズム**
```python
def update_ema_incremental(self, existing_df: pl.DataFrame, new_data: pl.DataFrame) -> pl.DataFrame:
    # 既存のEMAに新しいデータポイントを追加
    # バッファ管理（古いデータの削除）
```

#### **Step 6: エラーハンドリング**
- 入力データ検証
- 計算エラーのキャッチとログ
- デフォルト値での継続処理

### 🔗 参照ドキュメント
- 要件定義: `../.kiro/specs/Forex_procrssor/requirements.md` (要件2.2)
- 詳細設計: `../.kiro/specs/Forex_procrssor/design.md` (セクション2.2)
- 既存実装: `src/data_processing/processor.py` (PolarsProcessingEngine)
- 共通モデル: `src/common/models.py` (データ構造)

### ✅ 受け入れ条件（要件2.2より）
1. ✅ WHEN EMA計算を実行する THEN 5、20、50、100、200期間のEMAを同時計算する
2. ✅ WHEN polars-ta-extensionを使用する THEN ネイティブPolars表現で高速計算を実現する
3. ✅ WHEN 指標更新が要求される THEN 新しいデータポイントのみで増分計算を実行する
4. ✅ WHEN 複数指標を組み合わせる THEN 効率的なパイプライン処理で一括計算する
5. ✅ IF 計算エラーが発生 THEN エラー詳細をログに記録し、デフォルト値で継続する

### 🚀 実装の優先順位
1. **必須**: 基本的なEMA計算機能（Step 2-4）
2. **重要**: 増分計算（Step 5）
3. **推奨**: エラーハンドリングとログ（Step 6）
4. **任意**: パフォーマンス最適化と統合テスト

### 📝 技術的な決定事項
- **ライブラリ**: polars-ta-extension (https://github.com/Yvictor/polars_ta_extension)
- **データ型**: Float32統一（メモリ効率）
- **並列処理**: Polarsのネイティブ並列処理を活用
- **メモリ管理**: 既存のPolarsProcessingEngineのメモリ管理機能を継承

### ⚠️ 注意事項
- polars-ta-extensionの依存関係に注意（polars>=0.19.0が必要）
- 既存のprocessor.pyとの一貫性を保つ
- Float32型への変換を忘れない
- ログレベルの統一（既存コードと同じ規約）