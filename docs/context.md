# プロジェクト進捗
###  ファイル配置

```
src/
└── data_processing/
    ├── __init__.py        # RCICalculatorEngineをエクスポート追加
    ├── rci.py            # 新規作成（メイン実装）
    └── processor.py      # 既存（統合ポイント）
```

## . 実装ステップ

### Phase 1: 基礎実装
- [ ] Step 1.1: DifferentialRCICalculatorクラスの実装
  - dequeによるスライディングウィンドウ
  - Float32精度での計算
  - 基本的な差分更新ロジック

- [ ] Step 1.2: ランキング計算の最適化
  - scipy.stats.rankdataによる正確なタイ処理
  - NumPyフォールバック実装
  - パフォーマンステスト

### Phase 2: エンジン実装
- [ ] Step 2.1: RCICalculatorEngineクラスの実装
  - 複数期間の並列計算
  - バリデーションロジック
  - エラーハンドリング

- [ ] Step 2.2: バッチ/ストリーミングモード実装
  - モード切り替えロジック
  - 大規模データ用バッチ処理
  - リアルタイム用ストリーミング処理

### Phase 3: Polars統合
- [ ] Step 3.1: RCIProcessorクラスの実装
  - Polars Expression生成
  - LazyFrame対応
  - 最適化されたグループ処理

- [ ] Step 3.2: 既存processor.pyとの統合
  - PolarsProcessingEngineへの組み込み
  - パイプライン統合テスト

### Phase 4: テストとドキュメント
- [ ] Step 4.1: ユニットテスト作成
  - tests/unit/test_rci.pyの実装
  - 各クラスの単体テスト
  - エッジケースのテスト

- [ ] Step 4.2: 統合テストとベンチマーク
  - 大規模データでのパフォーマンステスト
  - メモリ使用量の測定
  - 精度検証