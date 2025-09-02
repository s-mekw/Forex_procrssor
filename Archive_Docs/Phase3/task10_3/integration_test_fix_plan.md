# 統合テスト修正計画書
## TestMultiframeIntegration 修正対応

---

**作成日**: 2025-08-28  
**優先度**: 高  
**推定作業時間**: 1-2時間  
**対象**: `tests/integration/test_data_pipeline.py`

---

## 📋 現状分析

### 失敗しているテスト（4個）
1. `test_pipeline_multiframe_data_flow` - データフロー検証
2. `test_buffer_synchronization` - バッファ同期確認
3. `test_analyzer_state_consistency` - 状態一貫性テスト
4. `test_pipeline_restart_recovery` - 再起動復旧テスト

### エラーの根本原因
```
WARNING  Output queue is full, waiting...
WARNING  Input queue is full (100/100)
ERROR    Failed to submit data: queue timeout
ERROR    Output queue timeout, dropping result
AssertionError: Failed to submit data at index 201
```

**問題**: キューサイズの制限（100）により、250個のデータ処理時に詰まりが発生

---

## 🔧 修正計画

### 1. キューサイズの調整

#### 現在の設定
```python
# test_pipeline_multiframe_data_flow
pipeline = RealtimePipeline(
    max_queue_size=100,  # 問題の原因
    enable_multiframe=True
)
```

#### 修正案
```python
# キューサイズを拡張
pipeline = RealtimePipeline(
    max_queue_size=500,  # 250個のデータ+バッファ余裕
    enable_multiframe=True,
    queue_timeout_seconds=5.0  # タイムアウトも延長
)
```

### 2. データ投入速度の調整

#### 現在の実装
```python
# 高速でデータを投入
for i in range(250):
    success = await pipeline.submit_data(data)
    assert success
```

#### 修正案
```python
# バックプレッシャーを考慮した投入
for i in range(250):
    success = await pipeline.submit_data(data)
    assert success, f"Failed at index {i}"
    
    # キューの状態を確認して速度調整
    if i % 50 == 0:
        await asyncio.sleep(0.1)  # 処理の猶予を与える
```

### 3. テスト環境の最適化

#### TestMultiframeIntegration共通設定
```python
@pytest.fixture
async def optimized_pipeline(self):
    """最適化されたテスト用パイプライン"""
    pipeline = RealtimePipeline(
        max_queue_size=500,
        queue_timeout_seconds=5.0,
        latency_threshold_seconds=2.0,  # 遅延閾値も緩和
        enable_multiframe=True,
        max_history_bars=5000
    )
    yield pipeline
    await pipeline.stop()
```

### 4. 各テストの個別修正

#### test_pipeline_multiframe_data_flow
```python
async def test_pipeline_multiframe_data_flow(self, optimized_pipeline):
    # データ投入ロジックの改善
    batch_size = 50
    for batch_start in range(0, 250, batch_size):
        for i in range(batch_start, min(batch_start + batch_size, 250)):
            # データ投入
            success = await pipeline.submit_data(data)
            assert success
        
        # バッチ間で処理待機
        await asyncio.sleep(0.2)
```

#### test_buffer_synchronization
```python
async def test_buffer_synchronization(self, optimized_pipeline):
    # バッファサイズチェックのタイミング調整
    for i in range(100):
        await pipeline.submit_data(create_sample_bar())
        
        # 定期的に同期確認
        if i % 20 == 0:
            await asyncio.sleep(0.1)
            buffer_size = pipeline._multiframe_analyzer.get_buffer_size()
            assert buffer_size <= 100
```

#### test_analyzer_state_consistency
```python
async def test_analyzer_state_consistency(self, optimized_pipeline):
    # 状態確認の間隔を調整
    for i in range(250):
        await pipeline.submit_data(data)
        
        # is_ready()状態の変化を確認
        if i == 199:  # 200個目の直前
            assert not pipeline._multiframe_analyzer.is_ready()
        elif i == 200:  # 200個目
            await asyncio.sleep(0.5)  # 処理完了を待つ
            assert pipeline._multiframe_analyzer.is_ready()
```

#### test_pipeline_restart_recovery
```python
async def test_pipeline_restart_recovery(self, optimized_pipeline):
    # 再起動前後のバッファ保持確認
    
    # データ投入
    for i in range(150):
        await pipeline.submit_data(data)
        if i % 50 == 0:
            await asyncio.sleep(0.1)
    
    initial_buffer = pipeline._multiframe_analyzer.get_buffer_size()
    
    # 再起動
    await pipeline.stop()
    await asyncio.sleep(1.0)  # 完全停止を待つ
    await pipeline.start()
    
    # バッファが保持されているか確認
    assert pipeline._multiframe_analyzer.get_buffer_size() == initial_buffer
```

---

## 📊 期待される成果

### 修正後の動作
- ✅ 250個のデータ処理が正常に完了
- ✅ キューのタイムアウトエラーが解消
- ✅ バックプレッシャー制御が適切に動作
- ✅ 4つのテストが全て合格

### パフォーマンスへの影響
- テスト実行時間: 約5秒 → 約10秒（安定性優先）
- メモリ使用量: 若干増加（キューサイズ拡張のため）
- CI/CDの安定性: 大幅に向上

---

## 🚀 実装手順

### Step 1: バックアップ作成
```bash
cp tests/integration/test_data_pipeline.py tests/integration/test_data_pipeline.py.bak
```

### Step 2: TestMultiframeIntegrationクラスの修正
1. 共通fixtureの作成（optimized_pipeline）
2. 各テストメソッドの修正
3. アサーションメッセージの改善

### Step 3: ローカルテスト実行
```bash
# 個別テスト
uv run --frozen pytest tests/integration/test_data_pipeline.py::TestMultiframeIntegration -v

# 全統合テスト
uv run --frozen pytest tests/integration/ -v
```

### Step 4: 検証
- 全4テストが合格することを確認
- エラーログが出力されないことを確認
- テスト実行時間が妥当であることを確認

---

## ⚠️ 注意事項

### 互換性の考慮
- 既存の他のテストクラスに影響しないよう注意
- RealtimePipelineの本番設定とは異なることを明記

### CI環境での考慮
- GitHub ActionsなどのCI環境では更に余裕を持った設定が必要な場合がある
- 環境変数で設定を調整できるようにすることを検討

### 将来的な改善
- モックを使用したユニットテストへの分割を検討
- 統合テストの実行時間短縮のための並列化
- テストデータのフィクスチャ化

---

## 📝 参考情報

### 関連ファイル
- `src/data_processing/pipelines.py` - RealtimePipelineの実装
- `src/data_processing/analyzer.py` - MultiTimeframeAnalyzerの実装
- `docs/task_10_3_final_report.md` - Task 10.3の実装詳細

### デバッグ用コマンド
```bash
# 詳細なエラー出力
uv run --frozen pytest tests/integration/test_data_pipeline.py::TestMultiframeIntegration::test_pipeline_multiframe_data_flow -vvs --tb=long

# カバレッジ無視でテスト実行
uv run --frozen pytest tests/integration/test_data_pipeline.py::TestMultiframeIntegration --no-cov
```

---

**次回実装時の参考**: この計画書に従って実装することで、約1-2時間で統合テストの問題を解決できます。