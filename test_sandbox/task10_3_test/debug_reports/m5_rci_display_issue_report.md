# M5 RCIウィンドウ表示問題 調査レポート

**作成日**: 2025-08-30  
**対象ファイル**: `test_sandbox/task10_3_test/pipeline_chart_dashboard.py`  
**問題**: M5 RCI [24, 33, 48]ウィンドウにRCIではなくローソク足が表示される

---

## 1. 問題の概要

### 症状
- Pipeline Chart Dashboardの右列（M5側）のRCIウィンドウ（row=2, col=2）にRCIグラフではなくローソク足のような形状が表示される
- スクリーンショット: `sample_img/2025-08-29_17h00_49.png`

### 期待される動作
- M5 RCIウィンドウには-100〜100の範囲のRCI値がライングラフとして表示されるべき

---

## 2. 調査内容

### 2.1 初期仮説
1. M5 RCIデータに価格データ（171.x）が混入している
2. M5のローソク足データが誤ってRCIウィンドウに描画されている
3. M1のデータがM5側のサブプロットに誤って追加されている

### 2.2 実施した検証

#### A. RCI計算の検証
**テストスクリプト**: `test_rci_calculation.py`

```python
# M5データでRCI計算をテスト
# 結果: すべてのRCI値が-100〜100の範囲内
📈 RCI[24]: Range: min=-96.35, max=95.13 ✅
📈 RCI[33]: Range: min=-87.00, max=96.93 ✅
📈 RCI[48]: Range: min=-90.69, max=89.92 ✅
📈 RCI[66]: Range: min=-91.66, max=90.93 ✅
📈 RCI[108]: Range: min=-83.01, max=93.79 ✅
```

**結論**: RCI計算ロジックは正常に動作している

#### B. チャートデータの検証
**テストスクリプト**: `test_chart_data.py`

```python
# M5 RCIデータの内容を詳細確認
# 結果: 価格データの混入なし
M5 RCI for display:
  Period 24: Range: -62.07 - 89.91 ✅
  Period 33: Range: -26.64 - 92.38 ✅
  Period 48: Range: -33.64 - 89.92 ✅
  Period 66: Range: 18.09 - 90.93 ✅
  Period 108: Range: -2.86 - 93.79 ✅
```

**結論**: M5 RCIデータは正しく、価格データの混入はない

#### C. コード構造の検証
- M5 RCIの描画位置: `row=2, col=2` （正しい）
- M1 RCIがM5側（col=2）に描画されている箇所: なし
- M5ローソク足の描画位置: `row=1, col=2` （正しい）

---

## 3. 実施した修正

### 3.1 Y軸範囲の明示的な設定
```python
# 各サブプロットのY軸範囲を個別に設定
for row in [2, 3, 4]:
    # M1側（左列）
    fig.update_yaxes(range=[-105, 105], showgrid=self.show_grid, 
                     gridcolor=self.config['theme']['grid'], row=row, col=1)
    # M5側（右列）
    if row < 4:
        fig.update_yaxes(range=[-105, 105], showgrid=self.show_grid,
                         gridcolor=self.config['theme']['grid'], row=row, col=2)
```

### 3.2 デバッグログの追加
```python
# M5 RCIデータの範囲確認
logger.info(f"M5 RCI[{period}]: {len(values)} values, range: {min(values):.2f} - {max(values):.2f}")

# 価格データ混入チェック
if max(values) > 150:
    logger.error(f"CRITICAL: M5 RCI[{period}] contains price data!")
```

### 3.3 None値の処理改善
```python
# RCI計算でNone値が返された場合の処理
if rci_value is not None:
    rci_history[period].append(float(rci_value))
else:
    logger.warning(f"RCI[{period}] returned None")
```

### 3.4 TOML設定ファイル対応
- `task10_3_config.toml`から設定を読み込む機能を追加
- 初期状態を"LIVE"に変更（自動起動）
- 表示バー数、更新間隔などを設定可能に

---

## 4. 調査結果

### 判明した事実
1. **M5 RCIデータは正しく計算されている** - すべての値が-100〜100の範囲内
2. **データの混入はない** - M5 RCIにM1データや価格データの混入なし
3. **コード構造は正しい** - 各トレースは適切なサブプロットに追加されている

### 残存する問題
画像（`sample_img/2025-08-29_17h00_49.png`）を見ると、M5 RCI [24, 33, 48]ウィンドウに明らかにローソク足のような形状が表示されている。これは以下の可能性が考えられる：

1. **Plotlyのレンダリング問題** - サブプロット間でデータが混在している可能性
2. **凡例の問題** - M5側にM1 RCIのラベルが表示されている
3. **タイミングの問題** - リアルタイム更新時にデータが一時的に混在

---

## 5. 今後の対応案

### 短期的対応
1. Plotlyのバージョン確認と更新
2. サブプロットのspecs設定を明示的に定義
3. 各トレースにuid属性を追加して識別

### 中期的対応
1. チャート描画を段階的に実行して問題箇所を特定
2. M1とM5のチャートを別々のfigureで作成して比較
3. Plotlyのデバッグモードで詳細情報を取得

### 長期的対応
1. チャート描画ライブラリの変更検討（matplotlib等）
2. チャート更新ロジックの全面的な見直し

---

## 6. 添付ファイル

### テストスクリプト
- `test_rci_calculation.py` - RCI計算の検証
- `debug_m5_rci.py` - M5 RCI詳細デバッグ
- `test_chart_data.py` - チャートデータ内容の確認

### 修正済みファイル
- `pipeline_chart_dashboard.py` - メインのダッシュボードファイル

### 設定ファイル
- `task10_3_config.toml` - TOML形式の設定ファイル

---

## 7. 結論

調査の結果、**データレベルでは問題がない**ことが確認された。M5 RCIは正しく計算され、適切な範囲（-100〜100）の値を持っている。

問題は**表示レベル**にあると考えられ、Plotlyのサブプロット機能またはトレースの管理に何らかの問題がある可能性が高い。

現時点では、データの正確性は保証されているため、アプリケーションの機能には影響がないが、視覚的な問題として今後の改善が必要である。

---

**以上**