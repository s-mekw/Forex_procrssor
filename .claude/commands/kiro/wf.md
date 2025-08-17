---
name: wf
description: ワークフローループ実行（計画→実装→レビューを全ステップ完了まで繰り返し）
---

## 実行内容

このコマンドが実行されたら、以下のループを全ステップ完了まで自動的に繰り返します。

### ワークフローループ
```
1. @agent-planner → 計画作成/更新（前処理: `docs/context.md` と `docs/plan.md` を読み込む）
2. @agent-executor → 1 ステップのみ実装（前処理: `docs/context.md` と `docs/plan.md` を読み込む）
3. @agent-reviewer → 実装をレビュー（前処理: `docs/context.md` と `docs/plan.md` を読み込む）
4. レビュー結果判定：
   - 必須修正あり → @agent-executorで修正 → 3へ
   - 修正なし → 次へ
5. 進捗確認：
   - 未完了 → 1へ戻る
   - 完了 → 終了
```

### 重要な動作仕様
- **1 回の実装 = 1 ステップのみ**
- **各エージェント呼び出し前に必ず `docs/context.md` と `docs/plan.md` を読み込む（最新状態を把握）**
- **各実装ステップ後に必ずレビュー**
- **レビュー後に必ず次の計画を更新**
- **`docs/context.md` で進捗・コンテキスト管理(UTF-8)**
- **`docs/plan.md` で実行計画管理(UTF-8)**
- **ユーザーの停止指示まで自動継続**

## 内部処理

コマンド受信時の処理：
```python
def execute_wf(task_description):
    print(f"🚀 ワークフロー開始: {task_description}")
    
    # 初回計画
    call("@agent-planner", task_description)
    step_count = 1
    
    while True:
        print(f"\n📍 Step {step_count}")
        
        # 実装
        print(f"🔨 実装中...")
        call("@agent-executor", f"docs/context.mdのStep {step_count}を実装")
        
        # レビュー
        print(f"👁️ レビュー中...")
        call("@agent-reviewer", "直前の実装をレビュー")
        
        # レビュー結果確認
        if has_critical_issues():
            print(f"🔧 修正中...")
            call("@agent-executor", "レビュー指摘を修正")
            call("@agent-reviewer", "修正内容を再レビュー")
        
        # 完了確認
        if all_steps_completed():
            print("✅ 全ステップ完了")
            break
        
        # 次のステップ
        print(f"📋 次のステップを計画中...")
        call("@agent-planner", "次のステップを計画")
        step_count += 1
    
    print("🎉 ワークフロー完了")
```

## 停止方法
- Ctrl+C または「停止」と入力
- 特定のステップで停止したい場合は「Step Nで停止」と指定

## 進捗確認
実行中の進捗は`docs/context.md`で確認できます。
