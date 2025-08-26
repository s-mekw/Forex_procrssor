"""
レポートジェネレーター

テスト結果のレポートを生成するユーティリティです。
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .metrics_analyzer import PerformanceReport


@dataclass
class TestResult:
    """テスト結果"""
    
    test_name: str
    status: str  # PASSED, FAILED, SKIPPED
    duration_seconds: float
    performance_report: Optional[PerformanceReport] = None
    error_message: Optional[str] = None
    logs: list[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.logs is None:
            self.logs = []
    
    def is_passed(self) -> bool:
        """テスト成功判定"""
        return self.status == "PASSED"


class ReportGenerator:
    """レポート生成器"""
    
    def __init__(self, output_dir: str = "test_reports"):
        """
        初期化
        
        Args:
            output_dir: レポート出力ディレクトリ
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.test_results = []
        
    def add_test_result(self, result: TestResult):
        """テスト結果を追加"""
        self.test_results.append(result)
    
    def generate_markdown_report(self, filename: Optional[str] = None) -> str:
        """
        Markdownレポートを生成
        
        Args:
            filename: 出力ファイル名（Noneの場合は自動生成）
            
        Returns:
            str: レポートのファイルパス
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"e2e_test_report_{timestamp}.md"
        
        filepath = self.output_dir / filename
        
        content = self._generate_markdown_content()
        filepath.write_text(content, encoding="utf-8")
        
        return str(filepath)
    
    def generate_json_report(self, filename: Optional[str] = None) -> str:
        """
        JSONレポートを生成
        
        Args:
            filename: 出力ファイル名（Noneの場合は自動生成）
            
        Returns:
            str: レポートのファイルパス
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"e2e_test_report_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        data = self._generate_json_data()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(filepath)
    
    def generate_html_report(self, filename: Optional[str] = None) -> str:
        """
        HTMLレポートを生成
        
        Args:
            filename: 出力ファイル名（Noneの場合は自動生成）
            
        Returns:
            str: レポートのファイルパス
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"e2e_test_report_{timestamp}.html"
        
        filepath = self.output_dir / filename
        
        content = self._generate_html_content()
        filepath.write_text(content, encoding="utf-8")
        
        return str(filepath)
    
    def _generate_markdown_content(self) -> str:
        """Markdownコンテンツを生成"""
        lines = [
            "# E2Eテストレポート",
            "",
            f"## 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## テスト結果サマリー",
            "",
            self._generate_summary_table(),
            "",
            "## 詳細結果",
            "",
        ]
        
        for result in self.test_results:
            lines.extend(self._generate_test_detail_markdown(result))
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_summary_table(self) -> str:
        """サマリーテーブルを生成"""
        passed = sum(1 for r in self.test_results if r.is_passed())
        failed = len(self.test_results) - passed
        
        lines = [
            f"- **合計テスト数**: {len(self.test_results)}",
            f"- **成功**: {passed}",
            f"- **失敗**: {failed}",
            f"- **成功率**: {passed/len(self.test_results)*100:.1f}%" if self.test_results else "- **成功率**: N/A",
            "",
            "| テスト名 | ステータス | 実行時間 | スループット | 成功率 |",
            "|---------|-----------|---------|-------------|--------|",
        ]
        
        for result in self.test_results:
            status_icon = "✅" if result.is_passed() else "❌"
            throughput = "N/A"
            success_rate = "N/A"
            
            if result.performance_report:
                throughput = f"{result.performance_report.throughput_per_sec:.2f} msg/s"
                success_rate = f"{result.performance_report.success_rate_percent:.1f}%"
            
            lines.append(
                f"| {result.test_name} | {status_icon} {result.status} | "
                f"{result.duration_seconds:.2f}s | {throughput} | {success_rate} |"
            )
        
        return "\n".join(lines)
    
    def _generate_test_detail_markdown(self, result: TestResult) -> list[str]:
        """テスト詳細のMarkdownを生成"""
        lines = [
            f"### {result.test_name}",
            "",
            f"- **ステータス**: {result.status}",
            f"- **実行時間**: {result.duration_seconds:.2f}秒",
            f"- **タイムスタンプ**: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if result.error_message:
            lines.extend([
                "",
                "#### エラーメッセージ",
                "```",
                result.error_message,
                "```",
            ])
        
        if result.performance_report:
            lines.extend([
                "",
                "#### パフォーマンスメトリクス",
                "",
                result.performance_report.get_summary(),
            ])
        
        if result.logs:
            lines.extend([
                "",
                "#### ログ出力（最新10件）",
                "```",
            ])
            lines.extend(result.logs[-10:])
            lines.append("```")
        
        return lines
    
    def _generate_json_data(self) -> dict[str, Any]:
        """JSONデータを生成"""
        data = {
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.test_results),
                "passed": sum(1 for r in self.test_results if r.is_passed()),
                "failed": sum(1 for r in self.test_results if not r.is_passed()),
            },
            "results": []
        }
        
        for result in self.test_results:
            result_data = {
                "test_name": result.test_name,
                "status": result.status,
                "duration_seconds": result.duration_seconds,
                "timestamp": result.timestamp.isoformat(),
                "error_message": result.error_message,
                "logs": result.logs[-10:] if result.logs else [],
            }
            
            if result.performance_report:
                result_data["performance"] = result.performance_report.to_dict()
            
            data["results"].append(result_data)
        
        return data
    
    def _generate_html_content(self) -> str:
        """HTMLコンテンツを生成"""
        passed = sum(1 for r in self.test_results if r.is_passed())
        failed = len(self.test_results) - passed
        
        html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E2Eテストレポート</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .summary-stats {{
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .passed {{
            color: #27ae60;
        }}
        .failed {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            background-color: white;
            border-collapse: collapse;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .test-detail {{
            background-color: white;
            padding: 20px;
            margin-top: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        pre {{
            background-color: #f8f8f8;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>E2Eテストレポート</h1>
    <p class="timestamp">実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>テスト結果サマリー</h2>
        <div class="summary-stats">
            <div class="stat">
                <div class="stat-value">{len(self.test_results)}</div>
                <div>合計テスト数</div>
            </div>
            <div class="stat">
                <div class="stat-value passed">{passed}</div>
                <div>成功</div>
            </div>
            <div class="stat">
                <div class="stat-value failed">{failed}</div>
                <div>失敗</div>
            </div>
            <div class="stat">
                <div class="stat-value">{passed/len(self.test_results)*100:.1f}%</div>
                <div>成功率</div>
            </div>
        </div>
    </div>
    
    <h2>テスト結果一覧</h2>
    <table>
        <thead>
            <tr>
                <th>テスト名</th>
                <th>ステータス</th>
                <th>実行時間</th>
                <th>スループット</th>
                <th>成功率</th>
            </tr>
        </thead>
        <tbody>
"""
        
        for result in self.test_results:
            status_class = "passed" if result.is_passed() else "failed"
            status_icon = "✅" if result.is_passed() else "❌"
            throughput = "N/A"
            success_rate = "N/A"
            
            if result.performance_report:
                throughput = f"{result.performance_report.throughput_per_sec:.2f} msg/s"
                success_rate = f"{result.performance_report.success_rate_percent:.1f}%"
            
            html += f"""
            <tr>
                <td>{result.test_name}</td>
                <td class="{status_class}">{status_icon} {result.status}</td>
                <td>{result.duration_seconds:.2f}s</td>
                <td>{throughput}</td>
                <td>{success_rate}</td>
            </tr>
"""
        
        html += """
        </tbody>
    </table>
"""
        
        # 詳細セクション
        for result in self.test_results:
            html += f"""
    <div class="test-detail">
        <h3>{result.test_name}</h3>
        <p><strong>ステータス:</strong> {result.status}</p>
        <p><strong>実行時間:</strong> {result.duration_seconds:.2f}秒</p>
"""
            
            if result.error_message:
                html += f"""
        <h4>エラーメッセージ</h4>
        <pre>{result.error_message}</pre>
"""
            
            if result.performance_report:
                html += f"""
        <h4>パフォーマンスメトリクス</h4>
        <pre>{result.performance_report.get_summary()}</pre>
"""
            
            html += """
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html
    
    def get_summary(self) -> dict[str, Any]:
        """テストサマリーを取得"""
        passed = sum(1 for r in self.test_results if r.is_passed())
        failed = len(self.test_results) - passed
        
        return {
            "total_tests": len(self.test_results),
            "passed": passed,
            "failed": failed,
            "success_rate": passed / len(self.test_results) * 100 if self.test_results else 0,
            "total_duration": sum(r.duration_seconds for r in self.test_results),
            "test_results": self.test_results,
        }