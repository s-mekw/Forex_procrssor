"""統一的なログユーティリティモジュール

このモジュールは、Forex Processor全体で使用される
構造化ログとパフォーマンス計測機能を提供します。
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

# 型変数の定義
T = TypeVar('T')


class StructuredLogger:
    """構造化ログ出力を提供するロガー
    
    このクラスは、JSON形式の構造化ログを出力し、
    機械的な解析を容易にします。
    
    Example:
        >>> logger = StructuredLogger("my_module")
        >>> logger.info("Processing started", {"items": 100, "mode": "batch"})
    """
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        output_format: str = "json",
        include_timestamp: bool = True,
        include_context: bool = True
    ):
        """
        初期化
        
        Args:
            name: ロガー名
            level: ログレベル
            output_format: 出力形式（"json" or "text"）
            include_timestamp: タイムスタンプを含めるか
            include_context: コンテキスト情報を含めるか
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.output_format = output_format
        self.include_timestamp = include_timestamp
        self.include_context = include_context
        self.context: Dict[str, Any] = {}
        
        # ハンドラーがない場合は追加
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            
            if output_format == "json":
                formatter = JsonFormatter(
                    include_timestamp=include_timestamp,
                    include_context=include_context
                )
            else:
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def set_context(self, **kwargs) -> None:
        """ログコンテキストを設定
        
        Args:
            **kwargs: コンテキスト情報
        """
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """ログコンテキストをクリア"""
        self.context.clear()
    
    def _format_message(
        self,
        message: str,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> Union[str, Dict[str, Any]]:
        """メッセージをフォーマット
        
        Args:
            message: ログメッセージ
            extra_data: 追加データ
            
        Returns:
            Union[str, Dict[str, Any]]: フォーマット済みメッセージ
        """
        if self.output_format == "json":
            log_data = {
                "message": message,
                "level": logging.getLevelName(self.logger.level)
            }
            
            if self.include_timestamp:
                log_data["timestamp"] = datetime.utcnow().isoformat()
            
            if self.include_context and self.context:
                log_data["context"] = self.context
            
            if extra_data:
                log_data["data"] = extra_data
            
            return log_data
        else:
            # テキスト形式
            if extra_data:
                return f"{message} - {json.dumps(extra_data)}"
            return message
    
    def debug(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """デバッグログを出力"""
        formatted = self._format_message(message, data)
        if self.output_format == "json":
            self.logger.debug(json.dumps(formatted))
        else:
            self.logger.debug(formatted)
    
    def info(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """情報ログを出力"""
        formatted = self._format_message(message, data)
        if self.output_format == "json":
            self.logger.info(json.dumps(formatted))
        else:
            self.logger.info(formatted)
    
    def warning(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """警告ログを出力"""
        formatted = self._format_message(message, data)
        if self.output_format == "json":
            self.logger.warning(json.dumps(formatted))
        else:
            self.logger.warning(formatted)
    
    def error(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """エラーログを出力"""
        formatted = self._format_message(message, data)
        if self.output_format == "json":
            self.logger.error(json.dumps(formatted))
        else:
            self.logger.error(formatted)
    
    def critical(self, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """クリティカルログを出力"""
        formatted = self._format_message(message, data)
        if self.output_format == "json":
            self.logger.critical(json.dumps(formatted))
        else:
            self.logger.critical(formatted)


class JsonFormatter(logging.Formatter):
    """JSON形式のログフォーマッター"""
    
    def __init__(
        self,
        include_timestamp: bool = True,
        include_context: bool = True
    ):
        """
        初期化
        
        Args:
            include_timestamp: タイムスタンプを含めるか
            include_context: コンテキスト情報を含めるか
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """ログレコードをJSON形式にフォーマット
        
        Args:
            record: ログレコード
            
        Returns:
            str: JSON形式のログ
        """
        # すでにJSON形式の場合はそのまま返す
        if isinstance(record.msg, str) and record.msg.startswith('{'):
            return record.msg
        
        log_data = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat()
        
        # 追加属性があれば含める
        if hasattr(record, 'data'):
            log_data["data"] = record.data
        
        if hasattr(record, 'context') and self.include_context:
            log_data["context"] = record.context
        
        return json.dumps(log_data, default=str)


class LogContext:
    """ログコンテキスト管理
    
    このクラスは、関連するログエントリーに
    共通のコンテキスト情報を付与します。
    
    Example:
        >>> context = LogContext()
        >>> with context.add(request_id="123", user_id="456"):
        ...     logger.info("Processing request")
    """
    
    def __init__(self):
        """初期化"""
        self._context_stack: list[Dict[str, Any]] = [{}]
    
    @property
    def current(self) -> Dict[str, Any]:
        """現在のコンテキストを取得
        
        Returns:
            Dict[str, Any]: 現在のコンテキスト
        """
        # スタック内のすべてのコンテキストをマージ
        merged = {}
        for context in self._context_stack:
            merged.update(context)
        return merged
    
    @contextmanager
    def add(self, **kwargs):
        """コンテキスト情報を追加
        
        Args:
            **kwargs: 追加するコンテキスト情報
            
        Yields:
            コンテキストマネージャー
        """
        # 新しいコンテキストをスタックに追加
        self._context_stack.append(kwargs)
        
        try:
            yield self
        finally:
            # コンテキストをスタックから削除
            self._context_stack.pop()
    
    def set_global(self, **kwargs) -> None:
        """グローバルコンテキストを設定
        
        Args:
            **kwargs: グローバルコンテキスト情報
        """
        self._context_stack[0].update(kwargs)
    
    def clear_global(self) -> None:
        """グローバルコンテキストをクリア"""
        self._context_stack[0].clear()


class PerformanceLogger:
    """パフォーマンス計測とログ出力
    
    このクラスは、処理時間やメモリ使用量を計測し、
    構造化ログとして出力します。
    
    Example:
        >>> perf = PerformanceLogger()
        >>> with perf.measure("data_processing"):
        ...     process_large_dataset()
    """
    
    def __init__(
        self,
        logger: Optional[Union[logging.Logger, StructuredLogger]] = None,
        include_memory: bool = True,
        auto_log: bool = True
    ):
        """
        初期化
        
        Args:
            logger: ロガーインスタンス
            include_memory: メモリ使用量を計測するか
            auto_log: 自動的にログ出力するか
        """
        self.logger = logger or StructuredLogger(__name__)
        self.include_memory = include_memory
        self.auto_log = auto_log
        self.measurements: Dict[str, Dict[str, Any]] = {}
    
    @contextmanager
    def measure(
        self,
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """処理時間とメモリを計測
        
        Args:
            operation_name: 操作名
            metadata: メタデータ
            
        Yields:
            計測コンテキスト
        """
        start_time = time.perf_counter()
        start_memory = None
        
        if self.include_memory:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        try:
            yield self
        finally:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            measurement = {
                "operation": operation_name,
                "elapsed_seconds": elapsed_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            if metadata:
                measurement["metadata"] = metadata
            
            if self.include_memory and start_memory is not None:
                import psutil
                process = psutil.Process()
                end_memory = process.memory_info().rss / (1024 * 1024)  # MB
                measurement["memory_start_mb"] = start_memory
                measurement["memory_end_mb"] = end_memory
                measurement["memory_delta_mb"] = end_memory - start_memory
            
            # 計測結果を保存
            self.measurements[operation_name] = measurement
            
            # 自動ログ出力
            if self.auto_log:
                self._log_measurement(measurement)
    
    def _log_measurement(self, measurement: Dict[str, Any]) -> None:
        """計測結果をログ出力
        
        Args:
            measurement: 計測結果
        """
        if isinstance(self.logger, StructuredLogger):
            self.logger.info("Performance measurement", measurement)
        else:
            # 通常のロガー
            message = (
                f"Performance: {measurement['operation']} "
                f"took {measurement['elapsed_seconds']:.3f}s"
            )
            
            if "memory_delta_mb" in measurement:
                message += f", memory Δ{measurement['memory_delta_mb']:+.1f}MB"
            
            self.logger.info(message)
    
    def get_measurement(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """計測結果を取得
        
        Args:
            operation_name: 操作名
            
        Returns:
            Optional[Dict[str, Any]]: 計測結果
        """
        return self.measurements.get(operation_name)
    
    def get_all_measurements(self) -> Dict[str, Dict[str, Any]]:
        """すべての計測結果を取得
        
        Returns:
            Dict[str, Dict[str, Any]]: すべての計測結果
        """
        return self.measurements.copy()
    
    def clear_measurements(self) -> None:
        """計測結果をクリア"""
        self.measurements.clear()
    
    def summary(self) -> Dict[str, Any]:
        """計測結果のサマリーを生成
        
        Returns:
            Dict[str, Any]: サマリー情報
        """
        if not self.measurements:
            return {"total_operations": 0}
        
        total_time = sum(
            m["elapsed_seconds"] for m in self.measurements.values()
        )
        
        summary_data = {
            "total_operations": len(self.measurements),
            "total_elapsed_seconds": total_time,
            "average_elapsed_seconds": total_time / len(self.measurements),
            "operations": list(self.measurements.keys())
        }
        
        # メモリ情報が含まれている場合
        memory_measurements = [
            m for m in self.measurements.values()
            if "memory_delta_mb" in m
        ]
        
        if memory_measurements:
            total_memory_delta = sum(
                m["memory_delta_mb"] for m in memory_measurements
            )
            summary_data["total_memory_delta_mb"] = total_memory_delta
        
        return summary_data


def log_execution_time(
    logger: Optional[Union[logging.Logger, StructuredLogger]] = None,
    level: int = logging.INFO,
    include_args: bool = False
):
    """実行時間をログ出力するデコレーター
    
    Args:
        logger: ロガーインスタンス
        level: ログレベル
        include_args: 引数をログに含めるか
        
    Example:
        >>> @log_execution_time()
        ... def slow_function():
        ...     time.sleep(1)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        nonlocal logger
        if logger is None:
            logger = StructuredLogger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            # 関数情報
            func_info = {"function": func.__name__}
            if include_args:
                func_info["args"] = str(args)
                func_info["kwargs"] = str(kwargs)
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                
                # 成功ログ
                log_data = {
                    **func_info,
                    "elapsed_seconds": elapsed,
                    "status": "success"
                }
                
                if isinstance(logger, StructuredLogger):
                    logger.info("Function executed", log_data)
                else:
                    logger.log(
                        level,
                        f"{func.__name__} executed in {elapsed:.3f}s"
                    )
                
                return result
                
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                
                # エラーログ
                log_data = {
                    **func_info,
                    "elapsed_seconds": elapsed,
                    "status": "error",
                    "error": str(e)
                }
                
                if isinstance(logger, StructuredLogger):
                    logger.error("Function failed", log_data)
                else:
                    logger.error(
                        f"{func.__name__} failed after {elapsed:.3f}s: {e}"
                    )
                
                raise
        
        return wrapper
    
    return decorator


def create_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    use_structured: bool = True
) -> Union[logging.Logger, StructuredLogger]:
    """統一的なロガーを作成
    
    Args:
        name: ロガー名
        level: ログレベル
        log_file: ログファイルパス（省略時は標準出力）
        use_structured: 構造化ログを使用するか
        
    Returns:
        Union[logging.Logger, StructuredLogger]: ロガーインスタンス
    """
    if use_structured:
        logger = StructuredLogger(name, level=level)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # ハンドラー設定
        if log_file:
            handler = logging.FileHandler(log_file)
        else:
            handler = logging.StreamHandler(sys.stdout)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
    
    return logger