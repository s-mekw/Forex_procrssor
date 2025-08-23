"""統一的なエラーハンドリングモジュール

このモジュールは、Forex Processor全体で使用される
共通例外クラスとエラーハンドリング機能を提供します。
"""

import json
import logging
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

from pydantic import ValidationError


# 型変数の定義
T = TypeVar('T')
ExceptionType = TypeVar('ExceptionType', bound=Exception)


# ========================================
# 基底例外クラス階層
# ========================================

class ForexProcessorError(Exception):
    """Forex Processor基底例外クラス
    
    すべてのカスタム例外はこのクラスを継承します。
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        初期化
        
        Args:
            message: エラーメッセージ
            error_code: エラーコード（ログ追跡用）
            details: 詳細情報の辞書
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """エラー情報を辞書形式で取得
        
        Returns:
            Dict[str, Any]: エラー情報
        """
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """エラー情報をJSON文字列で取得
        
        Returns:
            str: JSON形式のエラー情報
        """
        return json.dumps(self.to_dict(), default=str)


# ========================================
# データ処理関連の例外
# ========================================

class DataValidationError(ForexProcessorError):
    """データ検証エラー
    
    無効なデータ形式や値が検出された場合に発生します。
    """
    pass


class DataTypeError(DataValidationError):
    """データ型エラー
    
    期待されるデータ型と異なる場合に発生します。
    """
    pass


class DataRangeError(DataValidationError):
    """データ範囲エラー
    
    データが許容範囲外の場合に発生します。
    """
    pass


class DataIntegrityError(DataValidationError):
    """データ整合性エラー
    
    データの整合性が保たれていない場合に発生します。
    （例：タイムスタンプの逆転、欠損データ）
    """
    pass


# ========================================
# リソース関連の例外
# ========================================

class ResourceError(ForexProcessorError):
    """リソース関連の基底例外"""
    pass


class MemoryLimitError(ResourceError):
    """メモリ制限エラー
    
    メモリ使用量が制限を超えた場合に発生します。
    """
    pass


class FileOperationError(ResourceError):
    """ファイル操作エラー
    
    ファイルの読み書きに失敗した場合に発生します。
    """
    pass


# ========================================
# 接続関連の例外
# ========================================

class ConnectionError(ForexProcessorError):
    """接続エラー
    
    外部サービスへの接続に失敗した場合に発生します。
    """
    pass


class TimeoutError(ConnectionError):
    """タイムアウトエラー
    
    操作がタイムアウトした場合に発生します。
    """
    pass


# ========================================
# 設定関連の例外
# ========================================

class ConfigurationError(ForexProcessorError):
    """設定エラー
    
    設定ファイルや環境変数の問題で発生します。
    """
    pass


# ========================================
# エラーハンドラー
# ========================================

class ErrorHandler:
    """統一的なエラーハンドリング機能
    
    このクラスは、エラーの記録、変換、回復処理を
    統一的に管理します。
    
    Example:
        >>> handler = ErrorHandler()
        >>> with handler.handle_errors():
        ...     risky_operation()
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        raise_on_error: bool = True,
        log_level: int = logging.ERROR
    ):
        """
        初期化
        
        Args:
            logger: ロガーインスタンス
            raise_on_error: エラー時に例外を再発生させるか
            log_level: ログレベル
        """
        self.logger = logger or logging.getLogger(__name__)
        self.raise_on_error = raise_on_error
        self.log_level = log_level
        self.error_count = 0
        self.last_error: Optional[Exception] = None
    
    @contextmanager
    def handle_errors(
        self,
        operation_name: str = "operation",
        fallback_value: Any = None,
        error_types: Optional[tuple[Type[Exception], ...]] = None
    ):
        """エラーハンドリングコンテキストマネージャー
        
        Args:
            operation_name: 操作名（ログ用）
            fallback_value: エラー時の代替値
            error_types: ハンドリングする例外タイプ
            
        Yields:
            エラーハンドラーコンテキスト
            
        Example:
            >>> with handler.handle_errors("data_processing", fallback_value=[]):
            ...     result = process_data()
        """
        error_types = error_types or (Exception,)
        
        try:
            yield self
        except error_types as e:
            self.error_count += 1
            self.last_error = e
            
            # エラー情報を構造化
            error_info = self._format_error_info(e, operation_name)
            
            # ログ出力
            self.logger.log(self.log_level, json.dumps(error_info))
            
            # 詳細なトレースバック（デバッグレベル）
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
            
            # エラーを再発生させるか、代替値を返すか
            if self.raise_on_error:
                raise
            else:
                return fallback_value
    
    def _format_error_info(
        self,
        error: Exception,
        operation_name: str
    ) -> Dict[str, Any]:
        """エラー情報を構造化形式にフォーマット
        
        Args:
            error: 例外オブジェクト
            operation_name: 操作名
            
        Returns:
            Dict[str, Any]: 構造化されたエラー情報
        """
        error_info = {
            "event": "error_occurred",
            "operation": operation_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # ForexProcessorErrorの場合は詳細情報を追加
        if isinstance(error, ForexProcessorError):
            error_info.update({
                "error_code": error.error_code,
                "details": error.details
            })
        
        # ValidationErrorの場合はPydanticエラーの詳細を追加
        elif isinstance(error, ValidationError):
            error_info["validation_errors"] = [
                {
                    "field": ".".join(str(loc) for loc in e["loc"]),
                    "message": e["msg"],
                    "type": e["type"]
                }
                for e in error.errors()
            ]
        
        return error_info
    
    def wrap_function(
        self,
        func: Callable[..., T],
        operation_name: Optional[str] = None,
        fallback_value: Any = None,
        error_types: Optional[tuple[Type[Exception], ...]] = None
    ) -> Callable[..., T]:
        """関数をエラーハンドリングでラップ
        
        Args:
            func: ラップする関数
            operation_name: 操作名（省略時は関数名）
            fallback_value: エラー時の代替値
            error_types: ハンドリングする例外タイプ
            
        Returns:
            Callable: ラップされた関数
            
        Example:
            >>> @handler.wrap_function
            ... def risky_function():
            ...     return dangerous_operation()
        """
        operation_name = operation_name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.handle_errors(operation_name, fallback_value, error_types):
                return func(*args, **kwargs)
        
        return wrapper
    
    def reset_stats(self) -> None:
        """エラー統計をリセット"""
        self.error_count = 0
        self.last_error = None


# ========================================
# エラー回復機能
# ========================================

class ErrorRecovery:
    """エラー回復戦略の実装
    
    このクラスは、エラーからの自動回復機能を提供します。
    
    Example:
        >>> recovery = ErrorRecovery(max_retries=3)
        >>> result = recovery.retry_with_backoff(risky_operation)
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        max_backoff: float = 60.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        初期化
        
        Args:
            max_retries: 最大リトライ回数
            backoff_factor: バックオフ係数
            max_backoff: 最大バックオフ時間（秒）
            logger: ロガーインスタンス
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.logger = logger or logging.getLogger(__name__)
    
    def retry_with_backoff(
        self,
        func: Callable[..., T],
        *args,
        error_types: tuple[Type[Exception], ...] = (Exception,),
        **kwargs
    ) -> T:
        """指数バックオフでリトライ
        
        Args:
            func: 実行する関数
            *args: 関数の位置引数
            error_types: リトライする例外タイプ
            **kwargs: 関数のキーワード引数
            
        Returns:
            T: 関数の実行結果
            
        Raises:
            Exception: 最大リトライ回数を超えた場合
        """
        import time
        
        last_exception = None
        backoff = 1.0
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except error_types as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    sleep_time = min(backoff, self.max_backoff)
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                        f"Retrying in {sleep_time:.1f}s..."
                    )
                    time.sleep(sleep_time)
                    backoff *= self.backoff_factor
                else:
                    self.logger.error(
                        f"All {self.max_retries + 1} attempts failed. Last error: {e}"
                    )
        
        if last_exception:
            raise last_exception
        
        # ここには到達しないはずだが、型チェッカーのために
        raise RuntimeError("Unexpected state in retry_with_backoff")
    
    def with_fallback(
        self,
        primary_func: Callable[..., T],
        fallback_func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """プライマリ関数が失敗した場合にフォールバック関数を実行
        
        Args:
            primary_func: プライマリ関数
            fallback_func: フォールバック関数
            *args: 関数の位置引数
            **kwargs: 関数のキーワード引数
            
        Returns:
            T: 実行結果
        """
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(
                f"Primary function failed: {e}. Using fallback function."
            )
            return fallback_func(*args, **kwargs)


# ========================================
# デコレーター
# ========================================

def handle_errors(
    fallback_value: Any = None,
    error_types: tuple[Type[Exception], ...] = (Exception,),
    logger: Optional[logging.Logger] = None
):
    """エラーハンドリングデコレーター
    
    Args:
        fallback_value: エラー時の代替値
        error_types: ハンドリングする例外タイプ
        logger: ロガーインスタンス
        
    Example:
        >>> @handle_errors(fallback_value=[], error_types=(ValueError,))
        ... def get_data():
        ...     return risky_operation()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        handler = ErrorHandler(logger=logger, raise_on_error=False)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with handler.handle_errors(
                operation_name=func.__name__,
                fallback_value=fallback_value,
                error_types=error_types
            ):
                return func(*args, **kwargs)
            
            # コンテキストマネージャーがfallback_valueを返した場合
            return fallback_value
        
        return wrapper
    
    return decorator


def retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    error_types: tuple[Type[Exception], ...] = (Exception,)
):
    """リトライデコレーター
    
    Args:
        max_attempts: 最大試行回数
        backoff_factor: バックオフ係数
        error_types: リトライする例外タイプ
        
    Example:
        >>> @retry(max_attempts=3, error_types=(ConnectionError,))
        ... def connect_to_service():
        ...     return establish_connection()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        recovery = ErrorRecovery(
            max_retries=max_attempts - 1,
            backoff_factor=backoff_factor
        )
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return recovery.retry_with_backoff(
                func, *args, error_types=error_types, **kwargs
            )
        
        return wrapper
    
    return decorator