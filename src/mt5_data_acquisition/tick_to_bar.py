"""
ティックデータからバー（OHLCV）データへの変換モジュール

このモジュールは、リアルタイムティックデータを1分足などの
時間足バーデータに変換する機能を提供します。
"""

import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal

from pydantic import BaseModel, Field, ValidationError

# 共通のTickモデルとアダプターをインポート
from src.common.models import Tick
from src.mt5_data_acquisition.tick_adapter import TickAdapter


class Bar(BaseModel):
    """バー（OHLCV）データのモデル"""

    symbol: str = Field(..., description="通貨ペア")
    time: datetime = Field(..., description="バー開始時刻")
    end_time: datetime = Field(..., description="バー終了時刻")
    open: Decimal = Field(..., description="始値")
    high: Decimal = Field(..., description="高値")
    low: Decimal = Field(..., description="安値")
    close: Decimal = Field(..., description="終値")
    volume: Decimal = Field(..., description="期間中の総取引量")
    tick_count: int = Field(default=0, description="ティック数")
    avg_spread: Decimal | None = Field(default=None, description="平均スプレッド")
    is_complete: bool = Field(default=False, description="バー完成フラグ")


class TickToBarConverter:
    """ティックデータを時系列バー（OHLCV）に変換するコンバーター

    このクラスはリアルタイムのティックデータを受け取り、
    指定された時間枠（デフォルト1分）のOHLCVバーに変換します。

    Features:
        - リアルタイムティック処理
        - 未完成バーの継続更新
        - ティック欠損検知（デフォルト30秒以上）
        - エラーハンドリング（無効データ、タイムスタンプ逆転）
        - バー完成時のコールバック通知
        - メモリ管理（オプション）

    Example:
        >>> converter = TickToBarConverter("EURUSD")
        >>> converter.on_bar_complete = lambda bar: print(f"Bar completed: {bar}")
        >>>
        >>> tick = Tick(
        ...     symbol="EURUSD",
        ...     time=datetime.now(),
        ...     bid=Decimal("1.1234"),
        ...     ask=Decimal("1.1236"),
        ...     volume=Decimal("1.0")
        ... )
        >>> completed_bar = converter.add_tick(tick)
        >>>
        >>> # 未完成バーの取得
        >>> current_bar = converter.get_current_bar()
        >>> if current_bar:
        ...     print(f"Current bar: {current_bar}")

    Attributes:
        symbol: 処理対象の通貨ペア
        timeframe: バーの時間枠（秒単位）
        current_bar: 現在作成中のバー
        completed_bars: 完成したバーのリスト
        on_bar_complete: バー完成時のコールバック関数
        last_tick_time: 最後に受信したティックの時刻
        gap_threshold: ティック欠損警告の閾値（秒）
        max_completed_bars: 保持する完成バーの最大数

    Note:
        - ティックは時間順に処理される必要があります
        - タイムスタンプが逆転したティックは破棄されます
        - 無効な価格データ（負値、ゼロ）は拒否されます
    """

    def __init__(
        self,
        symbol: str,
        timeframe: int = 60,
        on_bar_complete: Callable[[Bar], None] | None = None,
        gap_threshold: int = 30,
        max_completed_bars: int | None = None,
    ):
        """
        初期化

        Args:
            symbol: 通貨ペア
            timeframe: バーの時間枠（秒単位、デフォルト60秒=1分）
            on_bar_complete: バー完成時のコールバック関数
            gap_threshold: ティック欠損と判定する秒数（デフォルト30秒）
            max_completed_bars: 保持する完成バーの最大数（Noneで無制限）
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.on_bar_complete = on_bar_complete
        self.gap_threshold = gap_threshold
        self.max_completed_bars = max_completed_bars
        self.current_bar: Bar | None = None
        self.completed_bars: list[Bar] = []
        self._current_ticks: list[Tick] = []
        self.last_tick_time: datetime | None = None
        self.logger = self._setup_logger()

    def add_tick(self, tick: Tick) -> Bar | None:
        """
        ティックを追加し、バーが完成した場合はそれを返す

        Args:
            tick: 追加するティックデータ

        Returns:
            完成したバー（完成していない場合はNone）
        """
        try:
            # タイムスタンプ逆転チェック
            if self.last_tick_time and tick.timestamp < self.last_tick_time:
                error_data = {
                    "event": "timestamp_reversal",
                    "symbol": self.symbol,
                    "current_time": tick.timestamp.isoformat(),
                    "last_tick_time": self.last_tick_time.isoformat(),
                }
                self.logger.error(json.dumps(error_data))
                return None

            # ティック欠損を検知
            self.check_tick_gap(tick.timestamp)

            # 現在のバーがない場合は新規作成
            if self.current_bar is None:
                self._create_new_bar(tick)
                self.last_tick_time = tick.timestamp
                return None

            # バー完成判定
            if self._check_bar_completion(tick.timestamp):
                # 現在のバーを完成させる
                self.current_bar.is_complete = True
                self.completed_bars.append(self.current_bar)

                # コールバック実行
                if self.on_bar_complete:
                    self.on_bar_complete(self.current_bar)

                completed_bar = self.current_bar

                # メモリ管理: 古いバーを削除
                if (
                    self.max_completed_bars
                    and len(self.completed_bars) > self.max_completed_bars
                ):
                    # FIFOで古いバーを削除（最新のmax_completed_bars個を保持）
                    self.completed_bars = self.completed_bars[
                        -self.max_completed_bars :
                    ]

                # 新しいバーを作成
                self._create_new_bar(tick)

                # 最後のティック時刻を更新
                self.last_tick_time = tick.timestamp

                return completed_bar
            else:
                # 現在のバーを更新
                self._update_bar(tick)
                # 最後のティック時刻を更新
                self.last_tick_time = tick.timestamp
                return None

        except ValidationError as e:
            error_data = {
                "event": "invalid_tick_data",
                "symbol": self.symbol,
                "error": str(e),
                "tick_data": {
                    "time": tick.timestamp.isoformat() if tick.timestamp else None,
                    "bid": str(tick.bid) if tick.bid else None,
                    "ask": str(tick.ask) if tick.ask else None,
                    "volume": str(tick.volume) if tick.volume else None,
                },
            }
            self.logger.error(json.dumps(error_data))
            return None

    def get_current_bar(self) -> Bar | None:
        """
        現在作成中のバーを取得

        Returns:
            現在作成中のバー（存在しない場合はNone）
        """
        return self.current_bar

    def get_completed_bars(self) -> list[Bar]:
        """
        完成したバーのリストを取得

        Returns:
            完成したバーのリスト
        """
        return self.completed_bars

    def clear_completed_bars(self) -> list[Bar]:
        """
        完成したバーをクリアして返す（メモリ解放用）

        バーのリストを外部に保存した後、メモリを解放したい場合に使用します。

        Returns:
            クリア前の完成バーのリスト

        Example:
            >>> # バーを外部に保存してメモリをクリア
            >>> bars_to_save = converter.clear_completed_bars()
            >>> database.save_bars(bars_to_save)
        """
        bars = self.completed_bars.copy()
        self.completed_bars.clear()
        return bars

    def reset(self) -> None:
        """
        コンバーターをリセット

        現在のバー、完成したバー、ティック履歴をすべてクリアします。
        設定（symbol、timeframe等）は保持されます。
        """
        self.current_bar = None
        self.completed_bars.clear()
        self._current_ticks.clear()
        self.last_tick_time = None

    def _setup_logger(self) -> logging.Logger:
        """
        構造化ログのためのロガー設定

        Returns:
            設定済みのロガー
        """
        logger = logging.getLogger(f"TickToBarConverter.{self.symbol}")
        logger.setLevel(
            logging.WARNING
        )  # WARNING以上のレベル（WARNING, ERROR, CRITICAL）を出力

        # JSONフォーマットのハンドラー設定（既存のハンドラーがない場合のみ）
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def check_tick_gap(self, current_time: datetime) -> float | None:
        """
        ティック欠損を検知して警告ログを出力

        Args:
            current_time: 現在のティック時刻

        Returns:
            欠損秒数（欠損がない場合はNone）
        """
        if self.last_tick_time is None:
            return None

        gap_seconds = (current_time - self.last_tick_time).total_seconds()

        if gap_seconds > self.gap_threshold:
            warning_data = {
                "event": "tick_gap_detected",
                "symbol": self.symbol,
                "gap_seconds": gap_seconds,
                "threshold": self.gap_threshold,
                "last_tick_time": self.last_tick_time.isoformat(),
                "current_time": current_time.isoformat(),
            }
            self.logger.warning(json.dumps(warning_data))
            return gap_seconds

        return None

    def _create_new_bar(self, tick: Tick) -> None:
        """
        新しいバーを作成（プライベートメソッド）

        Args:
            tick: バーの最初のティック
        """
        # 内部計算用にDecimal変換（精度保持）
        tick_decimal = TickAdapter.to_decimal_dict(tick)

        # ティックの時刻を分単位に正規化（秒とマイクロ秒を0に）
        bar_start = self._get_bar_start_time(tick.timestamp)

        # 終了時刻を計算（開始時刻 + 59.999999秒）
        bar_end = self._get_bar_end_time(bar_start)

        # OHLC値を最初のティックで初期化
        self.current_bar = Bar(
            symbol=self.symbol,
            time=bar_start,
            end_time=bar_end,
            open=tick_decimal["bid"],
            high=tick_decimal["bid"],
            low=tick_decimal["bid"],
            close=tick_decimal["bid"],
            volume=tick_decimal["volume"],
            tick_count=1,
            avg_spread=tick_decimal["ask"] - tick_decimal["bid"],
            is_complete=False,
        )

        # ティックリストをクリアして新しいティックを追加
        self._current_ticks = [tick]

    def _update_bar(self, tick: Tick) -> None:
        """
        現在のバーを更新（プライベートメソッド）

        Args:
            tick: バーに追加するティック
        """
        if not self.current_bar:
            return

        # 内部計算用にDecimal変換（精度保持）
        tick_decimal = TickAdapter.to_decimal_dict(tick)

        # High/Lowの更新（max/min）
        self.current_bar.high = max(self.current_bar.high, tick_decimal["bid"])
        self.current_bar.low = min(self.current_bar.low, tick_decimal["bid"])

        # Closeを最新のティック価格に
        self.current_bar.close = tick_decimal["bid"]

        # ボリューム累積
        self.current_bar.volume += tick_decimal["volume"]

        # ティックカウント増加
        self.current_bar.tick_count += 1

        # スプレッドの累積平均計算
        if self.current_bar.avg_spread is not None:
            total_spread = self.current_bar.avg_spread * (
                self.current_bar.tick_count - 1
            )
            new_spread = tick_decimal["ask"] - tick_decimal["bid"]
            self.current_bar.avg_spread = (
                total_spread + new_spread
            ) / self.current_bar.tick_count

        # ティックリストに追加
        self._current_ticks.append(tick)

    def _check_bar_completion(self, tick_time: datetime) -> bool:
        """
        バーが完成したかチェック（プライベートメソッド）

        Args:
            tick_time: チェック対象のティック時刻

        Returns:
            バーが完成した場合True
        """
        if not self.current_bar:
            return False

        # 現在のティック時刻がバーの終了時刻を超えているか判定
        return tick_time > self.current_bar.end_time

    def _get_bar_start_time(self, tick_time: datetime) -> datetime:
        """
        ティック時刻から対応するバーの開始時刻を計算（分単位への切り捨て）

        Args:
            tick_time: ティックの時刻

        Returns:
            バーの開始時刻（秒とマイクロ秒が0）
        """
        if self.timeframe == 60:
            # 1分足の場合、秒とマイクロ秒を0にする
            return tick_time.replace(second=0, microsecond=0)
        else:
            # 他のタイムフレームの場合（将来の拡張用）
            # エポックからの秒数を計算して、タイムフレームで切り捨て
            epoch = datetime(1970, 1, 1, tzinfo=tick_time.tzinfo)
            seconds_since_epoch = (tick_time - epoch).total_seconds()
            bar_start_seconds = (seconds_since_epoch // self.timeframe) * self.timeframe
            return epoch + timedelta(seconds=bar_start_seconds)

    def _get_bar_end_time(self, bar_start: datetime) -> datetime:
        """
        バー開始時刻から終了時刻を計算（開始時刻 + 59.999999秒）

        Args:
            bar_start: バーの開始時刻

        Returns:
            バーの終了時刻
        """
        return bar_start + timedelta(seconds=self.timeframe - 1, microseconds=999999)
