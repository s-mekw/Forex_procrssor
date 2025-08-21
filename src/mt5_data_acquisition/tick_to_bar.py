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

from pydantic import BaseModel, Field, ValidationError, field_validator


class Tick(BaseModel):
    """ティックデータのモデル"""

    symbol: str = Field(..., description="通貨ペア（例: EURUSD）")
    time: datetime = Field(..., description="ティックのタイムスタンプ")
    bid: Decimal = Field(..., description="Bid価格")
    ask: Decimal = Field(..., description="Ask価格")
    volume: Decimal = Field(..., description="取引量")

    @field_validator('bid', 'ask')
    @classmethod
    def validate_price(cls, v, info):
        """価格の妥当性を検証"""
        if v is None or v <= 0:
            raise ValueError(f"Invalid price: {v}")
        return v

    @field_validator('ask')
    @classmethod
    def validate_spread(cls, v, info):
        """スプレッドの妥当性を検証（ask >= bid）"""
        if 'bid' in info.data and v < info.data['bid']:
            raise ValueError(f"Invalid spread: ask ({v}) < bid ({info.data['bid']})")
        return v

    @field_validator('volume')
    @classmethod
    def validate_volume(cls, v):
        """取引量の妥当性を検証"""
        if v < 0:
            raise ValueError(f"Invalid volume: {v}")
        return v


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
    """ティックデータをバーデータに変換するコンバーター"""

    def __init__(
        self,
        symbol: str,
        timeframe: int = 60,
        on_bar_complete: Callable[[Bar], None] | None = None,
        gap_threshold: int = 30,
    ):
        """
        初期化

        Args:
            symbol: 通貨ペア
            timeframe: バーの時間枠（秒単位、デフォルト60秒=1分）
            on_bar_complete: バー完成時のコールバック関数
            gap_threshold: ティック欠損と判定する秒数（デフォルト30秒）
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.on_bar_complete = on_bar_complete
        self.gap_threshold = gap_threshold
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
            if self.last_tick_time and tick.time < self.last_tick_time:
                error_data = {
                    "event": "timestamp_reversal",
                    "symbol": self.symbol,
                    "current_time": tick.time.isoformat(),
                    "last_tick_time": self.last_tick_time.isoformat()
                }
                self.logger.error(json.dumps(error_data))
                return None

            # ティック欠損を検知
            self.check_tick_gap(tick.time)

            # 現在のバーがない場合は新規作成
            if self.current_bar is None:
                self._create_new_bar(tick)
                self.last_tick_time = tick.time
                return None

            # バー完成判定
            if self._check_bar_completion(tick.time):
                # 現在のバーを完成させる
                self.current_bar.is_complete = True
                self.completed_bars.append(self.current_bar)

                # コールバック実行
                if self.on_bar_complete:
                    self.on_bar_complete(self.current_bar)

                completed_bar = self.current_bar

                # 新しいバーを作成
                self._create_new_bar(tick)

                # 最後のティック時刻を更新
                self.last_tick_time = tick.time

                return completed_bar
            else:
                # 現在のバーを更新
                self._update_bar(tick)
                # 最後のティック時刻を更新
                self.last_tick_time = tick.time
                return None

        except ValidationError as e:
            error_data = {
                "event": "invalid_tick_data",
                "symbol": self.symbol,
                "error": str(e),
                "tick_data": {
                    "time": tick.time.isoformat() if tick.time else None,
                    "bid": str(tick.bid) if tick.bid else None,
                    "ask": str(tick.ask) if tick.ask else None,
                    "volume": str(tick.volume) if tick.volume else None
                }
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

    def _setup_logger(self) -> logging.Logger:
        """
        構造化ログのためのロガー設定

        Returns:
            設定済みのロガー
        """
        logger = logging.getLogger(f"TickToBarConverter.{self.symbol}")
        logger.setLevel(logging.WARNING)  # WARNING以上のレベル（WARNING, ERROR, CRITICAL）を出力

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
        # ティックの時刻を分単位に正規化（秒とマイクロ秒を0に）
        bar_start = self._get_bar_start_time(tick.time)

        # 終了時刻を計算（開始時刻 + 59.999999秒）
        bar_end = self._get_bar_end_time(bar_start)

        # OHLC値を最初のティックで初期化
        self.current_bar = Bar(
            symbol=self.symbol,
            time=bar_start,
            end_time=bar_end,
            open=tick.bid,
            high=tick.bid,
            low=tick.bid,
            close=tick.bid,
            volume=tick.volume,
            tick_count=1,
            avg_spread=tick.ask - tick.bid,
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

        # High/Lowの更新（max/min）
        self.current_bar.high = max(self.current_bar.high, tick.bid)
        self.current_bar.low = min(self.current_bar.low, tick.bid)

        # Closeを最新のティック価格に
        self.current_bar.close = tick.bid

        # ボリューム累積
        self.current_bar.volume += tick.volume

        # ティックカウント増加
        self.current_bar.tick_count += 1

        # スプレッドの累積平均計算
        if self.current_bar.avg_spread is not None:
            total_spread = self.current_bar.avg_spread * (
                self.current_bar.tick_count - 1
            )
            new_spread = tick.ask - tick.bid
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
