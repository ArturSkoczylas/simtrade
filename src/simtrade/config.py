"""Configuration models for the simulator.

All monetary values are in the simulation's base currency (GBP by default).
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CostModel(BaseModel):
    """Trading costs applied by the broker.

    Defaults approximate a UK retail investor trading London-listed shares.
    """

    model_config = ConfigDict(extra="forbid")

    commission_bps: float = Field(0.0, ge=0, description="Commission as basis points of trade value.")
    commission_fixed: float = Field(0.0, ge=0, description="Fixed commission per order (base currency).")
    commission_min: float = Field(0.0, ge=0, description="Minimum commission per order (base currency).")
    slippage_bps: float = Field(5.0, ge=0, description="Adverse price move per trade, in basis points.")
    half_spread_bps: float = Field(0.0, ge=0, description="Half bid-ask spread paid per trade, in basis points.")
    stamp_duty_rate: float = Field(
        0.005, ge=0, description="UK stamp duty / SDRT on purchases of eligible shares (0.5%)."
    )
    ptm_levy: float = Field(0.0, ge=0, description="Panel on Takeovers and Mergers levy per qualifying trade.")
    ptm_levy_threshold: float = Field(10_000.0, ge=0, description="Trade value above which the PTM levy applies.")
    borrow_rate: float = Field(0.0, ge=0, description="Annual borrow fee charged on short market value.")

    @classmethod
    def zero(cls) -> "CostModel":
        """A frictionless cost model (useful for debugging agents)."""
        return cls(slippage_bps=0.0, stamp_duty_rate=0.0)

    @classmethod
    def uk_retail(cls) -> "CostModel":
        """UK retail defaults: 0.5% stamp duty on eligible buys, 5 bps slippage, £0 commission."""
        return cls()


ExecutionMode = Literal["next_open", "next_close", "same_close"]


class BrokerConfig(BaseModel):
    """How orders are executed."""

    model_config = ConfigDict(extra="forbid")

    execution: ExecutionMode = Field(
        "next_open",
        description=(
            "When a decision made after the close of day t is filled. 'next_open' (default) and "
            "'next_close' fill on day t+1. 'same_close' fills at the close of day t, which is "
            "optimistic because the agent has already seen that close."
        ),
    )
    allow_short: bool = False
    max_gross_leverage: float = Field(1.0, gt=0, description="Cap on sum(|weights|).")
    fractional_shares: bool = True
    max_participation: Optional[float] = Field(
        0.1,
        gt=0,
        description=(
            "Max order size as a fraction of average daily traded value (None disables). "
            "Prevents unrealistic fills in illiquid names."
        ),
    )
    adv_window: int = Field(20, ge=1, description="Window (sessions) for average daily traded value.")
    min_trade_value: float = Field(0.0, ge=0, description="Skip orders smaller than this value.")
    cash_interest_rate: float = Field(0.0, description="Annual interest earned on positive cash.")
    costs: CostModel = Field(default_factory=CostModel)


RebalanceFreq = Union[Literal["daily", "weekly", "monthly", "quarterly"], int]


class SimulationConfig(BaseModel):
    """Parameters of a single simulation run."""

    model_config = ConfigDict(extra="forbid")

    start: Optional[date] = Field(None, description="First trading session (None = first available).")
    end: Optional[date] = Field(None, description="Last trading session (None = latest available).")
    initial_cash: float = Field(100_000.0, gt=0)
    base_currency: str = "GBP"
    lookback: int = Field(252, ge=1, description="Rows of history exposed to the agent in each observation.")
    rebalance: RebalanceFreq = Field(
        "daily",
        description="How often the agent is asked to act: daily, weekly, monthly, quarterly or every N sessions.",
    )
    use_adjusted_prices: bool = Field(
        False,
        description=(
            "False (default): trade on split-adjusted prices and credit cash dividends on the ex-date. "
            "True: trade on dividend-adjusted prices and ignore dividends (use when the data has no dividends)."
        ),
    )
    benchmark: Optional[str] = Field(None, description="Ticker used as benchmark, e.g. 'ISF.L'.")
    risk_free_rate: float = Field(0.0, description="Annual risk-free rate used in Sharpe/Sortino.")
    broker: BrokerConfig = Field(default_factory=BrokerConfig)

    @field_validator("rebalance")
    @classmethod
    def _check_rebalance(cls, v):
        if isinstance(v, int) and v < 1:
            raise ValueError("rebalance interval must be >= 1 session")
        return v
