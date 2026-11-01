"""Order execution: turns target weights into fills with costs, slippage and constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import BrokerConfig
from .portfolio import Portfolio

_EPS = 1e-9


@dataclass
class ExecutionReport:
    fills: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    traded_value: float = 0.0
    commission: float = 0.0
    stamp_duty: float = 0.0
    ptm_levy: float = 0.0
    slippage: float = 0.0

    @property
    def total_costs(self) -> float:
        return self.commission + self.stamp_duty + self.ptm_levy + self.slippage


class Broker:
    """Executes rebalances against a :class:`Portfolio`.

    Rules, in order:

    1. Target weights are cleaned (NaN -> 0), clipped at 0 unless shorting is allowed, and
       scaled down if gross exposure exceeds ``max_gross_leverage``.
    2. Untradable assets (no price or zero volume that day) keep their current holding.
    3. Orders are capped at ``max_participation`` x average daily traded value (known before
       the decision) and dropped if smaller than ``min_trade_value``.
    4. Sells execute before buys. If buys need more cash than available (including leverage
       headroom), all buys are scaled down proportionally.
    5. Each fill pays slippage + half spread through its price, plus commission, stamp duty on
       purchases of eligible shares (new or increased long positions only) and the PTM levy.
    """

    def __init__(self, config: BrokerConfig, tickers: Sequence[str], stamp_duty_mask: Optional[np.ndarray] = None):
        self.config = config
        self.tickers = list(tickers)
        n = len(self.tickers)
        self.stamp_mask = np.zeros(n, dtype=bool) if stamp_duty_mask is None else np.asarray(stamp_duty_mask, bool)

    # -- helpers ------------------------------------------------------------------
    def clean_weights(self, w: np.ndarray) -> np.ndarray:
        w = np.nan_to_num(np.asarray(w, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
        if not self.config.allow_short:
            w = np.clip(w, 0.0, None)
        gross = np.abs(w).sum()
        cap = self.config.max_gross_leverage
        if gross > cap + _EPS:
            w = w * (cap / gross)
        return w

    def _order_costs(self, shares: np.ndarray, px: np.ndarray, cur: np.ndarray):
        c = self.config.costs
        slip = (c.slippage_bps + c.half_spread_bps) / 1e4
        buy = shares > 0
        exec_px = np.where(buy, px * (1 + slip), px * (1 - slip))
        value = np.abs(shares) * exec_px
        active = np.abs(shares) > _EPS
        commission = np.where(
            active, np.maximum(c.commission_min, c.commission_fixed + c.commission_bps / 1e4 * value), 0.0
        )
        # Stamp duty on the part of a purchase that creates or increases a long position.
        new = cur + shares
        long_increase = np.clip(np.maximum(new, 0.0) - np.maximum(cur, 0.0), 0.0, None)
        stamp = np.where(buy & self.stamp_mask, c.stamp_duty_rate * long_increase * exec_px, 0.0)
        ptm = np.where(active & (value > c.ptm_levy_threshold), c.ptm_levy, 0.0)
        slippage = np.abs(shares) * px * slip
        # Cash flow: buys pay value + fees, sells receive value - fees.
        cash_flow = np.where(buy, -value, value) - commission - stamp - ptm
        cash_flow = np.where(active, cash_flow, 0.0)
        return exec_px, value, commission, stamp, ptm, slippage, cash_flow

    # -- main ---------------------------------------------------------------------
    def rebalance(
        self,
        pf: Portfolio,
        target: np.ndarray,
        px: np.ndarray,
        mark: np.ndarray,
        tradable: np.ndarray,
        adv: Optional[np.ndarray],
        date: pd.Timestamp,
    ) -> ExecutionReport:
        """Move ``pf`` towards ``target`` weights trading at prices ``px``.

        ``mark`` values untradable holdings when computing equity; ``adv`` is the average daily
        traded value per asset known before the decision (NaN = unknown, no cap).
        """
        cfg = self.config
        rep = ExecutionReport()
        val_px = np.where(np.isfinite(px) & (px > 0), px, mark)
        equity = pf.equity(val_px)
        if not np.isfinite(equity) or equity <= 0:
            rep.events.append({"date": date, "ticker": None, "event": "no_equity", "detail": f"equity={equity:.2f}"})
            return rep

        w = self.clean_weights(target)
        can = tradable & np.isfinite(px) & (px > 0)
        safe_px = np.where(can, px, 1.0)
        cur = pf.shares.copy()
        tgt = np.where(can, w * equity / safe_px, cur)
        if not cfg.fractional_shares:
            tgt = np.where(can, np.trunc(tgt), cur)
        delta = np.where(can, tgt - cur, 0.0)

        # Report wanted trades in untradable names.
        cur_w = np.where(np.isfinite(val_px), cur * np.nan_to_num(val_px) / equity, 0.0)
        blocked = (~can) & (np.abs(w - cur_w) > 1e-6)
        for j in np.flatnonzero(blocked):
            rep.events.append(
                {"date": date, "ticker": self.tickers[j], "event": "untradable", "detail": "no price/volume"}
            )

        # Liquidity cap.
        if cfg.max_participation is not None and adv is not None:
            cap_val = cfg.max_participation * np.asarray(adv, dtype="float64")
            cap_sh = np.where(np.isfinite(cap_val), cap_val / safe_px, np.inf)
            over = np.abs(delta) > cap_sh + _EPS
            if over.any():
                delta = np.sign(delta) * np.minimum(np.abs(delta), cap_sh)
                if not cfg.fractional_shares:
                    delta = np.trunc(delta)
                for j in np.flatnonzero(over):
                    rep.events.append(
                        {
                            "date": date,
                            "ticker": self.tickers[j],
                            "event": "participation_capped",
                            "detail": f"cap={cap_val[j]:.0f}",
                        }
                    )

        # Minimum trade size.
        delta[np.abs(delta * safe_px) < max(cfg.min_trade_value, _EPS)] = 0.0

        # Sells first.
        sells = np.where(delta < 0, delta, 0.0)
        buys = np.where(delta > 0, delta, 0.0)
        s_res = self._order_costs(sells, safe_px, cur)
        cash_after_sells = pf.cash + s_res[6].sum()
        cur_after_sells = cur + sells

        # Buys, scaled down if cash (plus leverage headroom) is insufficient.
        min_cash = -max(cfg.max_gross_leverage - 1.0, 0.0) * equity
        b_res = self._order_costs(buys, safe_px, cur_after_sells)
        need = -b_res[6].sum()
        available = cash_after_sells - min_cash
        if need > available + 1e-6 and need > 0:
            scale = max(available, 0.0) / need
            for _ in range(6):
                trial = buys * scale
                if not cfg.fractional_shares:
                    trial = np.trunc(trial)
                b_res = self._order_costs(trial, safe_px, cur_after_sells)
                if -b_res[6].sum() <= available + 1e-6:
                    buys = trial
                    break
                scale *= 0.995
            else:
                buys = np.zeros_like(buys)
                b_res = self._order_costs(buys, safe_px, cur_after_sells)
            rep.events.append({"date": date, "ticker": None, "event": "buys_scaled", "detail": f"scale={scale:.4f}"})

        # Apply.
        for shares, res in ((sells, s_res), (buys, b_res)):
            exec_px, value, comm, stamp, ptm, slip, flow = res
            active = np.flatnonzero(np.abs(shares) > _EPS)
            if len(active) == 0:
                continue
            before = pf.shares[active].copy()
            pf.shares[active] += shares[active]
            pf.cash += float(flow[active].sum())
            for k, j in enumerate(active):
                q = shares[j]
                prev = before[k]
                if q > 0:
                    side = "COVER" if prev < 0 else "BUY"
                else:
                    side = "SHORT" if prev + q < -_EPS and prev <= _EPS else "SELL"
                rep.fills.append(
                    {
                        "date": date,
                        "ticker": self.tickers[j],
                        "side": side,
                        "shares": float(q),
                        "price": float(safe_px[j]),
                        "exec_price": float(exec_px[j]),
                        "value": float(value[j]),
                        "commission": float(comm[j]),
                        "stamp_duty": float(stamp[j]),
                        "ptm_levy": float(ptm[j]),
                        "slippage": float(slip[j]),
                    }
                )
            rep.traded_value += float(value[active].sum())
            rep.commission += float(comm[active].sum())
            rep.stamp_duty += float(stamp[active].sum())
            rep.ptm_levy += float(ptm[active].sum())
            rep.slippage += float(slip[active].sum())
        # Clean tiny residual positions from floating point.
        pf.shares[np.abs(pf.shares) < 1e-10] = 0.0
        return rep
