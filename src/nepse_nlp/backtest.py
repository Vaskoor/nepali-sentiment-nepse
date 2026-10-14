"""Walk-forward backtesting + NEPSE trading simulation (Phase 5).

This module exposes:

- :data:`NEPSE_COSTS` -- canonical NEPSE transaction cost structure.
- :class:`Backtester` -- a position-aware trading simulator that
  applies realistic NEPSE transaction costs on each entry / exit.
- :func:`backtest_strategy` -- a one-shot helper that combines the
  :func:`nepse_nlp.prediction.walk_forward` prediction loop with the
  :class:`Backtester` to produce a unified performance report.

Design notes
------------
- NEPSE has no short selling. A signal of ``-1`` is therefore treated
  as flat (``0``) -- this is documented inline.
- Round-trip transaction costs are applied at *entry* and *exit* (i.e.
  a buy+sell costs the trader the brokerage, SEBON, DP, and (optional)
  capital-gains tax once per side, multiplied by 2 for the full
  round-trip).
- The Sharpe ratio uses 252 trading days with a default 3% / yr
  risk-free rate.

License: MIT (see LICENSE at repo root).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Lazy / shared utils import.
try:  # pragma: no cover
    from nepse_nlp.utils import get_logger, set_seed

    _LOGGER = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    _LOGGER = logging.getLogger("nepse_nlp.backtest")

    def set_seed(seed: int = 42) -> None:  # type: ignore[no-redef]
        import random

        random.seed(seed)
        np.random.seed(seed)


# Lazy import of walk_forward / NEPSEPredictor (avoids hard dep at import
# time so the module parses cleanly when prediction.py is mid-edit).
try:  # pragma: no cover
    from nepse_nlp.prediction import NEPSEPredictor, walk_forward
except Exception:  # pragma: no cover
    walk_forward = None  # type: ignore[assignment]
    NEPSEPredictor = None  # type: ignore[assignment]


__all__ = ["NEPSE_COSTS", "Backtester", "backtest_strategy"]


# ---------------------------------------------------------------------------
# NEPSE transaction cost structure
# ---------------------------------------------------------------------------
#: Canonical NEPSE round-trip cost structure. All percentages are
#: expressed as fractions (e.g. 0.004 == 0.4 %). Capital gains tax is
#: only levied on realised profit.
NEPSE_COSTS: Dict[str, float] = {
    "brokerage_pct": 0.004,  # 0.4 % standard broker commission
    "sebon_pct": 0.00015,  # SEBON fee
    "dp_fee": 25.0,  # flat DP / CDS&Clearing fee (NPR)
    "capital_gains_tax": 0.05,  # 5 % on realised gains (optional)
}


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------
class Backtester:
    """Position-aware NEPSE trading simulator.

    Parameters
    ----------
    initial_capital : float, default 1_000_000
        Starting cash (NPR).
    costs : dict, optional
        Cost structure override; defaults to :data:`NEPSE_COSTS`.

    Notes
    -----
    The simulator takes a sequence of daily signals (``+1`` long /
    ``0`` flat / ``-1`` short-intent) and produces an equity curve with
    NEPSE transaction costs applied per entry/exit. Since NEPSE does not
    permit shorting, ``-1`` is silently mapped to ``0`` (flat) and the
    behaviour is documented here.
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        costs: Optional[Dict[str, float]] = None,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.costs = dict(costs) if costs is not None else dict(NEPSE_COSTS)
        # Validate cost keys.
        required = {"brokerage_pct", "sebon_pct", "dp_fee", "capital_gains_tax"}
        missing = required - set(self.costs)
        if missing:
            # Backfill from canonical defaults rather than raise.
            for k in missing:
                self.costs[k] = NEPSE_COSTS[k]

    # ------------------------------------------------------------------
    # Cost helpers
    # ------------------------------------------------------------------
    def _round_trip_cost_pct(self) -> float:
        """Per-side percentage cost (brokerage + SEBON) without flat fees."""
        # Per side: brokerage + sebon. Round-trip = 2x.
        per_side = self.costs["brokerage_pct"] + self.costs["sebon_pct"]
        return 2.0 * per_side

    def _entry_cost(self, notional: float) -> float:
        """Cost incurred when entering a position of size ``notional``."""
        return (
            self.costs["brokerage_pct"] * notional
            + self.costs["sebon_pct"] * notional
            + self.costs["dp_fee"]
        )

    def _exit_cost(self, notional: float, realised_pnl: float) -> float:
        """Cost incurred when exiting a position.

        ``realised_pnl`` is the per-unit PnL of the position; if
        positive, capital gains tax applies.
        """
        cost = (
            self.costs["brokerage_pct"] * notional
            + self.costs["sebon_pct"] * notional
            + self.costs["dp_fee"]
        )
        if realised_pnl > 0:
            cost += self.costs["capital_gains_tax"] * realised_pnl
        return cost

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def simulate(
        self,
        signals: Sequence[int],
        closes: pd.Series,
    ) -> Dict:
        """Run the trading simulation.

        Parameters
        ----------
        signals : sequence of int
            Daily signals, ``+1`` long / ``0`` flat / ``-1`` short-intent
            (treated as flat since NEPSE has no shorting).
        closes : pandas.Series
            Daily close prices aligned with ``signals``.

        Returns
        -------
        dict
            Keys: ``equity_curve``, ``n_trades``, ``final_equity``,
            ``total_return_pct``, ``sharpe``, ``max_drawdown``,
            ``win_rate``, ``positions``, ``trades``.
        """
        closes_arr = np.asarray(pd.Series(closes), dtype=float)
        n = len(closes_arr)
        # Coerce -1 -> 0 (NEPSE has no shorting).
        sig = np.array(
            [int(s) if int(s) == 1 else 0 for s in signals],
            dtype=int,
        )
        if len(sig) != n:
            raise ValueError(f"signals (len={len(sig)}) and closes (len={n}) must align")

        # State.
        cash = self.initial_capital
        position: Optional[int] = None  # +1 long, None flat.
        entry_price: float = 0.0
        n_units: float = 0.0  # units currently held
        equity_curve = np.zeros(n, dtype=float)
        trades: List[Dict] = []
        wins = 0

        # Walk through time.
        for t in range(n):
            price = closes_arr[t]
            target_sig = sig[t]

            # Mark-to-market equity before any action at t.
            if position == 1:
                cash + n_units * price

            # Exit if currently long and target is flat (or would short).
            if position == 1 and target_sig != 1:
                gross_proceeds = n_units * price
                realised_pnl = (price - entry_price) * n_units
                cost = self._exit_cost(gross_proceeds, realised_pnl)
                cash += gross_proceeds - cost
                trades.append(
                    {
                        "entry_date_idx": trades[-1]["entry_date_idx"] if False else None,
                        "exit_date_idx": t,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "units": n_units,
                        "pnl": realised_pnl - cost,
                        "cost": cost,
                    }
                )
                if realised_pnl - cost > 0:
                    wins += 1
                position = None
                n_units = 0.0
                entry_price = 0.0

            # Enter long if currently flat and target_sig is +1.
            if position is None and target_sig == 1:
                notional = cash  # invest all available cash
                if notional > 0:
                    cost = self._entry_cost(notional)
                    investable = notional - cost
                    n_units = investable / price if price > 0 else 0.0
                    cash = 0.0
                    position = 1
                    entry_price = price
                    trades.append(
                        {
                            "entry_date_idx": t,
                            "exit_date_idx": None,
                            "entry_price": price,
                            "exit_price": None,
                            "units": n_units,
                            "pnl": None,
                            "cost": cost,
                        }
                    )

            # Mark-to-market equity at end of period t.
            if position == 1:
                equity_curve[t] = cash + n_units * price
            else:
                equity_curve[t] = cash

        # Auto-close any still-open position at the final close.
        if position == 1:
            t_last = n - 1
            price = closes_arr[t_last]
            gross_proceeds = n_units * price
            realised_pnl = (price - entry_price) * n_units
            cost = self._exit_cost(gross_proceeds, realised_pnl)
            cash += gross_proceeds - cost
            if realised_pnl - cost > 0:
                wins += 1
            trades.append(
                {
                    "entry_date_idx": t_last,
                    "exit_date_idx": t_last,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "units": n_units,
                    "pnl": realised_pnl - cost,
                    "cost": cost,
                }
            )
            equity_curve[t_last] = cash

        # --- Final metrics ---
        final_equity = float(equity_curve[-1])
        total_return_pct = (final_equity / self.initial_capital - 1.0) * 100.0

        # Daily returns from the equity curve.
        rets = pd.Series(equity_curve).pct_change().fillna(0.0).values
        sharpe = self.sharpe_ratio(rets)

        # Number of completed round-trip trades.
        completed = [t for t in trades if t.get("pnl") is not None]
        n_trades = len(completed)
        win_rate = wins / max(n_trades, 1)

        # Max drawdown on the equity curve.
        mdd = self.max_drawdown(equity_curve)

        equity_series = pd.Series(
            equity_curve,
            index=pd.Series(closes).index,
            name="equity",
        )
        return {
            "equity_curve": equity_series,
            "n_trades": n_trades,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "win_rate": win_rate,
            "positions": sig.tolist(),
            "trades": trades,
        }

    # ------------------------------------------------------------------
    # Static metric helpers
    # ------------------------------------------------------------------
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, rf: float = 0.03 / 252) -> float:
        """Annualised Sharpe ratio from daily returns.

        Parameters
        ----------
        returns : array-like
            Daily returns.
        rf : float
            Daily risk-free rate (default = 3 % / 252).

        Returns
        -------
        float
            Annualised Sharpe (sqrt(252) * (mean_excess / std_excess)).
        """
        r = np.asarray(returns, dtype=float)
        if len(r) < 2:
            return 0.0
        excess = r - rf
        std = excess.std(ddof=1)
        if std == 0 or np.isnan(std):
            return 0.0
        return float(np.sqrt(252) * (excess.mean() / std))

    @staticmethod
    def max_drawdown(equity: np.ndarray) -> float:
        """Maximum drawdown (as a fraction, e.g. -0.20 == -20 %)."""
        e = np.asarray(equity, dtype=float)
        if len(e) < 2:
            return 0.0
        running_max = np.maximum.accumulate(e)
        drawdown = (e - running_max) / running_max
        # Drawdown is non-positive; return the worst trough.
        return float(drawdown.min())


# ---------------------------------------------------------------------------
# Strategy wrapper
# ---------------------------------------------------------------------------
def backtest_strategy(
    df: pd.DataFrame,
    predictor: Optional["NEPSEPredictor"] = None,
    model_type: str = "xgboost",
    target: str = "direction",
    min_train: int = 120,
    step: int = 1,
    initial_capital: float = 1_000_000.0,
    costs: Optional[Dict[str, float]] = None,
) -> Dict:
    """End-to-end strategy backtest.

    Pipeline
    --------
    1. Run :func:`nepse_nlp.prediction.walk_forward` to get
       out-of-sample predictions and dates.
    2. Translate predictions -> signals: a positive predicted return /
       positive class -> +1, otherwise 0 (no shorting).
    3. Run :class:`Backtester.simulate` on the aligned closes.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`nepse_nlp.features.build_feature_matrix`.
    predictor : NEPSEPredictor, optional
        Unused (kept for API symmetry); pass ``None``.
    model_type : str
        Underlying model for walk_forward.
    target : str
        "direction" or "return".
    min_train : int
        Minimum training rows.
    step : int
        Walk-forward stride.
    initial_capital : float
        Starting cash.
    costs : dict, optional
        Cost override.

    Returns
    -------
    dict
        ``{prediction_metrics, trading_metrics}``.
    """
    if walk_forward is None:  # pragma: no cover
        raise RuntimeError("nepse_nlp.prediction.walk_forward unavailable")

    # 1. Predictions out-of-sample.
    wf = walk_forward(
        df=df,
        model_type=model_type,
        target=target,
        min_train=min_train,
        step=step,
    )
    pred_metrics = wf["metrics"]
    predictions = wf["predictions"]
    dates = wf["dates"]

    # 2. Translate predictions -> signals aligned to closes.
    # Convert the dates back to datetime for indexing.
    date_idx = pd.to_datetime(dates)
    closes_full = df["close"]
    closes_aligned = closes_full.reindex(date_idx)

    # Signals: +1 for positive predictions, 0 otherwise.
    signals: List[int] = [int(np.sign(float(p)) > 0) for p in predictions]

    # 3. Backtest.
    bt = Backtester(initial_capital=initial_capital, costs=costs)
    sim = bt.simulate(signals, closes_aligned)

    return {
        "prediction_metrics": pred_metrics,
        "trading_metrics": {
            "n_trades": sim["n_trades"],
            "final_equity": sim["final_equity"],
            "total_return_pct": sim["total_return_pct"],
            "sharpe": sim["sharpe"],
            "max_drawdown": sim["max_drawdown"],
            "win_rate": sim["win_rate"],
        },
        "equity_curve": sim["equity_curve"],
        "trades": sim["trades"],
    }
