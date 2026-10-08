"""Turn a raw Cboe SPX snapshot into quotes a calibration can use.

Three things have to happen before the chain is worth fitting, and the order matters.

*Find the forward and the discount factor, per expiry, from the quotes themselves.* The
naive route is to take the index level and subtract an assumed dividend yield. That assumed
yield is then baked into every calibrated parameter, and at five years a fifty basis point
error in it moves the forward by two and a half percent, which the fit absorbs by tilting
rho. Put-call parity does better and costs nothing:

    C(K) - P(K) = D * (F - K)

so a straight line through the call-minus-put spread against strike has slope -D and
intercept D*F. Both come out of one regression per expiry, with no dividend assumption
anywhere. The fit quality is reported, because a slice where the line does not fit is a
slice with stale quotes in it.

*Keep only the out-of-the-money side.* An in-the-money option is mostly intrinsic value, so
its price carries almost no information about volatility while carrying the full bid-ask
spread as noise. Every surface fit in practice uses OTM quotes for this reason.

*Weight by vega.* A five-year put struck thirty percent below spot and a one-month
at-the-money call differ in price by three orders of magnitude, and an unweighted
least-squares fit to prices would fit the second and ignore the first. This liability lives
in the first. Dividing the price error by vega turns it into an approximate error in
volatility points, which is the unit the fit should be judged in anyway.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .heston_cos import implied_vol

DAYS_PER_YEAR = 365.0
MIN_STRIKES_FOR_PARITY = 6


@dataclass(frozen=True)
class ChainSnapshot:
    """One dated option-chain observation and what parity implied from it."""

    as_of: pd.Timestamp
    spot: float
    quotes: pd.DataFrame       # cleaned OTM quotes, ready to calibrate
    forwards: pd.DataFrame     # one row per expiry: forward, discount, fit diagnostics


def _snapshot_header(path) -> dict:
    """Read the timestamp and spot out of the committed file's comment header."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            stamp = re.search(r"timestamp=([\d\-]+ [\d:]+)", line)
            spot = re.search(r"spot=([\d.]+)", line)
            if stamp and spot:
                return {"as_of": pd.Timestamp(stamp.group(1)), "spot": float(spot.group(1))}
    raise ValueError(f"{path} has no '# timestamp=... spot=...' header line")


def implied_forwards(parity_path, as_of: pd.Timestamp) -> pd.DataFrame:
    """Forward and discount factor per expiry, from put-call parity.

    Mid prices are used on both legs. A slice needs at least six strikes to be fitted; the
    far-dated expiries in this chain have as few as three quoted pairs and are dropped
    rather than fitted on a line through three points with two-hundred-point spreads.
    """
    raw = pd.read_csv(parity_path, comment="#")
    raw["expiry"] = pd.to_datetime(raw["expiry"])
    raw["call_mid"] = 0.5 * (raw["call_bid"] + raw["call_ask"])
    raw["put_mid"] = 0.5 * (raw["put_bid"] + raw["put_ask"])
    # Relative spread on the wider of the two legs. A pair where one side is quoted two
    # hundred points wide carries no usable parity information.
    raw["spread"] = np.maximum(
        (raw["call_ask"] - raw["call_bid"]) / raw["call_mid"],
        (raw["put_ask"] - raw["put_bid"]) / raw["put_mid"],
    )

    rows = []
    for expiry, group in raw.groupby("expiry"):
        usable = group[group["spread"] < 0.10]
        if len(usable) < MIN_STRIKES_FOR_PARITY:
            rows.append(
                {
                    "expiry": expiry,
                    "maturity": (expiry - as_of).days / DAYS_PER_YEAR,
                    "n_pairs": int(len(usable)),
                    "forward": np.nan,
                    "discount": np.nan,
                    "parity_r2": np.nan,
                    "parity_max_resid": np.nan,
                    "used": False,
                }
            )
            continue

        strikes = usable["strike"].to_numpy(dtype=float)
        spread = (usable["call_mid"] - usable["put_mid"]).to_numpy(dtype=float)
        design = np.column_stack([np.ones_like(strikes), strikes])
        (intercept, slope), *_ = np.linalg.lstsq(design, spread, rcond=None)
        fitted = design @ np.array([intercept, slope])
        total = float(((spread - spread.mean()) ** 2).sum())
        residual = float(((spread - fitted) ** 2).sum())

        discount = -float(slope)
        forward = float(intercept) / discount if discount > 0 else np.nan
        rows.append(
            {
                "expiry": expiry,
                "maturity": (expiry - as_of).days / DAYS_PER_YEAR,
                "n_pairs": int(len(usable)),
                "forward": forward,
                "discount": discount,
                "parity_r2": 1.0 - residual / total if total > 0 else np.nan,
                "parity_max_resid": float(np.abs(spread - fitted).max()),
                "used": bool(discount > 0 and np.isfinite(forward)),
            }
        )

    return pd.DataFrame(rows).sort_values("maturity").reset_index(drop=True)


def load_chain(
    chain_path,
    parity_path,
    min_maturity: float = 0.05,
    max_maturity: float = 6.0,
    max_spread: float = 0.35,
    min_price: float = 0.50,
    min_moneyness: float = 0.75,
    max_moneyness: float = 1.10,
) -> ChainSnapshot:
    """Clean the chain and attach a forward and discount factor to every quote.

    The filters, and why each is there:

    * maturity floor - anything inside a few weeks is dominated by the pinning and
      microstructure of the front expiry and says nothing about a forty-year liability.
    * spread ceiling - a quote whose bid-ask is a third of its mid is not a price.
    * price floor - fifty cents on a seven-thousand-point index is one tick of noise, and
      the deep wing is where a fit goes to get misled.
    * moneyness window - the calibration should not be asked to reproduce strikes far
      outside the range the guarantee's payoff depends on.
    """
    meta = _snapshot_header(chain_path)
    as_of, spot = meta["as_of"], meta["spot"]

    forwards = implied_forwards(parity_path, as_of)
    usable = forwards[
        forwards["used"]
        & forwards["maturity"].between(min_maturity, max_maturity)
    ].set_index("expiry")

    raw = pd.read_csv(chain_path, comment="#")
    raw["expiry"] = pd.to_datetime(raw["expiry"])
    raw = raw[raw["expiry"].isin(usable.index)].copy()

    raw["mid"] = 0.5 * (raw["bid"] + raw["ask"])
    raw["spread"] = (raw["ask"] - raw["bid"]) / raw["mid"]
    raw["maturity"] = raw["expiry"].map(usable["maturity"])
    raw["forward"] = raw["expiry"].map(usable["forward"])
    raw["discount"] = raw["expiry"].map(usable["discount"])
    raw["is_call"] = raw["cp"].eq("C")
    raw["moneyness"] = raw["strike"] / raw["forward"]

    keep = (
        (raw["spread"] < max_spread)
        & (raw["mid"] >= min_price)
        & raw["moneyness"].between(min_moneyness, max_moneyness)
    )
    clean = raw[keep].copy()

    # Implied volatility from the mid, then vega at that volatility. Using the Cboe-published
    # iv instead would import their forward assumption, which is the thing parity just
    # replaced.
    clean["market_vol"] = [
        implied_vol(row.mid, row.forward, row.strike, row.maturity, row.discount, row.is_call)
        for row in clean.itertuples()
    ]
    clean = clean[np.isfinite(clean["market_vol"])].copy()
    clean["vega"] = _black_vega(
        clean["forward"].to_numpy(),
        clean["strike"].to_numpy(),
        clean["maturity"].to_numpy(),
        clean["market_vol"].to_numpy(),
        clean["discount"].to_numpy(),
    )
    clean = clean[clean["vega"] > 1e-6].copy()
    clean["price"] = clean["mid"]

    columns = [
        "expiry", "maturity", "strike", "moneyness", "is_call", "bid", "ask", "price",
        "spread", "forward", "discount", "market_vol", "vega", "open_interest",
    ]
    return ChainSnapshot(
        as_of=as_of,
        spot=spot,
        quotes=clean[columns].sort_values(["maturity", "strike"]).reset_index(drop=True),
        forwards=forwards,
    )


def _black_vega(forward, strike, maturity, vol, discount):
    """dPrice/dVol for one volatility point, in price units."""
    total = vol * np.sqrt(maturity)
    d1 = (np.log(forward / strike) + 0.5 * total**2) / total
    density = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    return discount * forward * density * np.sqrt(maturity)


def fit_report(snapshot: ChainSnapshot, params, n_terms: int = 256) -> pd.DataFrame:
    """Model against market, in volatility points, per quote.

    Reporting the error in volatility rather than in price is the only way to see whether a
    fit that looks good on the at-the-money options has gone badly wrong in the wing, which
    is where the guarantee's payoff is.
    """
    from .heston_cos import cos_price

    rows = []
    for maturity, group in snapshot.quotes.groupby("maturity"):
        model = cos_price(
            params,
            float(group["forward"].iloc[0]),
            group["strike"].to_numpy(dtype=float),
            float(maturity),
            float(group["discount"].iloc[0]),
            group["is_call"].to_numpy(dtype=bool),
            n_terms=n_terms,
        )
        for price, row in zip(model, group.itertuples()):
            model_vol = implied_vol(
                float(price), row.forward, row.strike, row.maturity, row.discount, row.is_call
            )
            rows.append(
                {
                    "maturity": row.maturity,
                    "strike": row.strike,
                    "moneyness": row.moneyness,
                    "is_call": row.is_call,
                    "market_price": row.price,
                    "model_price": float(price),
                    "market_vol": row.market_vol,
                    "model_vol": model_vol,
                    "vol_error": model_vol - row.market_vol,
                    "vega": row.vega,
                }
            )
    return pd.DataFrame(rows)
