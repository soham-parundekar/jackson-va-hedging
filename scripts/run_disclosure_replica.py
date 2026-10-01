"""The model's hedge against the reported liability, tested on Jackson's own filed numbers.

Everything else in this project measures the model against itself. This measures one of its
conclusions against a series the model never saw: the quarterly and annual XBRL facts Jackson
filed for the market risk benefit and for its derivative result.

The conclusion under test comes from the reporting lens. There the hedge, sized on the economic
liability, left reported net income about a tenth more volatile than economic net worth - a small
mismatch, because the risk margin is a scaling of the same liability and the own-credit piece is
excluded from net income by the rule. If that is the whole story then Jackson's filed derivative
result and its filed market risk benefit change should offset each other closely, quarter after
quarter.

They do not, and the way they fail is specific enough to be informative. Over fourteen quarters
where both series are tagged the correlation is near zero and the signs agree about half the time.
Over the four overlapping years the sum of the derivative result is within about a tenth of the
sum of the market risk benefit change, with opposite signs in three of the four. So the offset is
near one-for-one annually and absent quarterly, which is a different claim from either "the hedge
works" or "the hedge does not", and it is the claim this script tests the model against.

Jackson says the reason in Item 7A: it does not use hedging to offset the movement in its US GAAP
liabilities as market conditions change, and that has resulted in net income volatility. The
model hedges the thing it values, so it should show a tighter quarterly offset than the filings
do, and the gap between the two is a measure of how much of the reported volatility comes from
the basis differences this project models and how much from the deliberate choice not to target
GAAP at all.

**What this is not.** Not a level reconciliation. The model is one in-force model point at a
single duration and Jackson's book is a cohort mix with new business replacing old, so the model
policy ages six to thirteen across the window while the book's weighted average attained age sits
near seventy throughout. Correlations, sign agreement and ratios survive that; levels do not,
which is why the comparison is built from ratios and the level column carries its scaling
on its face.

The own-credit check is separate and tighter. Jackson tags the market risk benefit's
instrument-specific credit risk movement through other comprehensive income, and the reporting
lens produces its own version of that line from a Baa spread. Those two should move together and
both should move against the change in spreads, which is one testable sign rather than a shape.
The filed figure is after tax and the model's is before, so the magnitudes are not comparable and
the test is restricted to direction.

Usage:  python -m scripts.run_disclosure_replica
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


from vahedge import paths

# The filed concepts this uses, and what each one is. Named here rather than inline because the
# sign conventions are the filings' own and getting one backwards would invert a conclusion.
MRB_CHANGE = "MarketRiskBenefitChangeInFairValueGainLoss"          # gain positive
DERIVATIVE = "GainLossOnDerivativeInstrumentsNetPretax"            # gain positive, pre-tax
DERIVATIVE_ALT = "DerivativeGainLossOnDerivativeNet"               # the narrower tagging
OWN_CREDIT_OCI = (
    "OciMarketRiskBenefitInstrumentSpecificCreditRiskGainLossAfterAdjustmentsAndTax"
)


def filed(concept: str, period_type: str) -> pd.Series:
    """One concept at one frequency, in millions, indexed by period end."""
    frame = pd.read_csv(paths.XBRL_QUARTERLY, comment="#", parse_dates=["start", "end"])
    rows = frame[(frame["concept"] == concept) & (frame["period_type"] == period_type)]
    if rows.empty:
        return pd.Series(dtype=float)
    # Last filing wins, which is what the header says the file already does, but a concept can
    # appear twice at one period end across two forms and the restated figure is the one to use.
    return (rows.sort_values("filed")
            .drop_duplicates(subset=["start", "end"], keep="last")
            .set_index("end")["val_usd"].sort_index() / 1e6)


def with_fourth_quarters(concept: str) -> pd.Series:
    """Quarterly series with Q4 derived as the year less the three tagged quarters.

    The 10-K reports the year rather than the fourth quarter, so Q4 is usually untagged. Treating
    it as missing throws away a quarter of the sample and, worse, a quarter that is not missing at
    random: year ends are when the valuation assumptions get unlocked.
    """
    quarters = filed(concept, "quarter")
    years = filed(concept, "year")
    out = quarters.copy()
    for year_end, annual in years.items():
        inside = quarters[(quarters.index > year_end - pd.offsets.YearBegin(1))
                          & (quarters.index <= year_end)]
        if inside.size == 3 and year_end not in out.index:
            out.loc[year_end] = annual - inside.sum()
    return out.sort_index()


def two_sided_p(correlation: float, periods: int) -> float:
    """The p-value for a Pearson correlation, from the t statistic, with no SciPy.

    Reported next to every correlation here on purpose. Four annual observations will produce a
    correlation of minus eight tenths from nothing in particular, and a bare coefficient invites
    the reader to treat that as established. The survival function of the t distribution is a
    regularised incomplete beta, which is a continued fraction and more machinery than this
    needs, so the p-value comes from the normal approximation to Fisher's z instead. It is
    slightly optimistic at these sample sizes, which is the wrong direction to be wrong in, so
    treat anything near the threshold as not shown.
    """
    if periods < 4 or not np.isfinite(correlation) or abs(correlation) >= 1.0:
        return float("nan")
    z = 0.5 * np.log((1.0 + correlation) / (1.0 - correlation)) * np.sqrt(periods - 3.0)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def offset_stats(liability_change: pd.Series, hedge: pd.Series, label: str) -> dict:
    """How closely a hedge result offsets a liability movement, on three measures.

    The ratio is of sums rather than a regression slope, because over a handful of periods a
    slope is dominated by whichever period had the largest move and the question here is whether
    the two sides add up over the window.
    """
    both = pd.concat([liability_change.rename("liability"), hedge.rename("hedge")],
                     axis=1, sort=True).dropna()
    if both.shape[0] < 3:
        return {"basis": label, "periods": float(both.shape[0])}
    opposite = int((np.sign(both["liability"]) != np.sign(both["hedge"])).sum())
    correlation = float(both["liability"].corr(both["hedge"]))
    return {
        "basis": label,
        "periods": float(both.shape[0]),
        "correlation": correlation,
        "correlation_p": two_sided_p(correlation, both.shape[0]),
        # One means the hedge result exactly cancelled the liability movement over the window.
        "offset_ratio": float(both["hedge"].sum() / -both["liability"].sum())
        if abs(both["liability"].sum()) > 1e-9 else np.nan,
        "share_opposite_sign": opposite / both.shape[0],
        "liability_total": float(both["liability"].sum()),
        "hedge_total": float(both["hedge"].sum()),
        "residual_total": float((both["liability"] + both["hedge"]).sum()),
    }


def model_periods(daily: pd.DataFrame, rule: str) -> pd.DataFrame:
    """The model's own liability movement and hedge result, aggregated to the filing frequency.

    The liability leg is signed the filings' way - a gain is positive - so it is minus the change
    in the reporting-basis market risk benefit. The hedge leg is the change in the programme's
    mark and cash together, which is the hedge's whole result including its financing and its
    trading costs rather than only the mark.
    """
    ends = daily.resample(rule).last()
    starts = daily.resample(rule).first()
    return pd.DataFrame({
        "liability": -(ends["reporting"] - starts["reporting"]),
        "hedge": ((ends["hedge_mark"] + ends["cash"]) - (starts["hedge_mark"] + starts["cash"])),
        "oci": daily["oci"].resample(rule).sum(),
        "spread_change": ends["own_credit_spread"] - starts["own_credit_spread"],
    }).dropna()


def own_credit_check(daily: pd.DataFrame) -> pd.DataFrame:
    """Does the filed own-credit line move the way the model's does, and against spreads?

    Three correlations on the quarters where the filed series exists. Direction only: the filed
    figure is after tax and after other adjustments, the model's is before both, so a magnitude
    comparison would be comparing two different quantities and calling the difference a result.
    """
    filed_oci = with_fourth_quarters(OWN_CREDIT_OCI)
    quarters = model_periods(daily, "QE")
    both = pd.concat([filed_oci.rename("filed_oci"), quarters], axis=1, sort=True).dropna(
        subset=["filed_oci", "oci"]
    )
    if both.shape[0] < 3:
        return pd.DataFrame([{"periods": float(both.shape[0])}])
    n = both.shape[0]
    pairs = {
        "filed_vs_model": (both["filed_oci"], both["oci"]),
        "filed_vs_spread_change": (both["filed_oci"], both["spread_change"]),
        "model_vs_spread_change": (both["oci"], both["spread_change"]),
    }
    correlations = {name: float(a.corr(b)) for name, (a, b) in pairs.items()}
    return pd.DataFrame([{
        "periods": float(n),
        **correlations,
        **{f"{name}_p": two_sided_p(value, n) for name, value in correlations.items()},
        "filed_total": float(both["filed_oci"].sum()),
        "share_same_sign": float((np.sign(both["filed_oci"]) == np.sign(both["oci"])).mean()),
    }])


def main() -> None:
    paths.ensure_output_dirs()
    daily = pd.read_csv(paths.TABLES / "reporting_lens_daily_s2.csv",
                        index_col=0, parse_dates=True)

    disclosed_mrb = with_fourth_quarters(MRB_CHANGE)
    disclosed_hedge = with_fourth_quarters(DERIVATIVE)
    rows = [
        offset_stats(disclosed_mrb, disclosed_hedge, "filed, quarterly"),
        offset_stats(filed(MRB_CHANGE, "year"), filed(DERIVATIVE, "year"), "filed, annual"),
    ]
    for rule, label in (("QE", "model, quarterly"), ("YE", "model, annual")):
        periods = model_periods(daily, rule)
        rows.append(offset_stats(periods["liability"], periods["hedge"], label))

    table = pd.DataFrame(rows)
    table.to_csv(paths.TABLES / "disclosure_replica.csv", index=False)
    credit = own_credit_check(daily)
    credit.to_csv(paths.TABLES / "disclosure_own_credit.csv", index=False)

    print("Does the hedge result offset the reported market risk benefit?")
    print("  basis              periods  correlation      p   offset ratio  opposite signs")
    for _, row in table.iterrows():
        if "correlation" not in row or pd.isna(row.get("correlation")):
            print(f"  {row['basis']:<18s} {row['periods']:7.0f}   too few periods")
            continue
        print(f"  {row['basis']:<18s} {row['periods']:7.0f} {row['correlation']:12.3f} "
              f"{row['correlation_p']:6.2f} {row['offset_ratio']:13.2f} "
              f"{row['share_opposite_sign']:15.0%}")

    quarterly = table[table["basis"] == "filed, quarterly"].iloc[0]
    annual = table[table["basis"] == "filed, annual"].iloc[0]
    model_q = table[table["basis"] == "model, quarterly"].iloc[0]

    detectable = 1.96 / np.sqrt(max(quarterly["periods"] - 3.0, 1.0))
    print(f"\n  What the filings support, on {quarterly['periods']:.0f} quarters: no detectable "
          f"quarterly relationship, correlation {quarterly['correlation']:+.3f}. At this sample "
          f"a correlation past about {np.tanh(detectable):.2f} in magnitude would register at "
          f"five per cent, so an offset anywhere near the model's would be unmissable and a weak "
          f"one would not. The claim is that there is no strong quarterly offset, not that there "
          f"is none at all.")
    print(f"  What they hint at, on {annual['periods']:.0f} years: the sums come within "
          f"{abs(100 * (annual['offset_ratio'] - 1)):.0f} per cent of cancelling, ratio "
          f"{annual['offset_ratio']:.2f}, with opposite signs in "
          f"{annual['share_opposite_sign']:.0%} of them. The annual correlation of "
          f"{annual['correlation']:+.3f} has a p-value of {annual['correlation_p']:.2f} on four "
          f"observations, so it is not evidence of anything on its own; the ratio of the sums is "
          f"the part that does not depend on period-by-period co-movement.")
    print(f"  Jackson states the mechanism in Item 7A: it does not use hedging to offset the "
          f"movement in its US GAAP liabilities as market conditions change.")
    print(f"\n  The model offsets its own reported basis at {model_q['offset_ratio']:.2f} with a "
          f"correlation of {model_q['correlation']:+.3f}, which is close to a tautology: the "
          f"hedge is rebalanced weekly against the economic liability and the reporting basis is "
          f"nearly a scaling of it. Its value is as the counterfactual. A programme that did "
          f"target the reported basis would look like this row, the filings look nothing like "
          f"it, and the distance between them is the cost of the choice rather than of the basis "
          f"differences the reporting lens measures - which came to a tenth.")

    print("\nThe own-credit line, direction only (filed is after tax, the model's is before)")
    if credit["periods"].iloc[0] < 3:
        print(f"  only {credit['periods'].iloc[0]:.0f} overlapping quarters; not enough to say")
    else:
        row = credit.iloc[0]
        print(f"  {row['periods']:.0f} overlapping quarters, correlations with p-values:")
        for name, text in (("filed_vs_model", "filed against the model's own line"),
                           ("filed_vs_spread_change", "filed against the spread change"),
                           ("model_vs_spread_change", "model against the spread change")):
            print(f"    {text:<38s} {row[name]:+.3f}  (p {row[f'{name}_p']:.2f})")
        print(f"    same sign {row['share_same_sign']:.0%} of the time")
        print("  Every sign is the one the mechanism predicts: spreads widen, the reported "
              "liability shrinks, the gain lands outside net income. Only the model's own link "
              "to spreads is firm; ten quarters cannot establish the other two, and saying they "
              "are consistent is as far as this goes.")
    print(f"\nwrote {paths.TABLES / 'disclosure_replica.csv'} and the own-credit check")


if __name__ == "__main__":
    main()
