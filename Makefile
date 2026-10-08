.PHONY: all data calibrate proxy valuation greeks validate convexity hedge statutory macro reporting replica \
        offset netting real-world economics figures test clean help

PY ?= python3

help:
	@echo "Order matters in two places and nowhere else. calibrate writes the market state"
	@echo "every other step reads, and convexity writes the curvature surface the hedging"
	@echo "experiments size their option leg from. Both outputs are gitignored, so a fresh"
	@echo "clone has to build them before anything downstream will run."
	@echo ""
	@echo "data        validate the committed inputs and bootstrap the curve history"
	@echo "calibrate   fit the market state: curve, Heston surface, short rate, correlations"
	@echo "valuation   at-issue valuation, cash flows, robustness, convergence"
	@echo "greeks      Greeks on paired paths, and the moneyness profile"
	@echo "validate    disclosed shocks, in-force comparison, vintage portfolio, behaviour sweep"
	@echo "proxy       the regression proxy against nested simulation (~6 min)"
	@echo "convexity   the nested curvature surface the option leg is sized from (~8 min)"
	@echo "hedge       crisis replays, the cost frontier, the put sweep, model risk (~15 min)"
	@echo "statutory   real-world requirement at CTE(70) and CTE(90), and the surrender floor"
	@echo "macro       what the tail put spread buys, swept over size and strikes (~10 min)"
	@echo "reporting   economic against reported earnings, and the own-credit OCI split (~15 min)"
	@echo "replica     the model's offset against Jackson's filed XBRL series (fast, reads a table)"
	@echo "offset      the hedge book against the disclosed derivative sensitivities"
	@echo "netting     what the index-linked book absorbs before any hedge (~10 min)"
	@echo "real-world  the hedge over bootstrap reorderings of the decade, two drift arms (~1h)"
	@echo "economics   fee income against hedge cost and breakage, by cohort (~15 min)"
	@echo "figures     redraw every figure from the tables, which takes seconds"
	@echo "test        the test suite"
	@echo "all         everything, in dependency order"
	@echo "clean       remove generated outputs, leaving data/raw alone"

all: data calibrate valuation greeks validate proxy convexity hedge macro statutory \
     reporting replica offset netting real-world economics figures test

data:
	$(PY) -m scripts.build_dataset

calibrate: data
	$(PY) -m scripts.run_calibration

# Both read the saved market calibration rather than the raw panel, so neither runs on a fresh
# clone until calibrate has written it.
valuation: calibrate
	$(PY) -m scripts.run_valuation

greeks: calibrate
	$(PY) -m scripts.run_greeks

validate: calibrate
	$(PY) -m scripts.run_shock_validation
	$(PY) -m scripts.run_portfolio_validation
	$(PY) -m scripts.run_behaviour_reconciliation

proxy: calibrate
	$(PY) -m scripts.run_proxy_validation

# Writes data/processed/gamma_surface.csv, which the hedging experiments read. Gitignored, so
# this is not optional on a fresh clone - the experiments fall back to the regression's own
# curvature, which the proxy validation shows is wrong by about its own size.
convexity: calibrate
	$(PY) -m scripts.run_convexity_surface

hedge: convexity
	$(PY) -m scripts.run_hedge_experiments

statutory: calibrate
	$(PY) -m scripts.run_statutory

# Reads the same curvature surface the experiments do, and reuses their setup, so it carries the
# same prerequisite rather than a looser one.
macro: convexity
	$(PY) -m scripts.run_macro_frontier

# Four regression fits off one simulation, so it is the slowest step that is not a backtest.
reporting: convexity
	$(PY) -m scripts.run_reporting_lens

# Reads the daily table the reporting lens writes rather than refitting, so it is seconds.
replica: reporting
	$(PY) -m scripts.run_disclosure_replica

# Sizes a hedge at four past balance-sheet dates and shocks it, so it needs the market state and
# nothing the backtest builds. The disclosed half of it is arithmetic on two filed tables.
offset: calibrate
	$(PY) -m scripts.run_hedge_disclosure

# One simulation of the index alone, reused across every segment on the grid, so it needs the
# market state and nothing the liability engine builds.
netting: calibrate
	$(PY) -m scripts.run_rila_netting

# Reuses the hedging experiments' setup and its curvature surface, so it carries the same
# prerequisite.
real-world: convexity
	$(PY) -m scripts.run_real_world

# Replays five cohorts through three strategies on five windows and then prices the guarantee
# across the fee grid, so it needs both the curvature surface and the valuation engine.
economics: convexity
	$(PY) -m scripts.run_rider_economics

# Figures read the committed tables and nothing else, so this is the one step that is cheap to
# rerun and the one that has to be rerun whenever a number moves.
figures:
	$(PY) -m scripts.run_figures

test:
	$(PY) -m tests.run_tests

clean:
	rm -rf data/processed/* reports/tables/* reports/figures/*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
