.PHONY: all data calibrate proxy valuation greeks validate convexity hedge statutory \
        backtest accounting test clean help

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
	@echo "backtest    the earlier single-policy weekly backtest (~10 min)"
	@echo "accounting  economic against reported earnings (needs backtest first)"
	@echo "test        the test suite"
	@echo "all         everything, in dependency order"
	@echo "clean       remove generated outputs, leaving data/raw alone"

all: data calibrate valuation greeks validate proxy convexity hedge statutory backtest accounting test

data:
	$(PY) -m scripts.build_dataset

calibrate: data
	$(PY) -m scripts.run_calibration

valuation:
	$(PY) -m scripts.run_valuation

greeks:
	$(PY) -m scripts.run_greeks

validate:
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

backtest:
	$(PY) -m scripts.run_hedge_backtest

accounting: backtest
	$(PY) -m scripts.run_gaap_comparison

test:
	$(PY) -m tests.run_tests

clean:
	rm -rf data/processed/* reports/tables/* reports/figures/*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
