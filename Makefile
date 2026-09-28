.PHONY: all data valuation greeks validate backtest accounting test clean help

PY ?= python3

help:
	@echo "data        validate inputs and bootstrap the curve history (run this first)"
	@echo "valuation   at-issue valuation, cash flows, robustness, convergence"
	@echo "greeks      Greeks and the moneyness profile"
	@echo "validate    disclosed shocks, in-force comparison, vintage portfolio, behaviour sweep"
	@echo "backtest    weekly hedging backtest over ten years (slow, around 10 minutes)"
	@echo "accounting  economic against reported earnings (needs backtest first)"
	@echo "test        run the test suite"
	@echo "all         everything, in order"
	@echo "clean       remove generated outputs, leaving raw data alone"

all: data valuation greeks validate backtest accounting test

data:
	$(PY) -m scripts.build_dataset

valuation:
	$(PY) -m scripts.run_valuation

greeks:
	$(PY) -m scripts.run_greeks

validate:
	$(PY) -m scripts.run_shock_validation
	$(PY) -m scripts.run_portfolio_validation
	$(PY) -m scripts.run_behaviour_reconciliation

backtest:
	$(PY) -m scripts.run_hedge_backtest

accounting:
	$(PY) -m scripts.run_gaap_comparison

test:
	$(PY) -m tests.run_tests

clean:
	rm -rf data/processed/* reports/tables/* reports/figures/*
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
