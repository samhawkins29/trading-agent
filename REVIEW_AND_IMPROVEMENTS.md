# AI Trading Agent — Back-Check & Improvement Review

**Date:** 2026-06-10
**Scope:** Architecture, code health, trading-robustness gaps, prioritized improvements, live-trading risk, and sharpest next moves.
**Method:** Static read of the codebase (no live APIs, no orders placed). Grounded in the actual source and the real trade/log artifacts in the repo.

> **Bottom line up front:** This is an ambitious, well-documented, and genuinely thoughtful system — the risk module, the regime-aware learning, and the Claude decision layer are all above hobby-grade. But there is a wide gap between the polished backtests/README and what the system actually does in paper. The **real paper account has drifted from $100,000 to ~$97,250 (≈ −2.75%) over ~2 months** (`logs/trades.csv`), while the backtests advertise positive CAGR. The codebase has accreted into **two parallel engines** (`agent.py` and `live_trader.py`) that have diverged, a self-improvement loop running on a near-empty sample, and a learned-parameter feedback loop that silently drops half of what it learns. **None of this is ready for real money, and several guardrails that must exist before real stakes do not exist yet.** Details below.

---

## 0. Remediation Status — 2026-06-11

A first remediation pass was completed (offline only — **no live trading was run
and no orders were placed; all broker interaction is mocked or dry-run**). Each
item below was committed as its own change with tests. Test suite at end of
pass: **39 failed, 248 passed, 1 skipped** (was 40 failed / 200 passed). The
one fixed failure is the RL-freeze test; the +47 passing tests are new coverage.
The remaining 39 failures are **all pre-existing stale assertions** that encode
config/API values which drifted before this work (e.g. `experience_replay_size==500`,
`backtest_start=="2023-01-01"`, 3% stop, `max_pct=0.05`, an `equity`→`strategy_equity`
column rename, and `test_sentiment.py` which targets a removed text-sentiment
API). They are **not regressions** from this pass and are catalogued as deferred
test-hygiene below.

> **Run the suite:** `python -m pytest -q` (pytest is now installed for the
> active interpreter). It runs fully offline in ~7s.

### Fixed in this pass

| # | Item | What changed | Tests |
|---|------|--------------|-------|
| **P1** | Learned stop/TP silently dropped (§1.5) | `live_trader._apply_learned_params` now applies `stop_loss_pct`/`take_profit_pct` onto the shared `config` singleton (which `risk_manager` reads), bounds-checked + logged. No more reporting changes that don't take effect. | `test_live_trader.py::TestLearnedParamsApplication` |
| **P2** | Polled-only stops + no real-time breaker (§2.1, Guardrails 1–2) | Broker-native **GTC resting stop** submitted on every open (`_submit_resting_stop`), cancelled/resynced on close/scale-out. **Real-time kill switch** (`_check_kill_switch`/`_trip_kill_switch`) on real equity: cancels orders, liquidates, writes a persistent `KILL_SWITCH.flag` that survives restarts. Checked at the top of every cycle. | `test_live_trader.py::TestNativeRestingStops`, `::TestKillSwitch` |
| **P3** | Backtest look-ahead + zero costs (§2.2) | Execution moved to **next-bar open** via a pending-order queue (`_execute_pending_orders`); **commission + slippage** charged on every fill. Synthetic/survivorship backtests now print a loud banner and tag saved JSON with `synthetic_data=true` + `trust_warning`. | `test_backtester_integrity.py` |
| **P5** | Concentration, sizing, leverage (§2.1) | **Per-sector exposure cap** (`config.sector_map`, `max_sector_exposure=0.40`) in `calculate_position_size`. Removed the unsafe "round-to-0 ⇒ force 1 share" hack (now rounds to nearest, skips genuine zeros). **Leverage default 2.0→1.0×, max 5.0→2.0×.** | `test_risk_caps.py` |
| **P6** | RL starved + mislabeled (§1.4, §2.3) | Weight-learning **frozen** until `min_trades_for_learning=30` closed round-trips (both `self_improver.update_weights` and `weekly_review`). Win/loss now **net-of-cost and short-direction-correct** in `record_experience`. | `test_self_improver_p6.py` |
| **P4** | Untested real-money path (§2.7) | Full engine merge deferred (too risky in one pass); instead the live path is now consistent with the tested path on the shared risk core, and wrapped in tests: `live_trader`, `agent_brain` (brain-JSON parsing), `weekly_review` (Opus validation), `leverage_manager` (modes + circuit breaker). | `test_live_trader.py`, `test_live_path_modules.py` |

### Deferred (targeted safety fix done where noted; full work outstanding)

- **Collapse the two engines into one (§1.3).** Not done — too risky in one pass.
  Mitigation in place: `agent.py` and `live_trader.py` share the same
  `RiskManager`/`config`, so every risk-core fix above applies to both; a role
  note was added to `agent.py`. Full merge (make `live_trader` the only
  execution path, `agent.py` logic a shared library) remains the top structural
  task.
- **Fully-independent kill-switch watchdog.** The kill switch is real-time
  *relative to cycles* (checked first, every cycle) and persists across
  restarts, but a always-on monitor thread/process that liquidates *between*
  cycles is still TODO. The per-cycle check + resting broker stops cover the
  gap in the meantime.
- **Per-name slippage/spread.** P3 uses a flat 5 bps one-way default; wide-spread
  names (COIN, MSTR, VXX, URA, COPX) need per-symbol cost models.
- **Walk-forward / out-of-sample backtest discipline (§2.2).** Still none.
- **Partial-fill handling (§2.4).** Still records requested qty, not `filled_qty`.
- **Correlation *haircut* (beyond sector caps).** Only a coarse sector cap is in;
  a true beta/correlation-matrix haircut is outstanding. The `sector_map` is a
  first-pass bucketing and should be refined.
- **Data-quality guards, `_daily_returns` population, P&L reconciliation,
  comment/config drift (§1.6, §1.7, §2.5).** Untouched; low-risk hygiene.
- **Test hygiene.** 39 stale assertions remain (old config values, the removed
  text-sentiment API). They should be re-synced to current behavior or deleted;
  they were left untouched here to avoid masking real signal, except where a
  test directly covered changed code.

---

## 1. Architecture & Code Health

### 1.1 The shape of the system

The repo is ~15,800 lines of Python across 44 tracked files (much larger than the "~2,599 lines" framing — that number appears to describe only the core path). The intended pipeline (`README.md`, `agent.py:12`) is:

```
data → indicators → regime detection → strategy signals →
regime-weighted aggregation → Kelly+vol sizing → risk check → execute → log → self-improve
```

Core modules:

| Layer | File | Notes |
|---|---|---|
| Decision engine (legacy) | `agent.py` (481 lines) | The README's "central decision engine." Used by `main.py` and most tests. |
| Decision engine (real) | `live_trader.py` (~2,186 lines) | What actually paper-trades. Has the Claude brain, reconciliation, learned-params, news. |
| Strategies | `strategies/*.py` (7 files) | mean_reversion, momentum, sentiment, pattern_recognition + 3 unused (dual_momentum, cross_asset_signals, adaptive_allocation). |
| Risk | `risk_manager.py` (644 lines) | Kelly + vol targeting + Chandelier stops + drawdown halt. |
| Self-improvement | `self_improver.py` (438 lines) | Thompson Sampling + James-Stein shrinkage + per-regime profiles. |
| Decision overlay | `agent_brain.py` (318 lines) | Calls Claude Sonnet (`claude-sonnet-4-6`) per cycle for final BUY/SELL. |
| Weekly meta-review | `weekly_review.py` (598 lines) | Opus review → writes `learned_params.json`. |
| Data | `data_fetcher.py` (248 lines) | yfinance primary, Alpha Vantage fallback, Alpaca for live price. |
| Leverage | `leverage_manager.py` | fixed / kelly / vol_target modes, circuit breaker. |
| Backtest | `backtester.py`, `backtest_30yr.py`, `backtest_leverage.py`, `run_backtest_standalone.py` | **Two of the headline backtests run on synthetic data.** |

### 1.2 Strategy abstraction — *weak*

There is **no base class or interface** for strategies. `strategies/__init__.py` just re-exports concrete classes. The `Signal` dataclass is defined inside `mean_reversion.py` and imported by the others (`momentum.py:40`), so mean-reversion is an implicit dependency of every strategy. Each strategy independently exposes `generate_signal(symbol, df) -> Signal` by convention only — nothing enforces the contract. **Three strategies (`dual_momentum`, `cross_asset_signals`, `adaptive_allocation`) are exported but never wired into the live engine** (`live_trader.py`/`agent.py` only instantiate the original four). That is dead weight and a maintenance trap.

**Recommendation:** introduce an explicit `Strategy` ABC (`name`, `generate_signal`) and move `Signal` to a shared `strategies/base.py`. Either wire in or delete the three orphan strategies.

### 1.3 Two divergent engines — *the biggest structural problem*

`agent.py` and `live_trader.py` are **two implementations of the same agent** that have drifted:

- `agent.py` is simpler, has the regime/weight logic, **no Claude brain, no Alpaca reconciliation, no learned-params, no news, no shorting execution path**, and a buggy `_portfolio_value_estimate` that values positions at entry price.
- `live_trader.py` is the real thing: Claude brain, position reconciliation, real-equity drawdown, momentum scale-outs, shorting, learned-params.

`main.py` and the bulk of the tests exercise `agent.py`; the thing that touches money exercises `live_trader.py`. **The tested code is not the code that trades, and the code that trades is barely tested.** This is the single most dangerous health issue in the repo. Pick one engine. Make `live_trader` the only execution path and have `agent.py`'s logic become a library the tests and backtests share with it.

### 1.4 The RL / self-improvement loop — *clever, but starved and partly disconnected*

`self_improver.py` is genuinely well-conceived: Thompson Sampling Beta posteriors per strategy (`ts_alpha`/`ts_beta`), James-Stein shrinkage toward equal weights that fades out over 200 trades (`self_improver.py:193`), per-regime weight profiles, and an experience-replay buffer. The math is sound and the citations are real.

Problems:

1. **It is learning from almost nothing.** `logs/trades.csv` has ~205 trade rows total and `learned_params.json`'s own Opus memo says the 7-day review covered **"only 1 completed trade."** Thompson Sampling with `Beta(1,1)` priors over 4 arms needs dozens of *closed round-trips per strategy* before the posteriors mean anything. Right now the weights are essentially still the prior.

2. **Win/loss is binary on raw price change** (`self_improver.py:146`): any `pnl > 0` is a "win," regardless of magnitude, holding period, or fees. A strategy that wins 1¢ 60% of the time and loses $5 40% of the time looks *good* to the Beta posterior. This rewards exactly the wrong behavior.

3. **`record_experience` uses `(exit-entry)/entry` and ignores short direction** (`self_improver.py:121`) — for a short, a price rise is a *loss* but is recorded as a positive pnl. Shorts are mislabeled in the learner.

4. **Two different update rules coexist.** Overall weights use Thompson Sampling; per-regime weights use the *old* softmax scorer (`self_improver.py:215-229`, `_softmax_update`). The docstring says softmax was replaced, but it's still running for regimes — confusing and untested.

### 1.5 The learned-parameter feedback loop — *silently drops half its output*

`weekly_review.py` (Opus) writes `learned_params.json` with `strategy_weights`, `buy_threshold`, `sell_threshold`, **`stop_loss_pct`, and `take_profit_pct`**. But `live_trader._apply_learned_params` (`live_trader.py:1458-1481`) **only applies weights and the two thresholds — it never reads `stop_loss_pct` or `take_profit_pct`.** The current file on disk has `stop_loss_pct: 0.04, take_profit_pct: 0.10`, which differ sharply from `config.py`'s `0.08 / 0.20`. So the weekly "learning" believes it has tightened stops to 4%, but the live risk manager still uses 8%. **The self-improvement loop reports changes that never take effect.** This is a real correctness bug, not a style nit.

### 1.6 Config / secrets handling — *good*

- `.env` is correctly listed in `.gitignore` and **is not tracked** (verified: `git ls-files` shows only `.env.example`). Good.
- `config.py` loads via `python-dotenv` and reads everything from env with empty-string defaults. Clean.
- One stale guard: several modules check `if ALPACA_API_KEY == "YOUR_ALPACA_API_KEY"` (`agent.py:418`, `data_fetcher.py:234`) but the default is now `""`, so the placeholder check never fires — harmless but dead.
- The config has drifted out of sync with its own docstring: the header says "Stop-loss widened from 3% to 6%" but the value is 8%; "half-Kelly" but `kelly_fraction=0.6`; "backtest 10 years" alongside a 30-year synthetic backtest. Comments are now archaeology, not documentation.

### 1.7 Tech-debt hotspots (summary)

- **Dual engines** (`agent.py` vs `live_trader.py`) — §1.3.
- **`live_trader.py` is a 2,186-line god-object** doing connection, sizing, execution, polling, reconciliation, review, brain orchestration, and logging in one class.
- **Orphan strategies and orphan helper file** `live_trader_fix.py` (a 2.5 KB snippet sitting in the root).
- **Comment/Config drift** — docstrings describe a system that no longer exists.
- **`_daily_returns` is initialized and passed to the leverage manager but never populated** (`agent.py:77`, `live_trader.py`), so any vol-target leverage mode silently falls back to 1.0x.
- **Logs are enormous and committed-adjacent** — `logs/agent_20260402.log` alone is 3.5 MB; `logs/` is gitignored but bloats the working tree.

---

## 2. Gaps That Matter For Trading

### 2.1 Risk management

**Strengths (real):** per-trade cap (`max_portfolio_pct_per_trade=0.10`), total levered-exposure cap (`max_total_exposure=0.80`, enforcing a 20% cash reserve), drawdown halt computed on *real Alpaca equity* in `live_trader._check_real_drawdown`, Chandelier ATR trailing stops, time-based exits, and momentum scale-outs. The `_portfolio_value_estimate` short double-count bug was found and fixed (`risk_manager.py:582-600`). This is the most mature part of the system.

**Gaps:**

- **Stops are polled, not resting at the broker.** `check_stop_loss_take_profit` runs once per cycle (default 5–15 min). Between cycles — and *entirely overnight, on weekends, and during the gap before a restart* — there is **no stop**. A gap-down through the stop level fills at the open, not the stop. For a levered book this is the difference between a 8% loss and a 25% loss. **Native bracket/stop orders at Alpaca are mandatory before real money.**
- **Position sizing is multiplicatively fragile.** Size = Kelly% × signal × vol_scalar × regime × leverage × sentiment_multiplier (`risk_manager.calculate_position_size` + `live_trader._execute_buy`). Five multipliers compound; at the bottom it's `int()`-floored, which both silently zeros small positions *and* triggered a "if it rounds to 0, make it 1" hack (`live_trader.py` ~1900) that can place unintended 1-share trades.
- **Kelly is estimated from ~30 trades** (`kelly_min_trades=30`) on a binary, fee-free, look-ahead-tainted history. Kelly is notoriously sensitive to win-rate/payoff estimates; estimating it from a tiny, biased sample and then running it at 0.6× (above half-Kelly) is a recipe for over-sizing. **The config calls it "half-Kelly" but uses 0.6** — and the raw Kelly is already clamped to 25%, so a bad streak's estimate can still demand large size.
- **No portfolio-level correlation/exposure control.** The 80% cap is on gross notional, but the universe is heavily tech/beta-correlated (AAPL/MSFT/NVDA/QQQ/XLK/TQQQ all move together). The agent can be "diversified" across 12 names that are really one bet. There is no sector cap, no beta budget, no correlation haircut.
- **Leverage default is 2× fixed** (`config.leverage.fixed_multiplier=2.0`) with a 5× hard cap, applied on top of an unproven edge. In paper this is free; live, with polled stops, it is the fastest path to a margin event.

### 2.2 Backtesting rigor — *do not trust the headline numbers*

- **Two of the three backtests run on *synthetic* data** (`backtest_30yr.py`, `run_backtest_standalone.py` generate calibrated random series, e.g. `generate_synthetic_spy(seed=42)`). Synthetic price paths are smooth and mean-reverting in exactly the way these strategies exploit. **A backtest of a strategy on data generated by a similar stochastic process is close to self-fulfilling.** The "30-year" results are a sanity check at best, not evidence.
- **Same-bar look-ahead.** The real-data backtester computes a signal from the bar's close and then *executes at that same close* (`backtester.py:222` → `open_position(... price)`). In reality you cannot trade the close you used to decide. This alone typically inflates returns by tens of bps per trade.
- **Survivorship bias is severe.** The universe (`config.py:44`) is hand-picked 2026 winners — NVDA, MSTR, PLTR, COIN, META, TQQQ. COIN didn't trade until 2021 and PLTR until late 2020, yet they sit in a "10-year" / "30-year" backtest. There is no point-in-time constituent handling and no delisted names. The backtest is curated to win.
- **Zero costs.** `commission_per_trade=0.0` and **no slippage or spread model anywhere.** For the wide-spread names in the universe (COIN, MSTR, VXX, URA, COPX) this understates costs by 10–50 bps per round trip.
- **No walk-forward / out-of-sample discipline.** Strategy parameters (lookback windows, thresholds, stop multiples) were chosen by the author and evaluated on the same period. There is no train/validate/test split, so there is no defense against the parameters being fit to the sample.

**Net:** the backtests are useful for *directional* hypothesis checking only. The real paper record (−2.75% in two months) is far more informative — and it disagrees with the backtests.

### 2.3 Overfitting risk in the RL self-improvement

Covered in §1.4. The compounding risk is that the agent has **many degrees of freedom** (4 strategy weights × 4 regimes, buy/sell thresholds, stops, Kelly fraction, leverage) and **very little data** to fit them. Every weekly Opus review nudges parameters on the basis of a handful of trades and qualitative log-reading. The `learned_params.json` memo is candid about this: *"only 1 completed trade — making statistically robust conclusions impossible."* Yet the loop still rewrote thresholds. **This is curve-fitting to noise, dressed as learning.** Worse, because the learner's "win" definition is fee-free and look-ahead-tainted (§1.4.2), what it optimizes toward is not what makes money live.

### 2.4 Slippage / fees / fill realism

- **Backtests:** none (§2.2).
- **Live/paper:** `live_trader._submit_order` does something reasonable — passive limit at ±0.1%, poll up to 2 min, fall back to market after 3 consecutive cancels — and logs the real `filled_avg_price`. That's good. But:
  - **Partial fills are accepted as full.** When Alpaca reports `filled`, the code records the *requested* qty, not `filled_qty`. A 50-of-100 fill leaves local and broker state mismatched (partly patched by the "insufficient qty" cover-retry hack, but not systematically).
  - **Paper fills are optimistic.** Alpaca paper fills don't model real queue position or impact; the wide-spread names will behave very differently with real size.

### 2.5 Data quality

- **Single primary source (yfinance)** with a thin Alpha Vantage fallback. yfinance is unofficial, rate-limited, and occasionally returns adjusted/unadjusted inconsistencies. There is no validation that a returned bar is fresh, non-stale, or sane (no spike/zero checks).
- **5-minute cache TTL** (`data_fetcher.py:31`) is fine for slow cycles but means the "latest price" used for stop checks can be up to 5 minutes stale.
- **Mixed adjustment semantics:** yfinance returns split/dividend-adjusted closes; the Alpha Vantage fallback path (`TIME_SERIES_DAILY`) returns **unadjusted** closes. Falling back mid-run silently changes the price basis.
- **Regime detection downloads 2 years of SPY every cycle** with no timeout guard (`agent.py:203`), so a slow data call can stall the whole cycle.

### 2.6 Paper-vs-live gap

- Paper has no real slippage, impact, borrow availability (shorts), locate fees, or PDT constraints. The agent shorts freely in paper; live, many of these names are hard-to-borrow or non-shortable.
- Margin/leverage works differently: 2× in paper is frictionless; live it needs a margin account, ≥ $2k equity, and trips Pattern-Day-Trader rules at this trade frequency on accounts < $25k.
- No reconciliation of the *daily P&L tally* against the broker; `total_pnl` is an in-process accumulator that resets on restart.

### 2.7 Test-coverage gaps

- **The trading engine is largely untested.** Tests target `agent.py`, strategies, `risk_manager`, `data_fetcher`, `self_improver`, backtester. There is **no test file for `live_trader.py`, `agent_brain.py`, `weekly_review.py`, `leverage_manager.py`, `entry_filter.py`, or `news_fetcher.py`** — i.e. the entire real execution + decision + learning-application path is uncovered.
- Untested critical behaviors: order submission/partial-fill handling, position reconciliation after restart, drawdown halt on real equity, short cover, learned-params application, brain-JSON parsing edge cases.
- I could **not execute the suite in this environment** (`pytest` is not installed for the active interpreter; run was hard-stopped per instructions). The ~264 tests should be run in the project venv; treat the above as a structural-coverage observation, not a pass/fail claim.

---

## 3. Improvements (Prioritized: Impact vs Effort)

### P0 — Do before anything else (high impact)

1. **Place native stop/bracket orders at the broker** instead of polling. *Why:* polled stops do not exist overnight, on weekends, or during downtime; this is the largest uncontrolled loss vector, and it is multiplied by leverage. *Effort:* medium.
2. **Fix the learned-params disconnect.** Either apply `stop_loss_pct`/`take_profit_pct` in `_apply_learned_params`, or stop writing them in `weekly_review.py`. *Why:* the system currently lies to itself about its own risk settings (`live_trader.py:1458` vs `learned_params.json`). *Effort:* trivial.
3. **Collapse to one engine.** Make `live_trader` the only execution path; refactor shared logic into tested libraries; delete or demote `agent.py` to a thin wrapper. *Why:* you are testing code that doesn't trade and trading code that isn't tested. *Effort:* medium-high but foundational.
4. **Add costs to the backtest** (commission + per-name slippage/spread) and **move execution to next-bar open.** *Why:* current backtests are structurally optimistic; you cannot calibrate risk on them. *Effort:* low-medium.

### P1 — High value, moderate effort

5. **Fix the RL reward.** Make "win" magnitude- and fee-aware (e.g. risk-adjusted, net-of-cost return), and **direction-correct for shorts** in `record_experience`. *Why:* the learner currently optimizes the wrong objective (§1.4). *Effort:* low.
6. **Gate self-improvement and weekly param changes on a minimum sample** (e.g. ≥ 30–50 *closed round-trips per strategy* before weights move; freeze thresholds/stops until then). *Why:* stop curve-fitting to 1–5 trades. *Effort:* low.
7. **Add portfolio-level correlation / sector exposure caps.** *Why:* the universe is one big tech-beta factor; "12 positions" can be one bet. *Effort:* medium.
8. **Tests for the live path.** Mock Alpaca; cover order submission, partial fills, reconciliation-after-restart, drawdown halt, short cover, learned-params application, and brain-JSON parsing. *Why:* this is the code that can lose money. *Effort:* medium.
9. **Handle partial fills explicitly** (record `filled_qty`, reconcile remainder). *Effort:* low-medium.

### P2 — Robustness / hygiene

10. **Strategy ABC + shared `Signal`;** wire in or delete the 3 orphan strategies. *Effort:* low.
11. **Data-quality guards:** staleness check, spike/zero sanity, unify adjusted-vs-unadjusted across sources, timeout on the per-cycle SPY download. *Effort:* low-medium.
12. **Populate `_daily_returns`** so vol-target leverage actually works (or remove the mode). *Effort:* low.
13. **Reconcile P&L against the broker daily;** persist a daily P&L file. *Effort:* low.
14. **Prune comment/config drift** so docstrings match reality. *Effort:* low.

---

## 4. Risks If Run Live

- **Financial:** With 2× default leverage, polled (not resting) stops, no overnight protection, and a correlated tech-beta book, a single gap-down morning can blow well past the 8% per-trade stop and the 15% drawdown halt before any cycle runs. The drawdown circuit breaker only fires *when a cycle executes* — it is not real-time.
- **Edge is unproven.** The only honest performance data — the live paper record — is **negative (−2.75% over ~2 months)**. The positive backtests are synthetic and/or biased (§2.2). There is currently **no evidence of a real edge**, so live trading would be risking capital on a hypothesis the data does not yet support.
- **Overfitting:** the weekly Opus loop rewrites parameters from tiny samples; over time this can wander into configurations that fit past noise and fail forward.
- **API / data failure mid-trade:** if `get_latest_prices` returns `{}`, the stop check silently finds no triggers and positions ride unmanaged that cycle; if the process dies between order-fill and `open_position`, local state and broker state diverge until the next restart reconciliation (which guesses entry time and ATR). A yfinance outage degrades both signals and stop checks simultaneously.
- **Execution realism:** partial fills counted as full; paper fills don't reflect real spread/impact for the wide-spread names; shorts assume borrow that may not exist live.

### Guardrails that MUST exist before real money

1. Native broker stop/bracket orders on every position (real-time, not polled).
2. A hard, real-time max-loss kill switch (per-day and peak-to-trough) that liquidates and halts — independent of the cycle loop.
3. Pre-flight order validation: shortability/borrow check, buying-power check, PDT-rule check, max-order-notional sanity cap.
4. Partial-fill and broker-state reconciliation on every order, not just startup.
5. A demonstrated, cost-and-slippage-inclusive **positive** track record over a statistically meaningful number of *closed* trades and a real out-of-sample window — before a single real dollar.
6. Start real money (if ever) at a tiny fraction of capital with leverage = 1.0×, and only scale after the live edge is demonstrated net of costs.

---

## 5. Sharpest Next Moves

1. **Believe the live tape, not the backtest.** The paper account is down ~2.75%; the backtests are synthetic/biased. Treat the live paper record as the only ground truth and instrument it: cost-inclusive P&L attribution per strategy and per regime. *(Highest leverage: it tells you whether there is anything here at all.)*
2. **Fix the two self-deceptions first** — the silently-dropped learned stops/TPs (`_apply_learned_params`) and the look-ahead/zero-cost backtest. These are cheap and they're currently making the whole system report numbers it doesn't actually achieve.
3. **Put stops at the broker and a real-time kill switch in front of everything.** This is the one change that converts "interesting paper experiment" into "won't catastrophically fail," and it's a prerequisite for every later step.
4. **Unify on one engine and wrap the live path in tests.** You cannot safely iterate on a system where the trading code is the untested code.
5. **Freeze the learning until you have data.** Stop letting the weekly loop rewrite parameters off 1–5 trades. Set a hard minimum sample, fix the reward to be cost- and direction-aware, and only then let the RL move weights.

---

*Prepared as analysis only. No live trading was run and no orders were placed during this review. Figures cited (trade counts, portfolio trajectory, parameter values) are read directly from `logs/trades.csv`, `learned_params.json`, and the source files referenced inline.*
