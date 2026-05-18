# Empirical Limits of Risk-Averse Reinforcement Learning versus Closed-Form Routing in Deterministic DeFi Execution

Risk-averse reinforcement learning for optimal liquidation on a Uniswap V2
constant-product pool under a rational MEV sandwich adversary, benchmarked
against a closed-form convex oracle.

The agent is PPO with an optional Rockafellar–Uryasev CVaR penalty and a
Chow–Ghavamzadeh dual-head critic. Gas is sampled from an AR(1)-lognormal
process calibrated to live EIP-1559 fee history, and background pool flow
replays real Swap events fetched from mainnet at a pinned block. Scope is
the temporal trade-splitting problem; spatial multi-pool routing is the
convex programme of Angeris and Chitra (2022) and is intentionally out of
scope.

The repository is a deterministic, reproducible empirical bound on what a
model-free MLP policy attains against an omniscient analytical oracle in
mechanically closed-form DeFi mechanics. The headline number is **79%**
of the oracle, and the experimental design captures a documented instance
of specification gaming that inflated apparent returns threefold before it
was patched.

---

## Quickstart

Tested on Python 3.10–3.14 on macOS and Linux.

```bash
git clone <repo-url> defi_execution_project
cd defi_execution_project
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python3 -m pytest tests/                 # 93 tests, fully offline
python3 -m scripts.run_full_live         # end-to-end live mainnet RPC
```

The executed demo notebook is at `notebooks/demo.ipynb` (with figures
embedded); `notebooks/demo.html` is a self-contained kernel-free export.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `env/` | AMM environment, MEV sandwich adversary, AR(1) gas sampler, flow replayer |
| `data/` | JSON-RPC client, pool reserves, fee history, Swap event logs, frozen snapshots |
| `agent/` | Baselines (single dump, TWAP, gas-aware greedy, convex no-MEV) and PPO + CVaR stack |
| `scripts/` | Calibration, training sweeps, evaluation, figure rendering, live end-to-end |
| `tests/` | 93 deterministic offline tests |
| `notebooks/` | Executed demo notebook and HTML export |
| `artifacts/` | Pareto figures from the sweeps and the publication figures |
| `paper/` | IEEE conference-style write-up (LaTeX + compiled PDF) |

All files in `data/snapshots/` are frozen mainnet captures (blocks
25,103,456 and 25,096,742–25,106,741). The test suite runs entirely
against these snapshots and never touches the network.

---

## Mathematical model

**Constant-product execution.** Selling $q$ units of token $X$ into a pool
with reserves $(x, y)$ and fee retention $\gamma$ yields

$$\Delta y \;=\; \frac{y \gamma q}{x + \gamma q}.$$

The invariant $k = xy$ is monotone non-decreasing and strict whenever
$\gamma < 1$ and $q > 0$.

**MEV adversary.** A rational arbitrageur sandwiches the trader's pending
swap iff the optimised payoff clears the gas cost:

$$p_t \cdot \max_{\delta_{\text{in}}} \Pi_{\text{MEV}}(q_t, \delta_{\text{in}}) \;>\; c_t,$$

with $p_t = y_t / x_t$ the pre-attack mid. The one-dimensional maximisation
is solved with Brent's method on a bracket $\bigl(10^{-9},\,
\max(10^3 q_t,\, 10 x_t)\bigr)$; a bind-assertion guarantees the optimum is
interior, never wall-pinned.

**Gas process.** Joint AR(1) fitted on the most recent 200 EIP-1559
fee-history records:

$$\log b_{t+1} = \mu_b + \phi_b (\log b_t - \mu_b) + \sigma_b \varepsilon_t^b,$$

$$\log p_t^{\text{prio}} = \alpha + \beta \log b_t + \sigma_p \varepsilon_t^p.$$

**MDP.** State $s_t = (Q_t, \tau_t, x_t, c_t) \in \mathbb{R}^4$
(optionally extended with a three-block gas history). Action
$u_t \in [0, 1]$, quantity $q_t = u_t Q_t$. Reward
$R_t = \Delta y_t^{\text{actual}}$ post-sandwich. Objective:

$$J(\pi) = \mathbb{E}\!\left[\textstyle\sum_t R_t\right]
  - \lambda \cdot \mathrm{CVaR}_\alpha\!\left(-\textstyle\sum_t R_t\right),$$

via the Rockafellar–Uryasev dual representation.

A residual $Q_T$ at the terminal block is force-liquidated **through the
MEV adversary**, not around it. The earlier version did the opposite; that
loophole is documented below.

---

## Specification gaming

A 1M-step sweep on the buggy environment reported a mean reward near
168,000 USDC across every CVaR penalty value with cross-seed standard
deviation under 50 USDC for the most risk-averse policies. A trace of
what the trained policies did:

| $\lambda$ | per-block action $u$ | terminal $q$ | terminal share of reward |
|---|---|---|---|
| 0.0 | 0.0139, constant | 121,464 | 72% |
| 1.0 | 0.0107, constant | 130,964 | 78% |
| 3.0 | 0.0053, constant | 148,635 | 88% |

The policy did not learn MEV avoidance. It learned to defer most of its
inventory to the terminal block, where the earlier specification routed
reward through plain CFMM with no adversary check. Higher CVaR penalty
$\Rightarrow$ more aggressive deferral: per-block trades carried
stochastic MEV exposure (variance), the terminal dump did not
(deterministic). The risk objective rewarded the loophole perfectly.

The patch was a one-line correction in `env/amm.py`: terminal forced
liquidation now flows through the same `mev_adversary` hook as any
in-horizon block. Re-evaluating the same 1M-step checkpoints on the
corrected environment dropped headline rewards from ~168k to between
19k and 52k USDC, with the most-deferral runs taking the largest hit.

This finding is preserved in the repository rather than quietly removed.
It is the most useful engineering lesson the project produced.

---

## Headline results (corrected environment)

Pool $1\text{e}6 \times 1\text{e}6$, $\gamma = 0.997$, $Q_0 = 10\%$ of
pool depth, $T = 20$. Five seeds per $\lambda$, evaluated on a disjoint
10,000-block TEST window over 64 episodes.

### Analytical baselines

| Policy | TEST mean USDC | std |
|---|---:|---:|
| `single_dump` | 82 | 0 |
| `twap` | 37,847 | 2,466 |
| `convex_no_mev` | 37,847 | 2,466 |
| **`gas_aware_greedy` (oracle)** | **58,786** | **616** |

### PPO sweeps, 1M training steps

**Sweep A** — base config, $\sigma_b = 0.5$, observation $\mathbb{R}^4$:

| $\lambda$ | TEST mean USDC | cross-seed std |
|---|---:|---:|
| 0.0 | **45,863** | 1,357 |
| 0.1 | 31,952 | 764 |
| 0.3 | 32,742 | 1,075 |
| 1.0 | 34,646 | 1,443 |
| 3.0 | 27,342 | 6,708 |

**Sweep B** — $\sigma_b = 1.5$, observation extended with 3-block gas
history ($\mathbb{R}^7$):

| $\lambda$ | TEST mean USDC | cross-seed std |
|---|---:|---:|
| 0.0 | **46,542** | 673 |
| 0.1 | 31,356 | 619 |
| 0.3 | 30,921 | 1,706 |
| 1.0 | 30,954 | 4,166 |
| 3.0 | 24,732 | 5,303 |

### What this means

The strongest PPO cell (Sweep B, $\lambda = 0$) attains
$46{,}542 / 58{,}786 = 79.2\%$ of the analytical oracle.

The oracle is not a stochastic baseline. It is an omniscient closed-form
bisection at every block for the largest $q$ such that the MEV trigger
remains false, given full knowledge of the current gas cost. The
model-free PPO agent, starting from zero domain knowledge and observing
only the MDP state, recovers $\sim 79\%$ of that absolute mathematical
maximum. Tripling AR(1) gas volatility and adding gas history did not
close the remaining gap. This empirical bound — not a "best mean" — is
the primary publication finding.

Four hypotheses for the residual gap:

1. AR(1) gas variation in absolute USDC at this $Q_0$ is small relative
   to $\Pi_{\text{MEV}}$. The trigger boundary moves less than assumed.
2. The MLP trunk (two layers, 64 hidden) may be too shallow to discover
   short temporal patterns.
3. The policy collapses to a near-deterministic per-block action early in
   training and then ignores the observation. PPO's importance-ratio clip
   prevents recovery once the action distribution is narrow.
4. The reward landscape on fixed-rate strategies is roughly convex in this
   regime; `gas_aware_greedy` sits at the global algebraic optimum that a
   stochastic gradient method cannot perturb without recurrent memory.

A recurrent (LSTM/GRU) policy or deeper trunk is the natural follow-up.
Neither is implemented here.

---

## Publication figures

| File | Content |
|------|---------|
| `artifacts/publication_fig1_specgaming.png` | Specification-gaming collapse: same 1M-step checkpoints on buggy vs. fixed environment, against the `gas_aware_greedy` reference |
| `artifacts/publication_fig2_pareto.png` | Headline Pareto on the corrected environment, both sweeps and four baselines |
| `artifacts/publication_fig3_lambda_response.png` | Lambda response, Sweep A vs. Sweep B, with `twap` and `gas_aware_greedy` reference lines |

The IEEE conference paper that wraps these results is at
`paper/main.pdf` (source `paper/main.tex`).

---

## Reproducing the sweeps

```bash
python3 -m pytest tests/

# Corrected 1M-step sweep with gas history (Sweep B)
python3 -m scripts.train_test_eval \
    --seeds 42 43 44 45 46 \
    --lambdas 0.0 0.1 0.3 1.0 3.0 \
    --total-timesteps 1000000 \
    --Q0 100000 --T 20 \
    --out-dir artifacts/sweep_1M_gas_history

python3 -m scripts.render_phase3e \
    --manifest artifacts/sweep_1M_gas_history/manifest.json \
    --output artifacts/pareto_gas_history.png

python3 -m scripts.render_publication
```

Live calibration against the current mainnet WETH/USDC pool:

```bash
python3 -m scripts.calibrate --pool weth-usdc --Q0 10.0 --T 10 \
    --eth-quote-price 2221.5
```

End-to-end live run with fresh RPC calls:

```bash
python3 -m scripts.run_full_live
```

---

## Known limitations

1. The AR(1) shock standard deviation is overridden from the calibrated
   $\sigma_b \approx 0.05$ to 0.5 (Sweep A) or 1.5 (Sweep B). The
   calibrated value is too tight to flip the MEV trigger at the
   experimental trade size. A sensitivity sweep over the calibrated value
   is open.
2. Single-pool scope. Multi-pool spatial routing composes externally with
   the temporal policy at deployment.
3. The adversary is a marginal-utility sandwich attacker. JIT liquidity
   and cyclic atomic arbitrage are richer and not modelled.
4. Mean-field assumption: the trader's swap does not perturb subsequent
   background Swap arrivals. Standard but not innocuous on illiquid pools.
5. Stochastic gas/flow streams use a fixed evaluation seed. Cross-seed
   standard deviation in the tables measures training convergence, not
   evaluation robustness; the two would be worth disambiguating.

---

## Tests

```bash
python3 -m pytest tests/ -v
```

93 tests, fully offline. Coverage:

- CFMM math against hand-computed golden values
- MEV bracket-bind assertion (catches wall-pinned optima)
- AR(1) gas sampler under fixed seeds
- Cyclic Swap replayer
- Dual-head CVaR critic with parameter-difference guard
- End-to-end env step under the gas-history observation
- Save/load round-trip for PPO checkpoints

---

## References

- Angeris, G. and Chitra, T. (2020). *Improved Price Oracles: Constant
  Function Market Makers.* AFT '20.
- Angeris, G., Agrawal, A., Evans, A., Chitra, T. and Boyd, S. (2022).
  *Optimal Routing for Constant Function Market Makers.* EC '22.
- Rockafellar, R. T. and Uryasev, S. (2000). *Optimization of Conditional
  Value-at-Risk.* Journal of Risk, 2(3), 21–41.
- Chow, Y. and Ghavamzadeh, M. (2014). *Algorithms for CVaR Optimization
  in MDPs.* NeurIPS.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.*
  arXiv:1707.06347.
- Almgren, R. and Chriss, N. (2001). *Optimal Execution of Portfolio
  Transactions.* Journal of Risk, 3(2), 5–40.
- Daian, P. et al. (2020). *Flash Boys 2.0: Frontrunning in Decentralized
  Exchanges, Miner Extractable Value, and Consensus Instability.* IEEE S&P.

---

## License

Research code released for reproducibility. Use at your own risk; the
findings are intentionally negative for model-free RL in this regime and
should not be deployed as a live execution strategy without substantial
additional engineering.
