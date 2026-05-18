# Empirical Limits of Risk-Averse Reinforcement Learning versus Gas-Aware Routing in Deterministic DeFi Execution

Reproducible empirical benchmark of a CVaR-regularised PPO agent against a
myopic gas-aware analytical oracle for optimal liquidation on a Uniswap V2
constant-product pool under a rational MEV sandwich adversary.

The agent is PPO with a Rockafellar–Uryasev CVaR penalty and a
Chow–Ghavamzadeh dual-head critic. Gas is sampled from an AR(1)-lognormal
process calibrated to live EIP-1559 fee history; background pool flow
replays real Swap events fetched from mainnet at a pinned block. The main
sweeps intentionally use stress-volatility gas shocks ($\sigma_b \in
\{0.5,\,1.5\}$) so the MEV trigger is informative at the tested trade size.
Scope is the temporal trade-splitting problem; spatial multi-pool routing
is the convex programme of Angeris and Chitra (2022) and is intentionally
out of scope.

The headline empirical finding is that the strongest tested PPO variant
captures roughly **79% of the myopic gas-aware oracle** at $Q_0 = 10\%$ of
pool depth, and this ceiling persists across a $\sim 16\times$ trunk
capacity expansion and a variance-controlled gas-history ablation. We
find no empirical evidence that the tested model-free MLP policies can
dominate this myopic oracle, which can be evaluated in
$O(\log \varepsilon^{-1})$ time per block. We do not claim a structural
inferiority for the reinforcement-learning paradigm; we report what the
tested policies attain under a deliberately favourable environment for the
oracle.

This repository documents a reproducible specification-gaming pathology in
the reward-hacking sense of Amodei et al. (2016), captured before patching
the underlying environment bug.

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
| `agent/` | Baselines (single dump, TWAP, gas-aware greedy, convex no-MEV) and the PPO + CVaR stack |
| `scripts/` | Calibration, training sweeps, evaluation, statistical tests, figure rendering, live end-to-end |
| `tests/` | 93 deterministic offline tests |
| `notebooks/` | Executed demo notebook and HTML export |
| `artifacts/` | Pareto figures, sweep manifests, and publication figures |
| `paper/paper-2/` | IEEE conference-style write-up (LaTeX source `main_2.tex` + compiled `main_2.pdf`) |

All files in `data/snapshots/` are frozen mainnet captures (blocks
25,103,456 and 25,096,742–25,106,741). The test suite runs entirely
against these snapshots and never touches the network.

---

## Mathematical model

**Constant-product execution.** Selling $q$ units of token $X$ into a pool
with reserves $(x, y)$ and fee retention $\gamma$ yields

$$\Delta y \;=\; \frac{y \, \gamma \, q}{x + \gamma \, q}.$$

The invariant $k = xy$ is monotone non-decreasing and strict whenever
$\gamma < 1$ and $q > 0$.

**MEV adversary.** A rational arbitrageur sandwiches the trader's pending
swap iff the optimised payoff clears the gas cost:

$$p_t \cdot \max_{\delta_{\text{in}}} \Pi_{\text{MEV}}(q_t, \delta_{\text{in}}) \;>\; c_t,$$

with $p_t = y_t / x_t$ the pre-attack mid. The one-dimensional concave
maximisation is solved with Brent's method on a bracket
$\bigl(10^{-9},\, \max(10^3 q_t,\, 10 x_t)\bigr)$; a bind-assertion
guarantees the optimum is interior, never wall-pinned.

**Gas process.** Joint AR(1) fitted on the most recent 200 EIP-1559
fee-history records:

$$\log b_{t+1} = \mu_b + \phi_b (\log b_t - \mu_b) + \sigma_b \varepsilon_t^b,$$

$$\log p_t^{\text{prio}} = \alpha + \beta \log b_t + \sigma_p \varepsilon_t^p.$$

**MDP.** State $s_t = (Q_t, \tau_t, x_t, c_t) \in \mathbb{R}^4$
(optionally extended with a three-block gas history). Action
$u_t \in [0, 1]$, quantity $q_t = u_t Q_t$. Reward
$R_t = \Delta y_t^{\text{actual}}$ post-sandwich. Objective:

$$J(\pi) \;=\; \mathbb{E}\!\left[\textstyle\sum_t R_t\right]
  \;-\; \lambda \cdot \mathrm{CVaR}_\alpha\!\left(-\textstyle\sum_t R_t\right),$$

via the Rockafellar–Uryasev dual representation.

A residual $Q_T$ at the terminal block is force-liquidated **through the
MEV adversary**, not around it. An earlier version of the environment did
the opposite; that loophole is documented below as a reproduced instance
of reward hacking / specification gaming.

---

## Specification gaming (reward-hacking instance)

A first 1M-step sweep on the buggy environment reported a mean reward near
168,000 USDC across every CVaR penalty value, with cross-seed standard
deviation under 50 USDC for the most risk-averse policies. Trace analysis
of the trained policies:

| $\lambda$ | per-block action $u$ | terminal $q$ | terminal share of reward |
|---|---|---:|---:|
| 0.0 | 0.0139, constant | 121,464 | 72% |
| 1.0 | 0.0107, constant | 130,964 | 78% |
| 3.0 | 0.0053, constant | 148,635 | 88% |

The policy did not learn MEV avoidance. It learned to defer most of its
inventory to the terminal block, where the earlier specification routed
reward through plain CFMM with no adversary check. Higher CVaR penalty
$\Rightarrow$ more aggressive deferral: per-block trades carried
stochastic MEV exposure (variance), the terminal dump did not
(deterministic). The risk objective rewarded the loophole. Empirically,
the reward surface behaved as a smooth deferral incentive in the fraction
of inventory withheld, so gradient descent reliably found it.

The patch is a one-line correction in `env/amm.py`: terminal forced
liquidation now flows through the same `mev_adversary` hook as any
in-horizon block. Re-evaluating the same 1M-step checkpoints on the
corrected environment dropped headline rewards from ~168k to between
19k and 52k USDC; the most-deferral runs took the largest hit.

This finding is preserved in the repository rather than quietly removed.
It is the most useful engineering lesson the project produced.

---

## Headline results (corrected environment)

Pool $1\text{e}6 \times 1\text{e}6$, $\gamma = 0.997$, $Q_0 = 10\%$ of
pool depth, $T = 20$. Five seeds per $\lambda$, evaluated on a disjoint
10,000-block TEST window over 64 episodes.

### Analytical baselines

| Policy | TEST mean (USDC) | std |
|---|---:|---:|
| `single_dump` | 82 | 0 |
| `twap` | 37,847 | 2,466 |
| `convex_no_mev` | 37,847 | 2,466 |
| **`gas_aware_greedy` (oracle)** | **58,786** | **616** |

### Sweep A — base PPO, 2×64 trunk, $\sigma_b = 0.5$, obs $\in \mathbb{R}^4$

| $\lambda$ | TEST mean USDC | cross-seed std |
|---|---:|---:|
| 0.0 | **45,863** | 1,357 |
| 0.1 | 31,952 | 764 |
| 0.3 | 32,742 | 1,075 |
| 1.0 | 34,646 | 1,443 |
| 3.0 | 27,342 | 6,708 |

### Sweep B — base PPO, 2×64 trunk, $\sigma_b = 1.5$, obs $\in \mathbb{R}^7$ (3-block gas history)

| $\lambda$ | TEST mean USDC | cross-seed std |
|---|---:|---:|
| 0.0 | **46,542** | 673 |
| 0.1 | 31,356 | 619 |
| 0.3 | 30,921 | 1,706 |
| 1.0 | 30,954 | 4,166 |
| 3.0 | 24,732 | 5,303 |

### Sweep C — pure ablation: 3×256 deep trunk, $\sigma_b = 0.5$, obs $\in \mathbb{R}^7$

Isolates the observation extension from the variance change.

| $\lambda$ | TEST mean USDC | 95% CI | $p$ vs oracle |
|---|---:|---:|---:|
| 0.0 | **44,202** | [37,630, 50,775] | $3.4\times 10^{-3}$ |
| 0.1 | 33,937 | [26,459, 41,415] | $7.5\times 10^{-4}$ |
| 0.3 | 31,909 | [29,948, 33,870] | $2.8\times 10^{-6}$ |
| 1.0 | 30,111 | [27,875, 32,347] | $3.6\times 10^{-6}$ |
| 3.0 | 23,536 | [16,410, 30,662] | $1.6\times 10^{-4}$ |

### Sweep D — capacity control: 3×256 deep trunk, $\sigma_b = 0.5$, obs $\in \mathbb{R}^4$

Holds the observation fixed, expands the trunk by $\sim 16\times$.

| $\lambda$ | TEST mean USDC | 95% CI | $p$ vs oracle |
|---|---:|---:|---:|
| 0.0 | **41,800** | [31,738, 51,862] | $9.1\times 10^{-3}$ |
| 0.1 | 32,658 | [29,706, 35,610] | $1.6\times 10^{-5}$ |
| 0.3 | 29,905 | [26,738, 33,072] | $1.4\times 10^{-5}$ |
| 1.0 | 26,671 | [20,784, 32,558] | $1.1\times 10^{-4}$ |
| 3.0 | 25,977 | [14,109, 37,845] | $1.5\times 10^{-3}$ |

The oracle constant used for Welch's $t$ in Sweeps C and D is
$58{,}929$ USDC, recomputed under the control-sweep evaluation window and
therefore slightly different from the $58{,}786$ USDC value in Table II of
the main sweep.

### What this means

The strongest tested PPO cell (Sweep B, $\lambda = 0$) attains
$46{,}542 / 58{,}786 \approx 79.2\%$ of the analytical oracle.

The oracle is privileged. It is an omniscient closed-form bisection at
every block for the largest $q$ such that the MEV trigger remains false,
with perfect access to the adversary's threshold structure and the
current $c_t$. The model-free PPO agent, starting from zero domain
knowledge and observing only the MDP state, recovers roughly $79\%$ of
that benchmark ceiling.

- **Capacity is not the binding constraint.** Sweep D, with a $\sim 16\times$
  deeper-and-wider trunk under the same configuration, attains $70.9\%$ of
  the oracle — *lower* than the original 2×64 trunk, with substantially
  larger cross-seed variance. Deeper alone does not help.
- **The gas-history extension is not an artefact of the variance bump.**
  Sweep C holds $\sigma_b = 0.5$ and adds the three-block gas history; it
  attains $75.0\%$, within four percentage points of the prior
  $\sigma_b = 1.5$ result. The history feature alone is approximately
  neutral.
- **Gas history does not rescue PPO.** Sweep B improves only the
  $\lambda = 0$ mean by $679$ USDC versus Sweep A and is lower for every
  positive $\lambda$ (row differences from $-3{,}692$ to $+679$ USDC).
- **The lambda response is not strictly monotone.** The unpenalised runs
  are strongest in every configuration; positive CVaR weights reduce mean
  return by roughly $11{,}000$–$22{,}000$ USDC versus $\lambda = 0$, but
  intermediate weights show seed-scale fluctuations rather than a clean
  monotone trend.
- **The gap is statistically reliable.** Welch's $t$ against the
  deterministic oracle returns $p < 10^{-2}$ at every tested $\lambda$
  across both deep-trunk sweeps. The weakest rejection among the controls
  is $p = 9.1 \times 10^{-3}$ at $\lambda = 0$ with the deep trunk, where
  cross-seed variance is largest; tail-$\lambda$ runs reach $p \sim 10^{-6}$.

We do not claim that *no* model-free policy can beat the oracle in this
regime; we observe that the tested PPO baselines, under three controlled
sweeps spanning capacity, observation, and gas-volatility settings, do not.

A recurrent (LSTM/GRU) policy, off-policy methods with prioritised replay
on boundary events, or risk-sensitive formulations beyond CVaR are the
natural follow-ups. None is implemented here.

---

## Reproducing the sweeps

```bash
python3 -m pytest tests/

# Sweep A: base PPO, σ_b = 0.5, obs = 4 (2×64 trunk; earlier commit)
# Sweep B: base PPO, σ_b = 1.5, obs = 7 (2×64 trunk; earlier commit)

# Sweep C: deep PPO, pure ablation
python3 -m scripts.train_test_eval \
    --seeds 42 43 44 45 46 \
    --lambdas 0.0 0.1 0.3 1.0 3.0 \
    --total-timesteps 1000000 \
    --Q0 100000 --T 20 \
    --sigma-b 0.5 --obs-dim 7 \
    --out-dir artifacts/sweep_1M_pure_ablation

# Sweep D: deep PPO, capacity control
python3 -m scripts.train_test_eval \
    --seeds 42 43 44 45 46 \
    --lambdas 0.0 0.1 0.3 1.0 3.0 \
    --total-timesteps 1000000 \
    --Q0 100000 --T 20 \
    --sigma-b 0.5 --obs-dim 4 \
    --out-dir artifacts/sweep_1M_deep_trunk

# 95% CIs and Welch's t against the oracle, per λ
python3 -m scripts.stats_test --manifest artifacts/sweep_1M_pure_ablation/manifest.json
python3 -m scripts.stats_test --manifest artifacts/sweep_1M_deep_trunk/manifest.json
```

The deep-trunk runs above used `HIDDEN = 256`, `N_TRUNK_LAYERS = 3` in
`agent/ppo/policy.py`. The original 2×64 results came from `HIDDEN = 64`,
`N_TRUNK_LAYERS = 2`.

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

## Publication figures

| File | Content |
|------|---------|
| `artifacts/publication_fig1_specgaming.png` | Reward-hacking collapse: same 1M-step checkpoints on buggy vs.\ fixed environment, against the `gas_aware_greedy` reference |
| `artifacts/publication_fig2_pareto.png` | Headline Pareto on the corrected environment, both base sweeps and four analytical baselines |
| `artifacts/publication_fig3_lambda_response.png` | Lambda response, Sweep A vs.\ Sweep B, with `twap` and `gas_aware_greedy` reference lines |

The IEEE conference paper that wraps these results, including the
$\sim 16\times$ deep-trunk capacity control, the variance-controlled
ablation, the 95% CIs, and the Welch's $t$ tests, is at
[`paper/paper-2/main_2.pdf`](paper/paper-2/main_2.pdf) (source
`paper/paper-2/main_2.tex`).

---

## Known limitations

1. **AR(1) volatility override.** The AR(1) shock standard deviation is
   overridden from the calibrated $\sigma_b \approx 0.045$ to 0.5
   (Sweeps A, C, D) or 1.5 (Sweep B). The calibrated value is too tight
   to flip the MEV trigger at the experimental trade size; the overrides
   are stress-volatility design choices that make the MEV trigger
   informative under the sweeps. A sensitivity sweep over the calibrated
   value remains open.
2. **Privileged oracle.** The `gas_aware_greedy` benchmark has exact
   access to the adversary's threshold structure, the trigger boundary,
   and the current $c_t$. The 79% capture rate is the gap to a strong
   ceiling, not a proof of structural inferiority for the RL paradigm.
3. **Single-pool scope.** Multi-pool spatial routing composes externally
   with the temporal policy at deployment.
4. **Marginal-utility adversary.** JIT liquidity and cyclic atomic
   arbitrage are richer adversaries and are not modelled.
5. **Mean-field assumption.** The trader's swap does not perturb
   subsequent background Swap arrivals.
6. **Evaluation seed.** Stochastic gas/flow streams use a fixed
   evaluation seed. Cross-seed standard deviation in the tables measures
   training convergence, not evaluation robustness; the two would be
   worth disambiguating.

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

- Amodei, D. et al. (2016). *Concrete Problems in AI Safety.*
  arXiv:1606.06565.
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

## Acknowledgments

The author acknowledges the academic support and research environment
provided by the School of Engineering, Jawaharlal Nehru University.

## License

Research code released for academic reproducibility. Use at your own
risk; the findings are intentionally negative for the tested model-free
PPO policies in this regime and should not be deployed as a live
execution strategy without substantial additional engineering.
