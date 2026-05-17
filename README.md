# defi_execution_project

A research codebase for optimal liquidation of an inventory on a Uniswap V2
style constant-product pool, in the presence of a rational MEV sandwich
adversary. The agent is PPO with an optional Rockafellar-Uryasev CVaR
penalty. Gas is sampled from an AR(1)-lognormal model calibrated to live
EIP-1559 fee history, and the background pool flow replays real Swap
events fetched from mainnet.

The scope is deliberately the temporal problem: how to split a large order
across blocks. Spatial routing across multiple pools is a separate convex
program solved well by Angeris and Chitra (2022), and is out of scope here.

This repository serves as a rigorous empirical benchmark, demonstrating both
the capabilities and the structural limits of model-free reinforcement
learning against closed-form convex oracles in deterministic DeFi
environments.


## Quickstart

Tested on Python 3.10 to 3.14 on macOS and Linux.

```
git clone <repo-url>
cd defi_execution_project
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python3 -m pytest tests/             # 93 tests, all offline
python3 -m scripts.run_full_live     # full pipeline on live mainnet RPC
```

The demo notebook in `notebooks/demo.ipynb` is already executed, with all
figures embedded. `notebooks/demo.html` is a self-contained browser export
that opens in any browser without a kernel.


## How the codebase is organised

```
env/        AMM environment, MEV sandwich bot, gas sampler, flow replayer
data/       JSON-RPC client, pool reserves, fee history, Swap-event logs, snapshots
agent/      baseline policies and the PPO + CVaR-PPO stack
scripts/    calibration, training sweeps, evaluation, figure rendering
tests/      93 tests, no network required
notebooks/  demo.ipynb (executed) and demo.html
artifacts/  Pareto figures from the sweeps and the publication figures
```

Everything in `data/snapshots/` is a frozen offline capture of mainnet
state from blocks 25,103,456 and 25,096,742-25,106,741. The tests run
entirely against these snapshots so the suite stays deterministic and
offline.


## The math, briefly

For a CFMM with fee retention gamma, selling q of token X into a pool with
reserves (x, y) yields

    delta_y = y * gamma * q / (x + gamma * q)

The MEV sandwich bot acts iff its optimal arbitrage profit clears the gas
cost: `p_t * max_delta Pi_MEV(q_t, delta) > c_t`, with `p_t = y_t / x_t`
the pre-attack mid. The optimisation runs `scipy.optimize.minimize_scalar`
on a wide bracket and asserts the optimum is strictly interior post-solve
(if it ever binds, the model is silently understating the adversary).

The agent's MDP has state `(Q_t, tau_t, x_t, c_t)`, optionally extended
with a 3-block gas history. Action `u_t in [0, 1]` produces `q_t = u_t * Q_t`.
Reward is the realised `delta_y` after the bot has done whatever it does.

A residual `Q_T` at the terminal block is force-liquidated in a single
trade that DOES go through the MEV adversary. (The earlier spec routed
this through plain CFMM. The agent immediately discovered the loophole.
See below.)


## The headline finding: PPO will exploit any hole you leave

The first 1M-step sweep on the original specification reported a mean
reward around 168,000 USDC across every CVaR penalty value, with cross-seed
standard deviation under 50 USDC for the most risk-averse policies. The
numbers were too clean.

A trace of what the trained policies actually did:

| lambda | per-block action u | terminal q | terminal share of reward |
|---|---|---|---|
| 0.0 | 0.0139, constant | 121,464 | 72% |
| 1.0 | 0.0107, constant | 130,964 | 78% |
| 3.0 | 0.0053, constant | 148,635 | 88% |

The policy did not learn MEV avoidance. It learned to defer most of its
inventory to the terminal block, where the original spec routed reward
through a plain CFMM trade with no MEV adversary. The higher the CVaR
penalty, the more aggressive the deferral, because per-block trades
carried stochastic MEV exposure (variance) while the terminal dump was
deterministic (no variance). The risk objective rewarded the loophole
perfectly.

This is specification gaming. The agent solved the optimisation problem
written down in the code. The optimisation problem was wrong.

The fix was small: terminal forced liquidation now flows through the same
MEV adversary as any normal block. Re-evaluating the same 1M-step
checkpoints on the corrected environment dropped headline rewards from
~168k to between 19k and 52k, exactly as predicted, with the policies
that had deferred most aggressively taking the largest hit.

I have kept this finding visible in the repo rather than quietly fixing
it. It is the most useful engineering lesson from the project.


## What the corrected sweeps say

Two 1M-step sweeps were run on the fixed environment. Pool is 1e6 x 1e6
with gamma = 0.997. `Q0 = 1e5` (10% of pool depth), `T = 20`. Each sweep
runs five seeds across five lambda values, evaluated on a disjoint 10,000
block TEST window over 64 episodes.

Baselines on the same regime:

| Policy | TEST mean USDC | std |
|---|---|---|
| single_dump | 82 | 0 |
| twap | 37,847 | 2,466 |
| gas_aware_greedy | 58,786 | 616 |
| convex_no_mev | 37,847 | 2,466 |

Sweep A, sigma_b = 0.5, observation = `(Q, tau, x, c)`:

| lambda | TEST mean USDC | std across 5 seeds |
|---|---|---|
| 0.0 | 45,863 | 1,357 |
| 0.1 | 31,952 | 764 |
| 0.3 | 32,742 | 1,075 |
| 1.0 | 34,646 | 1,443 |
| 3.0 | 27,342 | 6,708 |

Sweep B, sigma_b = 1.5, observation extended with a 3-block gas history:

| lambda | TEST mean USDC | std across 5 seeds |
|---|---|---|
| 0.0 | 46,542 | 673 |
| 0.1 | 31,356 | 619 |
| 0.3 | 30,921 | 1,706 |
| 1.0 | 30,954 | 4,166 |
| 3.0 | 24,732 | 5,303 |

PPO does not beat the myopic `gas_aware_greedy` baseline at this trade
size. It is worth being precise about what this baseline is: an omniscient
oracle that evaluates the closed-form algebraic limits of the CFMM and the
sandwich adversary at every block, with full knowledge of the trigger
boundary. The model-free PPO agent, starting from zero domain knowledge
and observing only the standard MDP state, learns to capture roughly 79%
of that absolute mathematical maximum (46,542 / 58,786). Tripling the
AR(1) gas volatility and adding a gas history did not close the remaining
gap. This empirical limit, not a "best mean", is the primary publication
finding.

A few honest hypotheses for why:

1. AR(1) gas variation in absolute USDC, at this Q0, may be small relative
   to the MEV profit Pi_MEV. The trigger boundary moves less than I
   assumed.
2. The MLP trunk (two layers, 96 hidden) may be too shallow to discover a
   short temporal pattern.
3. The policy may collapse to a near-deterministic per-block action early
   in training and then ignore the observation. PPO's importance ratio
   clip prevents recovery once the action distribution is too narrow.
4. The reward landscape on fixed-rate strategies may be roughly convex at
   this regime, with `gas_aware_greedy` sitting at the global algebraic
   optimum that a stochastic gradient method cannot perturb without
   recurrent memory.

Distinguishing these requires a recurrent (LSTM) policy or a deeper trunk.
Neither is implemented here.


## Publication figures

- `artifacts/publication_fig1_specgaming.png` shows the bug-fix delta:
  same 1M-step checkpoints, evaluated on the buggy environment and on the
  fixed one, against the `gas_aware_greedy` reference. This is the
  specification-gaming figure.
- `artifacts/publication_fig2_pareto.png` is the headline Pareto on the
  corrected environment. Both sweeps and the four baselines are plotted.
- `artifacts/publication_fig3_lambda_response.png` is a side-by-side bar
  chart of Sweep A versus Sweep B, with `twap` and `gas_aware_greedy` as
  reference lines. It is the negative-result figure.


## Reproducing the sweeps

```
cd defi_execution_project
python3 -m pytest tests/

python3 -m scripts.train_test_eval --seeds 42 43 44 45 46 \
    --lambdas 0.0 0.1 0.3 1.0 3.0 --total-timesteps 1000000 \
    --Q0 100000 --T 20 \
    --out-dir artifacts/sweep_1M_gas_history

python3 -m scripts.render_phase3e \
    --manifest artifacts/sweep_1M_gas_history/manifest.json \
    --output artifacts/pareto_gas_history.png

python3 -m scripts.render_publication
```

Live calibration against the current mainnet WETH/USDC pool:

```
python3 -m scripts.calibrate --pool weth-usdc --Q0 10.0 --T 10 \
    --eth-quote-price 2221.5
```

End-to-end live run with fresh RPC calls:

```
python3 -m scripts.run_full_live
```


## Known limitations

1. The AR(1) shock standard deviation is overridden from the calibrated
   value of around 0.05 to 0.5 (or 1.5 in Sweep B). The calibrated value
   is too tight to flip the MEV trigger at the demo trade size, so the
   override is necessary for the experiment to be interesting at all. A
   sensitivity analysis over the calibrated value is open.
2. Scope is one pool. Multi-pool spatial routing composes externally with
   the temporal policy.
3. The adversary is the marginal-utility sandwich attacker. JIT liquidity
   and cyclic atomic arbitrage are richer adversaries and are not
   modelled.
4. Empirical Swap replay assumes the agent's own trade does not perturb
   future Swap arrivals, which is the standard mean-field assumption.
5. Evaluation uses a fixed seed for the stochastic gas and flow streams.
   The cross-seed standard deviation reported in the tables measures
   convergence consistency across training seeds, not robustness across
   evaluation seeds. The two would be useful to disambiguate but are not
   reported here.


## Tests

```
python3 -m pytest tests/ -v
```

93 tests, all offline. They cover the CFMM math against hand-computed
golden values, the MEV bracket-bind assertion, the AR(1) sampler under
fixed seeds, the cyclic Swap replayer, the dual-head CVaR critic, and the
end-to-end env step under the gas-history observation.
