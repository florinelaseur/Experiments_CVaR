# Stochastic-dominance screening — findings, sampling problem, and solution

This document summarizes the investigation into "everything is INFEASIBLE" in the
stochastic-dominance (SD) screening loop, why it happens, and the adequacy-cut +
feasibility-mean-shift solution that was implemented.

---

## 1. Issues found (root-cause investigation)

The SD loop fixes one investment vector and solves the operational problem against the
selected scenarios. Every Sobol sample, `bounds.mean`, and even a scenario's "own optimum"
came back INFEASIBLE. We chased this to ground; here is what turned out to be true.

### 1.1 What was NOT the bug
- **Variable mapping / units** — already fixed and re-confirmed. Investments reach the
  correct `assets_investment` variables in correct model units via the indices-based
  alignment (`ScenarioReduction/src/investment_mapping.jl`). 48/48 unit tests pass.
- **`is_seasonal`** — *refuted*. The SD loop omits the `UPDATE asset SET is_seasonal =
  false` that the per-scenario generator, `main.jl` benchmark, and `run_solve!` all run.
  Toggling it changes **nothing**: single- and multi-scenario solves are identical in
  status *and* objective to 7 significant figures, and no seasonal/inter-period storage
  constraint appears in the infeasibility conflict (IIS). Under `dummy_cluster!` the
  seasonal storage formulation is non-binding here.
- **Demand scale** — the model solves against ~51–55 GW, which is `peak_demand = 1.5`
  (on `e_demand` in `base-input-data/RIDM-case-study/asset-milestone.csv`) × the raw
  ~34–36 GW profile peak. **The 1.5 IS applied** — confirmed two ways:
  - Official docs (Mathematical Formulation): the consumer balance is
    `Σ inflow − Σ outflow {=,≥} demand_profile × peak_demand`, with `peak_demand` = "peak
    demand of the consumer (MW)" and `demand_profile` a normalized (p.u.) time series.
  - TEM source `constraints/consumer.jl:44`: `incoming_flow − outgoing_flow −
    demand_agg * row.peak_demand ∈ balance_sense`, with `peak_demand` from `asset_milestone`
    and the schema describing it as "Value that multiplies the demand profile time series".
  - Exact arithmetic: the IIS hour had RHS 54572.4; scenario 129's raw demand there is
    36381.6, and `36381.6 × 1.5 = 54572.4` exactly.
  - Convention note: TEM expects `demand_profile` normalized (p.u.) with `peak_demand` =
    the peak MW. Here the profile is already in MW and `peak_demand = 1.5`, so it acts as a
    literal **1.5× uplift on absolute-MW profiles**. The model applies `profile ×
    peak_demand` identically either way; the adequacy cuts use the same product, so the
    cut's demand term equals the LP's consumer-balance demand exactly (hence 0 false
    rejections).
  - Note: `investable = false` on `e_demand` does **not** disable `peak_demand`.
    `investable` only governs capacity expansion (you never invest in a demand node);
    `peak_demand` is a separate consumer demand-scaling parameter, applied regardless.
  This multiplier lives in the shared input folder, so it applies identically everywhere;
  the per-scenario deterministic solves were OPTIMAL *with* it. Real, but not the cause.

### 1.2 What IS going on — genuine capacity-adequacy screening (not a code bug)
- **A scenario's own optimum is feasible against its own scenario.** When properly
  isolated (single scenario, same model setup that produced the optimum), it solves to
  OPTIMAL. The earlier "self-optimum infeasible" result came from solving against the SD
  loop's *scenario pair*, not the single scenario.
- **A provably-adequate portfolio is feasible across the pair.** The element-wise max of
  the two selected scenarios' optima solves to OPTIMAL across both — so the 2-scenario
  model is structurally sound; only the portfolio *size* is the issue.
- **The IIS is a power-balance shortfall:** `consumer_balance[e_demand, h] == ~54.6 GW`
  at a peak-demand / low-solar hour, with supply capped by `flow ≤ availability·investment`
  and only `flow[ens] ≤ 2` of slack. The fixed portfolio physically cannot meet demand.

**Quantified** (128 mean-centered samples, scenarios [96,129]):

| scenario | feasible | infeasible |
|---|---|---|
| 96 | 32/128 (25%) | 75% |
| 129 (binding) | 7/128 (5.5%) | 94.5% |
| joint (both) | 7/128 (5.5%) | 94.5% |

`bounds.mean` itself is infeasible for *both* scenarios individually.

### 1.3 Reproducibility gotcha (separate from the above)
The project's `asset.csv` uses the **old** TEM schema (`storage_method_energy` as a string
enum). Registry **TulipaEnergyModel v0.21.0** expects a BOOLEAN and crashes in
`populate_with_defaults!`. Only `main.jl:10` pins the compatible git rev
(`227a80f…`). `Manifest.toml` is gitignored, so standalone scripts must pin the same rev
or they crash. (`test_stochastic_dominance.jl` now pins it.)

---

## 2. The sampling problem

The SD loop draws investments from a Sobol-Gaussian cloud **centered on `bounds.mean`**
with the empirical covariance over the 144 per-scenario optima.

- `bounds.mean` is averaged over **all 144 scenarios**, which include many solar-rich
  ones, so it is **skewed toward solar** and **light on firm + wind**.
- The randomly selected screening pair (seed 19990907 → **scenarios [96,129]**) is
  **low-solar**, needing more `ccgt`/`wind`. So `bounds.mean` lies **outside** the feasible
  region for that pair.
- A Gaussian centered on an infeasible point spills most of its mass outside the feasible
  region → **~94.5% of samples are infeasible**, and *each* one costs a full LP solve to
  discover. The screen wastes almost all of its work and yields very few comparable
  (feasible) portfolios.

This is a methodology problem (poor sample placement), not a modeling error.

---

## 3. Current solution

Two complementary mechanisms, both grounded in the model's own per-hour power balance.

### 3.1 Cheap per-scenario "demanding-hour" adequacy cuts
`ScenarioReduction/src/adequacy_cuts.jl`

For a fixed investment `x` (MW, `INVESTABLE_ASSETS` order), an upper bound on deliverable
supply at hour `h` is
```
supply_ub_h(x) = x_ccgt + x_ocgt
               + a_solar·x_solar + a_won·x_wind + a_woff·x_wind_offshore
               + x_battery + HYDRO_CAP + ENS_CAP
required_h     = PEAK_DEMAND · demand_h
```
If `supply_ub_h(x) < required_h` for **any** hour, `x` is **provably infeasible** (even the
most optimistic supply can't meet the least optimistic demand). This is a **sound one-sided
filter**: every supply term is an over-estimate and the required side drops optional loads
(electrolyzer / battery charging), so it **never rejects a truly feasible portfolio** — it
only cheaply discards hopeless ones, leaving the rest for the LP.

The 8760 hourly inequalities reduce to the **non-dominated "demanding hours"**: hour `h` is
redundant if some `h'` has `demand_{h'} ≥ demand_h` and `availability_{r,h'} ≤
availability_{r,h}` for every renewable `r`. The surviving frontier (tens of hours) is your
"max demand" and "max demand when renewables ≈ 0" hours, made rigorous. Persisted to
`outputs/adequacy_cuts.csv` (`scenario, timestep, demand, a_solar, a_won, a_woff, rhs`).

### 3.2 Feasibility mean-shift (LP)
`ScenarioReduction/src/adequacy_center.jl::feasibility_center`

The cuts define a polytope `P` of not-provably-infeasible portfolios. `bounds.mean` lies
outside `P`. A tiny LP finds the **minimal upward shift** back into it:
```
min Σ_k shift_k    s.t.  A_s·x ≥ b_s ∀ selected s,  x = bounds.mean + shift,  shift ≥ 0,  x ≤ ub
```
The optimum `μ*` becomes the new sampling centre, so the cloud sits on the feasibility
frontier. The LP doubles as a diagnostic: if it is infeasible, **no** in-bounds portfolio
can serve the scenarios (the scenario set / `ub` is the real constraint). Persisted to
`outputs/feasibility_center.csv`. (Zero-LP fallback: element-wise max of the selected
scenarios' optima, `feasibility_center_max_optima`.)

### 3.3 Reject-to-target sampling in the loop
`ScenarioReduction/src/stochastic_dominance.jl` + `src/sampling.jl`

With cuts on, the loop builds the accept predicate `x -> adequacy_verdict(x, cuts).passed`
and the sampler **keeps drawing until `num_samples` cut-passing samples are produced per
seed** — `sobol_gaussian_reject_to_target` (Gaussian around μ*) or
`scrambled_sobol_uniform_reject_to_target` (uniform over `[0, ub]`, no mean — the cuts
steer). So the solver always receives a full pool of "probable" candidates; the in-loop
`passes_adequacy` check becomes a safety net (should report 0). `num_samples` (default 512)
and the number of scramble seeds `number_of_samples_sequences` (default 5) are both
configurable. Outputs: `outputs/sampling_stats.csv` (per-seed draws + acceptance rate).
This sampling stage is **Phase A** of `stochastic_dominance` and is independent of how the
kept samples are later evaluated (Phase B, §3.5).

### 3.5 Phase B — per-scenario operational cost matrix
`ScenarioReduction/src/stochastic_dominance.jl` + `src/single_scenario.jl`

SD screening compares the *distribution of operational cost across scenarios* for each
candidate investment, i.e. the matrix `C_i(z_k)` = cost of sample `k` under scenario `i`. A
single **multi-scenario** solve only yields the probability-weighted sum, which cannot be
decomposed back into the per-scenario costs. So Phase B replaces the old one-model loop with
**N single-scenario solves**:

- **Phase A (unchanged)** draws the cut-passing samples (§3.1–3.4).
- **Phase B** loops over the selected scenarios. For each scenario it builds one
  single-scenario, full-hourly operational model via
  `build_single_scenario_model` (`src/single_scenario.jl`) — same low-level TEM pipeline
  (`create_internal_tables!` → `compute_*_indices` → `prepare_profiles_structure` →
  `create_model`) and `is_seasonal=false` setup the per-scenario generator and the decisive
  test use — fixes **every** sample's investment on it (`fix_variables_from_sample`), solves,
  and records the full model objective (`NaN` if not OPTIMAL) into that scenario's column.
  The model + its DuckDB connection are then **freed before the next scenario is built**, so
  only **one** operational model is ever in memory (not N). Solving all samples consecutively
  on the same model also maximises dual-simplex warm-start reuse (`disable_presolve!` after the
  first solve).

**Solver load = `scenarios × num_samples × seeds`** single-scenario LPs (each far cheaper than
the old multi-scenario LP). Outputs:
- `outputs/screening_diagnostics.csv` — long format, one row per *(sequence, sample, scenario)*:
  `sequence, sample_id, scenario, passed_cut, lp_status, objective, solve_elapsed_sec`.
- `outputs/cost_matrix.csv` — `sample_id, sequence, scenario_<s>…`; each cell is the operational
  objective for that sample/scenario (`NaN` if infeasible). This is the direct input to the SSD
  pairwise check in the next pipeline step.

The per-scenario `passes_adequacy` check inside Phase B is a safety net — reject-to-target
already guarantees every kept sample passes every scenario's cuts. `stochastic_dominance`
returns a NamedTuple `(scenarios, cost_matrix, cost, diagnostics, center, cuts,
scenario_dominance, optimal, infeasible, rejected)`.

### 3.3 Phase C — scenario dominance from the cost matrix

After Phase B, `dominating_scenarios` (`src/scenario_dominance.jl`) compares **scenario columns**
of `cost_matrix.csv` across all samples (rows):

- Scenario **A** dominates **B** iff `cost(A,k) >= cost(B,k)` for every sample `k`, with strict
  `>` for at least one `k`.
- **`NaN` → `+Inf`** (worst outcome); two `NaN`s at the same sample compare equal (no strict
  inequality from that row).

Output: `outputs/scenario_dominance.csv` — long format `dominator, dominated` for each pair.
Return field `scenario_dominance` includes `scenarios`, `dominates` (N×N Bool matrix),
`pairs`, and `undominated` (scenario ids not dominated by any other).

### 3.4 Verification (`ScenarioReduction/old_scripts/verify_adequacy_screening.jl`)
Against ground-truth LP for scenarios [96,129]:

| centre | cut-pass | LP-feasible yield | false rejections |
|---|---|---|---|
| `bounds.mean` | 14.1% | **6.2%** | **0** |
| μ* (shifted) | 50.0% | **31.2%** | **0** |

- **Soundness holds** — 0 false rejections at both centres (no cut-rejected sample was
  LP-feasible).
- **Yield lifts ~5×** (6.2% → 31.2%). For this pair μ* = `bounds.mean` + **7594 MW ccgt**
  (LP = OPTIMAL); cuts = 118 (scn 96) + 68 (scn 129) demanding hours.
- End-to-end (8 samples): 5 cheap-rejected, 3 LP-solved → all OPTIMAL.

---

## 4. Limitations and possible next steps
- The cuts are **necessary, not sufficient**: some cut-passing samples still fail the LP
  (storage energy limits, ramping, the H2 chain). That's why the LP remains the final
  check; the cuts only prune the provably-doomed.
- The sampler now uses **reject-to-target** (keeps drawing until `num_samples` cut-passing
  samples are found per seed), so the solver gets a full pool of probable candidates rather
  than the ~half that survive a mean-centred Gaussian. Remaining knobs to raise the *raw*
  acceptance rate (fewer draws): shift a margin above `μ*`, or shrink the sampling covariance.
- Alternative to hard rejection: raise the `ens`/VOLL slack so shortfalls are **costed**
  rather than infeasible, making every sample comparable and ranked by cost.
- The screening pair [96,129] is a random draw; a more representative or larger scenario
  set may be worth investigating.

---

## 5. File map
| File | Role |
|---|---|
| `src/adequacy_cuts.jl` | cut construction, dominance reduction, `passes_adequacy`, CSV writers (solver-free) |
| `src/adequacy_center.jl` | `feasibility_center` LP (μ*) |
| `src/scenario_dominance.jl` | `dominating_scenarios` — pairwise scenario dominance on the cost matrix (Phase C) |
| `src/stochastic_dominance.jl` | Phase A (cuts + mean-shift + reject-to-target sampling) + Phase B (per-scenario evaluation, cost matrix) + Phase C call |
| `src/single_scenario.jl` | `build_single_scenario_model` — one full-hourly single-scenario operational model (Phase B) |
| `src/sampling.jl` | reject-to-target samplers (`sobol_gaussian_reject_to_target`, `scrambled_sobol_uniform_reject_to_target`); `:reject` `accept` predicate |
| `test/test_adequacy_cuts.jl`, `test/test_sampling.jl`, `test/test_scenario_dominance.jl` | unit tests (cuts; sampling; scenario dominance) |
| `test_uniform_adequacy_sampling.jl` | uniform-vs-gaussian acceptance + uniform cut→LP precision |
| `old_scripts/verify_adequacy_screening.jl` | ground-truth LP validation (soundness + yield) |
| `old_scripts/{decisive_single_scenario_test,multiscenario_seasonal_test,quantify_infeasibility}.jl` | root-cause diagnostic harness (archived) |
| `outputs/{adequacy_cuts,feasibility_center,sampling_stats}.csv` | Phase A artifacts |
| `outputs/{screening_diagnostics,cost_matrix,scenario_dominance}.csv` | Phase B–C artifacts |

### Run
```
julia --project=. ScenarioReduction/test_stochastic_dominance.jl               # full SD screen
julia --project=. ScenarioReduction/test_uniform_adequacy_sampling.jl          # uniform vs gaussian acceptance + precision
julia --project=. ScenarioReduction/old_scripts/verify_adequacy_screening.jl   # soundness + yield (archived)
julia --project=ScenarioReduction/test ScenarioReduction/test/runtests.jl      # unit tests
```
