# Tests — Scenario Reduction / Adequacy-Cut Screening

## A. Unit tests (TestItems framework, no solver required)
Run: `julia --project=ScenarioReduction/test ScenarioReduction/test/runtests.jl`
Status: **59 assertions pass** (0 failed, 0 errored).

### test/test_adequacy_cuts.jl  (capacity-adequacy cuts)
| Test | Verifies |
|---|---|
| cut coefficients & rhs (MW/availability units) | LHS row = `[ccgt,ocgt,a_solar,a_won,a_woff,0(electrolizer),1(battery)]`; `rhs = peak_demand·demand − hydro_cap − ens_cap` |
| non-dominated reduction drops dominated hours | an hour with ≤ demand and ≥ availability everywhere is pruned |
| keeps one of exact duplicates | identical demanding hours collapse to one |
| `passes_adequacy` is a sound reject filter | under-provisioned x rejected; generous x passes; electrolyzer (load) gives 0 supply credit; exact-rhs passes within tol |
| `adequacy_verdict` reports first binding scenario | returns `(passed, binding_scenario)` correctly |
| max-of-optima centre satisfies every scenario's cuts | element-wise max of per-scenario optima passes all cuts (monotonicity) |

### test/test_sampling.jl  (quasi-random samplers)
Existing: sample shape, mean recovery, covariance recovery, determinism, seed independence,
`:clip` bound enforcement, `:reject` bounds+count, `shrink_covariance` endpoints/convexity,
rank-deficient Σ (jitter).
New (reject-to-target):
| Test | Verifies |
|---|---|
| uniform reject-to-target hits target | returns exactly N, all in `[lb,ub]`, all pass `accept`, `n_drawn ≥ N` |
| gaussian reject-to-target honors box + accept | returns N, all `≥0`, `≤ub`, all pass `accept` |
| deterministic per seed | same seed → identical samples & draw count |
| unsatisfiable accept + cap | returns < N (no throw); soft cap (rounds up to power-of-two ≥ max_draws) |

### test/test_investment_mapping.jl  (pre-existing)
Sample→container alignment, permutation audit, MW→model-unit conversion.

## B. Integration / verification scripts (require Gurobi + TEM git rev 227a80f)

### test_uniform_adequacy_sampling.jl  (current)
Purpose: how Sobol UNIFORM sampling (no Gaussian mean) behaves with the cuts.
Results (scenarios [96,129], target 256):
- Uniform `[0,ub]` acceptance **12.5%** (256 kept / 2048 draws)
- Gaussian around μ* acceptance **6.25%** (256 / 4096)
- Uniform cut→LP precision **45.8%** (22/48 jointly LP-feasible)

### old_scripts/verify_adequacy_screening.jl
Purpose: validate the cuts + mean-shift against ground-truth LP.
Results: **0 false rejections** at both centres (sound); LP-feasible yield
**6.2% (bounds.mean) → 31.2% (μ*)**; μ* = bounds.mean + 7594 MW ccgt; cuts = 118 (scn96) + 68 (scn129).

### old_scripts/decisive_single_scenario_test.jl
Purpose: is a scenario's own optimum feasible against itself? (root-cause test)
Result: scenarios 1 & 19, `is_seasonal` on/off → **all OPTIMAL** ⇒ mapping/units fine,
`is_seasonal` irrelevant under `dummy_cluster!`.

### old_scripts/multiscenario_seasonal_test.jl
Purpose: isolate the cause in the multi-scenario solve.
Result (scenarios [96,129]): `bounds.mean` → **INFEASIBLE** (both `is_seasonal` settings);
max-of-optima → **OPTIMAL** (both) ⇒ genuine capacity adequacy, not a bug; `is_seasonal` A≡B.

### old_scripts/quantify_infeasibility.jl
Purpose: quantify the infeasible fraction of the mean-centred Sobol cloud.
Result (128 samples): scenario 96 feasible 25%, scenario 129 feasible 5.5%, joint 5.5%;
`bounds.mean` infeasible for both individually.

## C. End-to-end smoke (small SD-loop run)
`stochastic_dominance(conn; sampling_mode=:uniform, num_samples=16, number_of_samples_sequences=2, input_data_path=INPUT_DATA_PATH)`
Two-phase run: **Phase A** draws cut-passing samples (per-seed acceptance reported in
`outputs/sampling_stats.csv`; sampler delivers only cut-passing samples). **Phase B** builds one
single-scenario model at a time and solves every sample against it, writing
`outputs/screening_diagnostics.csv` (long: one row per *sequence × sample × scenario*) and
`outputs/cost_matrix.csv` (`sample_id, sequence, scenario_<s>…`; each cell the full objective,
`NaN` if infeasible). `input_data_path` is required (re-read per single-scenario model).

## Key takeaways
- The "everything INFEASIBLE" symptom is **genuine capacity-adequacy screening, not a bug**
  (mapping/units/`is_seasonal` ruled out by A/B/C above).
- Adequacy cuts are a **sound** pre-filter (never reject a truly feasible portfolio).
- Mean-shift + reject-to-target raise the usable-sample yield ~5× and guarantee a full pool.

---
See `ScenarioReduction/SCREENING_NOTES.md` for the full root-cause analysis and methodology.
