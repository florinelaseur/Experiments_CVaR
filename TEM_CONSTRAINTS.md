# TulipaEnergyModel.jl — Constraint & Math-Model Map (for Experiments_CVaR)

> **Purpose.** A sourced map of how TulipaEnergyModel.jl (TEM) builds its optimization
> constraints and objective, tied to the `Experiments_CVaR` RIDM case study and its data.
> Every constraint below is cited to (a) the TEM source `file:line` and (b) the official
> [Mathematical Formulation](https://tulipaenergy.github.io/TulipaEnergyModel.jl/stable/40-scientific-foundation/40-formulation/)
> docs.

**Investigated artifacts**

| Thing | Location | Identity |
|---|---|---|
| TEM clone (read here) | `C:\Users\rogin\Research course\TulipaEnergyModel.jl` | HEAD `88d7accd` (v0.21.0) |
| TEM rev the project solves with | pinned in [`main.jl:10`](main.jl) | `227a80f7907e2c7178edb0697874cfb6666ad644` |
| Clustering | `C:\Users\rogin\Research course\TulipaClustering.jl-main` | `dummy_cluster!`, `cluster!` |
| Consuming project | `C:\Users\rogin\Experiments_CVaR` | `main.jl`, `ScenarioReduction/` |
| Case-study data | `base-input-data/RIDM-case-study/` | RIDM (electricity + H2) |

TEM paths below are **relative to the TEM clone root** (e.g. `src/constraints/consumer.jl`).
Project paths are relative to `Experiments_CVaR`.

---

## 0. Version reconciliation — pinned rev vs. this clone (READ THIS FIRST)

The pinned rev `227a80f7` is **not an ancestor of this clone's `main`**. It is the *tip of the
CVaR feature branch* (PR [#1575](https://github.com/TulipaEnergy/TulipaEnergyModel.jl/pull/1575),
"Apply suggestions from code review", 2026-05-05). That branch was later **squash-merged** into
`main` as commit `b9378a5d "Add CVaR scenario tail excess constraints (#1575)"`. Their common
ancestor is `f00de485`. Current `HEAD` (`88d7accd`) is 5 commits past that merge.

```
f00de485 ──┬── 227a80f7  (PINNED: CVaR PR branch tip → what Experiments_CVaR solves)
           └── b9378a5d ── … ── 88d7accd  (this clone's main / HEAD)
                (#1575 squashed)   (#1596 rename, #1602 uc-param, docs)
```

**What this means for the constraints below.** I diffed every constraint-relevant file
between the pinned rev and HEAD. The **math is unchanged**; the differences are renames and a
parameter refactor:

| File | pinned `227a80f7` vs HEAD `88d7accd` | Effect on the model |
|---|---|---|
| `src/constraints/consumer.jl` | **byte-identical** | none |
| `src/constraints/conversion.jl` | **byte-identical** | none |
| `src/constraints/conditional_value_at_risk.jl` | **byte-identical** | none |
| `src/constraints/energy.jl` | **byte-identical** | none |
| `src/constraints/capacity.jl` | renamed only (#1596) | none on math |
| `src/constraints/storage.jl` | renamed only (#1596) | none on math |
| `src/input-schemas.json`, `asset` table | **`investment_method` → `vintage_method`** | **schema-breaking name change** |

The big one is PR **#1596 "Refactor and rename investment methods"**:

| Pinned rev `227a80f7` (what the data uses) | HEAD `88d7accd` |
|---|---|
| asset column **`investment_method`** | asset column **`vintage_method`** |
| value `simple` / `none` | value `aggregated` |
| value `compact` | value `compact_profiles` |
| value `semi-compact` | value `compact_efficiencies` |
| `*_simple_method` constraints/expressions | `*_aggregated_vintage_method` |
| `*_compact_method` | `*_compact_vintage_method` |
| `*_semi_compact_method` | `*_compact_efficiencies_vintage_method` |

> ⚠️ **The case-study `asset.csv` uses the OLD column `investment_method` with values `simple`/`none`**
> (see [`asset.csv`](base-input-data/RIDM-case-study/asset.csv) header). That matches the **pinned
> rev** and is *incompatible with this clone's HEAD*, where the column is `vintage_method` with
> renamed values. So **do your reading of constraint *logic* against this clone (it's identical),
> but be aware that the SQL identifiers, expression keys, and the asset-table column name differ**.
> Throughout this doc I cite HEAD line numbers and give the pinned-rev name in‑line where it differs
> (e.g. "`available_asset_units_aggregated_vintage_method`" = pinned "`…_simple_method`").

Also relevant but not affecting these constraints: PR **#1602** refactored the `unit_commitment`
parameter, and the unit-commitment builder was split into
`unit-commitment-{logic,start-up-upper-bound,shut-down-upper-bound}.jl`. The RIDM case sets
`unit_commitment = false` and `ramping = false` for every asset, so UC/ramping constraints are
not generated at all here.

---

## 1. Consumer / demand balance

**Source:** [`src/constraints/consumer.jl:6-107`](src/constraints/consumer.jl) — `add_consumer_constraints!`
**Docs:** *Balance Constraint for Consumers* → `#Constraints-for-Energy-Consumer-Assets`

### JuMP expression built (consumer.jl:42-47)
```julia
@constraint(model,
    incoming_flow - outgoing_flow - demand_agg * row.peak_demand in consumer_balance_sense)
```
where (consumer.jl:35-41)
```julia
demand_agg = _profile_aggregate(profiles.rep_period,
                (row.profile_name, row.milestone_year, row.rep_period),
                row.time_block_start:row.time_block_end,
                Statistics.mean,   # ← profile aggregated over the block by MEAN
                1.0)               # ← default if no 'demand' profile attached
```
`consumer_balance_sense` (consumer.jl:18-24) is read from `asset.consumer_balance_sense`:
`"=="` → `EqualTo(0)`, `">="` → `GreaterThan(0)`, anything else → `LessThan(0)`.

(There is also a special branch, consumer.jl:25-32, for a self-loop flow `asset→asset` where
the "demand" is itself a flow variable times `peak_demand`; not used in RIDM.)

### Demand = profile × peak_demand; how the profile is aggregated
- `peak_demand` = `asset_milestone.peak_demand`.
- `demand_agg` = **arithmetic mean** of the `demand`-type profile over the timestep block
  `time_block_start:time_block_end`. With a single 8760-step rep period and full resolution,
  each block is one timestep, so `demand_agg` = that hour's profile value.
- `_profile_aggregate` ([`src/utils.jl:30-47`](src/utils.jl)) looks the profile up by
  `(profile_name, year, rep_period)`; **if missing it returns `mean(repeat(1.0))` = 1.0**, i.e.
  demand falls back to `peak_demand` itself.

### DuckDB tables/columns feeding it (`_create_consumer_table`, consumer.jl:57-107)
| Table | Columns used |
|---|---|
| `cons_balance_consumer` | `id, asset, milestone_year, rep_period, time_block_start, time_block_end` |
| `asset` | `type, consumer_balance_sense` |
| `asset_milestone` | `peak_demand` |
| `assets_profiles` | `profile_name` joined **`ON … AND profile_type = 'demand'`** (consumer.jl:103) |
| `var_flow` | self-loop var id (`cte_loop`) |
- Incoming/outgoing flow sums come from precomputed expressions
  `cons.expressions[:incoming]` / `[:outgoing]` (consumer.jl:49-50).
- The join on `'demand'` is in the `ON` clause (an OUTER join) on purpose so consumers without a
  demand profile still get a row (then `demand_agg` defaults to 1.0). See the note at consumer.jl:57-69.

### Matches the formulation
$$\sum_{f\in\mathcal F^{in}} v^{flow}_f - \sum_{f\in\mathcal F^{out}} v^{flow}_f \;\{=,\ge\}\; p^{\text{demand profile}}\cdot p^{\text{peak demand}}$$
The prior session's derivation is **confirmed exactly** (mean-aggregated profile × peak_demand,
balance sense from the asset row).

### RIDM specifics
- `e_demand`: `consumer_balance_sense = "=="`, `peak_demand = 1.5`, demand profile `demand`.
- `h2_demand`: `"=="`, `peak_demand = 0.1`, **no** demand profile attached
  ([`assets-profiles.csv`](base-input-data/RIDM-case-study/assets-profiles.csv) has no `h2_demand`
  row) → `demand_agg` defaults to **1.0**, so H2 demand is the constant `0.1` every hour.
- `spillage`: `consumer_balance_sense = ">="`, `peak_demand = 0` → a free **sink** (lets hydro
  spill: `incoming − outgoing ≥ 0`).

---

## 2. Capacity / maximum output & input flow limits

**Source:** [`src/constraints/capacity.jl`](src/constraints/capacity.jl) — `add_capacity_constraints!`
**Docs:** *Capacity Constraints* → `#cap-constraints` (Maximum Output / Input Flows Limit)

### The two pieces: an availability×capacity *expression*, then the *constraint*

**(a) `profile_times_capacity` expression** (aggregated/"simple" method, capacity.jl:46-68):
```julia
@expression(model,
    row.capacity *
    _profile_aggregate(profiles.rep_period,
        (row.profile_name, row.milestone_year, row.rep_period),
        row.time_block_start:row.time_block_end,
        Statistics.mean, 1.0) *                          # availability_agg (default 1.0)
    expr_avail_aggregated_vintage_method[row.avail_id])  # available units = initial + invested
```
**(b) the constraint** (capacity.jl:265-290, suffix loop):
```julia
@constraint(model, outgoing_flow ≤ profile_times_capacity)   # max_output_flows_limit_*
```
Incoming (storage charging) is symmetric: expression at capacity.jl:155-178, constraint
`incoming_flow ≤ profile_times_capacity` at capacity.jl:317-340.

So the realized limit is:
$$\text{flow} \le \underbrace{\overline{p^{\text{avail}}}}_{\text{mean of availability profile, default }1}\cdot\; p^{\text{capacity}}_a \cdot v^{\text{available units}}_{a,y}$$
matching the docs' Maximum Output/Input Flows Limit.

### How availability profiles & investment enter — thermal vs VRE
- **`row.capacity`** = `asset.capacity` (per **unit** of investment).
- **`expr_avail_…[avail_id]`** = available units = `initial_units + Σ assets_investment − decommissioned`.
  This is where `assets_investment` enters the capacity limit (see §5).
- **`availability_agg`** = `Statistics.mean` of the `'availability'` profile over the block, **default 1.0**:
  - **VRE (solar/wind/wind_offshore):** an `'availability'` profile is attached in
    [`assets-profiles.csv`](base-input-data/RIDM-case-study/assets-profiles.csv)
    (`solar`, `wind_onshore`, `wind_offshore`), values in `[0,1]` → time-varying derate < 1.
  - **Thermal / dispatchable (ccgt, ocgt, ens, smr_ccs):** *no* availability profile → `availability_agg`
    defaults to **1.0** → `flow ≤ capacity × units`. **This is exactly the "thermal availability = 1"
    assumption from the prior session.**
- Storage assets get extra `*_with_binary` variants (capacity.jl:70-152, 180-258) that also use the
  `is_charging` binary to forbid simultaneous charge/discharge — but RIDM sets
  `use_binary_storage_method` empty, so those are inactive.

### Capacity index query (`_append_capacity_data_to_indices_aggregated_vintage_method`, capacity.jl:505-560)
Joins `cons_capacity_outgoing_aggregated_vintage_method` (pinned: `…_simple_method`) ← `asset`
← `asset_commission` ← `expr_available_asset_units_aggregated_vintage_method`
(pinned: `…_simple_method`) ← `assets_profiles` (`profile_type='availability'`), filtering
**`WHERE asset.vintage_method = 'aggregated'`** (pinned: `asset.investment_method in ('simple','none')`).

### RIDM specifics
- Investable producers (`ccgt, ocgt, solar, wind, wind_offshore`) and `battery` use `simple`
  investment (= `aggregated`). Non-investable assets (`ens, smr_ccs, hydro_reservoir, h2_storage,
  water_borrower`) use `none` (also handled by the aggregated path).
- `capacity_coefficient = 0` on the `hydro_reservoir→spillage` and `water_borrower→hydro_reservoir`
  flows ([`flow-commission.csv`](base-input-data/RIDM-case-study/flow-commission.csv)) → those
  flows are **uncapped** (safety valves; see §4/§9).
- No flow is investable (`flow_milestone.investable = false` everywhere), so transport-capacity
  variables (`src/constraints/transport.jl`) are not generated.

---

## 3. Storage — intra-rep-period, inter-period (seasonal), and the `is_seasonal` toggle

**Source:** [`src/constraints/storage.jl`](src/constraints/storage.jl) — `add_storage_constraints!`;
energy-capacity expression in [`src/expressions/storage.jl`](src/expressions/storage.jl).
**Docs:** `#rep-period-storage-balance`, `#accumulated-intra-period-storage-balance`,
`#inter-period-storage-balance`.

### 3a. Intra-rep-period balance (non-seasonal storage) — storage.jl:30-159
First block with a defined initial level (storage.jl:57-65):
```julia
var_storage_level[id] == initial_storage_level
    + profile_agg * row.storage_inflows                       # inflows
    + storage_charging_efficiency * incoming_flow
    - outgoing_flow / storage_discharging_efficiency
```
Subsequent blocks (storage.jl:90-98) replace `initial_storage_level` with
`computed_storage_loss_coef * previous_level`, where
`computed_storage_loss_coef = (1 - storage_loss_from_stored_energy)^duration` (storage.jl:84-88).
- `profile_agg` = `_profile_aggregate(..., 'inflows' profile, sum, 0.0)` — **summed** over the block
  (default 0), storage.jl:39-45.
- Max/min level (storage.jl:108-158):
  `var_storage_level ≤/≥ {max,min}_storage_level_agg * available_energy_capacity[…]`,
  where the level profile is mean-aggregated (default 1.0 for max, 0.0 for min).

### 3b. Accumulated intra-period (seasonal only) — storage.jl:279-331
For `is_seasonal=true` assets, an *accumulated* intra-period level is built per rep period
(starts at 0 in block 1; storage.jl:298-306), with the same inflow/charge/discharge/loss terms.
This is the per-rep-period net contribution that the inter-period balance then chains.

### 3c. Inter-period / seasonal balance — storage.jl:164-276
```julia
var_storage_level_inter_period[id] ==
    computed_storage_loss_coef * (initial_storage_level | previous_level)
    + accumulated_intra_period
```
- Indexed by **`(asset, milestone_year, scenario, period_block)`** — note the **scenario** index
  (storage.jl:191), not rep_period.
- `accumulated_intra_period` is the `cons.expressions[:accumulated_intra_period]` expression,
  i.e. `Σ_{k} map_weight · (accumulated intra-period level at last block of rep period k)`.
- `computed_storage_loss_coef = (1 - storage_loss_from_stored_energy)^duration_period_block`
  (storage.jl:178-182), with `duration_period_block = Σ timeframe_data.num_timesteps` over the
  period block (storage.jl:366-393 — a temp table `t_duration_inter_period`).
- Max/min inter-period levels use **`profiles.inter_period`** and **`assets_timeframe_profiles`**
  (keyed by `milestone_year, scenario`), storage.jl:225-275, 340-352.

Matches the docs:
$$v^{\text{inter}}_{a,s,p_y}=(1-p^{\text{loss}})^{p^{\text{dur}}_y}v^{\text{inter}}_{a,s,p_y-1}+\sum_{k_y}p^{\text{map}}_{s,p_y,k_y}\,v^{\text{accum}}_{a,k_y,b^{\text{last}}}$$

### What `is_seasonal` actually toggles
It selects **which storage variables exist**, and therefore which constraints get built. From
[`src/sql/create-variables.sql`](src/sql/create-variables.sql):
- `is_seasonal = false` (line 276) → creates **`var_storage_level_rep_period`** → only §3a runs.
- `is_seasonal = true` (lines 314, 352) → creates **`var_storage_level_inter_period`** and
  **`var_accumulated_storage_level_intra_period`** → §3b + §3c run (and the intra-rep-period
  §3a is replaced by the accumulated form).

The constraint tables `cons_balance_storage_inter_period` and `cons_accumulated_storage_intra_period`
are created in [`src/sql/create-constraints.sql:511-527`](src/sql/create-constraints.sql) — they are
empty unless seasonal storage exists, so the constraints simply don't materialize otherwise.

### Energy capacity (the RHS of the level limits) — `src/expressions/storage.jl:41-69`
```julia
if storage_method_energy == "optimize_storage_capacity":
    capacity_storage_energy * initial_storage_units
        + capacity_storage_energy * (avail_storage_units - initial_storage_units)   # uses assets_investment_energy
elseif storage_method_energy == "use_fixed_energy_to_power_ratio":
    capacity_storage_energy * initial_storage_units
        + energy_to_power_ratio * capacity_asset * (avail_asset_units - initial_asset_units)  # ties to assets_investment
else:  # 'none'
    capacity_storage_energy * initial_storage_units
```
So:
- **`capacity_storage_energy`** = MWh per energy unit (used directly in the energy method).
- **`energy_to_power_ratio`** = hours; converts power capacity → energy capacity for storage that
  invests in *power* (not a separate energy variable).
- **`initial_storage_level`** seeds the first block and (in TEM's cycling form) bounds the last block.
- **`storage_inflows`** scales the inflow profile in the balance.

### Why a single `dummy_cluster!` period makes seasonal storage degenerate
`dummy_cluster!` ([`TulipaClustering.jl-main/src/convenience.jl:166-186`](C:\Users\rogin\Research course\TulipaClustering.jl-main\src\convenience.jl)) sets
`period_duration = MAX(timestep)` (= 8760 for an hourly year) and calls `cluster!(conn, 8760, 1)`
— **one representative period equal to the whole horizon, and one timeframe period**. Therefore:
- `rep_periods_data`: a single rep period, `num_timesteps = 8760`, `resolution = 1`.
- `rep_periods_mapping`: a single period → the single rep period (per scenario), `weight = 1`.
- The **timeframe has exactly one period**, so the inter-period balance (§3c) has a single
  `period_block` with no predecessor — there is nothing to "carry between periods." The
  full-year chronology is already represented inside the one 8760-step rep period via the
  intra-rep-period balance (§3a). Seasonal modelling therefore buys nothing over the single long
  rep period, and the inter-period level variable collapses to (loss-decayed) initial level +
  the single accumulated intra-period total.

This is exactly why **`main.jl` forces `is_seasonal = false` in the benchmark** right after
`dummy_cluster!`:
```julia
TC.dummy_cluster!(connection_benchmark; layout=layout)
TEM.populate_with_defaults!(connection_benchmark)
DuckDB.query(connection_benchmark, "UPDATE asset SET is_seasonal = false")   # main.jl:152-154
```
(For the *clustered* case studies — `cluster!` with `rp ∈ {30,60,90}` — there are many periods in
the timeframe, so `hydro_reservoir` and `h2_storage` (both `is_seasonal=true` in `asset.csv`)
genuinely use the inter-period balance.)

### RIDM storage assets
| Asset | `is_seasonal` | `storage_method_energy` | `capacity_storage_energy` | `energy_to_power_ratio` | charge/discharge η |
|---|---|---|---|---|---|
| `battery` | false | `use_fixed_energy_to_power_ratio` | 0.1 | 2.0 | 0.95 / 0.95 |
| `h2_storage` | true | `none` | 16.8 | 0.0 | 0.65 / 0.65 |
| `hydro_reservoir` | true | `none` | 403.2 | 0.0 | 1.0 / 1.0 |
`hydro_reservoir` has `initial_storage_level = 201.6`, `storage_inflows = 0.1`
([`asset-milestone.csv`](base-input-data/RIDM-case-study/asset-milestone.csv)) and an `inflows`
profile `hydro_inflow`.

---

## 4. Conversion balance + the H2 chain

**Source:** [`src/constraints/conversion.jl:6-49`](src/constraints/conversion.jl)
**Docs:** *Balance Constraint for Conversion Assets* → `#conversion-balance-constraints`

### JuMP expression (conversion.jl:17-21)
```julia
@constraint(model, conversion_efficiency * incoming_flow == outgoing_flow)
```
`conversion_efficiency` = `asset_commission.conversion_efficiency` (conversion.jl:41-45). Equality
sense. (TEM's full formulation also carries per-flow `conversion_coefficient`; here every coefficient
is 1.0, so the constraint reduces to the simple form above.)

### The electrolyzer
`electrolizer` (`type = conversion`, `conversion_efficiency = 0.65`,
[`asset-commission.csv`](base-input-data/RIDM-case-study/asset-commission.csv)):
$$0.65 \cdot v^{flow}_{(\text{e\_demand}\to\text{electrolizer})} = v^{flow}_{(\text{electrolizer}\to\text{h2\_demand})}$$
i.e. 1 MWh electricity → 0.65 MWh-H₂ (LHV). It is also fed directly by `wind_offshore→electrolizer`.

### The full H2 chain (from [`flow-milestone.csv`](base-input-data/RIDM-case-study/flow-milestone.csv) + `asset.csv`)
```
              electricity                         hydrogen
 e_demand ───────────────► electrolizer ──0.65──► h2_demand ◄── smr_ccs   (commodity_price 250)
 wind_offshore ──────────► electrolizer            ▲   │
                                                   │   ▼
                                            h2_storage (seasonal, η=0.65, E-cap 16.8)
```
- **`h2_demand`** (consumer, `peak_demand=0.1`, no profile → constant 0.1/h): balanced by
  `electrolizer→h2_demand` + `smr_ccs→h2_demand` + `h2_storage→h2_demand`, minus `h2_demand→h2_storage`.
- **`smr_ccs`** (producer, capacity 0.5, `none`/non-investable): backup H₂ from steam-methane-
  reforming-with-CCS, priced at `commodity_price = 250` on `smr_ccs→h2_demand` — the H₂ analogue of
  loss-of-load. `main.jl` counts hours with `smr_ccs→h2_demand > 0` as `num_loss_of_load_h2_demand`.
- **`h2_storage`** seasonal storage charges from `h2_demand→h2_storage` and discharges via
  `h2_storage→h2_demand`.

---

## 5. Investment variables/constraints; MW ↔ units; the `capacity` column

**Source:** investment variables in [`src/sql/create-variables.sql`](src/sql/create-variables.sql);
available-units expressions in `src/expressions/` (`available_asset_units_*`, `available_energy_*`);
investment limits/costs in `src/constraints/capacity.jl` & `src/objectives/`.
**Docs:** *Constraints for Investments*, *Expressions for the Objective Function*.

### Variables
- **`assets_investment`** = `v^{inv}_{a,y}` — number of **units** invested in asset `a`
  (integer if `investment_integer=true`, else continuous; RIDM uses `false` → continuous).
- **`assets_investment_energy`** = `v^{inv energy}_{a,y}` — invested **energy** units for storage that
  uses `storage_method_energy = optimize_storage_capacity` (none of the RIDM assets do; `battery`
  uses the fixed E/P ratio path, so its energy capacity rides on `assets_investment`).

### How `capacity` (the `asset.csv` column) relates units → physical capacity
The **available-units** expression (docs, and `src/expressions/`) is
$$v^{\text{available units}}_{a,y}=p^{\text{initial units}}_{a,y}+\sum_i v^{\text{inv}}_{a,i}-\sum_i v^{\text{decom}}_{a,i}$$
and physical power capacity is **`p^{capacity}_a · v^{available units}`**. So `asset.capacity` is
**capacity per investment unit** (the conversion factor between the integer/continuous unit count and
physical MW/GW). It appears:
- in the capacity limit RHS (§2): `flow ≤ availability · capacity · available_units`;
- in the storage energy capacity (§3, fixed-E/P path): `energy_to_power_ratio · capacity · units`;
- in every investment **cost** term: `investment_cost · capacity · v^{inv}` (so cost is per physical
  MW/GW, not per unit) — see §6.

### Investment limit constraint (docs *Constraints for Investments*)
$$v^{\text{inv}}_{a,y}\le \frac{p^{\text{inv limit}}_{a,y}}{p^{\text{capacity}}_a}$$
RIDM leaves `investment_limit` blank in
[`asset-commission.csv`](base-input-data/RIDM-case-study/asset-commission.csv) → unbounded
investment (only the budget/feasibility caps it). `investment_group` is empty → no group limits.

### Fixing investments (the screening hook in this project)
`main.jl` and `ScenarioReduction/src/dominance.jl` **fix** `assets_investment` (and
`assets_investment_energy`) to candidate values and re-solve the *operational* problem
(`fix_variables_from_solution!` / `fix_variables_from_sample`, dual-simplex warm-start). This is the
"capacity-adequacy screening" referenced in the prior session — the investment variables become
parameters and only the §1–§4 operational constraints + objective remain active.

---

## 6. Objective + CVaR / risk

**Source:** [`src/objectives/create.jl:19-71`](src/objectives/create.jl) — `add_objective!`;
CVaR term in [`src/objectives/conditional_value_at_risk_term.jl`](src/objectives/conditional_value_at_risk_term.jl);
per-scenario operational cost in [`src/expressions/operational-costs.jl`](src/expressions/operational-costs.jl);
per-scenario total cost in [`src/expressions/conditional_value_at_risk.jl`](src/expressions/conditional_value_at_risk.jl);
tail-excess constraint in [`src/constraints/conditional_value_at_risk.jl`](src/constraints/conditional_value_at_risk.jl).
**Docs:** *Objective Function* → `#math-objective-function`; *Conditional Value at Risk Constraints* → `#cvar-constraints`.

### Objective assembled (create.jl:31-70)
$$\min\;(1-\lambda)\Big[\underbrace{\text{inv}+\text{fixed}}_{\text{1st stage}}+\sum_{s}p^{\text{prob}}_s\big(\text{flows\_op}_s+\text{unit\_on}_s\big)\Big]+\lambda\cdot\text{CVaR}_\alpha$$
Each cost builder receives `lambda` and adds **`(1 - lambda) × term`** (e.g.
[`flows_operational_cost.jl:19`](src/objectives/flows_operational_cost.jl)); the CVaR builder adds
**`lambda × CVaR`**. With `λ=0` the model is risk-neutral expected cost.

### CVaR term (conditional_value_at_risk_term.jl:46-57)
```julia
conditional_value_at_risk_term =
    value_at_risk_threshold_mu + (1/(1-alpha)) * sum(prob_s * tail_excess_slack_xi[s])
# added as  lambda * conditional_value_at_risk_term
```
Skipped when `lambda <= 0` **or** `n_scenarios <= 1` (conditional_value_at_risk_term.jl:38-40).

### Tail-excess constraint (conditional_value_at_risk.jl:42-49)
$$\xi_s \ge \text{total\_cost}_s - \mu \qquad \forall s$$
- `var_tail_excess_slack_xi[s]` ≥ 0, one per scenario; `var_value_at_risk_threshold_mu` is a single
  free variable. **Both are created only when `n_scenarios>1 AND lambda>0`**
  ([`create-variables.sql:562-566, 595-598`](src/sql/create-variables.sql)). `μ` is a *decision
  variable*, not an input — there is no `value_at_risk_threshold_mu` column in the data; `main.jl`
  reads its solved value from `var_value_at_risk_threshold_mu`.

### Per-scenario total cost (`expressions/conditional_value_at_risk.jl:34-69`)
```
total_cost_per_scenario[s] = base_cost                              # scenario-independent:
                                 (assets_investment_cost + assets_fixed_* +
                                  storage_energy_inv/fixed + flows_inv/fixed)
                           + flows_operational_cost_per_scenario[s]
                           + vintage_flows_operational_cost_per_scenario[s]
                           + units_on_operational_cost_per_scenario[s]
```

### Per-scenario operational cost (`expressions/operational-costs.jl:53-128`, 265-379)
```julia
cost_coefficient = weight_for_operation_discounts        # discount factor (objectives/create.jl:81-216)
                 * total_weight_per_scenario             # = SUM(rep_periods_mapping.weight)  per (year,rp,scenario)
                 * rep_periods_data.resolution
flow_cost[s] = Σ_flows cost_coefficient * total_variable_cost * (block length) * var_flow
```
- `total_variable_cost = commodity_price / producer_efficiency + operational_cost`
  (`t_objective_flows`, objectives/create.jl:233-234).
- The **scenario probability** multiplies each scenario's op-cost in the expected-cost bracket
  (flows_operational_cost.jl:13). **Representative-period weight** (`rep_periods_mapping.weight`
  summed per scenario) and `resolution` and block length scale a rep period up to the full timeframe.

### RIDM cost parameters
- Fuel/penalty via `commodity_price`: `ccgt=0.05`, `ocgt=0.07`, `ens=180` (electricity VoLL),
  `smr_ccs=250` (H₂ backup), `water_borrower=300` (water-borrow penalty)
  ([`flow-milestone.csv`](base-input-data/RIDM-case-study/flow-milestone.csv)).
- Variable O&M `operational_cost`: `wind=0.002`, `wind_offshore=0.004`, `hydro→spillage=0.001`.
- `producer_efficiency = 1.0` on all flows; `units_on_cost` empty → `unit_on_cost = 0`.
- Investment/fixed costs from [`asset-commission.csv`](base-input-data/RIDM-case-study/asset-commission.csv)
  (`investment_cost` 250–2852, `fixed_cost` 7–25, annualized with `discount_rate=0.05`,
  `economic_lifetime=20`; salvage handled in objectives/create.jl:185-204).

---

## 7. Scenario & representative-period representation

**Source (project):** [`main.jl`](main.jl); helpers in `utils/` and `ScenarioReduction/src/`.
**Source (TEM):** scenario probabilities in `stochastic_scenario`; rep periods in
`rep_periods_data` / `rep_periods_mapping` / `profiles_rep_periods` (written by TulipaClustering).
**Docs:** `#stochastic-concept`, `#representative-periods`, `#cvar-concept`.

### Scenarios
- `config.toml`: `number_of_scenarios = 2`.
- `main.jl:58-64` reads `create-scenarios/profiles-wide-all-scenarios.csv`, selects
  `number_of_scenarios` via `get_scenario_set`, **renumbers** them `1..N`, and writes
  `profiles-wide.csv`.
- `main.jl:66-70` writes `stochastic-scenario.csv` with **uniform probability `1/N`**
  (here `0.5, 0.5`). This becomes the `stochastic_scenario(scenario, probability)` table that drives
  the expected-cost weighting (§6) and the CVaR sum.
- `main.jl:131-139, 259-267` overwrites `model_parameters.risk_aversion_weight_lambda` and
  `risk_aversion_confidence_level_alpha` from the config (`λ=0.1`, `α=0.95`) — these **override** the
  values shipped in [`model-parameters.csv`](base-input-data/RIDM-case-study/model-parameters.csv)
  (which has `λ=0.1, α=0.90`).

### Representative periods → profiles
1. `TC.transform_wide_to_long!` melts `profiles_wide`
   (`solar, wind_offshore, wind_onshore, demand, hydro_inflow`) into long
   `profiles(milestone_year, scenario, timestep, profile_name, value)`.
2. Clustering (`dummy_cluster!` for the benchmark; `cluster!` with `rp∈{30,60,90}` for case studies)
   writes `profiles_rep_periods`, `rep_periods_mapping(period, rep_period, weight)`,
   `rep_periods_data(rep_period, num_timesteps, resolution)`.
3. Two stochastic structures are used (`main.jl:292-363`):
   - **`:per_scenario`** — `cols_to_groupby=[:milestone_year, :scenario]` → each scenario gets its
     own rep periods (block-diagonal `rep_periods_mapping.scenario`).
   - **`:cross_scenario`** — `cols_to_crossby=[:scenario]` → rep periods shared across scenarios.
4. Indexing: operational variables/constraints are indexed by `(…, milestone_year, rep_period,
   time_block)`; the scenario dimension enters via `rep_periods_mapping.scenario` (op-cost weights,
   §6) and via the inter-period storage `scenario` index (§3c). `profiles.rep_period` and
   `profiles.inter_period` are the in-memory profile dicts consumed by `_profile_aggregate`.

### `ScenarioReduction` (context)
`dominance.jl` builds the model **once** (with `dummy_cluster!`, single rep period),
then Sobol/Gaussian-samples investment vectors, **fixes `assets_investment`** to each sample, and
re-solves with dual-simplex warm-start — a screening loop over investment candidates.

---

## 8. Units — what the constraints actually use, and reconciling the README

**The claim:** [`README.md`](README.md) states *Power: GW, Energy: GWh, Cost: MEUR, Time: hours*.

**The data:**
| Quantity | Source | Magnitude |
|---|---|---|
| demand profile | `profiles-wide.csv` `demand` | **17,184 – 36,428** (mean 26,372), 8760 h × 2 scen |
| `peak_demand` (e_demand) | `asset-milestone.csv` | **1.5** |
| `peak_demand` (h2_demand) | `asset-milestone.csv` | 0.1 (no profile → constant) |
| asset `capacity` | `asset.csv` | **0.05 – 2.0** |
| `capacity_storage_energy` | `asset.csv` | battery 0.1, h2_storage 16.8, hydro 403.2 |
| `investment_cost` | `asset-commission.csv` | 250 – 2852 |
| `fixed_cost` | `asset-commission.csv` | 7 – 25 |
| `commodity_price` (VoLL `ens`) | `flow-milestone.csv` | 180 |

**What the constraints do with units.** Every operational constraint is *unit-agnostic*: it only
requires the quantities it relates to share one unit.
- Consumer balance (§1): `Σflows = demand_agg × peak_demand`. With `demand_agg ≈ 26,000` and
  `peak_demand = 1.5`, the **effective electricity demand is ≈ 26,000 × 1.5 ≈ 39,000 per hour** (and
  the *split* between profile and `peak_demand` is arbitrary — only the product enters).
- Capacity limit (§2): `flow ≤ capacity × units`. So flows live on the **same scale as
  `capacity × assets_investment`**.
- Cost (§6): `investment_cost × capacity × v_inv`.

**The reconciliation / the inconsistency.**
- The **cost side is internally consistent with GW/MEUR**: `investment_cost ≈ 10³` reads as
  **MEUR/GW = €/kW** (e.g. ccgt 1100 ≈ €1100/kW, a realistic CCGT capex), `fixed_cost` ≈ 7–25
  MEUR/GW/yr, `capacity ≈ 0.05–2 GW/unit`, energy capacities (hydro 403.2 GWh) plausible.
- The **demand profile is *not* on that GW scale.** 26,000 "GW" of demand is ~26 TW — physically
  impossible for one system. It is consistent with **MW** (17–36 GW peak). So there is a **~1000×
  unit mismatch between the demand profile (MW) and the capacity/cost basis (GW)**, amplified a
  further 1.5× by `peak_demand`.
- **Why it still solves and what it means.** Because the consumer balance ties `flow` to the
  (MW-scale) demand, and the capacity limit ties `flow` to `capacity × units`, the LP simply chooses
  `units` so that `capacity(≈1) × units ≈ 39,000` — i.e. it invests ~10⁴–10⁵ "units." The objective
  then multiplies by `investment_cost`, so the **absolute objective value is in mixed units and is
  ~10³× inflated relative to a clean GW/MEUR reading**. Nothing about the *structure* breaks; the
  solution's **dispatch shares, scenario ordering, loss-of-load counts, μ, and relative comparisons
  (scenario↔scenario, rp↔rp) are all meaningful** — only the *absolute* capacities and €-objective
  should not be read literally in the README's units.
- **To make the README literally true**, the demand profile would need to be divided by 1000 (or
  `peak_demand` set to ~1.5e-3) so that demand lands on the GW scale of `capacity`.

> Bottom line: the constraints use a single self-consistent linear unit; the *cost/capacity* inputs
> are genuinely GW/MEUR, but the *demand profile* is in MW, so the README's "GW everywhere" does not
> hold for demand. The mismatch is harmless to optimality/feasibility but inflates absolute
> magnitudes by ~10³.

---

## Appendix A — Constraint → file:line → DuckDB table quick index

| Constraint (HEAD name) | Builder file:line | Constraint table | Key inputs |
|---|---|---|---|
| Consumer balance | `consumer.jl:42-47` | `cons_balance_consumer` | `asset_milestone.peak_demand`, `assets_profiles('demand')` |
| Max output flow | `capacity.jl:265-290` | `cons_capacity_outgoing_aggregated_vintage_method` | `asset.capacity`, `assets_profiles('availability')`, `assets_investment` |
| Max input flow | `capacity.jl:317-340` | `cons_capacity_incoming_aggregated_vintage_method` | same (storage charging) |
| Storage balance (rep) | `storage.jl:57-98` | `cons_balance_storage_rep_period` | `initial_storage_level`, `storage_inflows`, η, `assets_profiles('inflows')` |
| Storage max/min level (rep) | `storage.jl:108-158` | same | `expr_available_energy_capacity_*` |
| Accumulated intra (seasonal) | `storage.jl:298-325` | `cons_accumulated_storage_intra_period` | inflows, η |
| Storage balance (inter/seasonal) | `storage.jl:186-216` | `cons_balance_storage_inter_period` | `timeframe_data`, `assets_timeframe_profiles` |
| Conversion balance | `conversion.jl:17-21` | `cons_balance_conversion` | `asset_commission.conversion_efficiency` |
| CVaR tail-excess | `conditional_value_at_risk.jl:42-49` | `cons_scenario_tail_excess` | `var_tail_excess_slack_xi`, `var_value_at_risk_threshold_mu`, `total_cost_per_scenario` |

## Appendix B — Project-specific quirks (cross-cutting)

- **`is_seasonal=false` forced in benchmark** after `dummy_cluster!` (`main.jl:154`) — single
  rep period makes seasonal pointless (§3).
- **`use_ratio`** (`config.toml` = `false` here) — when `true`, `main.jl:269-380` divides VRE/inflow
  profiles by `demand` for clustering and multiplies them back afterward (a clustering-weighting
  trick, net no-op on values; relies on there being a single demand node). **Off in the shipped config.**
- **Safety-valve assets:** `ens→e_demand` (electricity VoLL 180), `smr_ccs→h2_demand` (H₂ backup 250),
  `water_borrower→hydro_reservoir` (water-borrow penalty 300, `capacity_coefficient=0` → uncapped),
  `spillage` (`>=` sink). `main.jl` errors if `water_borrowed > 0` in a case study (a feasibility guard).
- **`fix_level_storage=true`** — when benchmarking, seasonal storage levels of `hydro_reservoir`
  and `h2_storage` are fixed from the clustered solution before re-solving (`main.jl:434-453`).
- **α discrepancy:** `model-parameters.csv` has `α=0.90` but `config.toml`/`main.jl` overwrite it to
  `0.95`; `λ=0.1` either way. CVaR is therefore active (λ>0 and 2 scenarios).
- **Vintage/UC features unused:** all RIDM assets are single-vintage (`commission_year=2030`),
  `unit_commitment=false`, `ramping=false`, no investable flows, no investment limits/groups → the
  compact-vintage, UC, ramping, transport, and investment-group constraint families are not generated.

## Appendix C — Sources

- TEM source: this clone, HEAD `88d7accd` (v0.21.0); pinned rev `227a80f7` (CVaR PR #1575 branch tip).
- Official formulation:
  <https://tulipaenergy.github.io/TulipaEnergyModel.jl/stable/40-scientific-foundation/40-formulation/>
  (rendered from `docs/src/40-scientific-foundation/40-formulation.md` in the clone).
- Concepts (rep periods, timeframe, seasonal storage, two-stage stochastic, CVaR):
  <https://tulipaenergy.github.io/TulipaEnergyModel.jl/stable/30-concepts/>
  (`docs/src/30-concepts.md`).
- TulipaClustering `dummy_cluster!`/`cluster!`: `TulipaClustering.jl-main/src/convenience.jl`.
