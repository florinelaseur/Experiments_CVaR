# Pure, solver-free helpers for the hybrid dominance + Kantorovich driver
# (test_dominance_kantorovich.jl). Kept separate from the driver so the
# deterministic logic (id mapping, resume detection, results readback,
# optimality-gap arithmetic, investment cross-check) is unit-testable without
# the TEM/Gurobi/DuckDB stack.
#
# Needs CSV and DataFrame from the includer's scope (the driver imports both;
# the test setup module does too). No top-level package use, so this file
# includes cleanly in a solver-free test environment.

ids_to_str(ids) = join(ids, " ")

# Map LOCAL scenario ids (1..N after renumbering) back to this run's SOURCE ids.
map_local_to_source(local_ids, source_ids_sorted) =
    Int[Int(source_ids_sorted[i]) for i in local_ids]

# A solve stage is complete (resumable-skippable) when every solver wrote results.csv.
stage_complete(stage_dir, solvers) =
    all(isfile(joinpath(stage_dir, string(s), "results.csv")) for s in solvers)

# Read (objective_value, termination_status) from a stage's one-row results.csv.
# Returns (NaN, "MISSING") when the file is absent or empty so callers stay NaN-safe.
function read_objective_status(path)
    isfile(path) || return (NaN, "MISSING")
    df = CSV.read(path, DataFrame)
    isempty(df) && return (NaN, "MISSING")
    obj = hasproperty(df, :objective_value) ? Float64(df.objective_value[1]) : NaN
    status = hasproperty(df, :termination_status) ? String(df.termination_status[1]) : "UNKNOWN"
    return (obj, status)
end

# Out-of-sample optimality gap (percent):
#   OG(%) = (F(z^reduced, full) − F(z^full, full)) / F(z^full, full) × 100
# NaN when either objective is NaN (skipped/infeasible stage) or the benchmark is 0.
function optimality_gap_percent(F_fixed::Real, F_bench::Real)
    (isnan(F_fixed) || isnan(F_bench) || F_bench == 0) && return NaN
    return (F_fixed - F_bench) / F_bench * 100
end

# Cross-check that two investment MW vectors (INVESTABLE_ASSETS order) agree, e.g.
# the reduced solve's stored investment vs. what the fixed-investment full solve
# reports back. Returns (match::Bool, max_abs_diff::Float64).
function investment_mw_matches(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}; tol::Real=1e-3)
    length(a) == length(b) || return (match=false, max_abs_diff=Inf)
    isempty(a) && return (match=true, max_abs_diff=0.0)
    d = maximum(abs.(Float64.(a) .- Float64.(b)))
    return (match=d <= tol, max_abs_diff=d)
end
