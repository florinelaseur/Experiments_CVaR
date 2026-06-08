# Append infeasibility / non-optimal solve diagnostics as JSONL (stochastic-dominance screening).
# Requires JuMP, JSON, and collect_infeasibility_conflict from utils/infeasibility_conflict.jl in scope.

"""
    append_infeasibility_conflict_record!(path, model; scenario, sample_mw, sequence, sample_id, ...)

Append one JSONL record for a non-OPTIMAL per-scenario LP solve during SD screening.

Top-level fields: `scenario`, `sample`, `model_status`, `conflict` (IIS when infeasible).
Context: `source`, `sequence`, `sample_id`.
"""
function append_infeasibility_conflict_record!(
    path::AbstractString,
    model::JuMP.Model;
    scenario::Integer,
    sample_mw::AbstractVector{<:Real},
    sequence::Integer,
    sample_id::Integer,
    max_conflict_items::Int=200,
)
    status = JuMP.termination_status(model)
    primal = JuMP.primal_status(model)

    conflict = if status in (JuMP.INFEASIBLE, JuMP.INFEASIBLE_OR_UNBOUNDED)
        collect_infeasibility_conflict(
            model;
            max_constraints=max_conflict_items,
            max_bounds=max_conflict_items,
        )
    else
        nothing
    end

    record = Dict{String,Any}(
        "scenario" => scenario,
        "sample" => sample_vector_by_asset(sample_mw),
        "model_status" => Dict(
            "termination_status" => string(status),
            "primal_status" => string(primal),
        ),
        "conflict" => conflict,
        "source" => "stochastic_dominance",
        "sequence" => sequence,
        "sample_id" => sample_id,
    )

    dir = dirname(path)
    !isempty(dir) && mkpath(dir)

    open(path, "a") do io
        println(io, JSON.json(record))
    end
    return nothing
end
