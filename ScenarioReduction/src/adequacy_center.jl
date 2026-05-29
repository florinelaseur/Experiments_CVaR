# Feasibility mean-shift LP for SD screening — kept separate from adequacy_cuts.jl
# because JuMP *macros* expand at include time and so require JuMP in scope. This
# file is included by src/utils.jl (whose callers `using JuMP`), not by the
# solver-free unit-test setup. The cut types/functions it uses come from
# adequacy_cuts.jl, included just before this file.

"""
    feasibility_center(cuts_list, mean_mw, ub; optimizer, optimizer_parameters)

Minimal upward shift from `mean_mw` into the adequacy polytope (satisfy every
scenario's cuts within `[0, ub]`):

    min Σ_k shift_k
    s.t.  A_s · x ≥ b_s   ∀ scenario s
          x = mean_mw + shift,  shift ≥ 0,  x ≤ ub

Returns `(status, center, shift)`. If the LP is infeasible, no in-bounds portfolio
can meet the cuts — the scenario set / `ub` is the real constraint — and
`center == nothing`.
"""
function feasibility_center(
    cuts_list,
    mean_mw::AbstractVector{<:Real},
    ub::AbstractVector{<:Real};
    optimizer,
    optimizer_parameters=Dict(),
)
    n = length(mean_mw)
    model = JuMP.Model(optimizer)
    for pair in optimizer_parameters
        JuMP.set_optimizer_attribute(model, first(pair), last(pair))
    end
    JuMP.set_silent(model)

    JuMP.@variable(model, x[1:n] >= 0)
    JuMP.@variable(model, shift[1:n] >= 0)
    JuMP.@constraint(model, [k = 1:n], x[k] == mean_mw[k] + shift[k])
    JuMP.@constraint(model, [k = 1:n], x[k] <= ub[k])
    for cuts in cuts_list
        JuMP.@constraint(model, cuts.A * x .>= cuts.b)
    end
    JuMP.@objective(model, Min, sum(shift))
    JuMP.optimize!(model)

    status = JuMP.termination_status(model)
    if status != JuMP.OPTIMAL
        return (status=status, center=nothing, shift=nothing)
    end
    return (status=status, center=JuMP.value.(x), shift=JuMP.value.(shift))
end
