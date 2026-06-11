# Irreducible Infeasible Subsystem (IIS) collection for infeasible/non-optimal solves.
#
# Used by:
#   - ScenarioReduction/src/conflict_log.jl (collect_infeasibility_conflict -> JSONL record)
#   - ScenarioReduction/src/utils.jl::solve_model (print_infeasibility_conflict!)
#
# References JuMP (and JuMP.MOI) from the includer's scope; no other dependencies.
# All solver IIS calls are guarded: if the solver cannot compute a conflict, the
# collectors return `nothing` rather than throwing.

"""Readable label for a conflicting constraint/bound: its name if set, else its string form."""
function _conflict_item_string(ref)
    nm = try
        JuMP.name(ref)
    catch
        ""
    end
    (nm isa AbstractString && !isempty(nm)) && return nm
    return string(ref)
end

"""Set of constraint refs that are variable bounds (lower/upper/fixed) from `@variable`."""
function _variable_bound_refs(model::JuMP.Model)
    refs = Set{Any}()
    for v in JuMP.all_variables(model)
        JuMP.has_lower_bound(v) && push!(refs, JuMP.LowerBoundRef(v))
        JuMP.has_upper_bound(v) && push!(refs, JuMP.UpperBoundRef(v))
        JuMP.is_fixed(v) && push!(refs, JuMP.FixRef(v))
    end
    return refs
end

"""
    collect_infeasibility_conflict(model; max_constraints=200, max_bounds=200)

Compute an IIS (via `JuMP.compute_conflict!`) and return a JSON-serializable Dict:

    "counts"      => Dict("n_constraints"=>Int, "n_bounds"=>Int)
    "constraints" => Vector{String}   (named/general constraints in the conflict)
    "bounds"      => Vector{String}   (variable bounds in the conflict)

Returns `nothing` if the solver cannot compute a conflict or none is found. Variable
bounds set via `@variable` are reported under `bounds`; everything else (named
`@constraint`s, affine/quadratic/vector constraints) under `constraints`.
"""
function collect_infeasibility_conflict(
    model::JuMP.Model; max_constraints::Int=200, max_bounds::Int=200,
)
    try
        JuMP.compute_conflict!(model)
    catch err
        @debug "compute_conflict! failed" err
        return nothing
    end

    conflict_status = try
        JuMP.MOI.get(model, JuMP.MOI.ConflictStatus())
    catch err
        @debug "ConflictStatus unavailable" err
        return nothing
    end
    conflict_status == JuMP.MOI.CONFLICT_FOUND || return nothing

    bound_refs = _variable_bound_refs(model)

    constraints = String[]
    bounds = String[]
    n_constraints = 0
    n_bounds = 0

    for (F, S) in JuMP.list_of_constraint_types(model)
        for con in JuMP.all_constraints(model, F, S)
            st = try
                JuMP.MOI.get(model, JuMP.MOI.ConstraintConflictStatus(), con)
            catch
                continue
            end
            st == JuMP.MOI.IN_CONFLICT || continue

            if con in bound_refs
                n_bounds += 1
                length(bounds) < max_bounds && push!(bounds, _conflict_item_string(con))
            else
                n_constraints += 1
                length(constraints) < max_constraints &&
                    push!(constraints, _conflict_item_string(con))
            end
        end
    end

    return Dict{String,Any}(
        "counts" => Dict{String,Any}(
            "n_constraints" => n_constraints, "n_bounds" => n_bounds,
        ),
        "constraints" => constraints,
        "bounds" => bounds,
    )
end

"""
    print_infeasibility_conflict!(model)

Compute and print the IIS for an infeasible `model`. Returns the conflict Dict, or
`nothing` if no conflict could be computed.
"""
function print_infeasibility_conflict!(model::JuMP.Model)
    data = collect_infeasibility_conflict(model)
    if data === nothing
        @warn "No IIS available (compute_conflict! unsupported or no conflict found)"
        return nothing
    end
    @info "Infeasibility conflict (IIS)" n_constraints = data["counts"]["n_constraints"] n_bounds =
        data["counts"]["n_bounds"]
    for c in data["constraints"]
        println("  constraint: ", c)
    end
    for b in data["bounds"]
        println("  bound: ", b)
    end
    return data
end
