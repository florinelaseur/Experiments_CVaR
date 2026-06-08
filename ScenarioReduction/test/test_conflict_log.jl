@testitem "collect_infeasibility_conflict on tiny infeasible LP" tags=[:conflict_log] setup=[
    ConflictLogSetup,
] begin
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    JuMP.@variable(model, x >= 0)
    JuMP.@constraint(model, c, x <= -1)
    JuMP.optimize!(model)
    JuMP.termination_status(model) == JuMP.INFEASIBLE ||
        @test_skip "HiGHS did not report INFEASIBLE for trivial model"

    data = collect_infeasibility_conflict(model; max_constraints=50, max_bounds=50)
    data === nothing && @test_skip "Solver did not return an IIS (compute_conflict unsupported or failed)"
    @test data["counts"]["n_constraints"] >= 1
    @test !isempty(data["constraints"])
end

@testitem "append_infeasibility_conflict_record writes one JSONL line" tags=[:conflict_log] setup=[
    ConflictLogSetup,
] begin
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    JuMP.@variable(model, x >= 0)
    JuMP.@constraint(model, c, x <= -1)
    JuMP.optimize!(model)
    JuMP.termination_status(model) == JuMP.INFEASIBLE ||
        @test_skip "HiGHS did not report INFEASIBLE for trivial model"

    path = tempname() * ".jsonl"
    try
        append_infeasibility_conflict_record!(
            path,
            model;
            scenario=129,
            sample_mw=zeros(length(INVESTABLE_ASSETS)),
            sequence=1,
            sample_id=42,
            max_conflict_items=50,
        )
        lines = readlines(path)
        @test length(lines) == 1
        rec = JSON.parse(lines[1]; dicttype=Dict{String,Any})
        @test rec["scenario"] == 129
        @test rec["sequence"] == 1
        @test rec["sample_id"] == 42
        @test rec["source"] == "stochastic_dominance"
        @test rec["model_status"]["termination_status"] == "INFEASIBLE"
        @test haskey(rec, "sample")
        if rec["conflict"] !== nothing
            @test rec["conflict"]["counts"]["n_constraints"] >= 1
        end
    finally
        isfile(path) && rm(path; force=true)
    end
end
