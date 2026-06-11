using TOML: TOML, parsefile

struct ScenarioReductionConfig
    repo_root::String
    script_dir::String
    input_data_path::String
    profiles_wide_source::String
    output_dir::String
    conflict_log_path::String
    number_of_scenarios::Int
    lambda::Float64
    alpha::Float64
    solvers::Vector{Symbol}
    random_seed::Int
    run_filter::Bool
    run_solve::Bool
    run_verify_mapping::Bool
    verify_mapping_tol::Float64
    sd_solver::Symbol
    sampling_mode::Symbol
    num_samples::Int
    number_of_samples_sequences::Int
    use_adequacy_cuts::Bool
    mean_shift::Bool
    gaussian_shrinkage::Float64
    gaussian_nonneg_mode::Symbol
    record_conflicts::Bool
    max_runtime_sec::Union{Nothing,Float64}
    solve_time_limit_sec::Union{Nothing,Float64}
    dominance_method::Symbol
    dominance_pick_k::Int
    representative_periods::Int
    hours_per_period::Int
    clustering_method::Symbol
    clustering_distance::Symbol
    weight_type::Symbol
    clustering_learning_rate::Float64
    clustering_niters::Int
    weight_learning_rate::Float64
    weight_niters::Int
    use_names::Bool
    tulipa_energy_model_rev::Union{Nothing,String}
    experiment_enabled::Bool
    experiment_name::String
    experiment_base_dir::String
    experiment_num_runs::Int
    experiment_base_seed::Int
    experiment_num_input_scenarios::Int
    experiment_run_ground_truth::Bool
    experiment_run_selected_scenarios::Bool
    hybrid_enabled::Bool
    hybrid_name::String
    hybrid_base_dir::String
    hybrid_num_runs::Int
    hybrid_base_seed::Int
    hybrid_num_input_scenarios::Int
    hybrid_kantorovich_pick_n::Int
    hybrid_run_full_benchmark::Bool
    hybrid_run_fixed_full::Bool
end

function _optional_positive_seconds(x)::Union{Nothing,Float64}
    x == 0 && return nothing
    return Float64(x)
end

"""
    load_config(; config_path=nothing, script_dir=..., repo_root=...)

Load `ScenarioReduction/config.toml` into a `ScenarioReductionConfig`.
"""
function load_config(;
    config_path::Union{Nothing,String}=nothing,
    script_dir::String=joinpath(@__DIR__, ".."),
    repo_root::String=joinpath(script_dir, ".."),
)
    path = something(config_path, joinpath(script_dir, "config.toml"))
    cfg = parsefile(path)

    paths = cfg["paths"]
    sim = cfg["simulation"]
    run = cfg["run"]
    # [dominance] is the current section name; [stochastic_dominance] is the legacy alias.
    sd = if haskey(cfg, "dominance")
        cfg["dominance"]
    elseif haskey(cfg, "stochastic_dominance")
        cfg["stochastic_dominance"]
    else
        error("config.toml: missing [dominance] section (legacy alias: [stochastic_dominance])")
    end
    solve = cfg["solve"]
    project = get(cfg, "project", Dict{String,Any}())
    experiment = get(cfg, "experiment", Dict{String,Any}())
    hybrid = get(cfg, "hybrid_experiment", Dict{String,Any}())

    dominance_method = Symbol(get(sd, "dominance_method", "pointwise"))
    dominance_method in (:pointwise, :fsd, :ssd) ||
        error("config.toml: dominance_method must be \"pointwise\", \"fsd\", or \"ssd\"; got \"$dominance_method\"")

    output_dir = joinpath(script_dir, paths["output_dir"])

    experiment_base_dir = joinpath(
        script_dir, get(experiment, "output_dir", "outputs/experiments"),
    )

    hybrid_base_dir = joinpath(
        script_dir, get(hybrid, "output_dir", "outputs/experiments"),
    )

    return ScenarioReductionConfig(
        repo_root,
        script_dir,
        joinpath(repo_root, paths["input_data"]),
        joinpath(repo_root, paths["profiles_wide_source"]),
        output_dir,
        joinpath(output_dir, paths["conflict_log"]),
        Int(sim["number_of_scenarios"]),
        Float64(sim["risk_aversion_weight_lambda"]),
        Float64(sim["risk_aversion_confidence_level"]),
        [Symbol(s) for s in sim["solvers"]],
        Int(sim["random_seed"]),
        Bool(run["filter"]),
        Bool(run["solve"]),
        Bool(run["verify_mapping"]),
        Float64(run["verify_mapping_tol"]),
        Symbol(sd["solver"]),
        Symbol(sd["sampling_mode"]),
        Int(sd["num_samples"]),
        Int(sd["number_of_samples_sequences"]),
        Bool(sd["use_adequacy_cuts"]),
        Bool(sd["mean_shift"]),
        Float64(sd["gaussian_shrinkage"]),
        Symbol(sd["gaussian_nonneg_mode"]),
        Bool(sd["record_conflicts"]),
        _optional_positive_seconds(sd["max_runtime_sec"]),
        _optional_positive_seconds(sd["solve_time_limit_sec"]),
        dominance_method,
        Int(get(sd, "dominance_pick_k", 2)),
        Int(solve["representative_periods"]),
        Int(solve["hours_per_period"]),
        Symbol(solve["clustering_method"]),
        Symbol(solve["clustering_distance"]),
        Symbol(solve["weight_type"]),
        Float64(solve["clustering_learning_rate"]),
        Int(solve["clustering_niters"]),
        Float64(solve["weight_learning_rate"]),
        Int(solve["weight_niters"]),
        Bool(get(solve, "use_names", true)),
        get(project, "tulipa_energy_model_rev", nothing),
        Bool(get(experiment, "enabled", false)),
        String(get(experiment, "name", "sd_experiment")),
        experiment_base_dir,
        Int(get(experiment, "num_runs", 1)),
        Int(get(experiment, "base_seed", Int(sim["random_seed"]))),
        Int(get(experiment, "num_input_scenarios", Int(sim["number_of_scenarios"]))),
        Bool(get(experiment, "run_ground_truth", true)),
        Bool(get(experiment, "run_selected_scenarios", true)),
        Bool(get(hybrid, "enabled", false)),
        String(get(hybrid, "name", "dom_kant")),
        hybrid_base_dir,
        Int(get(hybrid, "num_runs", 1)),
        Int(get(hybrid, "base_seed", Int(sim["random_seed"]))),
        Int(get(hybrid, "num_input_scenarios", Int(sim["number_of_scenarios"]))),
        Int(get(hybrid, "kantorovich_pick_n", 2)),
        Bool(get(hybrid, "run_full_benchmark", true)),
        Bool(get(hybrid, "run_fixed_full", true)),
    )
end

"""
    copy_config_with(cfg; kwargs...)

Return a new `ScenarioReductionConfig` identical to `cfg` except for the fields
named in `kwargs`. Used to give each experiment run its own `output_dir`,
`conflict_log_path`, `number_of_scenarios`, and `random_seed`.
"""
function copy_config_with(cfg::ScenarioReductionConfig; kwargs...)
    overrides = Dict{Symbol,Any}(kwargs)
    field_names = fieldnames(ScenarioReductionConfig)
    unknown = setdiff(keys(overrides), field_names)
    isempty(unknown) ||
        error("copy_config_with: unknown field(s) $(collect(unknown))")
    values = [get(overrides, name, getfield(cfg, name)) for name in field_names]
    return ScenarioReductionConfig(values...)
end

"""
    save_config_toml(cfg, path)

Serialize the resolved per-run configuration to `path` as TOML.
"""
function save_config_toml(cfg::ScenarioReductionConfig, path::AbstractString)
    mkpath(dirname(path))
    data = Dict{String,Any}(
        "paths" => Dict(
            "input_data_path" => cfg.input_data_path,
            "profiles_wide_source" => cfg.profiles_wide_source,
            "output_dir" => cfg.output_dir,
            "conflict_log_path" => cfg.conflict_log_path,
        ),
        "simulation" => Dict(
            "number_of_scenarios" => cfg.number_of_scenarios,
            "risk_aversion_weight_lambda" => cfg.lambda,
            "risk_aversion_confidence_level" => cfg.alpha,
            "solvers" => String.(string.(cfg.solvers)),
            "random_seed" => cfg.random_seed,
        ),
        "run" => Dict(
            "filter" => cfg.run_filter,
            "solve" => cfg.run_solve,
            "verify_mapping" => cfg.run_verify_mapping,
            "verify_mapping_tol" => cfg.verify_mapping_tol,
        ),
        "dominance" => Dict(
            "solver" => string(cfg.sd_solver),
            "sampling_mode" => string(cfg.sampling_mode),
            "num_samples" => cfg.num_samples,
            "number_of_samples_sequences" => cfg.number_of_samples_sequences,
            "use_adequacy_cuts" => cfg.use_adequacy_cuts,
            "mean_shift" => cfg.mean_shift,
            "gaussian_shrinkage" => cfg.gaussian_shrinkage,
            "gaussian_nonneg_mode" => string(cfg.gaussian_nonneg_mode),
            "record_conflicts" => cfg.record_conflicts,
            "max_runtime_sec" => cfg.max_runtime_sec === nothing ? 0.0 : cfg.max_runtime_sec,
            "solve_time_limit_sec" => cfg.solve_time_limit_sec === nothing ? 0.0 : cfg.solve_time_limit_sec,
            "dominance_method" => string(cfg.dominance_method),
            "dominance_pick_k" => cfg.dominance_pick_k,
        ),
        "solve" => Dict(
            "representative_periods" => cfg.representative_periods,
            "hours_per_period" => cfg.hours_per_period,
            "clustering_method" => string(cfg.clustering_method),
            "clustering_distance" => string(cfg.clustering_distance),
            "weight_type" => string(cfg.weight_type),
            "clustering_learning_rate" => cfg.clustering_learning_rate,
            "clustering_niters" => cfg.clustering_niters,
            "weight_learning_rate" => cfg.weight_learning_rate,
            "weight_niters" => cfg.weight_niters,
            "use_names" => cfg.use_names,
        ),
        "experiment" => Dict(
            "enabled" => cfg.experiment_enabled,
            "name" => cfg.experiment_name,
            "output_dir" => cfg.experiment_base_dir,
            "num_runs" => cfg.experiment_num_runs,
            "base_seed" => cfg.experiment_base_seed,
            "num_input_scenarios" => cfg.experiment_num_input_scenarios,
            "run_ground_truth" => cfg.experiment_run_ground_truth,
            "run_selected_scenarios" => cfg.experiment_run_selected_scenarios,
        ),
        "hybrid_experiment" => Dict(
            "enabled" => cfg.hybrid_enabled,
            "name" => cfg.hybrid_name,
            "output_dir" => cfg.hybrid_base_dir,
            "num_runs" => cfg.hybrid_num_runs,
            "base_seed" => cfg.hybrid_base_seed,
            "num_input_scenarios" => cfg.hybrid_num_input_scenarios,
            "kantorovich_pick_n" => cfg.hybrid_kantorovich_pick_n,
            "run_full_benchmark" => cfg.hybrid_run_full_benchmark,
            "run_fixed_full" => cfg.hybrid_run_fixed_full,
        ),
    )
    open(path, "w") do io
        TOML.print(io, data)
    end
    return path
end
