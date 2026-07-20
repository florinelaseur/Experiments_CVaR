# Marker and color mappings for plots
const MARKER_MAP = Dict(
    "per_scenario" => :utriangle,
    "cross_scenario" => :circle,
    #"per_and_cross_scenario" => :star5
)

const COLOR_MAP_weight = Dict(
    "dirac" => :red,
    "convex" => :black,
    # "conical" => :green,
    "conical_bounded" => :yellow,
)
const COLOR_MAP_method = Dict(
    "convex_hull" => :black,
    "convex_hull_with_null" => :yellow,
    # "conical_hull" => :green
)

const FILLER_MAP = Dict(
    "dirac" => :white,
    "convex" => :black,
    "conical" => :green,
    "conical_bounded" => :yellow
)

const COLOR_MAP_method_weight_stmethod = Dict(
    "k_means_convex_per_scenario" => :gold,
    "k_means_convex_cross_scenario" => :orange,
    "k_means_dirac_per_scenario" => :gold,
    "k_means_dirac_cross_scenario" => :orange,
    "k_medoids_convex_per_scenario" => :blue,
    "k_medoids_convex_cross_scenario" => :purple,
    "k_medoids_dirac_per_scenario" => :blue,
    "k_medoids_dirac_cross_scenario" => :purple,
)

const COLOR_MAP_method_weight = Dict(
    "k_means_convex" => :gold,
    "k_means_dirac" => :gold,
    "k_medoids_convex" => :blue,
    "k_medoids_dirac" => :blue,
)
const LEGEND_MAP = Dict(
    "k_means_convex_per_scenario" => "K-means: per-scenario",
    "k_means_convex_cross_scenario" => "K-means: cross-scenario",
    "k_means_dirac_per_scenario" => "K-means: dirac weights, per-scenario",
    "k_means_dirac_cross_scenario" => "K-means: dirac weights, cross-scenario",
    "k_medoids_convex_per_scenario" => "K-medoids: per-scenario",
    "k_medoids_convex_cross_scenario" => "K-medoids: cross-scenario",
    "k_medoids_dirac_per_scenario" => "K-medoids: dirac weights, per-scenario",
    "k_medoids_dirac_cross_scenario" => "K-medoids: dirac weights, cross-scenario",
)

const VALUE_MAP = Dict(
    "rel_regret" => "Relative regret",
    "num_loss_of_load_e_demand" => "Number of timesteps with electricity loss of load",
    "num_loss_of_load_h2_demand" => "Number of timesteps with hydrogen loss of load",
    "loss_of_load_e_demand" => "Total electricity loss of load",
    "loss_of_load_h2_demand" => "Total hydrogen loss of load",
    "total_lol" => "Total amount of loss of load",
    "total_steps_loss_of_load" => "Total number of timesteps with loss of load",
    "time_to_cluster" => "Time to cluster (s)",
    "time_to_create" => "Time to create (s)",
    "time_to_solve" => "Time to solve (s)",
    "total_time" => "Total time (s)",
)
const LEGEND_METHOD_MAP = Dict(
    "convex_hull" => "Convex hull",
    "convex_hull_with_null" => "Bounded conical hull",
    "conical_hull" => "Conical hull",
)